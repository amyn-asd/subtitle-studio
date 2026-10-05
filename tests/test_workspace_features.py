import json
import sqlite3

import pytest
from fastapi.testclient import TestClient

from subtitle_studio.app import create_app
from subtitle_studio.filesystem import list_files
from subtitle_studio.storage import Store
from subtitle_studio.subtitles import group_words
from subtitle_studio.translation import source_signature, utterances


class FakeJobs:
    def create(self, pid, kind, settings=None, tracks=None, **extra):
        return {"id": "test-job", "project_id": pid, "kind": kind, "status": "queued", "settings": settings, "tracks": tracks, **extra}

    def close(self):
        pass


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    import subtitle_studio.app as module
    monkeypatch.setattr(module, "model_ready", lambda key: True)
    store = Store(tmp_path / "workspace.sqlite")
    source = tmp_path / "clip-日本語.mp4"
    source.write_bytes(b"fixture")
    media = {"path": str(source), "name": source.name, "fingerprint": "fixture", "duration": 120,
             "audio_tracks": [{"stream_index": 1, "audio_ordinal": 0, "languages": [], "detection": "sampled"}], "subtitle_tracks": []}
    pid = store.create_project(media)["id"]
    client = TestClient(create_app(store, FakeJobs(), "test-token"))
    return store, pid, client, {"X-Studio-Token": "test-token"}


def test_file_browser_filters_unicode_search_pagination_and_folder_mode(tmp_path):
    (tmp_path / "子フォルダ").mkdir()
    (tmp_path / "not-a-video.txt").write_text("private notes")
    for name in ["زبان.MP4", "日本語.mkv", "recording.OPUS"]:
        (tmp_path / name).write_bytes(b"media")
    listing = list_files(str(tmp_path), limit=2)
    assert listing["total"] == 4 and len(listing["entries"]) == 2
    assert listing["entries"][0]["kind"] == "folder"
    assert len(list_files(str(tmp_path), offset=2, limit=2)["entries"]) == 2
    assert [e["name"] for e in list_files(str(tmp_path), query="日本")["entries"]] == ["日本語.mkv"]
    assert [e["name"] for e in list_files(str(tmp_path), kind="folder")["entries"]] == ["子フォルダ"]
    assert list_files(str(tmp_path / "日本語.mkv"))["path"] == str(tmp_path.resolve())
    with pytest.raises(ValueError, match="does not exist"):
        list_files(str(tmp_path / "missing"))


def test_file_browser_handles_permission_errors(tmp_path, monkeypatch):
    import subtitle_studio.filesystem as module
    def denied(*args):
        raise PermissionError("denied")
    monkeypatch.setattr(module.os, "scandir", denied)
    with pytest.raises(ValueError, match="not readable"):
        list_files(str(tmp_path))


def test_file_api_requires_local_authorization_and_valid_parameters(workspace, tmp_path):
    _, _, client, headers = workspace
    assert client.get("/api/files", params={"path": str(tmp_path)}).status_code == 403
    assert client.get("/api/files", headers={**headers, "Origin": "https://untrusted.example"}).status_code == 403
    result = client.get("/api/files", headers=headers, params={"path": str(tmp_path)}).json()
    assert result["entries"][0]["name"] == "clip-日本語.mp4"
    assert client.get("/api/files", headers=headers, params={"offset": -1}).status_code == 422
    assert client.get("/api/files", headers=headers, params={"kind": "execute"}).status_code == 422


def seed_transcript(store, pid):
    words = [{"word": " Hello", "start": 0, "end": .7, "language": "en"},
             {"word": " again again.", "start": .8, "end": 1.8, "language": "en"},
             {"word": " سلام.", "start": 7, "end": 8, "language": "fa"},
             {"word": " こんにちは。", "start": 9, "end": 10, "language": "ja"}]
    cues = group_words(pid, 1, words)
    store.replace_cues(pid, 1, cues)
    for group in utterances(cues):
        for cue in group:
            store.save_translation(pid, "de", cue, "Wieder wieder." if cue["language"] == "en" else "Hallo.",
                                   source_signature=source_signature(group, "de", "test"), model_digest="test")
    return cues


def test_full_transcript_keeps_all_languages_repetition_and_word_order(workspace):
    store, pid, client, headers = workspace
    cues = seed_transcript(store, pid)
    result = client.get(f"/api/projects/{pid}/transcript", headers=headers, params={"track": 1, "language": "de"})
    assert result.status_code == 200
    data = result.json()
    assert data["cue_count"] == len(cues) and data["translated_count"] == len(cues)
    assert [b["text"] for b in data["blocks"]] == ["Hello again again.", "سلام.", "こんにちは。"]
    assert client.get(f"/api/projects/{pid}/transcript", params={"track": 1}).status_code == 403
    assert client.get(f"/api/projects/{pid}/transcript", headers=headers, params={"track": 99}).status_code == 400
    original = client.get(f"/api/projects/{pid}/transcript/download", headers=headers, params={"track": 1, "view": "original"})
    assert original.text == "Hello again again.\n\nسلام.\n\nこんにちは。\n"
    assert "filename*=UTF-8''" in original.headers["content-disposition"]
    translated = client.get(f"/api/projects/{pid}/transcript/download", headers=headers, params={"track": 1, "language": "de", "view": "translated"})
    assert translated.text.startswith("Wieder wieder.") and "سلام" not in translated.text
    parallel = client.get(f"/api/projects/{pid}/transcript/download", headers=headers, params={"track": 1, "language": "de", "view": "parallel"})
    assert "Original (Persian)\nسلام." in parallel.text and "German\nHallo." in parallel.text


def test_stale_or_missing_translation_is_visible_and_cannot_export_as_complete(workspace):
    store, pid, client, headers = workspace
    cues = seed_transcript(store, pid)
    store.update_cue(pid, cues[0]["id"], {"text": "A human correction.", "edited": True})
    data = client.get(f"/api/projects/{pid}/transcript", headers=headers, params={"track": 1, "language": "de"}).json()
    assert data["translated_count"] < data["cue_count"] and data["blocks"][0]["translation"] is None
    assert data["blocks"][0]["parts"][0]["translation"] is None
    result = client.get(f"/api/projects/{pid}/transcript/download", headers=headers, params={"track": 1, "language": "de", "view": "parallel"})
    assert result.status_code == 400 and "incomplete or stale" in result.json()["detail"]
    assert client.get(f"/api/projects/{pid}/transcript/download", headers=headers, params={"track": 1}).status_code == 200
