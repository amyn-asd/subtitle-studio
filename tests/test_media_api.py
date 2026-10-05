import json
import shutil
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from subtitle_studio.app import create_app
from subtitle_studio.media import probe, audio_window, preview, run, fingerprint
from subtitle_studio.storage import Store
from subtitle_studio.subtitles import group_words


class FakeJobs:
    def __init__(self):
        self.created = []
    def create(self, pid, kind, settings=None, tracks=None, **kwargs):
        job = {"id":"fake","project_id":pid,"kind":kind,"status":"queued","settings":settings,"tracks":tracks,**kwargs}
        self.created.append(job)
        return job
    def close(self):
        pass


@pytest.fixture(scope="module")
def multitrack(tmp_path_factory):
    if not shutil.which("ffmpeg"):
        pytest.skip("FFmpeg not installed")
    path = tmp_path_factory.mktemp("media") / "زبان-日本語.mkv"
    run([shutil.which("ffmpeg"), "-nostdin", "-v", "error", "-y",
         "-f", "lavfi", "-i", "color=c=black:s=160x90:r=10:d=5",
         "-f", "lavfi", "-i", "sine=frequency=440:duration=5",
         "-itsoffset", "1.2", "-f", "lavfi", "-i", "sine=frequency=880:duration=3.8",
         "-map", "0:v", "-map", "1:a", "-map", "2:a", "-c:v", "libx264", "-bf", "0", "-pix_fmt", "yuv420p",
         "-c:a", "aac", "-metadata:s:a:0", "language=eng", "-metadata:s:a:1", "language=fas", str(path)])
    return path


def peak_frequency(audio):
    return np.argmax(np.abs(np.fft.rfft(audio))) * 16000 / len(audio)


def test_audio_track_index_and_timeline_are_preserved(multitrack):
    media = probe(str(multitrack))
    assert [(t["stream_index"],t["audio_ordinal"]) for t in media["audio_tracks"]] == [(1,0),(2,1)]
    delayed = audio_window(str(multitrack), 2, 0, 5)
    assert np.max(np.abs(delayed[:16000])) < .005
    assert abs(peak_frequency(delayed[24000:40000]) - 880) < 8
    sought = audio_window(str(multitrack), 2, 1.5, 1)
    assert abs(peak_frequency(sought) - 880) < 8


def test_preview_uses_selected_audio_without_mutating_source(multitrack, tmp_path):
    before = fingerprint(multitrack)
    clip = preview(str(multitrack), 2, 1.5, 1, tmp_path / "review.mp4")
    selected = audio_window(str(clip), 1, 0, 1)
    assert abs(peak_frequency(selected) - 880) < 10
    assert fingerprint(multitrack) == before


def test_api_auth_selection_edits_export_and_translation_staleness(multitrack, tmp_path, monkeypatch):
    import subtitle_studio.app as module
    monkeypatch.setattr(module, "model_ready", lambda key: True)
    store = Store(tmp_path / "api.sqlite")
    manager = FakeJobs()
    client = TestClient(create_app(store, manager, "test-token"))
    headers = {"X-Studio-Token":"test-token"}
    assert client.get("/api/projects").status_code == 403
    assert client.get("/", headers={"Host":"untrusted.example"}).status_code == 400
    assert client.post("/api/projects", json={"path":str(multitrack)}, headers={**headers,"Origin":"https://other.example"}).status_code == 403
    project = client.post("/api/projects", json={"path":str(multitrack)}, headers=headers).json()
    pid = project["id"]
    assert len(project["media"]["audio_tracks"]) == 2
    assert client.post(f"/api/projects/{pid}/jobs", headers=headers, json={"tracks":[0],"settings":{}}).status_code == 400
    assert client.post(f"/api/projects/{pid}/jobs", headers=headers, json={"tracks":[2],"settings":{}}).status_code == 200
    cue = group_words(pid, 2, [{"word":" سلام.","start":1.3,"end":2.2,"language":"fa","flags":[],"chunk":0}])[0]
    store.replace_cues(pid,2,[cue])
    store.save_translation(pid,"en",cue,"Hello.")
    result = client.post(f"/api/projects/{pid}/export",headers=headers,json={"track":2,"directory":str(tmp_path)}).json()
    assert ".audio-2.original.srt" in result["path"]
    assert "سلام" in Path(result["path"]).read_text(encoding="utf-8")
    edited = client.patch(f"/api/projects/{pid}/cues/{cue['id']}",headers=headers,json={"text":"سلام دوباره."})
    assert edited.status_code == 200 and edited.json()["raw_text"] == "سلام."
    stale = client.post(f"/api/projects/{pid}/export",headers=headers,json={"track":2,"language":"en","directory":str(tmp_path)})
    assert stale.status_code == 400 and "stale" in stale.json()["detail"]
    assert client.patch(f"/api/projects/{pid}/cues/{cue['id']}",headers=headers,json={"end":1}).status_code == 400


def test_modified_source_is_rejected_before_preview_or_play(multitrack, tmp_path, monkeypatch):
    import subtitle_studio.app as module
    monkeypatch.setattr(module, "model_ready", lambda key:False)
    source = tmp_path / "copy.mkv"
    shutil.copyfile(multitrack, source)
    store = Store(tmp_path / "changed.sqlite")
    client = TestClient(create_app(store, FakeJobs(), "token"))
    headers = {"X-Studio-Token":"token"}
    pid = client.post("/api/projects",headers=headers,json={"path":str(source)}).json()["id"]
    with source.open("ab") as file:
        file.write(b"changed")
    for operation in ("preview", "play"):
        result = client.post(f"/api/projects/{pid}/{operation}",headers=headers,json={"track":1})
        assert result.status_code == 400 and "changed" in result.json()["detail"]
