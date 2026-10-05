import json
import sqlite3
import threading
from types import SimpleNamespace

import numpy as np

from subtitle_studio.types import Settings
from subtitle_studio.storage import Store
from subtitle_studio.subtitles import group_words
from test_workspace_features import FakeJobs, workspace


def test_turbo_default_needs_only_selected_recognizer(workspace, monkeypatch):
    import subtitle_studio.app as app
    _, pid, client, headers = workspace
    monkeypatch.setattr(app, "model_ready", lambda key: key == "turbo")
    monkeypatch.setattr(app.ollama, "start", lambda: (_ for _ in ()).throw(AssertionError("Transcription must not start a text model")))
    response = client.post(f"/api/projects/{pid}/jobs", headers=headers, json={"tracks": [1]})
    assert response.status_code == 200
    assert response.json()["settings"]["preset"] == "fast"
    assert response.json()["settings"]["enhance_audio"] is True
    disabled = client.post(f"/api/projects/{pid}/jobs", headers=headers,
        json={"tracks": [1], "settings": {"enhance_audio": False}})
    assert disabled.status_code == 200 and disabled.json()["settings"]["enhance_audio"] is False
    fallback = client.post(f"/api/projects/{pid}/jobs", headers=headers,
        json={"tracks": [1], "settings": {"preset": "accurate"}})
    assert fallback.status_code == 400 and "whisper" in fallback.json()["detail"]
    monkeypatch.setattr(app, "model_ready", lambda key: key == "whisper")
    fallback = client.post(f"/api/projects/{pid}/jobs", headers=headers,
        json={"tracks": [1], "settings": {"preset": "accurate"}})
    assert fallback.status_code == 200


def test_language_scan_accepts_turbo_without_large(workspace, monkeypatch):
    import subtitle_studio.app as app
    store, pid, client, headers = workspace
    scheduled = []
    create = FakeJobs.create
    def record(self, pid, kind, *args, **kwargs):
        scheduled.append((pid, kind))
        return create(self, pid, kind, *args, **kwargs)
    monkeypatch.setattr(FakeJobs, "create", record)
    monkeypatch.setattr(app, "model_ready", lambda key: key == "turbo")
    assert client.post(f"/api/projects/{pid}/scan", headers=headers).status_code == 200
    media = store.project(pid)["media"]
    media["audio_tracks"][0]["detection"] = "pending"
    store.update_media(pid, media)
    monkeypatch.setattr(app, "probe", lambda path: media)
    response = client.post("/api/projects", headers=headers, json={"path": media["path"]})
    assert response.status_code == 200
    assert scheduled == [(pid, "scan"), (pid, "scan")]
    manager = app.Jobs(store)
    try:
        requests = []
        manager.worker = lambda job, stage, manifest, callback: requests.append((stage, manifest))
        monkeypatch.setattr("subtitle_studio.jobs.model_ready", lambda key: key == "turbo")
        manager.controls["scan"] = threading.Event()
        manager._scan({"id": "scan", "project_id": pid, "kind": "scan"})
        assert requests[0][0] == "scan" and requests[0][1]["settings"]["preset"] == "fast"
    finally:
        manager.pool.shutdown()


def test_existing_database_keeps_words_edits_and_translations(tmp_path):
    path = tmp_path / "legacy.sqlite"
    store = Store(path)
    pid = store.create_project({"path": "clip.mp4", "fingerprint": "fixture"})["id"]
    cue = group_words(pid, 1, [{"word": " Original.", "start": 1, "end": 2, "language": "en"}])[0]
    # Simulate metadata written by an older release, without changing its words.
    cue.update(candidates=[{"id": "old", "text": "Original."}], decision={"selected": "old"},
        verification="Recognizers agree on spoken words", flags=["AI selected alternative", "Fast speech"])
    store.replace_cues(pid, 1, [cue])
    store.update_cue(pid, cue["id"], {"text": "Human correction.", "edited": True, "reviewed": True})
    store.save_translation(pid, "de", store.cues(pid)[0], "Korrektur.")
    store.save_job({"id": "old", "project_id": pid, "kind": "transcribe", "status": "paused",
        "settings": {"preset": "fast", "audio_profile": "original", "debate": True, "review_context": "Old notes"}})
    with sqlite3.connect(path) as db:
        db.execute("ALTER TABLE projects ADD COLUMN review_context TEXT NOT NULL DEFAULT ''")
        db.execute("PRAGMA user_version=0")
    migrated = Store(path)
    result = migrated.cues(pid)[0]
    assert result["raw_text"] == "Original." and result["text"] == "Human correction."
    assert result["reviewed"] and "decision" not in result and "candidates" not in result
    assert "verification" not in result and result["flags"] == ["Previously revised wording; check audio", "Fast speech"]
    assert migrated.translations(pid, "de")[cue["id"]]["text"] == "Korrektur."
    assert "review_context" not in migrated.project(pid)
    assert migrated.job("old")["settings"] == Settings(enhance_audio=False).model_dump()


def test_pipeline_uses_one_recognizer_and_separates_audio_and_model_caches(workspace, monkeypatch):
    from subtitle_studio import jobs as module
    store, pid, _, _ = workspace
    manager = module.Jobs(store)
    manager.controls["j"] = threading.Event()
    monkeypatch.setattr(module, "fingerprint", lambda path: "fixture")
    monkeypatch.setattr(module, "model_revision", lambda key: key)
    monkeypatch.setattr(module.ollama, "start", lambda: (_ for _ in ()).throw(AssertionError("No text model in transcription")))
    calls = []
    def worker(job, stage, manifest, callback):
        calls.append((stage, manifest))
        callback({"type": "chunk", "chunk": {"track": 1, "words": [
            {"word": " Hello.", "start": 1, "end": 2, "language": "en"}]},
            "progress": 1, "message": "Done"})
    manager.worker = worker
    try:
        for settings in [Settings(), Settings(enhance_audio=False), Settings(preset="accurate")]:
            manager._transcribe({"id": "j", "project_id": pid, "kind": "transcribe", "tracks": [1],
                "settings": settings.model_dump()})
        assert [stage for stage, _ in calls] == ["primary"] * 3
        assert len({manifest["cache"] for _, manifest in calls}) == 3
        assert store.cues(pid)[0]["text"] == store.cues(pid)[0]["raw_text"] == "Hello."
    finally:
        manager.pool.shutdown()


def test_all_audio_prepared_before_model_loading_and_no_time_shift(tmp_path, monkeypatch):
    from subtitle_studio import worker
    order, events = [], []
    def prepare(path, track, start, duration, destination, profile):
        order.append(("prepare", track, profile))
        destination.parent.mkdir(parents=True, exist_ok=True)
        np.zeros(round(duration * 16000), dtype="<f4").tofile(destination)
        return np.memmap(destination, dtype="<f4", mode="r")
    class Pipeline:
        def transcribe(self, audio, **options):
            segment = SimpleNamespace(start=.2, end=.8, text=" Hello.", avg_logprob=-.1,
                compression_ratio=1, no_speech_prob=0,
                words=[SimpleNamespace(word=" Hello.", start=.2, end=.8, probability=.95)])
            return iter([segment]), SimpleNamespace(language="en", language_probability=1)
    def load(settings):
        order.append(("load", settings["preset"]))
        return object(), Pipeline()
    monkeypatch.setattr(worker, "prepare_track", prepare)
    monkeypatch.setattr(worker, "whisper_model", load)
    monkeypatch.setattr(worker, "Detector", lambda: object())
    monkeypatch.setattr(worker, "recover_speech", lambda *args, **kwargs: ([], []))
    monkeypatch.setattr(worker, "emit", lambda kind, **data: events.append({"type": kind, **data}))
    for enabled in (True, False):
        order.clear(); events.clear()
        worker.primary({"media": {"path": "unused", "duration": 50}, "tracks": [1, 2],
            "settings": Settings(language="en", enhance_audio=enabled, start_seconds=10, limit_seconds=4).model_dump(),
            "cache": str(tmp_path / str(enabled))})
        assert order == [("prepare", 1, "level" if enabled else "original"),
                         ("prepare", 2, "level" if enabled else "original"), ("load", "fast")]
        tokens = [w for event in events if event["type"] == "chunk" for w in event["chunk"]["words"]]
        assert all(w["start"] == 10.2 and w["end"] == 10.8 for w in tokens)
        assert [event["stage"] for event in events if event["type"] == "progress"][:3] == ["preparing_audio", "preparing_audio", "transcribing"]
