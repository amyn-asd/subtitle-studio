from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .config import DATA, ROOT, child_environment
from .debate import choose, apply_choice
from .media import NO_WINDOW, fingerprint, probe
from .embedded import extraction, remux_plan, remux_arguments, unchanged
from .subtitle_formats import cue_source
from .models import HF_MODELS, OLLAMA_MODELS, download_hf, model_ready, model_revision, ollama
from .storage import Store
from .subtitles import atomic_text, group_words, same_spoken_words, parse_srt
from .translation import translate
from .worker import QWEN_LANGUAGES
from .types import TRANSLATION_VERSION


class Jobs:
    def __init__(self, store: Store):
        self.store = store
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="studio-gpu")
        self.controls = {}
        self.processes = {}
        self.lock = threading.Lock()
        self.store.recover_jobs()

    def create(self, pid: str, kind: str, settings: dict | None = None, tracks: list[int] | None = None, **extra) -> dict:
        job = {"id": uuid.uuid4().hex, "project_id": pid, "kind": kind, "settings": settings or {}, "tracks": tracks or [],
               "status": "queued", "stage": "queued", "progress": 0, "message": "Waiting for the processing slot", "warnings": [],
               "created": time.time(), "elapsed": 0, **extra}
        self.store.save_job(job)
        self.controls[job["id"]] = threading.Event()
        self.pool.submit(self._run, job)
        return job

    def update(self, job, **changes):
        job.update(changes)
        if job.get("started"):
            job["elapsed"] = time.time() - job["started"]
        self.store.save_job(job)

    def pause(self, jid: str, cancel=False):
        job = self.store.job(jid)
        if job["status"] not in ("running", "queued", "pausing"):
            return job
        self.controls.setdefault(jid, threading.Event()).set()
        process = self.processes.get(jid)
        if process and process.poll() is None:
            process.terminate()
        job["stop_as"] = "cancelled" if cancel else "paused"
        self.store.save_job(job)
        return job

    def resume(self, jid):
        job = self.store.job(jid)
        if job["status"] not in ("paused", "failed", "cancelled"):
            raise ValueError("Only an interrupted job can be resumed")
        job.pop("error", None)
        job.pop("stop_as", None)
        self.update(job, status="queued", message="Resuming from saved chunks")
        self.controls[jid] = threading.Event()
        self.pool.submit(self._run, job)
        return job

    def stopped(self, job):
        return self.controls[job["id"]].is_set()

    def check_stop(self, job):
        if self.stopped(job):
            raise InterruptedError("Paused")

    def worker(self, job: dict, stage: str, manifest: dict, callback):
        self.check_stop(job)
        file = DATA / "projects" / job["project_id"] / f"{job['id']}-{stage}.json"
        atomic_text(file, json.dumps(manifest, ensure_ascii=False))
        log_path = DATA / "logs" / f"{job['id']}-{stage}.log"
        with log_path.open("w", encoding="utf-8") as log:
            process = subprocess.Popen([sys.executable, "-m", "subtitle_studio.worker", "--stage", stage, "--manifest", str(file)],
                                        cwd=ROOT, env=child_environment(), stdout=subprocess.PIPE, stderr=log,
                                        text=True, encoding="utf-8", creationflags=NO_WINDOW)
            self.processes[job["id"]] = process
            try:
                for line in process.stdout:
                    if self.stopped(job):
                        process.terminate()
                        break
                    try:
                        event = json.loads(line)
                    except json.JSONDecodeError:
                        log.write(line)
                        continue
                    if event["type"] == "warning":
                        job["warnings"] = list(dict.fromkeys(job["warnings"] + [event["message"]]))
                    else:
                        callback(event)
                code = process.wait()
            finally:
                if process.poll() is None:
                    process.terminate()
                    process.wait(timeout=20)
                self.processes.pop(job["id"], None)
        self.check_stop(job)
        if code:
            message = log_path.read_text(encoding="utf-8", errors="replace")[-1800:]
            raise RuntimeError(f"{stage} failed. {message}")

    def _run(self, job):
        started = time.time()
        try:
            self.check_stop(job)
            self.update(job, status="running", started=started)
            if job["kind"] == "setup":
                self._setup(job)
            elif job["kind"] == "scan":
                self._scan(job)
            elif job["kind"] == "transcribe":
                self._transcribe(job)
            elif job["kind"] == "translate":
                self._translate(job)
            elif job["kind"] in ("import_subtitles", "extract_subtitles"):
                self._subtitles(job)
            elif job["kind"] == "remux":
                self._remux(job)
            else:
                raise ValueError("Unknown processing job")
            if not (job["kind"] == "remux" and job.get("result", {}).get("path")):
                self.check_stop(job)
            self.update(job, status="done", stage="complete", progress=1, message="Ready", finished=time.time())
        except InterruptedError:
            saved = self.store.job(job["id"])
            self.update(job, status=saved.get("stop_as", "paused"), message="Saved progress. Resume when ready.")
        except Exception as exc:
            self.update(job, status="failed", message="Processing stopped; completed work is saved", error=str(exc))
        finally:
            if job["kind"] in ("transcribe", "translate"):
                ollama.unload(OLLAMA_MODELS["context"])
                ollama.unload(OLLAMA_MODELS["translation"])

    def _setup(self, job):
        keys = job["model_ids"]
        for index, key in enumerate(keys):
            self.check_stop(job)
            def report(message, progress):
                self.check_stop(job)
                self.update(job, stage="downloading", message=message, progress=(index + min(1, progress)) / len(keys))
            if key in HF_MODELS:
                download_hf(key, report)
            elif key in OLLAMA_MODELS:
                ollama.pull(key, report)

    def _scan(self, job):
        project = self.store.project(job["project_id"])
        media = project["media"]
        def event(item):
            if item["type"] == "track":
                media["audio_tracks"] = [item["track"] if t["stream_index"] == item["track"]["stream_index"] else t for t in media["audio_tracks"]]
                self.store.update_media(project["id"], media)
            elif item["type"] == "progress":
                self.update(job, progress=item["progress"], message=item["message"])
        self.update(job, stage="detecting_languages", message="Checking speech languages across all audio tracks")
        self.worker(job, "scan", {"media": media, "settings": {"preset": "accurate"}}, event)

    def _transcribe(self, job):
        pid, settings = job["project_id"], job["settings"]
        media = self.store.project(pid)["media"]
        if fingerprint(Path(media["path"])) != media["fingerprint"]:
            raise ValueError("The source file has changed. Select it again to create a new project.")
        signature = hashlib.sha256(json.dumps({"fingerprint": media["fingerprint"], "settings": settings,
                                      "revision": model_revision("turbo" if settings["preset"] == "fast" else "whisper"),
                                      "lid_revision": model_revision("lid"), "qwen_revision": model_revision("qwen_asr"), "pipeline": 4}, sort_keys=True).encode()).hexdigest()[:16]
        cache = DATA / "projects" / pid / "runs" / signature
        all_words = {track: [] for track in job["tracks"]}
        def primary_event(item):
            if item["type"] != "chunk":
                return
            chunk = item["chunk"]
            track = chunk["track"]
            all_words[track].extend(chunk["words"])
            self.store.replace_cues(pid, track, group_words(pid, track, all_words[track]))
            self.update(job, progress=item["progress"] * .65, message=item["message"])
        manifest = {"project_id": pid, "media": media, "settings": settings, "tracks": job["tracks"], "cache": str(cache)}
        self.update(job, stage="transcribing", message="Loading the recognition model")
        self.worker(job, "primary", manifest, primary_event)
        # Each selected track retains a speech-language timeline for the processed range.
        for track in media["audio_tracks"]:
            index = track["stream_index"]
            if index in all_words:
                timeline = []
                for file in sorted(cache.glob(f"track-{index}-at-*.json")):
                    timeline.extend(json.loads(file.read_text(encoding="utf-8"))["languages"])
                track["language_timeline"] = timeline
                codes = sorted({w["language"] for w in all_words[index]})
                track["languages"] = [{"code": code} for code in codes]
                track["detection"] = "processed_range" if settings.get("limit_seconds") or settings.get("start_seconds") else "full_track"
        self.store.update_media(pid, media)
        if settings.get("recheck"):
            self._recheck(job, manifest)
        if settings.get("debate"):
            self._debate(job, cache)
        changed = [cue for cue in self.store.cues(pid) if cue["track"] in job["tracks"] and "Word timing needs alignment" in cue["flags"] and not cue.get("edited")]
        if changed:
            self.update(job, stage="aligning", message="Aligning selected alternatives with the audio", progress=.96)
            def alignment_event(item):
                if item["type"] == "alignment" and item["words"]:
                    cue = next(c for c in changed if c["id"] == item["cue_id"])
                    self.store.update_cue(pid, cue["id"], {"words": item["words"], "flags": [f for f in cue["flags"] if f != "Word timing needs alignment"]},
                                          automated=True, expected_text=cue["text"])
            try:
                self.worker(job, "align", {"media": media, "cues": changed}, alignment_event)
            except InterruptedError:
                raise
            except Exception as exc:
                job["warnings"].append(f"Word alignment unavailable; cue-level timing retained: {str(exc)[:200]}")

    def _recheck(self, job, manifest):
        pid = job["project_id"]
        cases = []
        for track in job["tracks"]:
            cues = self.store.cues(pid, track)
            for index, cue in enumerate(cues):
                if cue["flags"] and not cue.get("edited"):
                    cases.append({"cue": cue, "neighbors": cues[max(0, index - 3):index] + cues[index + 1:index + 4]})
        checked = 0
        def event(item):
            nonlocal checked
            if item["type"] != "recheck":
                return
            result = item["result"]
            cue = next(c["cue"] for c in cases if c["cue"]["id"] == result["cue_id"])
            if result["text"] and not same_spoken_words(result["text"], cue["raw_text"]):
                cue["candidates"].append({"id": result["candidate_id"], "text": result["text"], "engine": result["engine"],
                                           "language": result.get("language", cue["language"]),
                                           "source": "Same original audio, tight subtitle interval with neighboring text context"})
                cue["flags"] = sorted(set(cue["flags"] + ["Recognizer disagreement"]))
            else:
                cue["verification"] = "Recognizers agree on spoken words" if result["text"] else "Recheck returned no words"
                if result["text"] and result.get("language") and not job["settings"].get("language"):
                    cue["language"] = result["language"]
            self.store.update_cue(pid, cue["id"], {k: cue[k] for k in ("candidates", "flags", "verification", "language") if k in cue},
                                  automated=True, expected_text=cue["text"])
            checked += 1
            self.update(job, progress=.65 + checked / max(1, len(cases)) * .18, message=f"Checked uncertain speech {checked}/{len(cases)}")
        for use_whisper in (False, True):
            subset = [c for c in cases if (c["cue"]["language"] not in QWEN_LANGUAGES and not any(h in QWEN_LANGUAGES for h in c["cue"].get("language_hints", []))) == use_whisper]
            if subset:
                self.update(job, stage="checking", message=f"Rechecking {len(subset)} uncertain passages")
                self.worker(job, "recheck_whisper" if use_whisper else "recheck", {**manifest, "cases": subset}, event)

    def _debate(self, job, cache):
        pid = job["project_id"]
        cases = []
        for track in job["tracks"]:
            cues = self.store.cues(pid, track)
            for index, cue in enumerate(cues):
                if len(cue["candidates"]) > 1 and not cue.get("edited"):
                    cases.append((cue, cues[max(0, index - 4):index] + cues[index + 1:index + 5]))
        model = OLLAMA_MODELS["context"]
        model_digest = next((m.get("digest") for m in ollama.tags() if m["name"] == model), "unknown")
        try:
            for index, (cue, neighbors) in enumerate(cases):
                self.check_stop(job)
                self.update(job, stage="discussing", message=f"Two-agent review {index + 1}/{len(cases)}", progress=.83 + index / max(1, len(cases)) * .12)
                signature = hashlib.sha256(json.dumps({"cue": cue, "neighbors": neighbors, "protocol": 2, "model_digest": model_digest}, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:16]
                file = cache / "debates" / f"{cue['id']}-{signature}.json"
                decision = json.loads(file.read_text(encoding="utf-8")) if file.exists() else choose(cue, neighbors, ollama.chat, model)
                atomic_text(file, json.dumps(decision, ensure_ascii=False))
                revised = apply_choice(cue, decision)
                self.store.update_cue(pid, cue["id"], {k: revised[k] for k in ("text", "words", "flags", "decision", "reviewed", "language")},
                                      automated=True, expected_text=cue["text"])
        finally:
            ollama.unload(model)

    def _translate(self, job):
        target = job["target_language"]
        model = OLLAMA_MODELS["translation"]
        model_digest = next((m.get("digest") for m in ollama.tags() if m["name"] == model), "unknown")
        for index, track in enumerate(job["tracks"]):
            self.check_stop(job)
            cues = self.store.cues(job["project_id"], track)
            existing = self.store.translations(job["project_id"], target)
            existing = {cid: t for cid, t in existing.items() if t.get("version") == TRANSLATION_VERSION and t.get("model_digest") == model_digest}
            # Translation uses the complete track for context, while skipping valid saved outputs on resume.
            def save(cue, text, source_signature):
                self.store.save_translation(job["project_id"], target, cue, text, model_digest=model_digest, source_signature=source_signature)
            self.update(job, stage="translating", message="Loading the translation model")
            translate(cues, target, ollama, save,
                      lambda progress, message: self.update(job, progress=(index + progress) / len(job["tracks"]), message=message),
                      lambda: self.stopped(job), existing,
                      lambda message: job.update(warnings=list(dict.fromkeys(job["warnings"] + [message]))), model_digest)

    def media_command(self, job, args, duration=0):
        """Cancellable FFmpeg work; progress and stderr stay bounded, including for long movies."""
        self.check_stop(job)
        log_path = DATA / "logs" / f"{job['id']}-media.log"
        with log_path.open("w", encoding="utf-8") as log:
            process = subprocess.Popen(args[:1] + ["-progress", "pipe:1", "-nostats"] + args[1:],
                                       stdout=subprocess.PIPE, stderr=log, text=True, encoding="utf-8", creationflags=NO_WINDOW)
            self.processes[job["id"]] = process
            last = 0
            try:
                for line in process.stdout:
                    self.check_stop(job)
                    if duration and line.startswith("out_time_us=") and time.monotonic() - last > .5:
                        try:
                            progress = min(.96, max(0, int(line.strip().split("=", 1)[1]) / 1_000_000 / duration))
                            self.update(job, progress=progress)
                            last = time.monotonic()
                        except ValueError:
                            pass
                code = process.wait()
            finally:
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
                self.processes.pop(job["id"], None)
        self.check_stop(job)
        if code:
            raise RuntimeError("FFmpeg could not save this format. Try MKV for wider compatibility. " +
                               log_path.read_text(encoding="utf-8", errors="replace")[-1800:])

    def _subtitles(self, job):
        pid = job["project_id"]
        media = self.store.project(pid)["media"]
        unchanged(media)
        folder = DATA / "projects" / pid / "embedded"
        folder.mkdir(parents=True, exist_ok=True)
        results = []
        def extract(args, path):
            partial = path.with_name(f".{path.name}.{job['id']}.partial")
            try:
                self.media_command(job, args[:-1] + [str(partial)])
                unchanged(media)
                self.check_stop(job)
                partial.replace(path)
            finally:
                partial.unlink(missing_ok=True)
        importing = job["kind"] == "import_subtitles"
        for index, track in enumerate(job["tracks"]):
            self.check_stop(job)
            source = cue_source(media, track)
            self.update(job, stage="importing_subtitles" if importing else "extracting_subtitles",
                        message=f"Reading embedded subtitle track {index + 1}/{len(job['tracks'])}", progress=index/len(job["tracks"]))
            args, native = extraction(media, track, folder)
            extract(args, native)
            result = {"track": track, "path": str(native), "kind": source["kind"]}
            if importing:
                args, normalized = extraction(media, track, folder, normalized=True)
                extract(args, normalized)
                language = job.get("languages", {}).get(str(track), job.get("languages", {}).get(track, source["language"]))
                cues = parse_srt(normalized.read_text(encoding="utf-8-sig"), pid, track, language, source["codec"])
                if not cues:
                    raise ValueError("This subtitle track contains no readable text cues")
                unchanged(media)
                self.check_stop(job)
                self.store.replace_cues(pid, track, cues)
                for embedded in media["subtitle_tracks"]:
                    if embedded["cue_track"] == track:
                        embedded["language"] = language
                self.store.update_media(pid, media)
                result["cue_count"] = len(cues)
            results.append(result)
            self.update(job, result={"subtitles": results}, progress=(index+1)/len(job["tracks"]))
        unchanged(media)

    def _remux(self, job):
        plan = remux_plan(self.store, job["project_id"], job["selections"], job["output_path"], job["keep_embedded"])
        output = plan["output"]
        output.parent.mkdir(parents=True, exist_ok=True)
        partial = output.with_name(f".{output.stem}.{job['id']}.partial{output.suffix}")
        folder = DATA / "projects" / job["project_id"] / "mux" / job["id"]
        folder.mkdir(parents=True, exist_ok=True)
        self.update(job, stage="embedding_subtitles", message="Copying video and audio with the selected subtitles",
                    warnings=plan["warnings"])
        try:
            self.media_command(job, remux_arguments(plan, folder, partial), plan["media"]["duration"])
            unchanged(plan["media"])
            self.check_stop(job)
            saved = probe(str(partial))
            if (len(saved["audio_tracks"]) != len(plan["media"]["audio_tracks"]) or
                    len(saved["video_tracks"]) != len(plan["media"]["video_tracks"]) or
                    len(saved["subtitle_tracks"]) != len(plan["entries"])):
                raise RuntimeError("Saved video did not retain the expected tracks; no output was published")
            # A hard link atomically publishes the completed file and refuses to overwrite any existing path.
            try:
                os.link(partial, output)
            except FileExistsError:
                raise ValueError("The output appeared while copying. Choose a new filename; that file was preserved.")
            except OSError:
                if os.name != "nt":
                    raise
                # Windows rename also refuses existing targets, and works on drives without hard links.
                os.rename(partial, output)
            self.update(job, result={"path": str(output), "fingerprint": fingerprint(output),
                                     "subtitle_count": len(plan["entries"]), "size": output.stat().st_size})
        finally:
            partial.unlink(missing_ok=True)

    def close(self):
        for jid in list(self.controls):
            self.controls[jid].set()
        for process in list(self.processes.values()):
            if process.poll() is None:
                process.terminate()
        self.pool.shutdown(wait=False, cancel_futures=True)
        ollama.close()
