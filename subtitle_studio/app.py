from __future__ import annotations

import asyncio
import hashlib
import importlib.metadata
import json
import os
import secrets
import subprocess
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.trustedhost import TrustedHostMiddleware

from . import __version__
from .config import DATA, ROOT, WEB, MODELS, OLLAMA_DIRECTORY, PREFERENCES, binary, initialize, preferences
from .jobs import Jobs
from .media import NO_WINDOW, probe, preview, audio_window, vlc_arguments, fingerprint
from .embedded import extraction, remux_plan, subtitle_choices, unchanged
from .subtitle_formats import cue_source, preview_audio
from .models import discover_existing, model_ready, status as model_status, HF_MODELS, OLLAMA_MODELS, ollama
from .storage import Store
from .subtitles import srt, atomic_text
from .types import ProbeRequest, JobRequest, CueEdit, ExportRequest, PlayRequest, TRANSLATION_VERSION, SubtitleImportRequest, RemuxRequest


def create_app(store: Store | None = None, manager=None, token: str | None = None):
    testing = store is not None
    initialize()
    store = store or Store()
    jobs = manager or Jobs(store)
    token = token or secrets.token_urlsafe(32)

    @asynccontextmanager
    async def lifespan(app):
        discover_existing()
        yield
        jobs.close()

    app = FastAPI(title="Subtitle Studio", version=__version__, lifespan=lifespan)
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=["127.0.0.1", "localhost", "[::1]"] + (["testserver"] if testing else []))
    app.state.store, app.state.jobs, app.state.token = store, jobs, token

    def load_project(pid):
        project = store.project(pid)
        if "subtitle_tracks" not in project["media"]:
            unchanged(project["media"])
            media = probe(project["media"]["path"])
            media["audio_tracks"] = project["media"]["audio_tracks"]
            store.update_media(pid, media)
            project = store.project(pid)
        return project

    def processing(pid):
        return any(j["kind"] in ("transcribe", "translate", "import_subtitles", "remux") and
                   j["status"] in ("queued", "running", "pausing") for j in store.project(pid)["jobs"])

    @app.middleware("http")
    async def local_only(request: Request, call_next):
        if request.url.path.startswith("/api/"):
            supplied = request.headers.get("x-studio-token", request.query_params.get("k", ""))
            if not secrets.compare_digest(supplied, token):
                return JSONResponse({"detail": "Open Subtitle Studio from its launcher."}, status_code=403)
            origin = request.headers.get("origin")
            if origin and origin != str(request.base_url).rstrip("/"):
                return JSONResponse({"detail": "Cross-origin requests are not allowed."}, status_code=403)
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        return response

    @app.exception_handler(ValueError)
    async def value_error(request, exc):
        return JSONResponse({"detail": str(exc)}, status_code=400)

    @app.exception_handler(KeyError)
    async def key_error(request, exc):
        return JSONResponse({"detail": str(exc).strip("'")}, status_code=404)

    @app.exception_handler(FileNotFoundError)
    async def file_error(request, exc):
        return JSONResponse({"detail": str(exc)}, status_code=400)

    @app.get("/health")
    def health():
        return {"app": "Subtitle Studio", "version": __version__}

    @app.get("/api/system")
    def system():
        tools = {}
        for name in ("ffmpeg", "ffprobe", "vlc", "ollama"):
            try:
                tools[name] = binary(name)
            except FileNotFoundError:
                tools[name] = None
        gpu = None
        try:
            result = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,memory.free", "--format=csv,noheader,nounits"],
                                    capture_output=True, text=True, timeout=5, creationflags=NO_WINDOW)
            fields = result.stdout.strip().splitlines()[0].split(",")
            gpu = {"name": fields[0].strip(), "total_mb": int(fields[1]), "free_mb": int(fields[2])}
        except (OSError, IndexError, ValueError, subprocess.TimeoutExpired):
            pass
        try:
            ollama.start()
        except Exception:
            pass
        with store.connect() as db:
            setup_jobs = [json.loads(r[0]) for r in db.execute("SELECT body FROM jobs WHERE project_id='_setup' ORDER BY updated DESC LIMIT 5")]
        return {"version": __version__, "tools": tools, "gpu": gpu, "models": model_status(),
                "models_root": str(MODELS), "ollama_models": str(OLLAMA_DIRECTORY), "setup_jobs": setup_jobs,
                "preferences": preferences}

    @app.post("/api/settings")
    def settings(body: dict):
        allowed = {"models_root", "ollama_models", "local_ai_root", "input_directory", "tools", "model_search_paths"}
        if any(k not in allowed for k in body):
            raise ValueError("Unknown setting")
        for key in ("models_root", "ollama_models", "local_ai_root", "input_directory"):
            if body.get(key) and not Path(body[key]).is_absolute():
                raise ValueError("Folder paths must be absolute")
        if body.get("tools"):
            for key, path in body["tools"].items():
                if key not in ("ffmpeg", "ffprobe", "vlc", "ollama") or (path and not Path(path).is_file()):
                    raise ValueError("Select an existing tool executable")
        updated = {**preferences, **body}
        atomic_text(PREFERENCES, json.dumps(updated, indent=2))
        return {"restart_required": True, "message": "Saved. Close and reopen Subtitle Studio to use the new locations."}

    @app.post("/api/restart")
    def restart():
        with store.connect() as db:
            active = db.execute("SELECT COUNT(*) FROM jobs WHERE json_extract(body,'$.status') IN ('running','queued','pausing')").fetchone()[0]
        if active:
            raise ValueError("Pause or finish current processing before restarting")
        if not hasattr(app.state, "shutdown"):
            raise ValueError("Restart from the launcher")
        app.state.shutdown(True)
        return {"restarting": True}

    @app.post("/api/shutdown")
    def shutdown():
        if not hasattr(app.state, "shutdown"):
            raise ValueError("Close the server process")
        app.state.shutdown(False)
        return {"stopping": True}

    @app.post("/api/models/install")
    def install(body: dict):
        keys = body.get("models", [])
        if not keys or any(k not in HF_MODELS and k not in OLLAMA_MODELS for k in keys):
            raise ValueError("Select at least one available model")
        return jobs.create("_setup", "setup", model_ids=keys)

    @app.get("/api/library")
    def library():
        directory = Path(preferences.get("input_directory", ROOT.parent))
        extensions = {".mp4", ".mkv", ".mov", ".avi", ".webm", ".m4v", ".mp3", ".wav", ".flac"}
        return [{"path": str(p), "name": p.name, "size": p.stat().st_size} for p in directory.iterdir()
                if p.is_file() and p.suffix.lower() in extensions][:100] if directory.exists() else []

    @app.post("/api/browse")
    def browse(body: dict):
        if os.name != "nt":
            raise ValueError("Paste a file path on this operating system.")
        if body.get("kind") == "folder":
            script = "Add-Type -AssemblyName System.Windows.Forms; $d=New-Object System.Windows.Forms.FolderBrowserDialog; if($d.ShowDialog() -eq 'OK') {ConvertTo-Json -Compress $d.SelectedPath} else {'null'}"
        else:
            script = "Add-Type -AssemblyName System.Windows.Forms; $d=New-Object System.Windows.Forms.OpenFileDialog; $d.Filter='Media files|*.mp4;*.mkv;*.mov;*.avi;*.webm;*.m4v;*.wav;*.mp3;*.flac|All files|*.*'; if($d.ShowDialog() -eq 'OK') {ConvertTo-Json -Compress $d.FileName} else {'null'}"
        script = "[Console]::OutputEncoding = New-Object System.Text.UTF8Encoding; " + script
        result = subprocess.run(["powershell.exe", "-NoProfile", "-STA", "-Command", script], capture_output=True,
                                encoding="utf-8", timeout=600, creationflags=NO_WINDOW)
        if result.returncode:
            raise ValueError("The file picker could not open. Paste the path instead.")
        return {"path": json.loads(result.stdout.strip() or "null")}

    @app.get("/api/projects")
    def projects():
        return store.projects()

    @app.post("/api/projects")
    def create_project(body: ProbeRequest):
        project = store.create_project(probe(body.path.strip().strip('"')))
        if project["media"]["audio_tracks"] and model_ready("whisper") and any(t["detection"] == "pending" for t in project["media"]["audio_tracks"]):
            if not any(j["kind"] == "scan" and j["status"] in ("queued", "running") for j in project["jobs"]):
                jobs.create(project["id"], "scan")
                project = store.project(project["id"])
        return project

    @app.get("/api/projects/{pid}")
    def project(pid: str):
        return load_project(pid)

    @app.post("/api/projects/{pid}/subtitles/import")
    def import_subtitles(pid: str, body: SubtitleImportRequest):
        project = load_project(pid)
        unchanged(project["media"])
        tracks = project["media"]["subtitle_tracks"]
        available = {t["cue_track"] for t in tracks if t["kind"] == "text"}
        if not set(body.tracks) <= available or len(set(body.tracks)) != len(body.tracks):
            raise ValueError("Select distinct embedded text subtitles. Image subtitles need OCR before translation")
        if any(key not in body.tracks or value not in LANGUAGE_CODES for key, value in body.languages.items()):
            raise ValueError("Choose a supported source language for each imported track")
        if any(body.languages.get(t["cue_track"], t["language"]) == "und" for t in tracks if t["cue_track"] in body.tracks):
            raise ValueError("Choose the language of the subtitle text before importing")
        if processing(pid):
            raise ValueError("Finish or pause current processing before importing subtitles")
        return jobs.create(pid, "import_subtitles", tracks=body.tracks, languages=body.languages)

    @app.post("/api/projects/{pid}/subtitles/extract")
    def extract_subtitles(pid: str, body: SubtitleImportRequest):
        media = load_project(pid)["media"]
        unchanged(media)
        available = {t["cue_track"] for t in media["subtitle_tracks"]}
        if not set(body.tracks) <= available or len(set(body.tracks)) != len(body.tracks):
            raise ValueError("Select distinct embedded subtitle tracks")
        return jobs.create(pid, "extract_subtitles", tracks=body.tracks)

    @app.get("/api/projects/{pid}/subtitles/file")
    def embedded_file(pid: str, track: int):
        media = load_project(pid)["media"]
        unchanged(media)
        _, path = extraction(media, track, DATA / "projects" / pid / "embedded")
        if not path.exists():
            raise ValueError("Extract this subtitle track first")
        source = Path(media["path"])
        subtitle = cue_source(media, track)
        filename = f"{source.stem}.embedded-{subtitle['subtitle_ordinal'] + 1}.{path.suffix.lstrip('.')}"
        return FileResponse(path, filename=filename, media_type="application/octet-stream")

    @app.get("/api/projects/{pid}/subtitles/options")
    def embedded_options(pid: str):
        project = load_project(pid)
        source = Path(project["media"]["path"])
        output = source.with_name(source.stem + ".subtitled.mkv")
        number = 2
        while output.exists():
            output = source.with_name(f"{source.stem}.subtitled-{number}.mkv")
            number += 1
            if number > 10000:
                raise ValueError("Choose a different output folder; many subtitled copies already exist")
        return {"subtitles": subtitle_choices(store, pid), "suggested_output": str(output)}

    @app.post("/api/projects/{pid}/subtitles/embed")
    def embed_subtitles(pid: str, body: RemuxRequest):
        load_project(pid)
        if processing(pid):
            raise ValueError("Finish or pause current processing before saving a video")
        selections = [s.model_dump() for s in body.subtitles]
        if any(s["language"] != "original" and s["language"] not in TRANSLATION_CODES for s in selections):
            raise ValueError("Choose a supported subtitle translation language")
        plan = remux_plan(store, pid, selections, body.output_path, body.keep_embedded)
        return jobs.create(pid, "remux", selections=selections, output_path=str(plan["output"]), keep_embedded=body.keep_embedded)

    @app.post("/api/jobs/{jid}/play-output")
    def play_output(jid: str):
        job = store.job(jid)
        if job["kind"] != "remux" or job["status"] != "done" or not job.get("result", {}).get("path"):
            raise ValueError("Finish saving a subtitled video first")
        result = job["result"]
        if fingerprint(Path(result["path"])) != result["fingerprint"]:
            raise ValueError("The saved video has changed")
        subprocess.Popen([binary("vlc"), "--no-one-instance", result["path"]], creationflags=NO_WINDOW)
        return {"launched": True, "path": result["path"]}

    @app.post("/api/projects/{pid}/scan")
    def scan(pid: str):
        media = store.project(pid)["media"]
        if not media["audio_tracks"]:
            raise ValueError("This file has no audio tracks")
        if not model_ready("whisper"):
            raise ValueError("Install the transcription model in Models first")
        return jobs.create(pid, "scan")

    @app.post("/api/projects/{pid}/jobs")
    def start(pid: str, body: JobRequest):
        project = store.project(pid)
        available = {t["stream_index"] for t in project["media"]["audio_tracks"]}
        if not set(body.tracks) <= available or len(body.tracks) != len(set(body.tracks)):
            raise ValueError("Select valid, distinct audio tracks")
        settings = body.settings.model_dump()
        if settings["start_seconds"] >= project["media"]["duration"]:
            raise ValueError("The start time must be within the video")
        if settings["language"] and settings["language"] not in LANGUAGE_CODES:
            raise ValueError("Unsupported transcription language")
        required = ["turbo" if settings["preset"] == "fast" else "whisper"]
        if settings["recheck"]:
            required.append("qwen_asr")
        missing = [key for key in required if not model_ready(key)]
        if settings["debate"]:
            ollama.start()
            if not ollama.ready("context"):
                missing.append("context")
        if missing:
            raise ValueError("Install these models first: " + ", ".join(missing))
        if processing(pid):
            raise ValueError("Finish or pause current processing before transcribing this project")
        return jobs.create(pid, "transcribe", settings, body.tracks)

    @app.get("/api/jobs/{jid}")
    def job(jid: str):
        return store.job(jid)

    @app.get("/api/jobs/{jid}/events")
    async def events(jid: str):
        async def generate():
            last = None
            while True:
                current = store.job(jid)
                encoded = json.dumps(current, ensure_ascii=False)
                if encoded != last:
                    yield f"data: {encoded}\n\n"
                    last = encoded
                if current["status"] in ("done", "failed", "paused", "cancelled"):
                    break
                await asyncio.sleep(.7)
        return StreamingResponse(generate(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})

    @app.post("/api/jobs/{jid}/pause")
    def pause(jid: str):
        return jobs.pause(jid)

    @app.post("/api/jobs/{jid}/cancel")
    def cancel(jid: str):
        return jobs.pause(jid, True)

    @app.post("/api/jobs/{jid}/resume")
    def resume(jid: str):
        return jobs.resume(jid)

    @app.get("/api/projects/{pid}/cues")
    def cues(pid: str, track: int | None = None, language: str | None = None):
        items = store.cues(pid, track)
        if language and language != "original":
            from .translation import valid_translations
            translations = valid_translations(store.cues(pid), store.translations(pid, language), language)
            for cue in items:
                result = translations.get(cue["id"], {})
                cue["translated_text"] = result.get("text") if result.get("source_text") == cue["text"] and result.get("version") == TRANSLATION_VERSION else None
        return items

    @app.patch("/api/projects/{pid}/cues/{cid}")
    def edit(pid: str, cid: str, body: CueEdit):
        changes = body.model_dump(exclude_none=True)
        if any(key in changes for key in ("text", "start", "end")):
            changes["edited"] = True
        return store.update_cue(pid, cid, changes)

    @app.post("/api/projects/{pid}/cues/{cid}/candidate")
    def accept_candidate(pid: str, cid: str, body: dict):
        cue = next((c for c in store.cues(pid) if c["id"] == cid), None)
        if not cue:
            raise KeyError("Cue not found")
        selected = next((c for c in cue["candidates"] if c["id"] == body.get("candidate_id")), None)
        if not selected:
            raise ValueError("Unknown candidate")
        return store.update_cue(pid, cid, {"text": selected["text"], "edited": True, "reviewed": True,
                                         "language": selected.get("language", cue["language"]),
                                         "words": cue["words"] if selected["id"] == "primary" else []})

    @app.post("/api/projects/{pid}/translate")
    def translation(pid: str, body: dict):
        target = body.get("target_language")
        tracks = body.get("tracks", [])
        if target not in TRANSLATION_CODES or not tracks or len(set(tracks)) != len(tracks):
            raise ValueError("Choose a supported target language and at least one track")
        if not all(store.cues(pid, t) for t in tracks):
            raise ValueError("Import or transcribe selected tracks first")
        if any(c["language"] == "und" for c in store.cues(pid) if c["track"] in tracks):
            raise ValueError("Choose a source language when importing these subtitles")
        if processing(pid):
            raise ValueError("Finish or pause current processing before translating this project")
        ollama.start()
        if not ollama.ready("translation"):
            raise ValueError("Install the translation model in Models first")
        return jobs.create(pid, "translate", tracks=tracks, target_language=target)

    def export_file(pid, track, language, directory=None):
        project = load_project(pid)
        subtitle_source = cue_source(project["media"], track)
        selected = store.cues(pid, track)
        if not selected:
            raise ValueError("This track has no subtitles yet")
        if language != "original" and language not in TRANSLATION_CODES:
            raise ValueError("Unsupported subtitle language")
        source = Path(project["media"]["path"])
        folder = Path(directory) if directory else source.parent / "Subtitles" / source.stem
        label = f"audio-{subtitle_source['audio_ordinal'] + 1}" if track >= 0 else f"embedded-{subtitle_source['subtitle_ordinal'] + 1}"
        filename = f"{source.stem[:100]}.{label}.{language}.srt"
        path = folder / filename
        from .translation import valid_translations
        translated = valid_translations(selected, store.translations(pid, language), language) if language != "original" else None
        atomic_text(path, srt(selected, translated))
        store.set_export(pid, track, language, str(path))
        return path, subtitle_source

    @app.post("/api/projects/{pid}/export")
    def export(pid: str, body: ExportRequest):
        path, _ = export_file(pid, body.track, body.language, body.directory)
        return {"path": str(path), "cue_count": len(store.cues(pid, body.track))}

    @app.get("/api/projects/{pid}/download")
    def download(pid: str, track: int, language: str = "original"):
        path, _ = export_file(pid, track, language)
        return FileResponse(path, filename=path.name, media_type="application/x-subrip")

    @app.get("/api/projects/{pid}/backup")
    def backup(pid: str):
        path = DATA / "projects" / pid / "project.json"
        atomic_text(path, json.dumps({"project": store.project(pid), "cues": store.cues(pid)}, ensure_ascii=False, indent=2))
        return FileResponse(path, filename="subtitle-project.json", media_type="application/json")

    @app.post("/api/projects/{pid}/play")
    def play(pid: str, body: PlayRequest):
        media = store.project(pid)["media"]
        if fingerprint(Path(media["path"])) != media["fingerprint"]:
            raise ValueError("The source file has changed. Select it again.")
        path, _ = export_file(pid, body.track, body.language)
        audio = preview_audio(load_project(pid)["media"], body.track, body.audio_track)
        args = vlc_arguments(media["path"], str(path), audio["audio_ordinal"] if audio else 0, body.start_seconds)
        subprocess.Popen(args, creationflags=NO_WINDOW)
        return {"launched": True, "subtitle": str(path)}

    @app.post("/api/projects/{pid}/preview")
    def clip(pid: str, body: dict):
        project = load_project(pid)
        if fingerprint(Path(project["media"]["path"])) != project["media"]["fingerprint"]:
            raise ValueError("The source file has changed. Select it again.")
        source_track = int(body["track"])
        audio_track = preview_audio(project["media"], source_track, body.get("audio_track"))
        track = audio_track["stream_index"] if audio_track else None
        start = max(0, float(body.get("start", 0)))
        duration = min(60, max(1, float(body.get("duration", 15))), project["media"]["duration"] - start)
        if duration <= 0:
            raise ValueError("Preview start is outside the media")
        key = hashlib.sha256(f"{project['media']['fingerprint']}:{track}:{start:.3f}:{duration:.3f}".encode()).hexdigest()
        path = DATA / "clips" / f"{key}.mp4"
        if not path.exists():
            preview(project["media"]["path"], track, start, duration, path)
        import numpy as np
        peaks = []
        if track is not None:
            audio = audio_window(project["media"]["path"], track, start, duration)
            block = max(1, len(audio) // 160)
            peaks = [round(float(np.max(np.abs(audio[n:n + block]))), 4) for n in range(0, len(audio), block)][:160]
        # Bound the review cache without touching source files.
        cached = sorted((DATA / "clips").glob("*.mp4"), key=lambda p: p.stat().st_mtime, reverse=True)
        total = 0
        for cached_path in cached:
            total += cached_path.stat().st_size
            if total > 2_000_000_000 and cached_path != path:
                try:
                    cached_path.unlink()
                except OSError:
                    pass
        return {"url": f"/api/clips/{key}?k={token}", "start": start, "duration": duration, "peaks": peaks, "audio_track": track}

    @app.get("/api/clips/{key}")
    def clip_file(key: str):
        if len(key) != 64 or any(c not in "0123456789abcdef" for c in key):
            raise ValueError("Invalid clip")
        path = DATA / "clips" / f"{key}.mp4"
        if not path.exists():
            raise KeyError("Clip not found")
        return FileResponse(path, media_type="video/mp4")

    @app.get("/api/languages")
    def languages():
        return {"transcription": LANGUAGE_CODES, "translation": TRANSLATION_CODES}

    if (WEB / "assets").exists():
        app.mount("/assets", StaticFiles(directory=WEB / "assets"), name="assets")

    @app.get("/")
    def index():
        if not (WEB / "index.html").exists():
            return HTMLResponse("<h1>Subtitle Studio</h1><p>Run Setup.ps1 to build the interface.</p>")
        html = (WEB / "index.html").read_text(encoding="utf-8")
        return HTMLResponse(html.replace("<!--STUDIO_CONFIG-->", f"<script>window.STUDIO_TOKEN={json.dumps(token)};</script>"),
                            headers={"Cache-Control": "no-store"})

    return app


LANGUAGE_CODES = "en zh de es ru ko fr ja pt tr pl ca nl ar sv it id hi fi vi he uk el ms cs ro da hu ta no th ur hr bg lt la mi ml cy sk te fa lv bn sr az sl kn et mk br eu is hy ne mn bs kk sq sw gl mr pa si km sn yo so af oc ka be tg sd gu am yi lo uz fo ht ps tk nn mt sa lb my bo tl mg as tt haw ln ha ba jw su yue".split()
TRANSLATION_CODES = "en fa de fr es it pt nl sv da fi no pl cs sk sl hr sr ro hu el bg ru uk tr ar he hi bn ur ja ko zh vi th id ms ca et lv lt sw af am az be gu kk kn ml mr ne ta te".split()
