from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path

import numpy as np

from .config import binary
from .subtitle_formats import subtitle_description

NO_WINDOW = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0


def run(args: list[str], timeout: float = 120) -> subprocess.CompletedProcess:
    result = subprocess.run(args, capture_output=True, timeout=timeout, creationflags=NO_WINDOW)
    if result.returncode:
        message = result.stderr.decode("utf-8", "replace")[-3000:]
        raise RuntimeError(message or f"Media command failed ({result.returncode})")
    return result


def fingerprint(path: Path) -> str:
    stat = path.stat()
    digest = hashlib.sha256(f"{path.resolve()}|{stat.st_size}|{stat.st_mtime_ns}".encode())
    with path.open("rb") as source:
        digest.update(source.read(1048576))
        if stat.st_size > 1048576:
            source.seek(max(0, stat.st_size - 1048576))
            digest.update(source.read(1048576))
    return digest.hexdigest()


def probe(path: str) -> dict:
    media = Path(path).expanduser().resolve(strict=True)
    if not media.is_file():
        raise ValueError("Select a video or audio file.")
    result = run([binary("ffprobe"), "-v", "warning", "-show_format", "-show_streams", "-show_chapters", "-of", "json", str(media)])
    info = json.loads(result.stdout)
    fmt = info.get("format", {})
    duration = float(fmt.get("duration", 0) or 0)
    audio = []
    video = []
    subtitles = []
    for stream in info.get("streams", []):
        tags = stream.get("tags", {})
        if stream.get("codec_type") == "audio":
            audio.append({"stream_index": stream["index"], "audio_ordinal": len(audio),
                          "codec": stream.get("codec_name"), "channels": stream.get("channels", 1),
                          "channel_layout": stream.get("channel_layout", ""), "title": tags.get("title", ""),
                          "metadata_language": tags.get("language", "und"), "languages": [],
                          "start_time": float(stream.get("start_time", 0) or 0), "detection": "pending"})
        elif stream.get("codec_type") == "subtitle":
            subtitles.append(subtitle_description(stream, len(subtitles)))
        elif stream.get("codec_type") == "video" and not stream.get("disposition", {}).get("attached_pic"):
            video.append({"stream_index": stream["index"], "codec": stream.get("codec_name"),
                          "width": stream.get("width"), "height": stream.get("height")})
    if duration <= 0:
        duration = max((float(s.get("duration", 0) or 0) for s in info.get("streams", [])), default=0)
    if duration <= 0:
        raise ValueError("The media has no readable duration. Try repairing the container with FFmpeg.")
    return {"path": str(media), "name": media.name, "duration": duration, "size": media.stat().st_size,
            "fingerprint": fingerprint(media), "audio_tracks": audio, "video_tracks": video,
            "subtitle_tracks": subtitles, "chapters": info.get("chapters", []),
            "auxiliary_tracks": [{"stream_index": s["index"], "type": s.get("codec_type"), "codec": s.get("codec_name"),
                                  "chapter_track": bool("mov" in fmt.get("format_name", "") and info.get("chapters") and
                                                        s.get("codec_name") == "bin_data" and s.get("codec_tag_string") == "text" and
                                                        s.get("nb_frames") == str(len(info["chapters"])))}
                                 for s in info.get("streams", []) if s.get("codec_type") not in ("audio", "video", "subtitle")],
            "warnings": list(dict.fromkeys(result.stderr.decode("utf-8", "replace").strip().splitlines()))}


def audio_window(path: str, track: int, start: float, duration: float, dialogue_channel: bool = False) -> np.ndarray:
    filters = "aresample=16000:async=1:first_pts=0,apad"
    if dialogue_channel:
        filters = "pan=mono|c0=FC," + filters
    result = run([binary("ffmpeg"), "-nostdin", "-v", "error", "-ss", str(max(0, start)), "-i", path,
                  "-map", f"0:{track}", "-vn", "-t", str(duration), "-af", filters,
                  "-ac", "1", "-ar", "16000", "-f", "f32le", "pipe:1"], timeout=max(90, duration * 3))
    return np.frombuffer(result.stdout, dtype="<f4").copy()


def preview(path: str, track: int | None, start: float, duration: float, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(".partial.mp4")
    args = [binary("ffmpeg"), "-nostdin", "-v", "error", "-y", "-ss", str(max(0, start)), "-i", path,
            "-map", "0:v:0?"]
    if track is not None:
        args += ["-map", f"0:{track}"]
    run(args + ["-t", str(min(60, duration)),
         "-vf", "scale=-2:480", "-c:v", "libx264", "-preset", "veryfast", "-crf", "25",
         "-af", "aresample=async=1:first_pts=0", "-c:a", "aac", "-b:a", "128k", "-movflags", "+faststart",
         str(partial)], timeout=180)
    partial.replace(destination)
    return destination


def vlc_arguments(path: str, subtitle: str, ordinal: int, start: float = 0) -> list[str]:
    return [binary("vlc"), "--no-one-instance", "--no-sub-autodetect-file", f"--audio-track={ordinal}",
            f"--sub-file={subtitle}", f"--start-time={start:.3f}", path]
