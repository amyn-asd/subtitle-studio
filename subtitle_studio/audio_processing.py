"""Conservative, time-preserving recognition audio filters."""
from __future__ import annotations

from pathlib import Path
import json
import numpy as np

AUDIO_VERSION = 1

# Explicit profiles prevent arbitrary filter expressions from entering media jobs.
# Keep the source media untouched; no silence removal or speech synthesis.
AUDIO_PROFILES = {
    "original": "",
    "level": "highpass=f=60,dynaudnorm=f=250:g=7:p=0.9:m=6:r=0.12",
    "gentle": "highpass=f=60,afftdn=nr=6:nf=-50:tn=1:gs=3,dynaudnorm=f=250:g=7:p=0.9:m=6:r=0.12",
    "speech": "highpass=f=80,lowpass=f=7500,afftdn=nr=9:nf=-45:tn=1:gs=5,dynaudnorm=f=250:g=7:p=0.9:m=6:r=0.12",
}


def audio_filters(profile: str = "original") -> str:
    try:
        return AUDIO_PROFILES[profile]
    except KeyError:
        raise ValueError(f"Unknown audio preparation profile: {profile}") from None


def process_waveform(audio: np.ndarray, profile: str) -> np.ndarray:
    """Filter a 16 kHz mono waveform without changing its timebase."""
    filters = audio_filters(profile)
    if not filters:
        return audio.copy()
    from .config import binary
    # FFmpeg handles the PCM stream. Keep the exact input length: some filters
    # buffer the last frame, and sample rounding must never shift subtitle times.
    import subprocess
    from .media import NO_WINDOW
    result = subprocess.run([binary("ffmpeg"), "-nostdin", "-v", "error",
        "-f", "f32le", "-ar", "16000", "-ac", "1", "-i", "pipe:0",
        "-af", filters, "-f", "f32le", "pipe:1"], input=np.asarray(audio,dtype="<f4").tobytes(),
        capture_output=True, timeout=max(90,len(audio)/16000*3), creationflags=NO_WINDOW)
    if result.returncode:
        raise RuntimeError(result.stderr.decode("utf-8", "replace")[-2000:])
    output = np.frombuffer(result.stdout,dtype="<f4").copy()
    if abs(len(output)-len(audio)) > 16:
        raise RuntimeError("Audio preparation changed the duration")
    output = np.pad(output, (0,max(0,len(audio)-len(output))))[:len(audio)]
    if not np.all(np.isfinite(output)):
        raise RuntimeError("Audio preparation produced invalid samples")
    return output


def prepare_track(path: str, track: int, start: float, duration: float,
                  destination: Path, profile: str = "level") -> np.memmap:
    """Stream preparation to disk, keeping long movies out of Python RAM."""
    from .config import binary
    from .media import run, fingerprint
    from .subtitles import atomic_text
    filters = audio_filters(profile)
    expected = round(duration * 16000)
    metadata = {"version": AUDIO_VERSION, "fingerprint": fingerprint(Path(path)),
                "track": track, "start": start, "samples": expected, "profile": profile}
    marker = destination.with_suffix(".json")
    reusable = False
    if destination.exists() and marker.exists():
        try:
            reusable = destination.stat().st_size == expected * 4 and json.loads(marker.read_text(encoding="utf-8")) == metadata
        except (ValueError, OSError):
            pass
    if not reusable:
        destination.parent.mkdir(parents=True, exist_ok=True)
        partial = destination.with_suffix(".partial.f32")
        # Downmix before the filters, as in process_waveform. Do not remove
        # silence: all exported times must stay on the source video's timeline.
        chain = "aresample=16000:async=1:first_pts=0,aformat=channel_layouts=mono"
        if filters:
            chain += "," + filters
        chain += ",apad"
        run([binary("ffmpeg"), "-nostdin", "-v", "error", "-y", "-ss", str(start), "-i", path,
             "-map", f"0:{track}", "-vn", "-t", f"{expected/16000:.8f}", "-af", chain,
             "-ac", "1", "-ar", "16000", "-f", "f32le", str(partial)], timeout=max(180,duration*3))
        if partial.stat().st_size != expected * 4:
            raise RuntimeError("Prepared audio does not cover the requested time range")
        partial.replace(destination)
        atomic_text(marker, json.dumps(metadata, ensure_ascii=False))
    return np.memmap(destination, dtype="<f4", mode="r", shape=(expected,))
