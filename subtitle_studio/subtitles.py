from __future__ import annotations

import hashlib
import re
import textwrap
import unicodedata
from pathlib import Path
from .types import TRANSLATION_VERSION


def same_spoken_words(first: str, second: str) -> bool:
    def tokens(text):
        text = unicodedata.normalize("NFKC", text).casefold().replace("’", "'")
        return re.findall(r"[^\W_]+(?:'[^\W_]+)*", text)
    return bool(tokens(first)) and tokens(first) == tokens(second)


def timestamp(seconds: float) -> str:
    milliseconds = max(0, round(seconds * 1000))
    hours, milliseconds = divmod(milliseconds, 3600000)
    minutes, milliseconds = divmod(milliseconds, 60000)
    sec, milliseconds = divmod(milliseconds, 1000)
    return f"{hours:02}:{minutes:02}:{sec:02},{milliseconds:03}"


def srt(cues: list[dict], translated: dict | None = None) -> str:
    lines = []
    for i, cue in enumerate(sorted(cues, key=lambda c: c["start"]), 1):
        text = cue["text"]
        if translated is not None:
            item = translated.get(cue["id"])
            if not item or item["source_text"] != cue["text"] or item.get("version") != TRANSLATION_VERSION:
                raise ValueError("Translation is incomplete or stale. Translate again before exporting.")
            text = item["text"]
        text = text.replace("\r", "").strip()
        # Wrapping is presentation only: never truncate or simplify spoken words.
        wrapped = "\n".join(textwrap.wrap(text, width=46, break_long_words=False, break_on_hyphens=False))
        lines.append(f"{i}\n{timestamp(cue['start'])} --> {timestamp(cue['end'])}\n{wrapped}\n")
    return "\n".join(lines)


def stable_id(pid: str, track: int, start: float, index: int) -> str:
    return hashlib.sha256(f"{pid}:{track}:{start:.3f}:{index}".encode()).hexdigest()[:20]


def group_words(pid: str, track: int, words: list[dict]) -> list[dict]:
    words = sorted(words, key=lambda w: (w["start"], w["end"]))
    unique = []
    for word in words:
        # Same word at the same time from overlapping windows, not repeated speech.
        duplicate = any(w["word"].strip() == word["word"].strip() and abs((w["start"] + w["end"]) - (word["start"] + word["end"])) < .25
                        and w.get("chunk") != word.get("chunk") for w in unique[-8:])
        if not duplicate:
            unique.append(word)
    groups, current = [], []
    for word in unique:
        if current and (word["language"] != current[-1]["language"] or word["start"] - current[-1]["end"] > .8
                        or word["end"] - current[0]["start"] > 5.5 or sum(len(w["word"]) for w in current) > 85):
            groups.append(current)
            current = []
        current.append(word)
        if re.search(r"[.!?。！？]$", word["word"].strip()) and len(current) >= 3:
            groups.append(current)
            current = []
    if current:
        groups.append(current)
    cues = []
    for i, group in enumerate(groups):
        text = "".join(w["word"] for w in group).strip()
        flags = sorted({flag for w in group for flag in w.get("flags", [])})
        duration = max(.12, group[-1]["end"] - group[0]["start"])
        if len(text) / duration > 28:
            flags.append("Fast speech")
        cue = {"id": stable_id(pid, track, group[0]["start"], i), "track": track,
               "start": group[0]["start"], "end": max(group[0]["start"] + .12, group[-1]["end"]),
               "text": text, "raw_text": text, "language": group[0]["language"], "words": group,
               "flags": sorted(set(flags)), "reviewed": False, "edited": False,
               "language_hints": sorted({w["language_hint"] for w in group if w.get("language_hint")}),
               "candidates": [{"id": "primary", "text": text, "engine": "Whisper", "source": "Original audio"}],
               "decision": None}
        cues.append(cue)
    return cues


def atomic_text(path: Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    partial.write_text(text, encoding="utf-8")
    partial.replace(path)
