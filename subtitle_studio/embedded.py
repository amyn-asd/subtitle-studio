"""Extract text/bitmap subtitles and plan a lossless A/V remux into a new file."""
from __future__ import annotations

from collections import Counter
from pathlib import Path

from .config import binary
from .media import fingerprint
from .subtitle_formats import MATROSKA_COPY, NATIVE_FORMATS, cue_source, language_tag
from .subtitles import srt
from .translation import valid_translations


def unchanged(media: dict):
    if fingerprint(Path(media["path"])) != media["fingerprint"]:
        raise ValueError("The source file has changed. Select it again to create a new project.")


def extraction(media: dict, track: int, folder: Path, *, normalized=False) -> tuple[list[str], Path]:
    source = cue_source(media, track)
    if source["source_kind"] != "embedded":
        raise ValueError("Select an embedded subtitle track")
    if normalized and source["kind"] != "text":
        raise ValueError("Image subtitles need OCR before translation. Extract or preserve the original track instead.")
    if normalized:
        extension, muxer, codec = "srt", "srt", "srt"
    elif source["codec"] in NATIVE_FORMATS:
        extension, muxer = NATIVE_FORMATS[source["codec"]]
        codec = "copy"
    elif source["kind"] == "text":
        extension, muxer, codec = "srt", "srt", "srt"
    else:
        extension, muxer, codec = "mks", "matroska", "copy"
    name = f"subtitle-{source['subtitle_ordinal'] + 1}{'-normalized' if normalized else ''}.{extension}"
    path = folder / name
    args = [binary("ffmpeg"), "-nostdin", "-v", "error", "-y", "-copyts", "-start_at_zero",
            "-i", media["path"], "-map", f"0:{source['stream_index']}", "-c:s", codec, "-f", muxer, str(path)]
    return args, path


def output_destination(media: dict, value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute() or path.suffix.lower() not in (".mkv", ".mp4"):
        raise ValueError("Choose an absolute path for a new MKV or MP4 video")
    path = path.resolve()
    source = Path(media["path"]).resolve()
    if path == source or (path.exists() and path.samefile(source)):
        raise ValueError("Choose a new filename; the original video cannot be overwritten")
    if path.exists():
        raise ValueError("The output already exists. Choose a new filename")
    return path


def subtitle_choices(store, pid: str) -> list[dict]:
    media = store.project(pid)["media"]
    all_cues = store.cues(pid)
    choices = []
    for track in media.get("subtitle_tracks", []):
        choices.append({"track": track["cue_track"], "language": "original", "label": track.get("title") or
                        f"Embedded {track['subtitle_ordinal'] + 1} · {track['metadata_language']} · {track['codec'].upper()}",
                        "kind": track["kind"], "native": not any(c.get("edited") for c in all_cues if c["track"] == track["cue_track"])})
    for track in media["audio_tracks"]:
        if any(c["track"] == track["stream_index"] for c in all_cues):
            choices.append({"track": track["stream_index"], "language": "original",
                            "label": (track.get("title") or f"Audio {track['audio_ordinal'] + 1}") + " · original words",
                            "kind": "text", "native": False})
    with store.connect() as db:
        languages = [row[0] for row in db.execute("SELECT DISTINCT language FROM translations WHERE project_id=?", (pid,))]
    for language in languages:
        valid = valid_translations(all_cues, store.translations(pid, language), language)
        for track in sorted({c["track"] for c in all_cues}):
            cues = [c for c in all_cues if c["track"] == track]
            if cues and all(c["id"] in valid for c in cues):
                source = cue_source(media, track)
                choices.append({"track": track, "language": language, "label": source["label"],
                                "kind": "text", "native": False})
    return choices


def remux_plan(store, pid: str, selections: list[dict], output: str, keep_embedded=True) -> dict:
    media = store.project(pid)["media"]
    unchanged(media)
    destination = output_destination(media, output)
    if not media["video_tracks"]:
        raise ValueError("Select a video before embedding subtitles")
    keys = [(s["track"], s["language"]) for s in selections]
    if not keys or len(set(keys)) != len(keys) or sum(s.get("default", False) for s in selections) > 1:
        raise ValueError("Choose distinct subtitle versions and at most one default track")
    all_cues = store.cues(pid)
    entries = []
    replaced = {s["track"] for s in selections if s["language"] == "original"}
    if keep_embedded:
        for track in media.get("subtitle_tracks", []):
            if track["cue_track"] not in replaced:
                entries.append({"source": cue_source(media, track["cue_track"]), "native": True,
                                "language": track["language"], "title": track.get("title", ""),
                                "default": track["default"], "forced": track["forced"]})
    for selection in selections:
        source = cue_source(media, selection["track"])
        cues = [c for c in all_cues if c["track"] == selection["track"]]
        native = source["source_kind"] == "embedded" and selection["language"] == "original" and not any(c.get("edited") for c in cues)
        entry = {"source": source, "native": native, "default": selection.get("default", False),
                 "forced": source.get("forced", False)}
        if native:
            entry.update(language=source["language"], title=source.get("title", ""))
        else:
            if not cues:
                raise ValueError("Import or transcribe this subtitle source first")
            language = selection["language"]
            translated = valid_translations(all_cues, store.translations(pid, language), language) if language != "original" else None
            # Render now to reject incomplete/stale translation before starting a potentially long copy.
            entry["text"] = srt(cues, translated)
            if language == "original":
                codes = Counter(c["language"] for c in cues)
                language = next(iter(codes)) if len(codes) == 1 else "und"
            entry.update(language=language, title=f"{language.upper()} · {source['label']}")
        entries.append(entry)
    warnings = []
    mp4 = destination.suffix.lower() == ".mp4"
    if not mp4 and any(t["type"] == "data" and not t.get("chapter_track") for t in media.get("auxiliary_tracks", [])):
        raise ValueError("MKV cannot preserve this video's data stream. Choose MP4 if its codecs are compatible")
    if mp4:
        if any(e["native"] and e["source"]["kind"] != "text" for e in entries):
            raise ValueError("MP4 cannot contain these image subtitle tracks. Choose MKV or exclude those tracks")
        if any(t["type"] == "attachment" for t in media.get("auxiliary_tracks", [])):
            raise ValueError("Choose MKV to preserve this video's font/file attachments")
        warnings.append("MP4 converts subtitles to timed text; advanced subtitle styling is not preserved.")
    elif any(e["native"] and e["source"]["kind"] == "text" and e["source"]["codec"] not in MATROSKA_COPY for e in entries):
        warnings.append("Text formats unsupported by MKV are converted to SRT; advanced styling may change.")
    if any(not e["native"] and e["source"]["source_kind"] == "embedded" for e in entries):
        warnings.append("Translated or edited subtitles use SRT styling. Retained original tracks keep their original styling.")
    if any(s.get("default") for s in selections):
        # Retained default tracks must not compete with the user's chosen default.
        for entry in entries[:len(entries) - len(selections)]:
            entry["default"] = False
    return {"media": media, "output": destination, "entries": entries, "warnings": warnings}


def remux_arguments(plan: dict, folder: Path, partial: Path) -> list[str]:
    args = [binary("ffmpeg"), "-nostdin", "-v", "error", "-n", "-copyts", "-start_at_zero", "-i", plan["media"]["path"]]
    inputs = {}
    for index, entry in enumerate(plan["entries"]):
        if not entry["native"]:
            path = folder / f"mux-subtitle-{index}.srt"
            path.write_text(entry["text"], encoding="utf-8")
            inputs[index] = len(inputs) + 1
            args += ["-i", str(path)]
    # Map every source stream explicitly, excluding only the old subtitles handled below.
    args += ["-map", "0", "-map", "-0:s?"]
    for track in plan["media"].get("auxiliary_tracks", []):
        if track.get("chapter_track"):
            # MOV stores chapters in an extra binary/text stream. Rebuild the chapter representation
            # from mapped chapter metadata; copying that data stream would break MKV or duplicate MP4 chapters.
            args += ["-map", f"-0:{track['stream_index']}"]
    args += ["-map_metadata", "0", "-map_chapters", "0", "-c", "copy"]
    mp4 = plan["output"].suffix.lower() == ".mp4"
    for index, entry in enumerate(plan["entries"]):
        source = entry["source"]
        args += ["-map", f"0:{source['stream_index']}" if entry["native"] else f"{inputs[index]}:s:0"]
        codec = "mov_text" if mp4 else "srt" if entry["native"] and source["kind"] == "text" and source["codec"] not in MATROSKA_COPY else "copy"
        dispositions = [name for name in source.get("dispositions", []) if name not in ("default", "forced")]
        dispositions += [name for name in ("default", "forced") if entry.get(name)]
        args += [f"-c:s:{index}", codec, f"-metadata:s:s:{index}", f"language={language_tag(entry['language'])}",
                 f"-metadata:s:s:{index}", f"title={entry['title']}", f"-disposition:s:{index}", "+".join(dispositions) or "0"]
    args += ["-avoid_negative_ts", "disabled"]
    if mp4:
        args += ["-movflags", "+faststart"]
    return args + ["-f", "mp4" if mp4 else "matroska", str(partial)]
