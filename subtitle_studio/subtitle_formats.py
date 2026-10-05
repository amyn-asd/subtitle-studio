"""Subtitle capabilities and container language metadata, independent of AI models."""
from __future__ import annotations

BITMAP_CODECS = {"hdmv_pgs_subtitle", "dvd_subtitle", "dvb_subtitle", "xsub", "arib_caption"}
TEXT_CODECS = {"subrip", "srt", "ass", "ssa", "webvtt", "mov_text", "text", "ttml", "dfxp",
               "microdvd", "mpl2", "jacosub", "sami", "realtext", "subviewer", "subviewer1",
               "vplayer", "pjs", "subviewer", "aqtitle", "eia_608", "hdmv_text_subtitle"}
MATROSKA_COPY = {"subrip", "ass", "ssa", "webvtt", "hdmv_pgs_subtitle", "dvd_subtitle", "dvb_subtitle"}
# Other codecs are preserved in a subtitle-only Matroska file instead of pretending they are SRT.
NATIVE_FORMATS = {"subrip": ("srt", "srt"), "ass": ("ass", "ass"), "ssa": ("ass", "ass"),
                  "webvtt": ("vtt", "webvtt"), "hdmv_pgs_subtitle": ("sup", "sup"),
                  "ttml": ("ttml", "ttml")}
ISO3 = dict(zip(
    "en fa de fr es it pt nl sv da fi no pl cs sk sl hr sr ro hu el bg ru uk tr ar he hi bn ur ja ko zh vi th id ms ca et lv lt sw af am az be gu kk kn ml mr ne ta te".split(),
    "eng fas deu fra spa ita por nld swe dan fin nor pol ces slk slv hrv srp ron hun ell bul rus ukr tur ara heb hin ben urd jpn kor zho vie tha ind msa cat est lav lit swa afr amh aze bel guj kaz kan mal mar nep tam tel".split()))
ISO2 = {value: key for key, value in ISO3.items()}
ISO2.update(dict(zip("per ger fre dut cze slo rum gre chi may alb arm baq bur ice mac mao tib wel".split(),
                     "fa de fr nl cs sk ro el zh ms sq hy eu my is mk mi bo cy".split())))


def language_code(value: str) -> str:
    value = value.lower().replace("_", "-").split("-")[0]
    return ISO2.get(value, value if len(value) == 2 else "und")


def language_tag(value: str) -> str:
    return ISO3.get(value, value if len(value) == 3 else "und")


def subtitle_description(stream: dict, ordinal: int) -> dict:
    codec = stream.get("codec_name", "unknown")
    tags, disposition = stream.get("tags", {}), stream.get("disposition", {})
    kind = "image" if codec in BITMAP_CODECS else "text" if codec in TEXT_CODECS else "unknown"
    extension, _ = NATIVE_FORMATS.get(codec, ("srt", "srt") if kind == "text" else ("mks", "matroska"))
    return {"stream_index": stream["index"], "subtitle_ordinal": ordinal,
            # Audio stream IDs remain unchanged. Negative IDs cannot collide with them.
            "cue_track": -stream["index"] - 1, "codec": codec, "kind": kind,
            "title": tags.get("title", tags.get("name", "")), "metadata_language": tags.get("language", "und"),
            "language": language_code(tags.get("language", "und")), "extract_extension": extension,
            "default": bool(disposition.get("default")), "forced": bool(disposition.get("forced")),
            "dispositions": [key for key, value in disposition.items() if value]}


def cue_source(media: dict, track: int) -> dict:
    if track >= 0:
        found = next((t for t in media["audio_tracks"] if t["stream_index"] == track), None)
        if found:
            return {**found, "source_kind": "audio", "cue_track": track,
                    "label": found.get("title") or f"Audio {found['audio_ordinal'] + 1}"}
    else:
        found = next((t for t in media.get("subtitle_tracks", []) if t["cue_track"] == track), None)
        if found:
            return {**found, "source_kind": "embedded",
                    "label": found.get("title") or f"Embedded subtitles {found['subtitle_ordinal'] + 1}"}
    raise ValueError("Subtitle source not found")


def preview_audio(media: dict, source_track: int, selected: int | None = None) -> dict | None:
    source = cue_source(media, source_track)
    audio = media["audio_tracks"]
    if selected is not None:
        found = next((t for t in audio if t["stream_index"] == selected), None)
        if not found:
            raise ValueError("Invalid preview audio track")
        return found
    if source["source_kind"] == "audio":
        return source
    return next((t for t in audio if language_code(t.get("metadata_language", "und")) == source["language"]
                 and source["language"] != "und"), audio[0] if audio else None)
