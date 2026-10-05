from __future__ import annotations

from .translation import LANGUAGE_NAMES


def transcript_blocks(cues: list[dict]) -> list[dict]:
    groups, current = [], []
    for cue in cues:
        if current and (cue["language"] != current[-1]["language"] or cue["start"] - current[-1]["end"] > 3
                        or sum(len(item["text"]) for item in current) >= 1000):
            groups.append(current)
            current = []
        current.append(cue)
    if current:
        groups.append(current)
    return [{"id": group[0]["id"], "language": group[0]["language"], "start": group[0]["start"],
             "text": " ".join(c["text"] for c in group),
             "translation": " ".join(c["translated_text"] for c in group) if all(c.get("translated_text") for c in group) else None,
             "parts": [{"text": c["text"], "translation": c.get("translated_text")} for c in group]} for group in groups]


def transcript_text(blocks: list[dict], view="original", target="en") -> str:
    if not blocks:
        raise ValueError("There is no transcript yet")
    if view != "original" and any(block["translation"] is None for block in blocks):
        raise ValueError("The translation is incomplete or stale. Translate the whole track before exporting it.")
    if view == "original":
        parts = [block["text"] for block in blocks]
    elif view == "translated":
        parts = [block["translation"] for block in blocks]
    elif view == "parallel":
        parts = [f"Original ({LANGUAGE_NAMES.get(block['language'], block['language'])})\n{block['text']}\n\n"
                 f"{LANGUAGE_NAMES.get(target, target)}\n{block['translation']}" for block in blocks]
    else:
        raise ValueError("Unsupported transcript view")
    return "\n\n".join(parts) + "\n"
