import re

import pytest

from subtitle_studio.subtitles import timestamp, srt, group_words
from subtitle_studio.storage import Store


def word(text, start, end, chunk=0):
    return {"word": text, "start": start, "end": end, "language": "en", "flags": [], "chunk": chunk}


def test_real_repetition_is_preserved_but_window_duplicates_are_removed():
    words = [word(" No", 1, 1.3), word(" no", 1.5, 1.8), word(" no!", 2, 2.3),
             word(" no!", 2.01, 2.31, 1)]
    cues = group_words("p", 1, words)
    assert " ".join(c["text"] for c in cues) == "No no no!"


def test_language_changes_and_pauses_split_cues_without_losing_words():
    words = [word(" Hello.", 0, .4), {**word(" سلام", .5, 1), "language": "fa"}, word(" Again.", 4, 4.6)]
    cues = group_words("p", 2, words)
    assert [c["language"] for c in cues] == ["en", "fa", "en"]
    assert [c["text"] for c in cues] == ["Hello.", "سلام", "Again."]
    assert len({c["id"] for c in cues}) == 3


def test_srt_handles_unicode_hours_and_never_truncates_words():
    text = "سلام hello " * 60
    cues = [{"id": "x", "start": 3600.9996, "end": 3603, "text": text}]
    exported = srt(cues)
    assert "01:00:01,000 --> 01:00:03,000" in exported
    assert exported.count("سلام") == 60
    assert timestamp(26 * 3600) == "26:00:00,000"


def test_stale_or_missing_translation_cannot_export():
    cue = {"id": "x", "start": 0, "end": 1, "text": "New text"}
    with pytest.raises(ValueError, match="stale"):
        srt([cue], {"x": {"text": "Old translation", "source_text": "Old text"}})
    with pytest.raises(ValueError):
        srt([cue], {})


def test_storage_keeps_raw_recognition_and_user_edits_across_resume(tmp_path):
    store = Store(tmp_path / "store.sqlite")
    cue = group_words("p", 1, [word(" Original.", 0, .9)])[0]
    store.replace_cues("p", 1, [cue])
    store.update_cue("p", cue["id"], {"text": "Manual correction.", "edited": True, "reviewed": True})
    store.replace_cues("p", 1, [cue])
    result = store.cues("p", 1)[0]
    assert result["raw_text"] == "Original."
    assert result["text"] == "Manual correction."
    assert result["reviewed"]
    with pytest.raises(ValueError):
        store.update_cue("p", cue["id"], {"end": -1})
