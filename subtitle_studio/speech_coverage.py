"""Find audible speech intervals missing from the word-timestamp coverage."""
from __future__ import annotations

import numpy as np
import re
import unicodedata


def repetitive_text(text: str) -> bool:
    tokens = re.findall(r"[^\W_]+", unicodedata.normalize("NFKC",text).casefold())
    if len(tokens) < 8:
        return False
    for size in (1,2,3):
        for at in range(len(tokens)-size*4+1):
            phrase = tokens[at:at+size]
            repeats = 4 if size > 1 else 6
            if all(tokens[at+size*i:at+size*(i+1)] == phrase for i in range(repeats)):
                return True
    return False


def uncovered_speech(speech: list[dict], words: list[dict], duration: float,
                     minimum_seconds: float = .7, maximum_seconds: float = 12) -> list[dict]:
    step = .02
    spoken = np.zeros(int(np.ceil(duration / step)), dtype=bool)
    covered = np.zeros_like(spoken)
    for region in speech:
        start, end = max(0,region["start"]), min(duration,region["end"])
        spoken[int(start/step):int(np.ceil(end/step))] = True
    for word in words:
        start, end = max(0,word["start"]-.15), min(duration,word["end"]+.15)
        # A misaligned long word must not hide several seconds of missing speech.
        if end-start > 2:
            end = min(end,start+.8)
        covered[int(start/step):int(np.ceil(end/step))] = True
    missing = spoken & ~covered
    changes = np.diff(np.pad(missing.astype(np.int8),(1,1)))
    starts, ends = np.flatnonzero(changes==1), np.flatnonzero(changes==-1)
    regions = []
    for start, end in zip(starts,ends):
        if (end-start)*step < minimum_seconds:
            continue
        for at in np.arange(start*step,end*step,maximum_seconds):
            stop = min(end*step,at+maximum_seconds)
            if stop-at >= minimum_seconds:
                regions.append({"start":float(at),"end":float(stop)})
    return regions
