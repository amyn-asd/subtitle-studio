"""Compare ASR text with a supplied reference; the reference is never an ASR prompt."""
from __future__ import annotations

import argparse
import json
import re
import unicodedata
from pathlib import Path


def words(text: str) -> list[str]:
    text = unicodedata.normalize("NFKC", text).casefold().replace("ي", "ی").replace("ك", "ک").replace("\u200c", " ").replace("\u0640", "")
    text = "".join(c for c in text if unicodedata.category(c) != "Mn")
    text = "".join(str(unicodedata.digit(c)) if c.isdigit() else c for c in text)
    return re.findall(r"[^\W_]+", text)


def compare(reference: str, hypothesis: str) -> dict:
    expected, actual = words(reference), words(hypothesis)
    width = len(actual) + 1
    # Store traceback directions, not a Python-object matrix of every distance.
    directions = bytearray((len(expected) + 1) * width)
    row = list(range(width))
    for i, token in enumerate(expected, 1):
        next_row = [i]
        for j, output in enumerate(actual, 1):
            diagonal = row[j - 1] + (token != output)
            deletion, insertion = row[j] + 1, next_row[j - 1] + 1
            best = min(diagonal, deletion, insertion)
            directions[i * width + j] = 0 if best == diagonal else (1 if best == deletion else 2)
            next_row.append(best)
        row = next_row
    substitutions = deletions = insertions = matches = 0
    i, j = len(expected), len(actual)
    while i or j:
        direction = directions[i * width + j] if i and j else (1 if i else 2)
        if direction == 0:
            if expected[i - 1] == actual[j - 1]:
                matches += 1
            else:
                substitutions += 1
            i -= 1; j -= 1
        elif direction == 1:
            deletions += 1; i -= 1
        else:
            insertions += 1; j -= 1
    return {"reference_words": len(expected), "output_words": len(actual),
        "word_count_gap": len(expected) - len(actual), "minimum_word_edits": row[-1],
        "reference_edit_percent": 100 * row[-1] / max(1, len(expected)),
        "matches": matches, "substitutions": substitutions, "deletions": deletions, "insertions": insertions,
        "note": "Edits to the supplied reference, which may itself contain errors; word count alone is not accuracy."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("hypotheses", type=Path, nargs="+")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    reference = args.reference.read_text(encoding="utf-8-sig")
    result = {str(path): compare(reference, path.read_text(encoding="utf-8-sig")) for path in args.hypotheses}
    text = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
