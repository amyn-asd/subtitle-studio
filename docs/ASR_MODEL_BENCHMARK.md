# Whisper Turbo and large-v3 reference comparisons

The application defaults to Turbo. Large-v3 is selectable as an optional fallback before processing. Both use identical recommended audio preparation, padded speech boundaries, timestamp-token decoding and same-model missed-speech recovery.

| Recording | Model | Processing time | Output words | Substitutions | Deletions | Insertions | Word error |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Persian, 16m 03s | Turbo | 39.12 s | 2,623 | 621 | 170 | 129 | 34.53% |
| Persian, 16m 03s | Large-v3 | 76.55 s | 2,510 | 555 | 270 | 116 | 35.32% |
| English, 5m 33s | Turbo | 14.10 s | 1,375 | 33 | 14 | 24 | 5.20% |
| English, 5m 33s | Large-v3 | 22.33 s | 1,370 | 29 | 17 | 22 | 4.98% |

References contain 2,664 and 1,365 normalized words respectively. Source language was supplied for each test. Recognition and export were frozen before reading the reference for scoring. No reference words were passed to recognition. The prepared waveform hash was identical between Turbo and large-v3 for each recording.

On the English recording, large-v3 made three fewer word edits and took about 58% longer. On the Persian recording, Turbo had fewer edits and was faster. These samples support keeping the choice available; they do not establish a universal ranking across languages or audio conditions.

Times are single fresh-cache local runs on an RTX 5080 with 16 GB VRAM, including model loading/preparation/recognition/recovery and excluding scoring/export. OS/driver caches were not flushed. No independently listened or human-timed ground truth is available. Word error is normalized substitutions + deletions + insertions divided by reference words, including spelling/number/censoring differences.

Source videos, supplied text and generated transcripts remain in ignored local storage. Public statistics contain no local paths or dialogue. `scripts/transcript_metrics.py` evaluates exported TXT; it never changes recognition output. See [audio preparation comparisons](AUDIO_RECOVERY.md) for the English on/off experiment.
