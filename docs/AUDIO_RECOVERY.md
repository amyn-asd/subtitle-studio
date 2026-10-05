# Audio preparation and missed-speech recovery

The previous recognition path dropped substantial dialogue in a Persian video. Comparing Turbo with a supplied transcript exposed the problem: the exported SRT had 1,496 normalized words against 2,664 in the reference.

The implementation now combines speech level adjustment, explicit timestamp-token decoding, padded speech boundaries of up to 16 seconds, and a second audio pass over uncovered speech. The same selected Whisper model handles recovery. There is no text reviewer, reference prompt or reference-based output rewriting in this path.

## Measured result

Full 16-minute 3-second video, Whisper Turbo FP16, RTX 5080 with 16 GB VRAM. Rechecks, context reviewers and translation disabled; forced Persian. The supplied transcript was used only to evaluate completed outputs.

| Metric | Previous output | Integrated output |
| --- | ---: | ---: |
| Normalized words | 1,496 | 2,623 |
| Shortfall against reference count | 1,168 (43.8%) | 41 (1.5%) |
| Minimum word edits against reference | 1,522 | 920 |
| Reference edit percentage | 57.1% | 34.5% |
| Deletions in minimum-edit alignment | 1,208 | 170 |
| Exact matches in that alignment | 1,182 | 1,873 |

The integrated run took 39.1 seconds including model imports/loading, audio preparation, recognition and recovery. The previous Turbo benchmark averaged 27.9 seconds. These are local measurements on one video, not performance guarantees. The original source, previous SRTs and existing application transcript were preserved. The new SRT was parsed and its timestamp bounds checked. A completed-cache replay produced an identical SRT.

A raw recovery experiment reached 2,654 words, but included repeated artifacts. Retrying those intervals with more audio context improved the reference comparison; the integrated output is the reported 2,623-word version. Matching a word count is insufficient to establish accuracy. The supplied reference can itself contain errors, and spelling, spacing, overlapping voices and damaged audio still affect the result.

## Controls

- **Even out quiet and loud speech** is the default. It uses a mild high-pass filter and dynamic level adjustment.
- **Original audio** provides a comparison without signal processing.
- **Light/stronger noise reduction** add FFT noise reduction. They are optional: neither outperformed level adjustment on this test.
- **Recover missed speech** checks intervals with detected speech but no corresponding word coverage. It adds only audio-derived recognition results and flags them for review.

Other tests included shorter/longer windows, stronger gain, denoising, VAD thresholds and slowed audio. Increasing the count sometimes increased errors; those combinations were not selected merely for their counts. No model weights were trained, downloaded or changed for these tests.

## Implementation

Audio preparation streams 16 kHz mono float32 PCM to a local cache and reads it through a memory map. Long movies do not need a complete Python copy of their waveform. Silence is retained and input/output duration is checked, keeping subtitles on the video timeline. The preparation cache includes source fingerprint, stream, range, profile and processing version.

Both primary recognition and recovery use actual audio. Recovery excludes already covered words and retries suspicious repetition with wider audio context. Cached completed segments and recovery results are reusable. A partial primary resume replays deterministic decoding up to saved segments, preserving completed results rather than seeking past an unfinished sentence. This can repeat some computation after interruption. Pipeline cache version 5 prevents older incomplete recognition results from being reused.

The source media, reference text, model weights, timings and generated subtitles stay in ignored local storage. Generic settings and test code are the only repository additions.

## Research used

- [FFmpeg dynamic audio normalization](https://ffmpeg.org/ffmpeg-filters.html#dynaudnorm) explains level adjustment across quiet and loud sections.
- [FFmpeg FFT denoising](https://ffmpeg.org/ffmpeg-filters.html#afftdn) documents the optional noise-reduction controls.
- [faster-whisper source and decoding options](https://github.com/SYSTRAN/faster-whisper/blob/master/faster_whisper/transcribe.py) distinguish word alignment from timestamp-token decoding.
- [Reported batched clip timestamp issue](https://github.com/SYSTRAN/faster-whisper/issues/1361) motivated tracing recognition versus export losses; the local audit showed that most missing text had already been omitted by decoding.
- [Speech enhancement evaluation with Whisper](https://arxiv.org/abs/2603.04710) reports that cleaner audio does not necessarily improve ASR. Its languages and enhancement model differ from this test; the local comparisons determine which profile is used here.

Reference comparison uses NFKC normalization, punctuation removal, Arabic/Persian Yeh and Kaf normalization, half-spaces as separators, and unified digit forms. For a local reference and exported TXT:

```powershell
.venv\Scripts\python.exe -X utf8 scripts\transcript_metrics.py reference.txt output.txt --output data\validation\transcript-comparison.json
```

This command scores existing texts. It never sends the reference to a recognizer.
