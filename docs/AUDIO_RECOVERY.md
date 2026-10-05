# Audio preparation and missed-speech recovery

The previous recognition path dropped substantial dialogue in a Persian video. Comparing Turbo with a supplied transcript exposed the problem: the exported SRT had 1,496 normalized words against 2,664 in the reference.

The implementation now combines speech level adjustment, explicit timestamp-token decoding, padded speech boundaries of up to 16 seconds, and a second audio pass over uncovered speech. The same selected Whisper model handles recovery. The supplied reference is used only to evaluate completed output.

## Measured result

Full 16-minute 3-second video, Whisper Turbo FP16, RTX 5080 with 16 GB VRAM. Transcription only; forced Persian. The supplied transcript was used only to evaluate completed outputs.

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

## One recommended option

**Audio enhancements — Recommended** is enabled by default. It uses a mild 60 Hz high-pass filter and dynamic level adjustment, preparing the audio before the recognition model loads. Turning it off retains original decoded audio with the required mono/16 kHz conversion. The interface offers no gain or noise-reduction tuning controls.

Padded speech boundaries, timestamp-token decoding and same-model missed-speech recovery are built into transcription in both modes. Uncovered audible intervals are re-recognized; suspicious repetition receives wider audio context. The source video and its timing stay intact.

Stronger filtering, varied window lengths, changed speech thresholds and slowed audio were tested during tuning. Their higher word counts sometimes increased errors, so the fixed recommended preparation uses the measured mild leveling method.

### English filter comparison

Same 5m 33s video, Turbo model/revision, forced English, 16-second windows, beam 5, batch 4, FP16 and missed-speech recovery. Only the audio profile changed; each mode used a fresh recognition cache. Reference: 1,365 normalized words.

| Audio | Time | Output words | Substitutions | Deletions | Insertions | Word error |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Recommended preparation | 14.10 s | 1,375 | 33 | 14 | 24 | 5.20% |
| Filters disabled | 14.29 s | 1,524 | 45 | 14 | 173 | 17.00% |

Repeated extra phrases were substantially worse without preparation. These are single measured runs; the 0.19-second timing difference is not a stable speed ranking. Timing includes worker import/model loading, preparation, recognition and recovery; excludes probing, export and scoring. Reference spelling, number formatting and censoring differences also count as edits.

## Implementation

Audio preparation streams 16 kHz mono float32 PCM to a local cache and reads it through a memory map. Long movies do not need a complete Python copy of their waveform. Silence is retained and input/output duration is checked, keeping subtitles on the video timeline. The preparation cache includes source fingerprint, stream, range, profile and processing version.

Both primary recognition and recovery use actual audio. Recovery excludes already covered words and retries suspicious repetition with wider audio context. Cached completed segments and recovery results are reusable. A partial primary resume replays deterministic decoding up to saved segments, preserving completed results rather than seeking past an unfinished sentence. This can repeat some computation after interruption. Pipeline cache version 7 separates this workflow from older processing runs.

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
