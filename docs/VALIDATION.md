# Verification scope

The backend test suite and production TypeScript build cover the transcription workflow, portable paths, Unicode media, timed subtitle preservation, manual edits, restart/cache behavior, full transcript exports and local translation. Embedded subtitle integration fixtures cover SRT, ASS, WebVTT, timed text, PGS and DVD tracks, packet preservation, styles, attachments, chapters, cancellation and output-file races.

The 0.2 workflow prepares audio before loading recognition, defaults to Turbo and uses large-v3 only when selected. Database migration removes retired processing metadata while preserving project media, raw text, user edits and translations. API tests verify transcription starts with only its selected recognition model available. Separate caches cover enabled/disabled enhancements and model changes.

The running 0.2 interface was checked for file selection, both recognizer choices and the single recommended enhancement toggle. A fresh complete 333-second English recording finished in 17.7 seconds through the application's job queue, including audio preparation, model loading, recognition, gap recovery and stored subtitle generation; the preceding language scan is excluded. Its 116-cue SRT and prepared audio exactly matched the earlier enhanced Turbo benchmark, with 1,375 normalized output words and 71 reference edits (5.20% WER).

Measured recognition tests include supplied Persian and English recordings. The model and filter results are documented in [the ASR comparison](ASR_MODEL_BENCHMARK.md) and [audio preparation](AUDIO_RECOVERY.md). References were only read after completed output was frozen. The small set cannot establish accuracy for every language, accent, music track or overlapping conversation.

Earlier versions processed full recordings of approximately 18 and 106 minutes, and public English/Persian/Japanese FLEURS excerpts. These older runs establish that long-file processing and multilingual paths worked in their tested versions, not that the current release has independently rerun every recording. Raw recordings, subtitle outputs, reference text, timing logs and machine-specific paths are excluded from Git.

Speech recognition can repeat or omit words on damaged, overlapping or nonverbal audio. Word timestamps and flags remain available for manual playback/editing. The original source and generated translations stay separate; exported words are never truncated to fit subtitle lines.
