# Subtitle Studio

A local Windows workspace for multilingual video transcription, translation, subtitle editing and playback. Models and media stay outside the Git repository.

## Workflow

1. Choose a video and its audio tracks.
2. Keep **Audio enhancements — Recommended** enabled to prepare quiet and loud speech before recognition. One toggle turns the fixed preparation on or off; subtitle timing stays on the source timeline.
3. Transcribe with **Whisper Turbo**, the default. Select **Large-v3** as an optional fallback when you want to try that recognizer instead.
4. Inspect flagged passages, edit subtitles, read/copy the full transcript, and export or play the video with subtitles in VLC.

The selected recognizer uses padded speech boundaries, explicit timestamp decoding and a pass over audible gaps to recover missed speech. Recovered words remain flagged for listening. Transcription finishes after recognition and subtitle generation. Recognition is a best hypothesis; unclear or overlapping voices can still need manual correction.

Existing embedded text subtitles can also be extracted, edited, translated and embedded into a new MKV/MP4 without transcribing the audio. Optional local translation uses TranslateGemma and keeps the source wording separate.

## Quick start

Requires Windows 10/11, an NVIDIA CUDA GPU and Node.js 20.19+ to build the interface. The setup creates an isolated Python 3.12 environment and installs the pinned CUDA runtime. The processing queue runs one GPU stage at a time for a 16 GB VRAM budget.

```powershell
.\Setup.ps1
.\Start.ps1
```

`Setup.bat` and `Start.bat` provide the same launchers. **Models & settings → Install recommended models** installs Turbo and the independent language detector. Large-v3 and the translation model are optional downloads. Recognition requires only the selected Whisper model; it can use Whisper's own language detection if the independent detector is unavailable. FFmpeg/FFprobe are installed by setup; install VLC separately for playback.

The app uses a loopback address and a per-run authorization token. Start opens the browser locally. Stop uses `Stop.bat`; completed recognition checkpoints and user edits are saved.

## Models and portable locations

| Purpose | Model | Availability |
| --- | --- | --- |
| Default transcription | Whisper large-v3-turbo, FP16 | Recommended |
| Alternative transcription | Whisper large-v3, FP16 | Optional fallback selected before processing |
| Independent language hints | SpeechBrain VoxLingua107 | Recommended; CPU, optional for recognition |
| Speech regions | Silero VAD | Local CPU detection |
| Translation | TranslateGemma 12B | Optional local translation |

Compatible Hugging Face snapshots are registered in place. In **Models & settings**, choose any recognition-model and translation-model storage folders and optional tool executable paths. Copy `studio-config.example.json` to `studio-config.json` for additional cache search paths. `SUBTITLE_STUDIO_MODELS` and `SUBTITLE_STUDIO_DATA` override model/project storage. Local paths, models, media, generated subtitles and project databases are ignored by Git.

Models retain their publishers' licenses. Core processing works offline once its required assets are installed.

## Audio preparation

The recommended toggle applies a mild 60 Hz high-pass filter and dynamic volume leveling before the recognition model loads. It retains silence and duration. Normal 16 kHz mono conversion is required in both modes. There are no noise-filter or gain tuning controls.

The tuned transcription also retains 16-second padded speech windows, word timestamps and same-model missed-speech recovery. Recovery does not synthesize or infer dialogue from a supplied transcript. Turning off enhancements disables the signal filters while keeping the recognition settings the same.

On one supplied 5m 33s English recording, Turbo took 14.1 seconds with enhancements and had a 5.20% normalized word error against the supplied reference. With filters disabled it took 14.3 seconds and had 17.00% word error, primarily from repeated extra words. These are single local measurements, not universal performance or accuracy guarantees. See [audio preparation results](docs/AUDIO_RECOVERY.md) and the [Turbo/large-v3 comparison](docs/ASR_MODEL_BENCHMARK.md).

## Editing, translation and exports

In **Review & export**, select a cue to preview its audio, edit words/times, and mark it reviewed. **Space** plays/pauses; **Up/Down** navigate cues. Spoken wording is retained until you edit it. Flagging helps locate uncertain passages; confidence values are not calibrated accuracy probabilities.

**Full transcript** includes every cue from the selected track, independent of editor filters. Display the original, a translation, or both. **Copy text** and **Download TXT** export the displayed version. Missing or stale translations stay visible and cannot be exported as complete translated text. Editing source text invalidates affected translations.

**Optional translation** handles supported source/target languages locally. Nearby same-language fragments are joined into bounded utterances and translated words are distributed over existing cue times. Translation remains separate from the source transcript. **Export SRT** writes to `Subtitles/<video name>/` beside the source. **Play in VLC** supplies the selected audio track and SRT explicitly.

## Embedded subtitles

**Subtitles already in this video** lists subtitle streams, languages, formats and default/forced flags. Choose a language and **Import & review** to load text/timing into the editor without speech recognition. Imported tracks have separate IDs from audio transcripts. **Extract original** saves an existing subtitle track.

| Format | Supported handling |
| --- | --- |
| SRT, ASS/SSA, WebVTT | Extract original; import normalized text/timing for editing and translation |
| MP4/MOV timed text and other supported FFmpeg text formats | Convert to SRT for editing/translation; original extraction where supported |
| PGS, DVD/DVB image tracks | Extract/preserve compatible streams; text translation unavailable |

OCR and translation of captions drawn into video images are outside the workflow. Existing styled/image subtitle streams can be preserved when the destination container supports them.

**Save video** embeds complete subtitle versions into a new MKV/MP4. MKV preserves compatible styled/image tracks, attachments, chapters and source audio/video. MP4 converts text subtitles to timed text and requires compatible source codecs; image tracks and attachments need MKV. Source media is never overwritten. Temporary output is published without replacing a file that appears while processing. Pause/failure can be retried; media copying starts again from the beginning.

Stream selection and format handling follow [FFmpeg's stream selection](https://ffmpeg.org/ffmpeg.html#Stream-selection), [subtitle support](https://ffmpeg.org/general.html#Subtitle-Formats) and [container documentation](https://ffmpeg.org/ffmpeg-formats.html).

## Development and verification

```powershell
.venv\Scripts\python.exe -m pytest
npm.cmd --prefix frontend run build
.venv\Scripts\python.exe -m subtitle_studio --no-browser
```

Tests cover audio preparation order and timing, recovery, Turbo defaults, optional large-v3 selection, cache separation, database upgrades that preserve words/edits/translations, multilingual cues, file selection, full transcripts, translation staleness, embedded extraction and safe remuxing. GitHub Actions runs backend tests and builds the interface without downloading model weights.

`scripts/validate.py` processes supplied media through the real API and stores ignored local reports. Use `--full` for complete files, `--start`/`--seconds` for excerpts, `--model large` for the fallback, or `--original-audio` to disable enhancements.

The optional FLEURS reference benchmark uses `scripts/fetch_references.py` and `scripts/benchmark_references.py`; install the `benchmark` extra for its dataset dependency. Recordings remain under ignored `data/references`. `--languages en fa ja` selects languages and `--noise-db 8` adds a reproducible noisy variant. Small samples cannot establish universal accuracy. See [verification scope](docs/VALIDATION.md).

The React/TypeScript UI is served by FastAPI. SQLite holds projects, original recognition, user edits, translations and restartable jobs. A single queue schedules CUDA recognition processes and local translation. Recognition processes exit after completing their work; the translation model unloads after its job. Cache keys include source fingerprints, settings, pipeline versions and model revisions.

The app prepares subtitles before playback. Live transcription and scene-image interpretation are outside this release.

## License

Source code is MIT licensed. The design was inspired by Caption Studio and Subtitle Mux Studio; their installers are not included. Dependencies/assets retain their own licenses: [Whisper](https://github.com/openai/whisper), [faster-whisper](https://github.com/SYSTRAN/faster-whisper), [SpeechBrain](https://speechbrain.github.io/), [Silero VAD](https://github.com/snakers4/silero-vad), [TranslateGemma](https://huggingface.co/google/translategemma-12b-it), [FFmpeg](https://ffmpeg.org/legal.html) and [VLC](https://www.videolan.org/legal.html). Optional reference recordings come from [Google FLEURS](https://huggingface.co/datasets/google/fleurs).
