# Subtitle Studio

A local video-to-subtitle workspace for Windows. Select a video, choose its audio tracks, transcribe multilingual speech, review uncertain passages, and export subtitles or play the original video in VLC.

![Local processing](https://img.shields.io/badge/processing-local-195b49) ![Python](https://img.shields.io/badge/python-3.12-195b49) ![License](https://img.shields.io/badge/license-MIT-195b49)

## Features

- Audio-track inspection and sampled language detection before processing.
- Language checks throughout selected audio, including changes in conversation.
- Accurate Whisper large-v3 recognition; optional faster Whisper turbo mode.
- Selective Qwen3-ASR rechecks for difficult passages.
- Two context reviewers with independent initial judgments and one bounded exchange.
- Verbatim wording, editable timing, Unicode/RTL text, alternatives, and review flags.
- A virtualized transcript editor, waveform, and short synchronized video previews.
- Separate optional translations through TranslateGemma, using complete short utterances.
- Resumable processing and SRT export, with explicit audio/subtitle selection in VLC.

Context reviewers select **only supplied recognition candidates**. They cannot generate replacement dialogue. Their agreement keeps the passage flagged; uncertainty scores are not presented as calibrated probabilities. Recognition is a best hypothesis and may still require listening, particularly for overlapping voices or damaged audio.

Accurate mode recognizes complete short audio windows, with overlap, so quiet words are not cut away by a second speech filter. Independent window decoding avoids skipping dialogue after pauses. Nonverbal filtering combines the recognizer's no-speech evidence with speech-region and recognition evidence; uncertain words remain flagged. Differences in punctuation alone retain the original wording and do not trigger a discussion. Human edits and review decisions take priority over AI results, including requests already in flight.

## Windows quick start

1. Install Node.js 20.19+ for the initial interface build. Normal use does not require Node.
2. Make FFmpeg/ffprobe and Ollama available on PATH, or use their standard Windows installations. VLC is optional. Other executable locations can be selected in Settings.
3. Run `Setup.bat`. It installs a private Python 3.12 environment and the locked runtime dependencies, then builds the interface.
4. Open `Start.bat` and use **Models & settings** to install missing models.
5. Choose a video, select audio tracks, and create subtitles. Review flagged passages before sharing important results.

`Stop.bat` closes the local server and saves completed processing chunks. Closing the browser alone leaves the server running. Reopening `Start.bat` reconnects to it.

An NVIDIA GPU with around 16 GB VRAM is recommended for Accurate mode. CUDA 12.8 PyTorch wheels are pinned for modern NVIDIA hardware, including Blackwell. Models run in sequential stages; the app never intentionally loads all large models together. Other GPU applications still affect available memory.

## Model and tool locations

The repository contains source code, configuration examples, and tests. It excludes model weights, videos, subtitle exports, databases, runtime environments, logs, and local settings.

The default recognition-model location is the user's local application-data folder. Choose any drive in **Models & settings**, save, then use **Restart app**. An Ollama model store can be shared with an existing installation; Subtitle Studio runs its own loopback Ollama server and does not change the user's Ollama service configuration.

Compatible Hugging Face snapshots are registered in place. For additional cache directories, copy `studio-config.example.json` to `studio-config.json` and add `model_search_paths` pointing to Hugging Face `hub` directories. `models_root`, `ollama_models`, and `tools` are optional overrides. `SUBTITLE_STUDIO_MODELS` and `SUBTITLE_STUDIO_DATA` environment variables can override model and project storage.

Local configuration is deliberately ignored by Git. Do not commit your filled-in settings file or exported project backups.

| Stage | Model | Notes |
| --- | --- | --- |
| Transcription | Whisper large-v3, FP16 | Broad language coverage through faster-whisper |
| Rechecks | Qwen3-ASR 1.7B | 30 main languages, including Persian; Whisper retries cover others |
| Language hints | SpeechBrain VoxLingua107 | Independent audio classifier; disagreements remain hints |
| Speech regions | Silero VAD | Conservative CPU detection with padding |
| Discussion | Qwen3.5-9B Q4 | Two logical reviewers share one model with separate conversations |
| Translation | TranslateGemma 12B Q4 | Optional; translation never overwrites original wording |
| Replacement timing | WhisperX aligners | Downloaded per language when needed; cue timing retained on failure |

Models have separate licenses. Downloads come from their publishers/model repositories; media is not uploaded. Core processing can run offline after the required models have been installed. A new alignment language may need an initial download.

## Review and exports

Select a cue to prepare a short preview with the correct audio track. Use **Space** to play/pause and **Up/Down** to navigate. Text saves when its editor loses focus. You can choose an alternative, edit text/times, and mark a passage reviewed.

Default exports are placed beside the source, under `Subtitles/<video name>/`:

```text
<video>.audio-1.original.srt
<video>.audio-1.fa.srt
```

Mixed-language originals retain the spoken languages. Translation joins nearby same-language sentence fragments into bounded utterances, then distributes the translated words over their original cue IDs/times. This avoids asking a translator to guess sentence boundaries from generated cue labels. Word boundaries in translated subtitles are approximate because languages reorder words. Edited source text invalidates translations for the affected utterance, preventing stale translation export. **Play in VLC** explicitly supplies the subtitle file and selected audio ordinal. The original media is never rewritten.

## Development and testing

```powershell
.venv\Scripts\python.exe -m pytest
npm.cmd --prefix frontend run build
.venv\Scripts\python.exe -m subtitle_studio --no-browser
```

The normal test suite checks candidate validation, bounded discussion, failure fallbacks, preservation of repeated words, timing offsets, multiple audio tracks, Unicode paths, source preservation, and translation staleness. Media tests require FFmpeg. GitHub Actions runs backend tests and the frontend build without downloading AI weights.

With the app running, `scripts/validate.py` processes supplied media through the real API and records runtime, total observed GPU memory, languages, review counts, and export results under ignored `data/validation/`. Use `--full` for full files or `--start`/`--seconds` for a sample. `--recognition-only` isolates long-file recognition performance. No dialogue is printed to the console.

The optional reference benchmark uses public FLEURS recordings and measures word/character errors against known transcriptions. Small benchmark results establish behavior on those samples, not universal accuracy across languages and recording conditions. See [validation results](docs/VALIDATION.md) for the tested scope and limitations.

Install `pyarrow` (`uv pip install --python .venv\Scripts\python.exe pyarrow`), run `scripts/fetch_references.py`, then `scripts/benchmark_references.py`. The download is about 1.8 GB and stays under ignored `data/references/`. `--noise-db 8` adds a reproducible noisy variant; `--languages en fa ja` selects languages, and `--recognition-only` isolates primary recognition. These recordings do not represent all conversational audio.

TranslateGemma uses its publisher's [source/target prompt format](https://ollama.com/library/translategemma). Each request contains one contiguous short utterance, without generated labels or unrelated neighboring text. Language switches, completed sentences, and longer pauses start new groups. Valid translations are reused on resume; cache entries include the model digest, translation version, and complete source-group signature. If a result is too short to distribute across its source cues, those cues are translated individually and a warning is recorded.

## Architecture

The React/TypeScript interface is served by a FastAPI loopback server. A single job queue schedules GPU stages. CUDA recognizers live in child processes that exit between stages; context and translation models are explicitly unloaded. SQLite stores projects, raw recognition, user edits, alternatives, decisions, and restartable job state. Cache keys include source fingerprints, settings, pipeline versions, and model revisions.

API requests require a per-run token. Host/origin checks protect local file and player operations. Source dialogue is quoted data and cannot invoke tools or change configuration.

The app prepares subtitles before playback. Live transcription, scene-image interpretation, video embedding, and required speaker-identification accounts are outside the current release.

## License and acknowledgments

Source code is MIT licensed. The design was inspired by Caption Studio and Subtitle Mux Studio; their installers are not included in this repository. Dependencies and model weights retain their own licenses: [Whisper](https://github.com/openai/whisper), [faster-whisper](https://github.com/SYSTRAN/faster-whisper), [Qwen3-ASR](https://github.com/QwenLM/Qwen3-ASR), [Qwen3.5](https://huggingface.co/Qwen/Qwen3.5-9B), [SpeechBrain](https://speechbrain.github.io/), [Silero](https://github.com/snakers4/silero-vad), [TranslateGemma](https://huggingface.co/google/translategemma-12b-it), [WhisperX](https://github.com/m-bain/whisperX), [FFmpeg](https://ffmpeg.org/legal.html), and [VLC](https://www.videolan.org/legal.html). FLEURS reference recordings are from [Google's dataset](https://huggingface.co/datasets/google/fleurs).
