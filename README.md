# Subtitle Studio

A local video-to-subtitle workspace for Windows. Transcribe multilingual audio or import existing embedded subtitles, review and translate the text, then export subtitles, save a subtitled video, or play it in VLC.

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
- Embedded text-subtitle inspection, original-format extraction, editing, and local translation.
- New MKV/MP4 video export with selected subtitle versions, default-track selection, and original audio/video quality.

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

### Embedded subtitles

After selecting a video, **Subtitles already in this video** lists its subtitle streams, language metadata, format, and default/forced flags. For text tracks, choose or correct **Text language**, then use **Import & review**. This loads the existing words and times into the editor without speech recognition. The imported track has its own source ID and cannot overwrite an audio transcript. **Optional translation** works the same way as for transcribed audio; translated text remains separate. For an imported track, the preview audio can be selected independently.

**Extract original** prepares a downloadable sidecar. The supported path depends on the subtitle codec and the installed FFmpeg build:

| Embedded format | Extraction and translation |
| --- | --- |
| SubRip/SRT | Original SRT, editable and translatable |
| ASS/SSA | Original ASS with styles; plain words and times imported for translation |
| WebVTT | Original VTT, editable and translatable |
| MP4/MOV timed text | Converted to UTF-8 SRT for editing/translation |
| Other recognized FFmpeg text formats, including TTML | Original format where a muxer is available; normalized SRT for editing/translation |
| Blu-ray PGS | Original SUP image track; extraction and MKV preservation |
| DVD/DVB image subtitles | Subtitle-only MKV (`.mks`); extraction and compatible MKV preservation |

Image subtitles contain pictures rather than words. OCR is not implemented; the interface identifies these tracks and keeps translation unavailable. You can preserve/extract them or transcribe the video's audio to create a separate text track. Captions drawn into the video image are not embedded subtitle streams.

Use **Save video** in the editor, or **Save a subtitled video** beside the embedded-track list. Choose one or more complete subtitle versions, an optional default track, and whether to retain the other embedded subtitles. The output must be a new absolute MKV/MP4 filename. A completed temporary file is published without replacing an existing file, including a file that appears during processing. Paused/failed copies can be retried; video copying restarts from the beginning.

**MKV** preserves compatible original styled/image tracks, attachments, chapters, and audio/video streams. Translated or edited subtitles use simple SRT styling; an unedited original is copied directly from the source, preserving its styling. **MP4** converts text subtitles to timed text and requires compatible source audio/video codecs. Image subtitles and attachments require MKV; advanced styles and overlapping presentation may change in MP4. Incompatible data/codecs produce an error rather than silently re-encoding or dropping streams. MOV chapter data is rebuilt from the retained chapter metadata for the destination container. **Play saved video in VLC** opens the completed output with its embedded default track.

Stream mapping and format handling follow the [FFmpeg stream-selection documentation](https://ffmpeg.org/ffmpeg.html#Stream-selection), [subtitle codec support](https://ffmpeg.org/general.html#Subtitle-Formats), and [container documentation](https://ffmpeg.org/ffmpeg-formats.html). Extracting and embedding do not require an AI model or GPU; local text translation uses the configured translation model.

## Development and testing

```powershell
.venv\Scripts\python.exe -m pytest
npm.cmd --prefix frontend run build
.venv\Scripts\python.exe -m subtitle_studio --no-browser
```

The normal test suite checks candidate validation, bounded discussion, failure fallbacks, preservation of repeated words, timing offsets, multiple audio tracks, Unicode paths, source preservation, translation staleness, and embedded subtitle extraction/import/remux. Real SRT, ASS, WebVTT, timed-text, PGS, and DVD fixtures verify video/audio packet preservation, styles, chapters, attachments, accessibility flags, cancellation/retry, and output-file races. Media tests require FFmpeg. GitHub Actions runs backend tests and the frontend build without downloading AI weights.

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
