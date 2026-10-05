# Validation

Tested on 5 October 2026 with Windows, an RTX 5080 (16,303 MiB VRAM), Python 3.12, and PyTorch 2.11 / CUDA 12.8. These are observed results on this machine, not minimum hardware guarantees.

## Automated and integration checks

- 45 backend tests passed; the TypeScript/production interface build passed.
- Locked installed dependencies passed compatibility checks.
- Real FFmpeg fixtures verified multiple audio tracks, delayed audio, Unicode paths, preview selection, and preservation of source files.
- Real VLC playback loaded the explicit UTF-8 SRT. Recording its audio output confirmed the first and second selected tracks at 440 Hz and 880 Hz.
- Browser checks verified editing and marking reviewed together, persisted changes, waveform/preview, original/Persian subtitle display, and exports.
- Start, stop, and restart passed from a directory containing spaces. Restart retained projects and refreshed the access token.
- Silence, white noise, and a pure tone produced no subtitle cues in the final recognition pipeline.
- Legacy translation caches were rejected. Editing any part of an utterance invalidates translations of its other fragments as well.
- Recognition, Qwen-ASR rechecks, bounded context discussions, and alignment ran on real media. SpeechBrain's registered model also loaded offline.

## Small reference diagnostic

The final Accurate recognizer was tested against three held-out public FLEURS recordings per language. The noisy version adds seeded white noise at 8 dB SNR separately to each recording. English and Persian use normalized word error rate (WER); Japanese uses character error rate (CER).

| Language | Reference units | Clean primary errors | 8 dB primary errors |
| --- | ---: | ---: | ---: |
| English | 73 words | 5 / 73 (6.85%) | 4 / 73 (5.48%) |
| Persian | 67 words | 8 / 67 (11.94%) | 16 / 67 (23.88%) |
| Japanese | 162 characters | 10 / 162 (6.17%) | 12 / 162 (7.41%) |

Language detection returned the expected language in all six cases. Recognition jobs took approximately 8–10 seconds, excluding initial language scans and queue delays. One queued noisy-Persian run took longer. The full recheck/discussion path was also tested on the final Japanese references; it retained the primary error rates and kept uncertain passages flagged.

This small diagnostic cannot rank models reliably or establish accuracy on every language, accent, background track, or conversational setting. Added noise occasionally changes recognition favorably on such a small sample. Context reviewers do not guarantee a lower error rate.

## Real-media scope

The final recognition pipeline processed 120-second excerpts from three supplied videos with recognition rechecks and context review enabled:

| Sample | Spoken language | Cues | Context discussions | Observed total GPU memory |
| --- | --- | ---: | ---: | ---: |
| 1 | German | 16 | 4 | 9,590 MiB |
| 2 | German | 8 | 4 | 10,110 MiB |
| 3 | Japanese | 25 | 21 | 10,760 MiB |

The third excerpt required substantially more review. Completion times were about 108, 145, and 214 seconds including queue waits. Unclear speech remained flagged.

Earlier integration runs processed two complete videos of roughly 18 minutes with rechecks/discussion, and one complete 106-minute video with primary recognition only. Those full-file runs used the preceding recognition-window policy. The final window-policy change was revalidated on the reference recordings and the three excerpts above; complete final-policy processing of the 106-minute file was not repeated. The highest total GPU memory observed in those earlier runs was 11,939 MiB.

The final translation policy was tested with English-to-Persian dialogue and complete German-to-English subtitle exports for both shorter files (249 and 119 cues). A fresh English-to-Persian run with the final 2,048-token translation context used 9,850 MiB total GPU memory and took 5.7 seconds for seven cues. Translation requests combine complete short utterances and distribute their words across existing cue intervals; translated word boundaries are approximate. Generated cue-label translation was replaced after testing exposed shifted and duplicated dialogue.

Detailed reports, transcripts, exports, recordings, and local machine paths are stored outside the committed source, under the ignored data directory. Models and media are excluded from Git. Context review uses neighboring dialogue and recognition evidence; the current release does not interpret scene images or identify speakers reliably in overlapping speech.

## File selection, full transcripts, and supplied context

The added tests cover Unicode media browsing, audio extensions, folder-only selection, search/pagination, missing/unreadable paths, local token/origin enforcement, old-database migration, context length limits and persistence, per-run snapshots, and protection against changing an active transcription's context. Changing notes preserves the recognition cache and invalidates the reviewer cache. Both reviewers receive the same supplied background; injected instructions cannot create a new candidate ID.

Full-transcript tests preserve multilingual words, repetition, reading order, and UTF-8 TXT output. Stale or missing translations are visible but cannot export as complete translations. Browser checks opened a real local fixture through the new picker, recovered from a missing directory, reopened saved context, copied the original/English reading view, downloaded its TXT file, switched original/translated views, and selected a different video-output folder without changing the output filename. Translation view selects the correct English/default track in the video-export dialog. The live route also returned all 249 and 119 cues with their saved English translations, and all 1,008 cues of the long video; flattening the reading paragraphs matched every stored source cue in order. These are presentation checks of existing transcripts, not new full-file recognition runs.

A smoke test with the installed Qwen3.5-9B reviewer model used two supplied candidate strings and scene notes. Both reviewers received the notes, completed their independent judgments and bounded exchange (four model calls, about 14 seconds), and returned an allowed candidate ID. This validates the updated reviewer protocol; it does not establish an accuracy improvement from background notes or replace the reference diagnostics above.

## Embedded subtitle validation

Real FFmpeg fixtures tested embedded SubRip, ASS, WebVTT, MP4/MOV timed text, PGS, and DVD subtitle tracks. The fixtures include overlapping subtitles, Persian/Japanese text, Unicode filenames, multiple audio tracks, delayed audio, ASS styles, chapters, and an attachment. Original PGS bitmap bytes are generated by the test; FFmpeg decoding to DVD subtitles verifies that the fixture contains a valid image rather than just a codec label. Bitmap tracks are extraction/preservation tests; OCR and captions burned into the video image are outside the translation scope.

MKV export retained bit-identical compressed video/audio packets and their timing, original ASS styles, chapters, attachments, forced/default flags, and the hearing-impaired flag. PGS/DVD subtitle packet hashes also matched after copying. MP4 export converted text to `mov_text`; MP4-to-MKV conversion produced readable SRT and rebuilt the chapter representation from preserved metadata. Incompatible image/data/attachment combinations were rejected. Tests also verified text import without AI models, videos without audio, stale/incomplete translation rejection, preservation of human edits, partial-file cleanup, retry after interruption, and protection against an output file appearing during copying.

In the running app, the GUI imported two Persian ASS cues and translated them to English with the installed local TranslateGemma model in **4.11 seconds**. Original intervals (0.75–2.00 and 1.50–2.50 seconds) were retained. Saving a new MKV took **0.21 seconds** for the five-second fixture and produced four subtitle tracks, two audio tracks, two chapters, and the original attachment. The English translation became the default subtitle track and the source fingerprint remained unchanged. These timings describe a tiny fixture, not movie-scale translation or copy speed.

The exported MKV played to completion in VLC with its subtitle decoder active. The headless software-decoding check produced no caption-blending errors; hardware decoding with a dummy video output was unsuitable for that diagnostic. Browser checks covered embedded-track selection, importing into the editor, separate source/translated text, source timing, English default-track selection, and the completed video-export view.
