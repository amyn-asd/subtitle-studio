# Validation

Tested on 5 October 2026 with Windows, an RTX 5080 (16,303 MiB VRAM), Python 3.12, and PyTorch 2.11 / CUDA 12.8. These are observed results on this machine, not minimum hardware guarantees.

## Automated and integration checks

- 24 backend tests passed; the TypeScript/production interface build passed.
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
