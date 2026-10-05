# Turbo with zero, one and two reviewers

**On this recording, neither review mode improved the supplied-reference score.** Turbo without reviewers was fastest and required the fewest word edits. A single context reviewer increased errors; the two-agent protocol mostly preserved the primary output and was much slower.

The source is the same 16-minute 3-second Persian recording and the same 2,664-word supplied transcript used in the [ASR comparison](ASR_MODEL_BENCHMARK.md). The exact measured, tuned Turbo output was reused: 2,623 words, 189 cues, 39.115 seconds of processing and 920 word edits. No reference text entered a recognition, review or alignment request.

## Final results

| Review mode | Total processing time* | Output words | Word edits | Word error ↓ | Reference accuracy ↑ |
| --- | ---: | ---: | ---: | ---: | ---: |
| No agents | **39.1 s** | 2,623 | **920** | **34.53%** | **65.47%** |
| One analysis agent | 373.5 s (6m 13s) | 2,591 | 949 | 35.62% | 64.38% |
| Two agents with debate | 610.7 s (10m 11s) | 2,620 | 923 | 34.65% | 65.35% |

Reference accuracy means **100% minus normalized word error**, not independently verified listening or semantic accuracy. Word error is `100 × (substitutions + deletions + insertions) / 2664`, with the same normalization as the previous comparison. The supplied reference can contain errors and spelling differences; one recording cannot establish universal reviewer performance.

One agent added **29 edits** against the reference, a 1.09 percentage-point error increase. Two agents added **three edits**, a 0.11-point increase. The small two-agent difference should not be treated as a general accuracy ranking, but its substantial processing cost was clear in this run.

## Where the time went

| Stage | No agents | One agent | Two agents |
| --- | ---: | ---: | ---: |
| Existing measured Turbo recognition/recovery | 39.115 s | 39.115 s | 39.115 s |
| Shared large-v3 candidate preparation | — | 52.082 s | 52.082 s |
| Model loading, reviewer startup and judgments | — | 233.570 s | 508.512 s |
| Replacement-word alignment on CPU | — | 48.683 s | 10.947 s |
| **Sum** | **39.115 s** | **373.451 s** | **610.656 s** |

**\*Total is the sum of these measured stages**, not three independently timed complete transcriptions. Turbo and candidate preparation were computed once and reused identically. Reference scoring, export, process startup and post-review model cleanup are excluded. Each review mode used a fresh private Ollama server and unloaded model, with no saved decision cache. Runs were sequential, one agent first; OS/driver caches were not flushed. The first single-agent call took 62.17 seconds, while the first two-agent call took 4.11 seconds. Startup/caching therefore affected the timing comparison. These are single-run measurements on an RTX 5080 with 16 GB VRAM, Windows and Python 3.12.

## What the reviewers actually did

Turbo flagged **70 cues**. Large-v3 rechecked those intervals in the same prepared audio, using the existing tight boundaries and reliable neighboring text. The waveform hash matched Turbo's prepared waveform exactly. **68 cues** had a distinct alternative; the remaining cues did not need an LLM judgment. Adding these alternatives alone did not change any output text.

Both modes received the same immutable candidate pool, original neighboring dialogue, language and flags. No scene notes were supplied. Reviewers had **text evidence only**, not direct audio access, and could select only an existing ASR candidate. Neither mode generated replacement dialogue or filled missing sentences from imagination.

Both used the existing `qwen3.5:9b` context model, digest `56671c2ab9385f9cfcb404638e32cd62d88e3501d44822208363c010179a3c90`, temperature 0, thinking disabled, context limit 8,192 and output limit 768 tokens. This is the separate text review model; Qwen3-ASR 1.7B has been retired.

- **One agent:** the existing context-review role made one judgment per cue, using the same context-role prompt and candidate presentation as the two-agent protocol. It made 68 calls and selected alternatives in **55 cues**. Two replies were invalid/incomplete JSON; the safe fallback retained Turbo's wording for those cues.
- **Two agents:** the existing recognition and context roles judged independently, then both reconsidered the same initial snapshot if they disagreed. There were **244 calls**, **54 initial disagreements out of 68 cases (79.4%)**, and **27 unresolved disagreements** after the bounded exchange. Unresolved cases retained the primary text. The final output changed **seven cues**; there were no failed decisions.

These are two logical roles sharing one model, not two independently trained models. Agreement is not acoustic verification. The two-agent result remained close to Turbo largely because it retained the original wording in most cases, while the single agent selected many more alternatives.

## Timing and source checks

All three outputs retained exactly the same **189 subtitle cue start/end boundaries**. Selected replacement words were aligned through WhisperX using the existing local Persian timing model. No alignment warning occurred. Aligning words did not change subtitle text or generate additional words. There is no human-timed reference, so this does not establish improved timestamp accuracy.

The three SRTs were parsed; timing bounds and exported word sequences were checked. Raw Turbo wording stayed preserved as `raw_text`, every replacement belonged to the shared candidate pool, the source video fingerprint stayed unchanged, and both reviewed sidecars matched their benchmark exports after newline normalization. The original application transcript and earlier subtitle exports were preserved.

## Current application and reproduction

The supported ASR choices are **Whisper Turbo and large-v3**. Large-v3 now rechecks uncertain passages using the prepared audio. Qwen ASR and both OmniASR variants, their runtime dependencies, and their adapters were removed from the active setup. Their local files were archived outside the active model directory. The historical comparison remains available as an experiment record.

The GUI's **Context review** control offers **One analysis agent** or **Two agents with debate**. Review mode belongs in the decision-cache signature; switching reviewer counts reuses the recognition cache but computes the appropriate judgments. Turning context review off retains the primary transcription. Local tests cover candidate guards, safe fallbacks, bounded debate, cache separation, large-v3 selection and prepared-audio timing offsets.

The no-agent table row disables both context review and the additional acoustic recheck. Keeping **Recheck uncertain speech** enabled still spends time preparing alternatives, even with context review off.

`scripts/benchmark_reviewers.py` records the three stages in separate processes. For the recorded experiment's source metadata, primary cache and scored baseline result:

```powershell
.venv\Scripts\python.exe -X utf8 -u scripts\benchmark_reviewers.py `
  --stage candidates --source "data\source.json" `
  --baseline-cache "data\turbo-primary-cache" `
  --baseline-result "data\turbo-scored-result.json" `
  --reference "D:\references\recording.txt" --output "data\reviewer-benchmark"
```

Run again with `--stage one`, then `--stage two`, using the same arguments and optional `--review-context` value. The candidate stage requires a new output folder; reviewer stages reuse its saved candidates without rereading audio or regenerating alternatives. Each reviewer output folder must also be new. This audit script reuses the recorded track-1 baseline and its measured runtime. Metrics and raw decisions stay local; only sanitized statistics and public model identities appear in the [machine-readable report](REVIEWER_BENCHMARK.json).
