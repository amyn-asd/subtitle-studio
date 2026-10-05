# ASR comparison on a Persian recording

**Historical experiment:** Qwen ASR and both OmniASR variants were retired after this comparison. Their weights are outside the active model folder, and their application/benchmark adapters have been removed. The supported recognizers are Whisper Turbo and large-v3.

On this recording, tuned Whisper Turbo remains the strongest speed/accuracy compromise. OmniASR LLM 3B v2 produced the smallest edit distance against the supplied transcript, but improved word error by only **1.13 percentage points** while taking **5.59 times longer**. This is one recording and one measured full run per model, not a general language ranking.

The full recording is **962.64 seconds** (16 minutes 3 seconds). The supplied reference contains **2,664 normalized words**. All five models received the identical prepared waveform: its SHA-256 was checked across the five runs. No model received the reference, a transcript prompt, scene context, another model's output, or a text reviewer. There was no translation.

## Final outputs, including missed-speech recovery

| Model | Processing time | Words | Substitutions | Deletions | Insertions | Word error against reference |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Whisper large-v3 Turbo | **39.1 s** | **2,623** | 621 | **170** | 129 | 34.53% |
| Whisper large-v3 | 76.5 s | 2,510 | 555 | 270 | 116 | 35.32% |
| Qwen3-ASR 1.7B | 202.1 s | 2,508 | 1,027 | 242 | 86 | 50.86% |
| OmniASR CTC 3B v2* | 40.7 s | 2,270 | 556 | 432 | **38** | 38.51% |
| OmniASR LLM 3B v2* | 218.8 s | 2,530 | 560 | 232 | 98 | **33.41%** |

**\*Meta results use an experimental Windows Transformers port of the official checkpoints. They have not been validated against the native fairseq2 runtime.** Every inference tensor was checked against its expected name and shape, with strict loading. RoPE conversion and reordered decoder-cache checks passed, but these checks do not establish full native-runtime parity.

Word error is `100 × (substitutions + deletions + insertions) / 2664`. A substitution is a different word, a deletion is a reference word absent from the output, and an insertion is an extra output word in the minimum-edit alignment. These are text-alignment counts, not an independently verified listening assessment. Alternative spellings, colloquial forms, and imperfect reference text can contribute to the score. A close word count alone does not establish accuracy.

Omni LLM needed 890 word edits, compared with Turbo's 920: **30 fewer edits**. Turbo omitted fewer reference words. CTC was close to Turbo's processing time but omitted substantially more. Qwen's larger substitution count made it the weakest reference match in this particular run. All measured processing times were shorter than the video duration.

## What recovery changed

| Alternative model | Primary words | Final words | Primary word error | Final word error |
| --- | ---: | ---: | ---: | ---: |
| Qwen3-ASR 1.7B | 2,442 | 2,508 | 50.90% | 50.86% |
| OmniASR CTC 3B v2 | 2,083 | 2,270 | 40.95% | 38.51% |
| OmniASR LLM 3B v2 | 2,437 | 2,530 | **32.43%** | 33.41% |

The same recovery rule helped CTC, barely changed Qwen's score, and **worsened Omni LLM's score by 26 edits**. Omni LLM's recovery removed some omissions but added substitutions and insertions. Its primary result is shown for diagnosis; the final comparison above consistently includes recovery for every model. This suggests testing a different recovery policy for that model rather than assuming recovery always improves fidelity.

## Hardware, decoding and timing

The machine used an RTX 5080 with 16,303 MiB VRAM, 64 GB system RAM, Windows, and Python 3.12. Models ran sequentially. Model downloads and installation time are excluded.

| Model | Precision | Audio batch | Observed device peak |
| --- | --- | ---: | ---: |
| Turbo | FP16 | 4 | Not recorded in the earlier tuned run |
| large-v3 | FP16 | 4 | 8,558 MiB |
| Qwen | BF16 | 4 | 9,009 MiB |
| Omni CTC | BF16 | 4 | 12,082 MiB |
| Omni LLM | BF16 | 1 | 15,587 MiB |

Device peaks include the desktop and other existing GPU allocations; they are not model-only memory requirements. Omni LLM's measured peak left about 716 MiB of physical headroom on this machine. No out-of-memory failure occurred during the reported full runs.

Processing time includes runtime/model loading, fresh audio preparation, recognition, word alignment where needed, and missed-speech recovery. It excludes Python startup/base utility imports, downloads, reference scoring, and SRT/TXT export. The Whisper rows reuse the previously measured tuned runs on the same source. Runs were not averaged or randomly interleaved; OS file caches were not flushed. Small timing differences should not be treated as definitive.

Audio preparation used the existing **level** profile: 16 kHz mono float32 PCM, mild high-pass filtering and speech level adjustment, with no denoising or tempo change. Primary input used the same 16-second VAD clipping/concatenation, threshold 0.25, 250 ms silence boundary and 400 ms speech padding: 69 clips for each new model. The existing uncovered-speech rule selected recovery intervals separately from each model's word timings.

Whisper used its native word timestamps. Qwen and both Meta models used the existing Persian wav2vec2 model through WhisperX to assign times to their recognized words. The aligner did not replace or generate transcription text. All recognized tokens survived alignment; no sequence-mismatch fallback occurred. SRT parsing, timestamp bounds, exported word sequences, sidecar equality, and preservation of the source video were checked.

Whisper, Qwen and Omni LLM used deterministic beam size 5. CTC used its architecture's greedy collapse with blank ID 0 and no external language model. Persian was supplied to Whisper/Qwen and the `fas_Arab` language embedding to Omni LLM; CTC has no equivalent language conditioning. Generative alternatives allowed 1,024 new tokens per clip; Omni LLM reported no truncated clips. These architecture differences, separate alignment, different recovery intervals, and Meta's runtime port limit how literally this can be called an identical decoder test. The audio and evaluation rules were identical.

## Reproduce the alternative runs

The commands below document the original experiment and refer to scripts retained in Git history at commit `b0c4843`. They are no longer present in the current checkout. Install the optional AI/benchmark dependencies into the project environment, with a compatible CUDA PyTorch/torchaudio installation. The measured environment used torch 2.11.0+cu128, transformers 4.57.6, qwen-asr 0.0.6, whisperx 3.7.2, faster-whisper 1.2.1 and pyarrow 23.0.1.

Keep checkpoints outside Git. For Meta, the model directory needs the selected official `omniASR-CTC-3B-v2.pt` or `omniASR-LLM-3B-v2.pt`, `omniASR_tokenizer_written_v2.model`, and (LLM only) the official `languges_lookup_table.parquet`. The published [model cards](https://github.com/facebookresearch/omnilingual-asr/blob/81f51e224ce9e74b02cc2a3eaf21b2d91d743455/src/omnilingual_asr/cards/models/rc_models_v2.yaml) and [language table](https://github.com/facebookresearch/omnilingual-asr/blob/81f51e224ce9e74b02cc2a3eaf21b2d91d743455/src/omnilingual_asr/models/wav2vec2_llama/languges_lookup_table.parquet) identify the assets. Qwen needs a local Qwen3-ASR-1.7B snapshot. The aligner argument points to a complete local `jonatasgrosman/wav2vec2-large-xlsr-53-persian` snapshot.

Example with generic paths:

```powershell
.venv\Scripts\python.exe -X utf8 -u scripts\benchmark_asr_alternatives.py `
  --video "D:\videos\recording.mp4" --reference "D:\references\recording.txt" `
  --model omni-llm --model-directory "D:\models\omniasr" `
  --aligner "D:\models\persian-aligner" `
  --output "data\benchmarks\new-omni-llm-run" --batch 1
```

Use `--model qwen` or `--model omni-ctc` with `--batch 4` and the appropriate model directory. Output directories must be new. The script writes local SRT/TXT/metrics and creates an additional `.audio-tuned-<model>.srt` beside the source. An existing sidecar is preserved; the new output remains available in the benchmark directory. References are read only after timed inference has finished. These scripts fix the recognition language to Persian for this comparison.

The [machine-readable report](ASR_MODEL_BENCHMARK.json) contains full numeric results, phase timing, model revisions, checkpoint hashes and method limitations. Source media, supplied reference, raw dialogue, subtitles and model weights remain local. The four optional adapter checks passed for the experiment and are retained in the same Git history.

## Publisher sources

- [Qwen3-ASR](https://github.com/QwenLM/Qwen3-ASR) and [Qwen3-ASR-1.7B model](https://huggingface.co/Qwen/Qwen3-ASR-1.7B).
- [Meta Omnilingual ASR](https://github.com/facebookresearch/omnilingual-asr), including its checkpoint configurations, language syntax and beam decoder.
- [fairseq2 Windows support](https://github.com/facebookresearch/fairseq2#installing-on-windows): its official runtime requires Linux/WSL rather than native Windows.

No model weights were trained or adjusted against this reference. These results measure the chosen models and decoding settings on this recording; they do not establish performance for every language or audio condition.
