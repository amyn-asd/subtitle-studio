"""Model subprocesses. The HTTP/UI process never loads CUDA models."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

from .config import DATA, MODELS
from .media import audio_window
from .models import model_path, model_ready
from .subtitles import atomic_text

QWEN_LANGUAGES = {"zh": "Chinese", "en": "English", "yue": "Cantonese", "ar": "Arabic", "de": "German",
                  "fr": "French", "es": "Spanish", "pt": "Portuguese", "id": "Indonesian", "it": "Italian",
                  "ko": "Korean", "ru": "Russian", "th": "Thai", "vi": "Vietnamese", "ja": "Japanese",
                  "tr": "Turkish", "hi": "Hindi", "ms": "Malay", "nl": "Dutch", "sv": "Swedish", "da": "Danish",
                  "fi": "Finnish", "pl": "Polish", "cs": "Czech", "fil": "Filipino", "tl": "Filipino", "fa": "Persian",
                  "el": "Greek", "hu": "Hungarian", "mk": "Macedonian", "ro": "Romanian"}
_dll_handles = []


def emit(kind: str, **data):
    print(json.dumps({"type": kind, **data}, ensure_ascii=False), flush=True)


def cuda():
    import torch
    torch.set_num_threads(4)
    if os.name == "nt":
        directory = str(Path(torch.__file__).parent / "lib")
        os.environ["PATH"] = directory + os.pathsep + os.environ["PATH"]
        _dll_handles.append(os.add_dll_directory(directory))
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Run Setup.ps1 to install the RTX-compatible PyTorch runtime.")
    return torch


class Detector:
    def __init__(self):
        import torch
        from silero_vad import load_silero_vad
        torch.set_num_threads(4)
        self.torch = torch
        self.vad = load_silero_vad(onnx=True)
        self.lid = None
        if model_ready("lid"):
            try:
                from speechbrain.inference.classifiers import EncoderClassifier
                from speechbrain.utils.fetching import LocalStrategy
                self.lid = EncoderClassifier.from_hparams(source=str(model_path("lid")), savedir=str(MODELS / "lid-runtime"),
                                                          overrides={"pretrained_path": str(model_path("lid"))},
                                                          run_opts={"device": "cpu"}, local_strategy=LocalStrategy.COPY)
            except Exception as exc:
                emit("warning", message=f"Independent language detector unavailable: {exc}")

    def speech(self, audio):
        from silero_vad import get_speech_timestamps
        return get_speech_timestamps(self.torch.from_numpy(audio.copy()), self.vad, sampling_rate=16000,
                                     threshold=.35, min_speech_duration_ms=180, min_silence_duration_ms=700,
                                     speech_pad_ms=350, return_seconds=True)

    def language(self, audio) -> dict | None:
        if self.lid is None or len(audio) < 32000:
            return None
        with self.torch.inference_mode():
            _, score, _, labels = self.lid.classify_batch(self.torch.from_numpy(audio.copy()).unsqueeze(0))
        confidence = float(score.flatten()[0])
        if confidence < 0:
            confidence = math.exp(confidence)
        return {"language": labels[0].split(":")[0].strip(), "score": min(1, confidence)}


def whisper_model(settings):
    cuda()
    from faster_whisper import WhisperModel, BatchedInferencePipeline
    key = "turbo" if settings.get("preset") == "fast" else "whisper"
    if not model_ready(key):
        raise RuntimeError(f"Download {key} in Models before processing.")
    model = WhisperModel(str(model_path(key)), device="cuda", compute_type="float16", cpu_threads=4, num_workers=1)
    return model, BatchedInferencePipeline(model)


def scan(manifest):
    media = manifest["media"]
    model, _ = whisper_model(manifest["settings"])
    detector = Detector()
    tracks = media["audio_tracks"]
    for n, track in enumerate(tracks):
        estimates = []
        duration = media["duration"]
        positions = sorted({max(0, min(duration - 12, duration * fraction)) for fraction in (.02, .12, .28, .48, .68, .86, .97)})
        for i, position in enumerate(positions):
            audio = audio_window(media["path"], track["stream_index"], position, min(12, duration - position))
            speech = detector.speech(audio)
            if not speech:
                continue
            start = int(speech[0]["start"] * 16000)
            sample = audio[start:start + 16000 * 10]
            language, probability, _ = model.detect_language(sample)
            independent = detector.language(sample)
            estimates.append({"time": position + start / 16000, "language": language,
                              "score": probability, "independent": independent})
            emit("progress", progress=(n + (i + 1) / len(positions)) / max(1, len(tracks)),
                 message=f"Sampling languages in audio {track['audio_ordinal'] + 1}")
        summary = {}
        for item in estimates:
            lang = item["language"]
            summary[lang] = max(summary.get(lang, 0), item["score"])
            other = item["independent"]
            if other and other["score"] >= .7:
                summary[other["language"]] = max(summary.get(other["language"], 0), other["score"])
        track.update(languages=[{"code": lang, "score": confidence} for lang, confidence in sorted(summary.items(), key=lambda kv: -kv[1])],
                     detection="sampled" if estimates else "no_speech_in_samples", samples=estimates)
        emit("track", track=track)
    emit("result", media=media)


def primary(manifest):
    settings, media = manifest["settings"], manifest["media"]
    model, pipeline = whisper_model(settings)
    detector = Detector()
    cache = Path(manifest["cache"])
    cache.mkdir(parents=True, exist_ok=True)
    # Keep the whole window, including its two-second margins, within Whisper's
    # 30-second acoustic input. Independent decoding avoids timestamp-driven
    # seeking that can skip an utterance after a pause.
    window_size = min(settings["chunk_seconds"], 26) if settings["preset"] == "accurate" else settings["chunk_seconds"]
    beginning = min(settings.get("start_seconds", 0), media["duration"])
    ending = min(media["duration"], beginning + (settings.get("limit_seconds") or media["duration"]))
    windows = [(track, start) for track in manifest["tracks"] for start in np.arange(beginning, ending, window_size)]
    batch = settings.get("batch_size", 4)
    for index, (track, start) in enumerate(windows):
        start = float(start)
        end = min(ending, start + window_size)
        file = cache / f"track-{track}-at-{start:.3f}.json"
        if file.exists():
            chunk = json.loads(file.read_text(encoding="utf-8"))
        else:
            offset = max(0, start - 2)
            audio = audio_window(media["path"], track, offset, min(media["duration"], end + 2) - offset)
            speech = detector.speech(audio)
            chunk = {"track": track, "start": start, "end": end, "words": [], "languages": [], "speech": speech}
            # Accurate mode lets the recognizer hear the complete bounded window. Applying
            # a second VAD trim was dropping quiet dialogue; VAD remains boundary evidence.
            has_signal = bool(np.mean(audio * audio) > 1e-7)
            if speech or (settings["preset"] == "accurate" and has_signal):
                while True:
                    try:
                        if settings["preset"] == "accurate":
                            iterator, info = pipeline.transcribe(audio, language=settings.get("language"), beam_size=5,
                                          word_timestamps=True, multilingual=not bool(settings.get("language")),
                                          vad_filter=False, clip_timestamps=[{"start": 0, "end": len(audio) / 16000}],
                                          batch_size=batch, temperature=0)
                        else:
                            iterator, info = pipeline.transcribe(audio, language=settings.get("language"), beam_size=5,
                                          word_timestamps=True, condition_on_previous_text=False, multilingual=not bool(settings.get("language")),
                                          vad_filter=True, vad_parameters={"threshold": .35, "min_silence_duration_ms": 700, "speech_pad_ms": 350},
                                          batch_size=batch, temperature=0)
                        segments = list(iterator)
                        break
                    except RuntimeError as exc:
                        if ("memory" in str(exc).lower() or "cublas_status_alloc" in str(exc).lower()) and batch > 1:
                            batch = max(1, batch // 2)
                            emit("warning", message=f"Reducing recognition batch to {batch}; retaining the same accuracy model.")
                        else:
                            raise
                spans = []
                for position in range(0, len(audio), 8 * 16000):
                    sample = audio[position:position + 8 * 16000]
                    if len(sample) >= 2 * 16000 and detector.speech(sample):
                        lid = detector.language(sample)
                        if lid:
                            spans.append({"start": offset + position / 16000, "end": offset + (position + len(sample)) / 16000, **lid})
                chunk["languages"] = spans or [{"start": start, "end": end, "language": info.language, "score": info.language_probability}]
                for segment in segments:
                    speech_overlaps = any(s["start"] < segment.end and s["end"] > segment.start for s in speech)
                    # Batched decoding does not apply Whisper's no-speech filter.
                    # Combine acoustic no-speech evidence with VAD/recognition
                    # evidence; this rejects noise without trimming quiet words.
                    if segment.no_speech_prob > .6 and (not speech_overlaps or segment.avg_logprob <= -1):
                        continue
                    for word in segment.words or []:
                        absolute_start = offset + word.start
                        absolute_end = offset + word.end
                        midpoint = (absolute_start + absolute_end) / 2
                        if not start <= midpoint < end:
                            continue
                        flags = []
                        relative_midpoint = midpoint - offset
                        if not any(s["start"] <= relative_midpoint <= s["end"] for s in speech):
                            flags.append("Uncertain speech boundary")
                        if word.probability < .5 or segment.avg_logprob < -.8:
                            flags.append("Uncertain recognition")
                        if segment.no_speech_prob > .5:
                            flags.append("Possible nonverbal audio")
                        if segment.compression_ratio > 2.4:
                            flags.append("Possible repeated artifact")
                        language = settings.get("language") or info.language
                        span = next((s for s in spans if s["start"] <= midpoint < s["end"]), None)
                        if not settings.get("language") and span and span["score"] > .75 and span["language"] != info.language:
                            flags.append("Language disagreement")
                        if info.language_probability < .7:
                            flags.append("Uncertain language")
                        chunk["words"].append({"word": word.word, "start": max(beginning, absolute_start),
                            "end": min(ending, absolute_end), "probability": word.probability, "language": language,
                            "language_hint": span["language"] if span else None,
                            "flags": flags, "chunk": index, "avg_logprob": segment.avg_logprob})
            atomic_text(file, json.dumps(chunk, ensure_ascii=False))
        emit("chunk", chunk=chunk, progress=(index + 1) / max(1, len(windows)),
             message=f"Audio {track}: {int(end)} / {int(ending)} seconds")


def recheck(manifest, use_whisper=False):
    settings, media = manifest["settings"], manifest["media"]
    cases = manifest["cases"]
    if not cases:
        return
    torch = cuda()
    if use_whisper:
        model, _ = whisper_model(settings)
    else:
        from qwen_asr import Qwen3ASRModel
        if not model_ready("qwen_asr"):
            raise RuntimeError("Download Qwen3-ASR in Models before enabling recognition rechecks.")
        model = Qwen3ASRModel.from_pretrained(str(model_path("qwen_asr")), dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
                                            device_map="cuda:0", attn_implementation="sdpa",
                                            max_inference_batch_size=1, max_new_tokens=384)
    cache = Path(manifest["cache"]) / ("whisper-rechecks" if use_whisper else "qwen-rechecks")
    cache.mkdir(parents=True, exist_ok=True)
    for index, case in enumerate(cases):
        cue = case["cue"]
        signature = hashlib.sha256(json.dumps(case, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:16]
        file = cache / f"{cue['id']}-{signature}.json"
        if file.exists():
            result = json.loads(file.read_text(encoding="utf-8"))
        else:
            start = max(0, cue["start"] - .12)
            duration = min(media["duration"], cue["end"] + .12) - start
            audio = audio_window(media["path"], cue["track"], start, duration)
            neighbors = case.get("neighbors", [])
            context = "\n".join(n["text"] for n in neighbors if not n.get("flags") or n.get("reviewed"))[-1500:]
            if use_whisper:
                segments, info = model.transcribe(audio, language=settings.get("language"),
                                 word_timestamps=True, beam_size=5, condition_on_previous_text=False, temperature=(0, .2),
                                 initial_prompt=context or None, vad_filter=True)
                text = "".join(segment.text for segment in segments).strip()
                detected_language = info.language
            else:
                recognized = model.transcribe(audio=(audio, 16000), context=context,
                                 language=QWEN_LANGUAGES.get(settings.get("language")))
                text = recognized[0].text.strip()
                detected_language = next((code for code, name in QWEN_LANGUAGES.items() if name.lower() == recognized[0].language.lower()), cue["language"])
            result = {"cue_id": cue["id"], "text": text, "engine": "Whisper contextual retry" if use_whisper else "Qwen3-ASR",
                      "candidate_id": "whisper_retry" if use_whisper else "qwen", "language": detected_language, "start": start, "end": start + duration}
            atomic_text(file, json.dumps(result, ensure_ascii=False))
        emit("recheck", result=result, progress=(index + 1) / len(cases), message=f"Checking uncertain speech {index + 1}/{len(cases)}")


def align(manifest):
    import torch
    import whisperx
    torch.set_num_threads(4)
    by_language = {}
    for cue in manifest["cues"]:
        by_language.setdefault(cue["language"], []).append(cue)
    done = 0
    for language, cues in by_language.items():
        try:
            model, metadata = whisperx.load_align_model(language_code=language, device="cpu", model_dir=str(MODELS / "alignment"))
        except Exception as exc:
            emit("warning", message=f"Word alignment unavailable for {language}: {exc}")
            continue
        for cue in cues:
            audio = audio_window(manifest["media"]["path"], cue["track"], cue["start"], cue["end"] - cue["start"])
            result = whisperx.align([{"start": 0, "end": len(audio) / 16000, "text": cue["text"]}], model, metadata, audio, "cpu")
            words = [{**word, "start": word["start"] + cue["start"], "end": word["end"] + cue["start"]}
                     for word in result.get("word_segments", []) if "start" in word and "end" in word]
            done += 1
            emit("alignment", cue_id=cue["id"], words=words, progress=done / max(1, len(manifest["cues"])))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", required=True, choices=["scan", "primary", "recheck", "recheck_whisper", "align"])
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if args.stage == "recheck_whisper":
        recheck(manifest, True)
    else:
        globals()[args.stage](manifest)
    emit("done")


if __name__ == "__main__":
    main()
