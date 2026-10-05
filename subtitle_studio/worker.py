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
from .models import model_path, model_ready, model_revision
from .subtitles import atomic_text
from .audio_processing import prepare_track
from .speech_coverage import uncovered_speech, repetitive_text

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

    def speech(self, audio, coverage=False):
        from silero_vad import get_speech_timestamps
        return get_speech_timestamps(self.torch.from_numpy(audio.copy()), self.vad, sampling_rate=16000,
                                     threshold=.25 if coverage else .35, min_speech_duration_ms=180,
                                     min_silence_duration_ms=250 if coverage else 700,
                                     speech_pad_ms=100 if coverage else 350, return_seconds=True)

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


def segment_words(segment, origin, duration, language, index, recovered=False):
    result = []
    for word in segment.words or []:
        start = max(0, min(duration, float(word.start)))
        end = max(start, min(duration, float(word.end)))
        flags = []
        if word.probability < .5 or segment.avg_logprob < -.8:
            flags.append("Uncertain recognition")
        if segment.no_speech_prob > .5:
            flags.append("Possible nonverbal audio")
        if segment.compression_ratio > 2.4:
            flags.append("Possible repeated artifact")
        if recovered:
            flags.append("Recovered speech; check audio")
        if end <= start or end-start > 2:
            flags.append("Uncertain word timing")
        result.append({"word": word.word, "start": origin+start, "end": origin+end,
            "probability": word.probability, "language": language, "language_hint": None,
            "flags": flags, "chunk": index, "avg_logprob": segment.avg_logprob})
    return result


def recover_speech(model, audio, existing, detector, language, origin=0):
    duration = len(audio)/16000
    speech = detector.speech(np.asarray(audio), coverage=True)
    local_words = [{**w,"start":w["start"]-origin,"end":w["end"]-origin} for w in existing]
    gaps = uncovered_speech(speech,local_words,duration)
    additions, evidence = [], []
    for index, gap in enumerate(gaps):
        def recognize(margin):
            offset, end = max(0,gap["start"]-margin), min(duration,gap["end"]+margin)
            iterator, info = model.transcribe(np.asarray(audio[round(offset*16000):round(end*16000)]),
                language=language, beam_size=5, temperature=0, word_timestamps=True,
                without_timestamps=False, condition_on_previous_text=False, vad_filter=False)
            segments = list(iterator)
            picked = [word for segment in segments
                for word in segment_words(segment,offset,end-offset,language or info.language,-index-1,True)
                if gap["start"] <= (word["start"]+word["end"])/2 <= gap["end"]]
            return segments,picked
        segments, picked = recognize(.4)
        retry = any(s.compression_ratio>2.4 or repetitive_text(s.text) for s in segments)
        if retry:
            segments, picked = recognize(2)
            if repetitive_text("".join(w["word"] for w in picked)):
                picked = []
        evidence.append({**gap,"retried":retry,"text":"".join(w["word"] for w in picked)})
        additions.extend({**w,"start":w["start"]+origin,"end":w["end"]+origin} for w in picked)
        emit("progress",progress=.85+.15*(index+1)/max(1,len(gaps)),
             message=f"Recovering audible speech {index+1}/{len(gaps)}")
    return additions,evidence


def primary(manifest):
    settings, media = manifest["settings"],manifest["media"]
    model,pipeline = whisper_model(settings)
    detector = Detector()
    cache = Path(manifest["cache"]); cache.mkdir(parents=True,exist_ok=True)
    beginning = min(settings.get("start_seconds",0),media["duration"])
    ending = min(media["duration"],beginning+(settings.get("limit_seconds") or media["duration"]))
    if ending <= beginning:
        raise ValueError("The requested range contains no audio")
    # Explicit timestamp-token decoding and padded speech boundaries are needed
    # for complete recognition. Word timestamps alone do not enable that mode.
    chunk_length = min(settings.get("chunk_seconds",16),16)
    for track_number,track in enumerate(manifest["tracks"]):
        emit("progress",progress=0,message=f"Preparing audio {track} ({settings.get('audio_profile','level')})")
        audio = prepare_track(media["path"],track,beginning,ending-beginning,
            cache/f"audio-{track}.f32",settings.get("audio_profile","level"))
        marker = cache/f"track-{track}-primary.json"
        saved = json.loads(marker.read_text(encoding="utf-8")) if marker.exists() else None
        all_words, files = [], []
        if saved and saved.get("complete"):
            for filename in saved["files"]:
                chunk = json.loads((cache/filename).read_text(encoding="utf-8"))
                all_words.extend(chunk["words"])
                emit("chunk",chunk=chunk,progress=.85*(track_number+1)/len(manifest["tracks"]),message="Reading saved recognition")
        else:
            # The generator yields completed segments for immediate checkpointing.
            # A partial resume reruns deterministic decoding and reuses saved
            # segments, rather than seeking past an unfinished spoken sentence.
            batch = settings.get("batch_size",4)
            while True:
                try:
                    iterator,info = pipeline.transcribe(audio,language=settings.get("language"),
                        beam_size=5,temperature=0,word_timestamps=True,without_timestamps=False,
                        multilingual=not bool(settings.get("language")),batch_size=batch,vad_filter=True,
                        vad_parameters={"threshold":.25,"min_silence_duration_ms":250,"speech_pad_ms":400},
                        chunk_length=chunk_length)
                    decoded = iter(iterator)
                    first = next(decoded,None)
                    break
                except RuntimeError as exc:
                    if batch>1 and ("memory" in str(exc).lower() or "cublas_status_alloc" in str(exc).lower()):
                        batch=max(1,batch//2)
                        from faster_whisper import BatchedInferencePipeline
                        pipeline=BatchedInferencePipeline(model)
                        emit("warning",message=f"Reducing recognition batch to {batch}; keeping the selected model.")
                    else:
                        raise
            import itertools
            for index,segment in enumerate(itertools.chain([] if first is None else [first],decoded)):
                filename=f"track-{track}-at-{beginning+segment.start:.3f}-{index:06}.json"
                file=cache/filename
                if file.exists():
                    chunk=json.loads(file.read_text(encoding="utf-8"))
                else:
                    language=settings.get("language") or info.language
                    words=segment_words(segment,beginning,len(audio)/16000,language,index)
                    spans=[]
                    if not settings.get("language"):
                        sample=np.asarray(audio[round(segment.start*16000):round(segment.end*16000)])
                        hint=detector.language(sample)
                        if hint:
                            spans=[{"start":beginning+segment.start,"end":beginning+segment.end,**hint}]
                            for word in words:
                                word["language_hint"]=hint["language"]
                                if hint["score"]>.75 and hint["language"]!=language:
                                    word["flags"].append("Language disagreement")
                    chunk={"track":track,"start":beginning+segment.start,"end":beginning+segment.end,
                        "words":words,"languages":spans or [{"start":beginning+segment.start,"end":beginning+segment.end,
                            "language":language,"score":info.language_probability}],"raw_text":segment.text}
                    atomic_text(file,json.dumps(chunk,ensure_ascii=False))
                files.append(filename); all_words.extend(chunk["words"])
                emit("chunk",chunk=chunk,progress=.85*(track_number+segment.end/(ending-beginning))/len(manifest["tracks"]),
                     message=f"Audio {track}: {int(beginning+segment.end)} / {int(ending)} seconds")
            atomic_text(marker,json.dumps({"complete":True,"files":files}))
        if settings.get("recover_speech",True):
            file=cache/f"track-{track}-recovery.json"
            if file.exists():
                recovery=json.loads(file.read_text(encoding="utf-8"))
            else:
                words,evidence=recover_speech(model,audio,all_words,detector,settings.get("language"),beginning)
                recovery={"track":track,"start":beginning,"end":ending,"words":words,"languages":[],"evidence":evidence}
                atomic_text(file,json.dumps(recovery,ensure_ascii=False))
            emit("chunk",chunk=recovery,progress=(track_number+1)/len(manifest["tracks"]),
                 message=f"Recovered {len(recovery['words'])} additional word entries in audio {track}")
        del audio


def recheck(manifest, use_whisper=True):
    settings, media = manifest["settings"], manifest["media"]
    cases = manifest["cases"]
    if not cases:
        return
    model, _ = whisper_model({**settings, "preset": "accurate"})
    cache = Path(manifest["cache"]) / "large-rechecks"
    cache.mkdir(parents=True, exist_ok=True)
    beginning = min(settings.get("start_seconds", 0), media["duration"])
    ending = min(media["duration"], beginning + (settings.get("limit_seconds") or media["duration"]))
    prepared = {}
    for index, case in enumerate(cases):
        cue = case["cue"]
        signature = hashlib.sha256(json.dumps({"case": case, "revision": model_revision("whisper"),
            "audio_profile": settings.get("audio_profile", "level"), "protocol": 2},
            sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:16]
        file = cache / f"{cue['id']}-{signature}.json"
        if file.exists():
            result = json.loads(file.read_text(encoding="utf-8"))
        else:
            track = cue["track"]
            if track not in prepared:
                prepared[track] = prepare_track(media["path"], track, beginning, ending - beginning,
                    Path(manifest["cache"]) / f"audio-{track}.f32", settings.get("audio_profile", "level"))
            start = max(beginning, cue["start"] - .12)
            end = min(ending, cue["end"] + .12)
            audio = np.asarray(prepared[track][round((start-beginning)*16000):round((end-beginning)*16000)])
            neighbors = case.get("neighbors", [])
            context = "\n".join(n["text"] for n in neighbors if not n.get("flags") or n.get("reviewed"))[-1500:]
            segments, info = model.transcribe(audio, language=settings.get("language"),
                             word_timestamps=True, beam_size=5, condition_on_previous_text=False, temperature=(0, .2),
                             initial_prompt=context or None, vad_filter=True)
            text = "".join(segment.text for segment in segments).strip()
            result = {"cue_id": cue["id"], "text": text, "engine": "Whisper large-v3 contextual retry",
                      "candidate_id": "whisper_retry", "language": info.language, "start": start, "end": end}
            atomic_text(file, json.dumps(result, ensure_ascii=False))
        emit("recheck", result=result, progress=(index + 1) / len(cases), message=f"Checking uncertain speech {index + 1}/{len(cases)}")


def align(manifest):
    import torch
    import whisperx
    import whisperx.alignment as alignment_module
    # Cue boundaries already define the subtitle span. Avoid downloading an
    # unrelated sentence tokenizer, and retain the complete recognized text.
    class WholeCue:
        def span_tokenize(self, text):
            return [(0, len(text))]
    alignment_module.nltk_load = lambda _: WholeCue()
    torch.set_num_threads(4)
    by_language = {}
    for cue in manifest["cues"]:
        by_language.setdefault(cue["language"], []).append(cue)
    done = 0
    for language, cues in by_language.items():
        try:
            from whisperx.alignment import DEFAULT_ALIGN_MODELS_HF
            repository = DEFAULT_ALIGN_MODELS_HF.get(language)
            local_model = None
            if repository:
                snapshots = MODELS / "alignment" / ('models--' + repository.replace('/', '--')) / 'snapshots'
                if snapshots.is_dir():
                    for snapshot in sorted(snapshots.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
                        if ((snapshot/'config.json').is_file() and (snapshot/'vocab.json').is_file()
                            and ((snapshot/'pytorch_model.bin').is_file() or (snapshot/'model.safetensors').is_file())):
                            local_model = str(snapshot)
                            break
            model, metadata = whisperx.load_align_model(language_code=language, device="cpu",
                model_name=local_model, model_dir=str(MODELS / "alignment"))
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
