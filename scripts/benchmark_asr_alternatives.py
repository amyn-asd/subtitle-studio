"""Benchmark alternative ASR on prepared audio; references are scoring-only.

Uses the app's level adjustment, Whisper's 16s VAD input clips and the app's
missed-speech recovery rules. A separate Persian CTC aligner supplies timings,
without replacing any recognized words. Models and reference stay outside Git.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import re
import subprocess
import sys
import threading
import time
import zlib
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from subtitle_studio.audio_processing import prepare_track
from subtitle_studio.media import NO_WINDOW, fingerprint
from subtitle_studio.speech_coverage import repetitive_text, uncovered_speech
from subtitle_studio.subtitles import group_words, parse_srt, srt
from scripts.transcript_metrics import compare, words


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument('--source', type=Path, help='Existing benchmark source metadata JSON')
    inputs.add_argument('--video', type=Path, help='Video to probe directly')
    parser.add_argument('--track', type=int, help='FFmpeg audio stream index; defaults to the first audio track')
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--model', choices=['qwen', 'omni-ctc', 'omni-llm'], required=True)
    parser.add_argument('--model-directory', type=Path, required=True)
    parser.add_argument('--aligner', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--batch', type=int, default=4)
    args = parser.parse_args()
    if args.batch < 1:
        parser.error('--batch must be positive')
    if args.model == 'omni-llm' and args.batch != 1:
        parser.error('The serial Omni LLM adapter requires --batch 1')
    args.output.mkdir(parents=True, exist_ok=False)
    if args.source:
        source = json.loads(args.source.read_text(encoding='utf-8'))
    else:
        from subtitle_studio.media import probe
        media = probe(str(args.video.resolve()))
        if not media['audio_tracks']:
            parser.error('The video has no audio track')
        source = dict(media=media, track=media['audio_tracks'][0]['stream_index'])
    if args.track is not None:
        source['track'] = args.track
    media, track = source['media'], source['track']
    if track not in [t['stream_index'] for t in media['audio_tracks']]:
        parser.error('Selected stream is not an audio track')
    assert fingerprint(Path(media['path'])) == media['fingerprint']
    phases, gpu, stopping = {}, [], threading.Event()
    def emit(**data):
        print(json.dumps(data, ensure_ascii=False), flush=True)
    def sample_gpu():
        while not stopping.is_set():
            try:
                result = subprocess.run(['nvidia-smi', '--query-gpu=memory.used,utilization.gpu',
                    '--format=csv,noheader,nounits'], capture_output=True, text=True,
                    timeout=3, creationflags=NO_WINDOW)
                if result.returncode == 0:
                    memory, utilization = map(int, result.stdout.strip().split(','))
                    gpu.append(dict(memory_mib=memory, utilization_percent=utilization))
            except (OSError, ValueError, subprocess.TimeoutExpired):
                pass
            stopping.wait(1)
    sampler = threading.Thread(target=sample_gpu, daemon=True)
    sampler.start()
    started = time.perf_counter()
    from subtitle_studio.worker import cuda, Detector
    torch = cuda()
    from faster_whisper.vad import VadOptions, get_speech_timestamps, collect_chunks, SpeechTimestampsMap
    import whisperx.alignment as alignment_module
    from whisperx.alignment import load_align_model, align
    # Sentence splitting only affects subtitle presentation. Align the full clip
    # as one span, avoiding NLTK's English tokenizer for Persian speech.
    class WholeClip:
        def span_tokenize(self, text):
            return [(0, len(text))]
    alignment_module.nltk_load = lambda _: WholeClip()
    if args.model == 'qwen':
        from qwen_asr import Qwen3ASRModel
        engine = Qwen3ASRModel.from_pretrained(str(args.model_directory), dtype=torch.bfloat16,
            device_map='cuda:0', attn_implementation='sdpa',
            max_inference_batch_size=args.batch, max_new_tokens=1024)
        engine.model.generation_config.do_sample = False
        engine.model.generation_config.num_beams = 5
        def recognize(clips):
            return [r.text for r in engine.transcribe(audio=[(np.asarray(a),16000) for a in clips],
                context='', language='Persian', return_time_stamps=False)]
    else:
        from benchmark_omni_torch import OmniRecognizer
        engine = OmniRecognizer(args.model_directory, args.model, beam=5)
        recognize = engine.transcribe
    alignment_model, alignment_metadata = load_align_model('fa', 'cuda', model_name=str(args.aligner))
    phases['runtime_and_models'] = time.perf_counter()-started
    began = time.perf_counter()
    audio = prepare_track(media['path'], track, 0, media['duration'], args.output/'audio.f32', 'level')
    with (args.output/'audio.f32').open('rb') as prepared:
        audio_sha = hashlib.file_digest(prepared, 'sha256').hexdigest()
    phases['audio_preparation'] = time.perf_counter()-began
    duration = len(audio)/16000
    options = VadOptions(threshold=.25, min_silence_duration_ms=250, speech_pad_ms=400,
        max_speech_duration_s=16)
    detected = get_speech_timestamps(audio, options)
    clips, metadata = collect_chunks(audio, detected, max_duration=16)
    timestamp_map = SpeechTimestampsMap(detected, 16000)
    records, all_words, alignment_failures = [], [], []
    phases['alignment'] = 0
    phases['recognition'] = 0

    def align_text(text, clip, index, recovered=False):
        if not text.strip():
            return []
        began = time.perf_counter()
        alignment = align([dict(start=0, end=len(clip)/16000, text=text)], alignment_model,
            alignment_metadata, np.asarray(clip).copy(), 'cuda', print_progress=False)
        phases['alignment'] += time.perf_counter()-began
        tokens = re.findall(r'\S+', text)
        aligned = alignment['word_segments']
        if [w['word'] for w in aligned] != tokens:
            # Keep the recognizer's exact text if sentence splitting/alignment fails.
            alignment_failures.append(dict(chunk=index, reason='word sequence mismatch'))
            aligned = [dict(word=token) for token in tokens]
        result = []
        for i, word in enumerate(aligned):
            lower = next((float(w['end']) for w in reversed(aligned[:i]) if 'end' in w), 0)
            upper = next((float(w['start']) for w in aligned[i+1:] if 'start' in w), len(clip)/16000)
            uncertain = 'start' not in word or 'end' not in word
            start = max(0,min(len(clip)/16000,float(word.get('start',lower))))
            end = max(start,min(len(clip)/16000,float(word.get('end',max(start,upper)))))
            flags = (['Estimated timing'] if uncertain else []) + (['Recovered speech; check audio'] if recovered else [])
            result.append(dict(word=(' ' if i else '')+word['word'], start=start, end=end,
                probability=float(word.get('score',0)), language='fa', language_hint=None,
                flags=flags, chunk=index, avg_logprob=None))
        assert words(''.join(w['word'] for w in result)) == words(text)
        return result

    def timed_recognition(batch):
        began = time.perf_counter()
        text = recognize(batch)
        phases['recognition'] += time.perf_counter()-began
        return text

    for offset in range(0,len(clips),args.batch):
        texts = timed_recognition(clips[offset:offset+args.batch])
        for index, text in enumerate(texts,offset):
            local_words = align_text(text, clips[index], index)
            for word in local_words:
                middle = metadata[index]['offset']+(word['start']+word['end'])/2
                original_index = timestamp_map.get_chunk_index(middle)
                word['start'] = timestamp_map.get_original_time(metadata[index]['offset']+word['start'], original_index)
                word['end'] = timestamp_map.get_original_time(metadata[index]['offset']+word['end'], original_index)
            all_words.extend(local_words)
            records.append(dict(index=index, text=text, metadata=metadata[index]))
        save(args.output/'primary-windows.json', records)
        emit(model=args.model, stage='primary', clips_done=min(offset+args.batch,len(clips)), clips=len(clips))
    primary_words = list(all_words)
    primary_cues = group_words('alternative-benchmark',track,primary_words)
    primary_text = '\n'.join(c['text'] for c in primary_cues)
    began_recovery = time.perf_counter()
    class BenchmarkDetector(Detector):
        def __init__(self):
            from silero_vad import load_silero_vad
            self.torch = torch
            self.vad = load_silero_vad(onnx=True)
            self.lid = None
    speech = BenchmarkDetector().speech(np.asarray(audio), coverage=True)
    gaps = uncovered_speech(speech, all_words, duration)
    recovery = []
    for index,gap in enumerate(gaps):
        def retry(margin):
            start, end = max(0,gap['start']-margin), min(duration,gap['end']+margin)
            clip = np.asarray(audio[round(start*16000):round(end*16000)])
            text = timed_recognition([clip])[0]
            timed_words = align_text(text,clip,-index-1,True)
            for word in timed_words:
                word['start'] += start
                word['end'] += start
            picked = [w for w in timed_words if gap['start'] <= (w['start']+w['end'])/2 <= gap['end']]
            return text,picked
        text,picked = retry(.4)
        encoded = text.encode('utf-8')
        repeated = repetitive_text(text) or len(encoded)/max(1,len(zlib.compress(encoded)))>2.4
        if repeated:
            text,picked = retry(2)
            if repetitive_text(''.join(w['word'] for w in picked)):
                picked = []
        all_words.extend(picked)
        recovery.append(dict(**gap, retried=repeated, text=text,
            selected_text=''.join(w['word'] for w in picked)))
        save(args.output/'recovery.json',recovery)
        emit(model=args.model, stage='recovery', gaps_done=index+1,gaps=len(gaps))
    phases['recovery_total'] = time.perf_counter()-began_recovery
    torch.cuda.synchronize()
    wall = time.perf_counter()-started
    stopping.set(); sampler.join(timeout=4)
    cues = group_words('alternative-benchmark',track,all_words)
    output = srt(cues)
    parsed = parse_srt(output,'validation',track,'fa','srt')
    assert len(parsed)==len(cues)>0
    assert all(0<=c['start']<c['end']<=duration+.15 for c in parsed)
    text = '\n'.join(c['text'] for c in cues)
    assert words('\n'.join(c['text'] for c in parsed)) == words(text)
    (args.output/'output.srt').write_text(output,encoding='utf-8')
    (args.output/'transcript.txt').write_text(text+'\n',encoding='utf-8')
    (args.output/'primary.txt').write_text(primary_text+'\n',encoding='utf-8')
    save(args.output/'cues.json',cues)
    # Reference never reaches a recognizer or timing model.
    reference = args.reference.read_text(encoding='utf-8-sig')
    metrics = compare(reference,text)
    sidecar = Path(media['path']).with_name(Path(media['path']).stem+f'.audio-tuned-{args.model}.srt')
    sidecar_status = 'created'
    try:
        with sidecar.open('x',encoding='utf-8',newline='\n') as target:
            target.write(output)
    except FileExistsError:
        sidecar = None
        sidecar_status = 'Existing sidecar preserved; new SRT is in the benchmark output directory'
    assert fingerprint(Path(media['path'])) == media['fingerprint']
    result = dict(model=args.model, wall_seconds=wall, phase_seconds=phases, batch_size=args.batch,
        model_directory=str(args.model_directory), aligner=str(args.aligner),
        source_duration_seconds=duration, audio_sha256=audio_sha, primary_clips=len(clips),
        recovery_gaps=len(gaps), cues=len(cues), source_unchanged=True,
        sidecar=str(sidecar) if sidecar else None, sidecar_status=sidecar_status,
        alignment_failures=alignment_failures,
        engine_metadata=getattr(engine, 'metadata', None),
        device_peak_memory_mib=max((g['memory_mib'] for g in gpu),default=None),
        torch_peak_allocated_mib=torch.cuda.max_memory_allocated()/1024**2,
        primary_metrics=compare(reference,primary_text),
        versions={p:importlib.metadata.version(p) for p in ['torch','transformers','qwen-asr','whisperx','faster-whisper']},
        **metrics)
    save(args.output/'telemetry.json',gpu)
    save(args.output/'result.json',result)
    emit(finished=result)


if __name__ == '__main__':
    main()
