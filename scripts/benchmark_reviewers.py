"""Compare zero/one/two reviewers on one immutable tuned Turbo output.

Large-v3 supplies a shared pool of audio-derived alternatives. Reference text
is read only after review, and never enters a recognition or review request.
Run each stage in a fresh process so ASR and review models do not coexist.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from subtitle_studio.subtitles import atomic_text, group_words, parse_srt, same_spoken_words, srt
from subtitle_studio.media import fingerprint
from scripts.transcript_metrics import compare, words


def save(path, value):
    atomic_text(path, json.dumps(value, ensure_ascii=False, indent=2))


def baseline(directory):
    index = json.loads((directory/'track-1-primary.json').read_text(encoding='utf-8'))
    assert index['complete']
    tokens = []
    for name in index['files']:
        tokens.extend(json.loads((directory/name).read_text(encoding='utf-8'))['words'])
    tokens.extend(json.loads((directory/'track-1-recovery.json').read_text(encoding='utf-8'))['words'])
    return group_words('turbo-review-benchmark', 1, tokens)


def cue_text(cues):
    return '\n'.join(c['text'] for c in cues)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=['candidates', 'one', 'two'], required=True)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--baseline-cache', type=Path, required=True)
    parser.add_argument('--baseline-result', type=Path, required=True)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--review-context', default='')
    args = parser.parse_args()
    source = json.loads(args.source.read_text(encoding='utf-8'))
    original = baseline(args.baseline_cache)
    original_result = json.loads(args.baseline_result.read_text(encoding='utf-8'))
    media = source['media']
    assert source['track'] == 1, 'This audit reuses the measured track-1 baseline'
    assert fingerprint(Path(media['path'])) == media['fingerprint']
    assert len(original) == original_result['cues']
    assert len(words(cue_text(original))) == original_result['output_words']
    settings = {**original_result['settings'], 'recheck': True}

    if args.stage == 'candidates':
        args.output.mkdir(parents=True, exist_ok=False)
        save(args.output/'baseline-cues.json', original)
        (args.output/'no-agent.srt').write_text(srt(original), encoding='utf-8')
        cases = [{'cue': cue, 'neighbors': original[max(0,i-3):i]+original[i+1:i+4]}
                 for i, cue in enumerate(original) if cue['flags'] and not cue.get('edited')]
        results = []
        began = time.perf_counter()
        from subtitle_studio import worker
        def event(kind, **data):
            if kind == 'recheck':
                results.append(data['result'])
                save(args.output/'rechecks.json', results)
                print(json.dumps(dict(stage='candidates', done=len(results), total=len(cases))), flush=True)
        worker.emit = event
        worker.recheck(dict(media=media, settings=settings, cases=cases, cache=str(args.output/'candidate-cache')))
        seconds = time.perf_counter()-began
        candidates = deepcopy(original)
        by_id = {c['id']: c for c in candidates}
        for result in results:
            cue = by_id[result['cue_id']]
            if result['text'] and not same_spoken_words(result['text'], cue['raw_text']):
                cue['candidates'].append(dict(id=result['candidate_id'], text=result['text'],
                    engine=result['engine'], language=result['language'],
                    source='Same cleaned audio, tight subtitle interval with neighboring text context'))
                cue['flags'] = sorted(set(cue['flags']+['Recognizer disagreement']))
        assert cue_text(candidates) == cue_text(original)
        save(args.output/'candidate-cues.json', candidates)
        with (args.output/'candidate-cache/audio-1.f32').open('rb') as f:
            digest = hashlib.file_digest(f,'sha256').hexdigest()
        with (args.baseline_cache/'audio-1.f32').open('rb') as f:
            assert digest == hashlib.file_digest(f,'sha256').hexdigest()
        info = dict(seconds=seconds, rechecked_cues=len(cases),
            reviewable_cues=sum(len(c['candidates'])>1 for c in candidates), audio_sha256=digest,
            context=args.review_context, reference_used_for_candidates=False)
        save(args.output/'candidate-preparation.json', info)
        print(json.dumps(info), flush=True)
        return

    count = 1 if args.stage=='one' else 2
    destination = args.output/f'{count}-agents'
    destination.mkdir(exist_ok=False)
    shared = json.loads((args.output/'candidate-cues.json').read_text(encoding='utf-8'))
    prep = json.loads((args.output/'candidate-preparation.json').read_text(encoding='utf-8'))
    assert args.review_context == prep['context']
    assert cue_text(shared) == cue_text(original)
    cases = [(cue, shared[max(0,i-4):i]+shared[i+1:i+5])
             for i, cue in enumerate(shared) if len(cue['candidates'])>1 and not cue.get('edited')]
    revised = deepcopy(shared)
    by_id = {c['id']: c for c in revised}
    logs, calls, timings = [], [], {}
    began = time.perf_counter()
    from subtitle_studio.models import ollama, OLLAMA_MODELS
    from subtitle_studio.debate import choose, apply_choice
    model = OLLAMA_MODELS['context']
    ollama.start()
    # A new private server per process, plus explicit unload, prevents warm model
    # and conversation state from favoring the second review mode.
    ollama.unload(model)
    tags = ollama.tags()
    model_info = next(m for m in tags if m['name']==model)
    def chat(model, messages, schema):
        started = time.perf_counter()
        try:
            return ollama.chat(model, messages, schema)
        finally:
            calls.append(dict(seconds=time.perf_counter()-started))
    try:
        for i, (cue, neighbors) in enumerate(cases):
            started = time.perf_counter()
            decision = choose(deepcopy(cue), deepcopy(neighbors), chat, model, args.review_context, count)
            by_id[cue['id']] = apply_choice(cue, decision)
            logs.append(dict(cue_id=cue['id'], seconds=time.perf_counter()-started, decision=decision))
            save(destination/'decisions.json', logs)
            print(json.dumps(dict(stage=f'{count}-agents', done=i+1, total=len(cases),
                selected=decision['selected'], status=decision['status'])), flush=True)
        timings['review_seconds'] = time.perf_counter()-began
    finally:
        ollama.unload(model)
        ollama.close()
    revised = [by_id[c['id']] for c in original]
    changed = [c for c in revised if c['text'] != c['raw_text']]
    began = time.perf_counter()
    alignment_warnings = []
    if changed:
        from subtitle_studio import worker
        def alignment_event(kind, **data):
            if kind == 'alignment' and data['words']:
                cue = by_id[data['cue_id']]
                cue['words'] = data['words']
                cue['flags'] = [f for f in cue['flags'] if f != 'Word timing needs alignment']
            elif kind == 'warning':
                alignment_warnings.append(data['message'])
        worker.emit = alignment_event
        try:
            worker.align(dict(media=media, cues=changed))
        except Exception as error:
            alignment_warnings.append(f'Alignment unavailable; cue timing retained: {type(error).__name__}: {error}')
    timings['alignment_seconds'] = time.perf_counter()-began
    output = srt(revised)
    parsed = parse_srt(output, 'validate', 1, 'fa', 'srt')
    assert len(parsed)==len(revised)==len(original)
    assert words(cue_text(parsed)) == words(cue_text(revised))
    assert all(0 <= c['start'] < c['end'] <= media['duration']+.15 for c in parsed)
    assert [(c['start'],c['end']) for c in revised] == [(c['start'],c['end']) for c in original]
    assert fingerprint(Path(media['path'])) == media['fingerprint']
    (destination/'output.srt').write_text(output,encoding='utf-8')
    (destination/'transcript.txt').write_text(cue_text(revised)+'\n',encoding='utf-8')
    save(destination/'cues.json', revised)
    # Nothing above reads the supplied reference.
    reference = args.reference.read_text(encoding='utf-8-sig')
    assert compare(reference,cue_text(original))['minimum_word_edits']==original_result['minimum_word_edits']
    metrics = compare(reference,cue_text(revised))
    sidecar = Path(media['path']).with_name(Path(media['path']).stem+f'.turbo-{count}-agents.srt')
    with sidecar.open('x',encoding='utf-8',newline='\n') as f:
        f.write(output)
    summary = dict(agent_count=count, baseline_seconds=original_result['wall_seconds'],
        candidate_preparation_seconds=prep['seconds'], **timings,
        total_seconds=original_result['wall_seconds']+prep['seconds']+sum(timings.values()),
        reviewable_cues=len(cases), changed_cues=len(changed), llm_calls=len(calls),
        disagreement_cues=sum(len(votes)==2 and votes[0]['candidate_id']!=votes[1]['candidate_id']
            for votes in ([v for v in d['decision']['votes'] if v['round']==1] for d in logs)),
        unresolved_cues=sum(d['decision']['status']=='unresolved' for d in logs),
        failed_cues=sum(d['decision']['status']=='failed' for d in logs),
        alignment_warnings=alignment_warnings,
        timings_unchanged=True, source_unchanged=True, sidecar=str(sidecar),
        model=model, model_digest=model_info['digest'], calls=calls,
        reference_used_only_for_scoring=True, **metrics)
    save(destination/'result.json',summary)
    print(json.dumps(summary,ensure_ascii=False),flush=True)


if __name__=='__main__':
    main()
