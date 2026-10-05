from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from subtitle_studio.audio_processing import process_waveform, prepare_track
from subtitle_studio.speech_coverage import uncovered_speech, repetitive_text
from scripts.transcript_metrics import compare


def test_level_preparation_preserves_timeline_and_raises_quiet_speech():
    rate = 16000
    time = np.arange(rate*12)/rate
    audio = (np.sin(2*np.pi*330*time)*np.where(time<6,.012,.45)).astype(np.float32)
    prepared = process_waveform(audio,"level")
    assert len(prepared)==len(audio) and np.all(np.isfinite(prepared))
    assert np.max(np.abs(prepared))<=1
    quiet = slice(rate,rate*4)
    assert np.sqrt(np.mean(prepared[quiet]**2))>2*np.sqrt(np.mean(audio[quiet]**2))
    # Transient markers on each side of the level change must keep their position.
    marked = np.zeros(rate*4,dtype=np.float32)
    marked[rate] = .3;marked[rate*3] = .5
    filtered = process_waveform(marked,"level")
    assert abs(np.argmax(np.abs(filtered[:rate*2]))-rate)<rate*.02
    assert abs(np.argmax(np.abs(filtered[rate*2:]))+rate*2-rate*3)<rate*.02


def test_prepared_track_is_reused_and_profile_change_invalidates_it(tmp_path):
    import soundfile as sf
    source = tmp_path/"زبان.wav"
    time = np.arange(16000*4)/16000
    sf.write(source,(np.sin(2*np.pi*440*time)*.025).astype(np.float32),16000)
    destination = tmp_path/"prepared.f32"
    original = prepare_track(str(source),0,1,2,destination,"original")
    original_copy = original.copy();del original
    stamp = destination.stat().st_mtime_ns
    reused = prepare_track(str(source),0,1,2,destination,"original")
    assert np.array_equal(reused,original_copy)
    assert destination.stat().st_mtime_ns==stamp
    del reused
    processed = prepare_track(str(source),0,1,2,destination,"level")
    assert len(processed)==32000 and not np.array_equal(processed,original_copy)
    del processed


def test_uncovered_speech_ignores_silence_and_does_not_hide_speech_under_bad_timings():
    speech = [{"start":1,"end":3},{"start":5,"end":8}]
    gaps = uncovered_speech(speech,[{"start":1,"end":2},{"start":5,"end":8}],10)
    assert any(g["start"]>=2 and g["end"]<=3 for g in gaps)
    assert any(g["start"]>5 and g["end"]<=8 for g in gaps)
    assert all(any(s['start']<=g['start']<g['end']<=s['end'] for s in speech) for g in gaps)
    assert not uncovered_speech([],[],10)
    assert all(g['end']-g['start']<=12.001 for g in uncovered_speech([{"start":0,"end":40}],[],40))


def test_recovery_retries_repeated_output_with_more_audio_context(monkeypatch):
    import subtitle_studio.worker as module
    calls=[]
    def segment(text,ratio):
        return SimpleNamespace(text=text,compression_ratio=ratio,no_speech_prob=0,avg_logprob=-.2,
            words=[SimpleNamespace(word=" "+text,start=2,end=3,probability=.9)])
    class Model:
        def transcribe(self,audio,**kwargs):
            calls.append((len(audio),kwargs))
            return iter([segment("word "*10,4) if len(calls)==1 else segment("Actual speech",1)]),SimpleNamespace(language="en")
    class Detector:
        def speech(self,audio,coverage=False):return [{"start":2,"end":5}]
    monkeypatch.setattr(module,"emit",lambda *args,**kwargs:None)
    existing=[{"word":" Keep.","start":0,"end":1}]
    recovered,evidence=module.recover_speech(Model(),np.zeros(16000*8,dtype=np.float32),existing,Detector(),"en")
    assert len(calls)==2 and calls[1][0]>calls[0][0]
    assert recovered and recovered[0]['word'].strip()=="Actual speech"
    assert existing==[{"word":" Keep.","start":0,"end":1}]
    assert all('initial_prompt' not in kwargs for _,kwargs in calls)
    assert evidence[0]['retried'] and repetitive_text("one two "*5)
    assert not repetitive_text("No no no, I meant something else.")


def test_reference_metrics_penalize_word_count_padding():
    result=compare("one two three","one two filler filler filler")
    assert result['output_words']>result['reference_words']
    assert result['minimum_word_edits']==3 and result['insertions']==2
    assert result['substitutions']==1 and result['matches']==2
    assert compare("مي\u200cروم ۲","می روم 2")['minimum_word_edits']==0


def test_partial_resume_reuses_completed_segments_and_keeps_range_offset(tmp_path,monkeypatch):
    import subtitle_studio.worker as module
    calls=[]
    def segment(index):
        return SimpleNamespace(start=index+.2,end=index+.8,text=f" Word{index}",
            avg_logprob=-.1,compression_ratio=1,no_speech_prob=0,
            words=[SimpleNamespace(word=f" Word{index}",start=index+.2,end=index+.8,probability=.95)])
    class Pipeline:
        def transcribe(self,audio,**kwargs):
            calls.append(kwargs)
            return iter([segment(0),segment(2)]),SimpleNamespace(language='en',language_probability=1)
    class Detector:
        def language(self,audio):return None
    def prepared(path,track,start,duration,destination,profile):
        assert start==10 and duration==4
        destination.parent.mkdir(parents=True,exist_ok=True)
        np.zeros(64000,dtype=np.float32).tofile(destination)
        return np.memmap(destination,dtype="<f4",mode="r",shape=(64000,))
    monkeypatch.setattr(module,'whisper_model',lambda settings:(object(),Pipeline()))
    monkeypatch.setattr(module,'Detector',Detector)
    monkeypatch.setattr(module,'prepare_track',prepared)
    manifest={'media':{'path':'unused','duration':100},'tracks':[1],
        'settings':{'language':'en','start_seconds':10,'limit_seconds':4,'recover_speech':False},'cache':str(tmp_path/'cache')}
    def interrupt(kind,**data):
        if kind=='chunk':raise InterruptedError('Pause after completed segment')
    monkeypatch.setattr(module,'emit',interrupt)
    with pytest.raises(InterruptedError):module.primary(manifest)
    checkpoint=next((tmp_path/'cache').glob('track-1-at-*.json'))
    stamp=checkpoint.stat().st_mtime_ns
    output=[]
    monkeypatch.setattr(module,'emit',lambda kind,**data:output.extend(data['chunk']['words']) if kind=='chunk' else None)
    module.primary(manifest)
    assert len(output)==2 and output[0]['start']==10.2 and output[1]['start']==12.2
    assert checkpoint.stat().st_mtime_ns==stamp
    assert calls[0]['word_timestamps'] and calls[0]['without_timestamps'] is False
    assert 'initial_prompt' not in calls[0]
    output.clear();module.primary(manifest)
    assert len(calls)==2 and len(output)==2
