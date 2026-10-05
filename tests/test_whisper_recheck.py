from types import SimpleNamespace

import numpy as np

from subtitle_studio import worker


def test_recheck_uses_large_and_prepared_audio_with_absolute_range_offsets(tmp_path, monkeypatch):
    loaded, recognized, prepared, events = [], [], [], []
    class Model:
        def transcribe(self, audio, **settings):
            recognized.append((audio.copy(), settings))
            return [SimpleNamespace(text=' corrected words')], SimpleNamespace(language='fa')
    def load(settings):
        loaded.append(settings)
        return Model(), None
    def prepare(*args):
        prepared.append(args)
        return np.arange(20*16000, dtype=np.float32)
    monkeypatch.setattr(worker, 'whisper_model', load)
    monkeypatch.setattr(worker, 'model_revision', lambda key: 'fixed-large-revision')
    monkeypatch.setattr(worker, 'prepare_track', prepare)
    monkeypatch.setattr(worker, 'emit', lambda kind, **data: events.append(data))
    manifest = dict(settings=dict(preset='fast', language='fa', start_seconds=10, limit_seconds=20, audio_profile='level'),
        media=dict(path='recording.mp4', duration=100), cache=str(tmp_path), cases=[dict(
        cue=dict(id='cue', track=1, start=12, end=14), neighbors=[
            dict(text='Reliable name.', flags=[]), dict(text='Uncertain story.', flags=['Uncertain recognition'])])])
    worker.recheck(manifest)
    assert loaded[0]['preset'] == 'accurate'
    assert prepared[0][2:4] == (10, 20)
    assert prepared[0][-1] == 'level'
    audio, settings = recognized[0]
    assert len(audio) == round(4.12*16000)-round(1.88*16000)
    assert audio[0] == round(1.88*16000)
    assert settings['initial_prompt'] == 'Reliable name.'
    assert settings['language'] == 'fa'
    assert events[0]['result']['engine'] == 'Whisper large-v3 contextual retry'
    assert events[0]['result']['start'] == 11.88
    assert events[0]['result']['end'] == 14.12
    worker.recheck(manifest)
    assert len(recognized) == len(prepared) == 1
