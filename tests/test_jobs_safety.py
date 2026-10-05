from subtitle_studio.storage import Store
from subtitle_studio.subtitles import group_words
from scripts.benchmark_references import units, distance


def test_simultaneous_text_save_and_review_keep_both_changes(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier
    store = Store(tmp_path / "concurrent.sqlite")
    cue = group_words("p",1,[{"word":" Original.","start":1,"end":2,"language":"en"}])[0]
    store.replace_cues("p",1,[cue])
    with ThreadPoolExecutor(max_workers=2) as pool:
        for iteration in range(8):
            store.update_cue("p",cue["id"],{"text":"Original.","reviewed":False})
            together = Barrier(2)
            def save(changes):
                together.wait()
                return store.update_cue("p",cue["id"],changes)
            text = f"Correction {iteration}."
            tasks = [pool.submit(save,{"text":text,"edited":True}),pool.submit(save,{"reviewed":True})]
            for task in tasks:
                task.result()
            saved = store.cues("p")[0]
            assert saved["text"] == text and saved["reviewed"]


def test_inflight_ai_result_cannot_overwrite_a_human_edit(tmp_path):
    store = Store(tmp_path / "studio.sqlite")
    cue = group_words("p", 1, [{"word":" Original.","start":1,"end":2,"language":"en"}])[0]
    store.replace_cues("p", 1, [cue])
    store.update_cue("p",cue["id"],{"text":"Human correction.","edited":True})
    result = store.update_cue("p",cue["id"],{"text":"AI alternative."},automated=True,expected_text="Original.")
    assert result["text"] == "Human correction."
    assert store.cues("p")[0]["raw_text"] == "Original."


def test_reference_metrics_use_unicode_and_count_insertions():
    assert distance(units("One two.","en"),units("One three two!","en")) == 1
    assert units("عربي كی\u200cي", "fa") == ["عربی", "کی", "ی"]
    assert distance(units("今日は。","ja"),units("今日です。","ja")) == 2


def test_punctuation_only_recheck_keeps_original_words_and_review_flag(tmp_path):
    from subtitle_studio.jobs import Jobs
    store = Store(tmp_path / "punctuation.sqlite")
    cue = group_words("p", 1, [{"word":" Oh, hello!","start":1,"end":2,"language":"en","flags":["Uncertain recognition"]}])[0]
    store.replace_cues("p",1,[cue])
    manager = Jobs(store)
    def worker(job, stage, manifest, callback):
        callback({"type":"recheck","result":{"cue_id":cue["id"],"text":"oh hello.","language":"en","engine":"test","candidate_id":"qwen"}})
    manager.worker = worker
    manager._recheck({"id":"j","project_id":"p","tracks":[1],"settings":{},"warnings":[],"kind":"transcribe"},{})
    result = store.cues("p")[0]
    assert result["text"] == "Oh, hello!" and len(result["candidates"]) == 1
    assert result["flags"] and not result["reviewed"]
    manager.pool.shutdown()
