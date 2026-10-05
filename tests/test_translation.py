import json

import pytest
from subtitle_studio.translation import utterances, distribute, translate, source_signature, valid_translations
from subtitle_studio.types import TRANSLATION_VERSION


def cue(cid, text, start, end, language="en"):
    return {"id": cid, "text": text, "start": start, "end": end, "language": language, "track": 1}


def test_fragments_form_complete_utterances_but_pauses_and_languages_split():
    cues = [cue("a", "The note was written", 0, 2), cue("b", "by Lord Byron.", 2, 3),
            cue("c", "Next", 3, 4), cue("d", "After a pause.", 7, 8), cue("e", "سلام", 8, 9, "fa")]
    assert [[c["id"] for c in g] for g in utterances(cues)] == [["a","b"],["c"],["d"],["e"]]
    assert cues[0]["text"] == "The note was written"


def test_distributing_translation_keeps_all_words_and_original_timings():
    cues = [cue("a", "First", 0, 3), cue("b", "Second.", 3, 4)]
    text = "این ترجمه شامل همه کلمات است."
    output = distribute(text, cues, "fa")
    assert " ".join(output.values()) == text
    assert all(output.values()) and list(output) == ["a","b"]
    assert [(c["start"],c["end"]) for c in cues] == [(0,3),(3,4)]
    japanese = "今日は良い天気です。"
    assert "".join(distribute(japanese, cues, "ja").values()) == japanese


def test_source_edit_invalidates_all_translations_of_its_utterance():
    cues = [cue("a", "A note", 0, 1), cue("b", "by Byron.", 1, 2)]
    signature = source_signature(cues, "fa", "model")
    existing = {c["id"]: {"source_text":c["text"],"text":"ترجمه", "version":TRANSLATION_VERSION,
                           "model_digest":"model","source_signature":signature} for c in cues}
    assert len(valid_translations(cues,existing,"fa")) == 2
    cues[0]["text"] = "A different note"
    assert not valid_translations(cues,existing,"fa")


def test_resume_reuses_groups_and_translation_requests_have_no_generated_labels():
    class Client:
        def __init__(self): self.requests=[]
        def chat(self, model, messages):
            self.requests.append(messages[0]["content"])
            return "A correct translation."
        def unload(self, model): pass
    cues = [cue("saved", "Hallo.", 0, 1, "de"), cue("new", "Noch einmal.", 1, 2, "de"),
            cue("fa", "سلام.", 2, 3, "fa"), cue("en", "Already English.", 3, 4)]
    first = cues[0]
    existing = {"saved": {"source_text":first["text"],"text":"Hello.","version":TRANSLATION_VERSION,
                           "model_digest":"model","source_signature":source_signature([first],"en","model")}}
    client = Client(); outputs={}
    translate(cues,"en",client,lambda c,t,s:outputs.update({c["id"]:t}),lambda *a:None,lambda:False,existing,model_digest="model")
    assert set(outputs) == {"new","fa","en"} and outputs["en"] == "Already English."
    assert len(client.requests)==2 and all("[SS_" not in p for p in client.requests)
    assert "German (de)" in client.requests[0] and "سلام" not in client.requests[0]
    assert "Persian (fa)" in client.requests[1] and "Hallo" not in client.requests[1]


def test_complete_sentence_is_one_request_and_response_does_not_depend_on_cue_labels():
    class Client:
        def __init__(self): self.requests=[]
        def chat(self,model,messages):
            self.requests.append(messages[0]["content"])
            return "این نوشته توسط لرد بایرون ثبت شد."
        def unload(self,model): pass
    cues=[cue("first","This was recorded",0,2),cue("last","by Lord Byron.",2,3)]
    client=Client(); outputs={}
    translate(cues,"fa",client,lambda c,t,s:outputs.update({c["id"]:t}),lambda *a:None,lambda:False)
    assert len(client.requests)==1 and client.requests[0].endswith("This was recorded by Lord Byron.")
    assert " ".join(outputs.values())=="این نوشته توسط لرد بایرون ثبت شد."
    assert all(c["text"] in client.requests[0] for c in cues)


def test_incomplete_or_punctuation_only_output_cannot_be_saved_as_a_group():
    with pytest.raises(ValueError): distribute(".",[cue("a","Hello",0,1)],"en")
    class Client:
        def __init__(self): self.calls=0
        def chat(self,model,messages):
            self.calls+=1
            return "Yes" if self.calls==1 else "One cue"
        def unload(self,model): pass
    cues=[cue("a","Yes",0,1),cue("b","again",1,2)]; outputs={};warnings=[]
    translate(cues,"en",Client(),lambda c,t,s:outputs.update({c["id"]:t}),lambda *a:None,lambda:False,warning=warnings.append)
    assert outputs=={"a":"Yes","b":"again"}  # same-language copies preserve every cue
    for c in cues: c["language"]="de"
    outputs.clear()
    translate(cues,"en",Client(),lambda c,t,s:outputs.update({c["id"]:t}),lambda *a:None,lambda:False,warning=warnings.append)
    assert outputs=={"a":"One cue","b":"One cue"} and warnings
