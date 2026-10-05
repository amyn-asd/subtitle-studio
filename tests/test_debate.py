import json

from subtitle_studio.debate import choose, apply_choice


def case():
    return {"id": "cue", "language": "en", "text": "Turn left.", "raw_text": "Turn left.", "flags": ["Uncertain recognition"],
            "words": [{"word": "Turn", "start": 0, "end": 1}], "reviewed": False,
            "candidates": [{"id": "primary", "text": "Turn left.", "engine": "Whisper"},
                           {"id": "qwen", "text": "Turn right.", "engine": "Qwen3-ASR"}]}


def answer(candidate):
    return json.dumps({"candidate_id": candidate, "evidence_ids": [candidate], "reason": "Supported candidate."})


def test_independent_initial_judgments_and_bounded_exchange():
    calls = []
    responses = iter([answer("primary"), answer("qwen"), answer("qwen"), answer("qwen")])
    def chat(model, messages, schema):
        calls.append(json.loads(messages[-1]["content"]))
        assert schema["properties"]["candidate_id"]["enum"] == ["primary", "qwen"]
        return next(responses)
    decision = choose(case(), [], chat, "local-model")
    assert len(calls) == 4
    assert "independent_judgments" not in calls[0] and "independent_judgments" not in calls[1]
    assert calls[2]["independent_judgments"] == calls[3]["independent_judgments"]
    assert decision["selected"] == "qwen" and decision["rounds"] == 2
    revised = apply_choice(case(), decision)
    assert revised["text"] == "Turn right."
    assert revised["raw_text"] == "Turn left."
    assert revised["flags"] and not revised["reviewed"] and not revised["words"]


def test_unresolved_disagreement_keeps_primary():
    replies = iter([answer("primary"), answer("qwen"), answer("primary"), answer("qwen")])
    result = choose(case(), [], lambda *args: next(replies), "local")
    assert result["selected"] == "primary" and result["status"] == "unresolved"


def test_unknown_words_ids_and_fake_evidence_are_rejected():
    for response in (answer("invented"), '{"candidate_id":"qwen","evidence_ids":["fake"],"reason":"made up"}', "not json"):
        result = choose(case(), [], lambda *args: response, "local")
        assert result["status"] == "failed" and result["selected"] == "primary"


def test_model_failure_is_safe_and_uncertainty_is_retained():
    def broken(*args):
        raise TimeoutError("Local model timed out")
    result = choose(case(), [], broken, "local")
    assert apply_choice(case(), result)["text"] == "Turn left."
    assert result["status"] == "failed"


def test_only_trustworthy_neighbor_dialogue_enters_context():
    packets = []
    def chat(model, messages, schema):
        packets.append(json.loads(messages[-1]["content"]))
        return answer("primary")
    neighbors = [{"id":"good","text":"A known name.","language":"en","flags":[]},
                 {"id":"uncertain","text":"An unreliable story.","language":"en","flags":["Uncertain recognition"]}]
    result = choose(case(), neighbors, chat, "local")
    assert result["rounds"] == 1 and len(packets) == 2
    assert [n["id"] for n in packets[0]["neighboring_dialogue"]] == ["good"]


def test_one_agent_makes_one_context_judgment_with_same_candidate_guard():
    calls = []
    def chat(model, messages, schema):
        calls.append(messages)
        return answer("qwen")
    decision = choose(case(), [], chat, "local", agent_count=1)
    assert len(calls) == 1
    assert decision['status'] == 'single_review'
    assert decision['selected'] == 'qwen'
    assert decision['votes'][0]['role'] == 'context'
    assert apply_choice(case(), decision)['raw_text'] == 'Turn left.'
    invalid = choose(case(), [], lambda *args: answer('invented'), 'local', agent_count=1)
    assert invalid['status'] == 'failed' and invalid['selected'] == 'primary'


def test_invalid_review_agent_count_is_rejected():
    import pytest
    with pytest.raises(ValueError, match='one or two'):
        choose(case(), [], lambda *args: answer('primary'), 'local', agent_count=3)
