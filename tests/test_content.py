import json

import httpx
import pytest

from pipeline.content import (ContentAnalyzer, RhetoricReport, extract_json, find_devices, find_transitions,
                              heuristic_rhetoric, llm_rhetoric, structure_checks)

SPEECH = ("Have you ever felt invisible in a meeting? We can change that. We can speak up. "
          "Research shows that 70 percent of people fear public speaking. "
          "However, the problem is not talent but practice. First, prepare your story. "
          "Then practise it with friends, family, and strangers. "
          "So remember, start practising today and let's make every voice heard.")


def _sentences():
    from pipeline.text import split_sentences

    return split_sentences(SPEECH)


def test_structure_detects_hook_and_call_to_action():
    s = structure_checks(_sentences())
    assert s["opener"]["type"] == "question"
    assert s["closer"]["type"] == "call-to-action"


def test_heuristic_devices():
    kinds = {d.type for d in find_devices(_sentences())}
    assert {"rhetorical_question", "anaphora", "statistic", "antithesis", "tricolon"} <= kinds


def test_transitions_and_appeals():
    transitions, quality = find_transitions(_sentences())
    assert {t.category for t in transitions} >= {"contrast", "sequence"}
    assert 0 < quality <= 1
    report = heuristic_rhetoric(_sentences(), structure_checks(_sentences()))
    a = report.appeals
    assert abs(a.ethos + a.pathos + a.logos - 1) < 0.01


def test_schema_normalises_llm_output():
    raw = {"devices": [{"type": "Rhetorical Question", "text": "Have you ever?"}, {"type": "zeugma", "text": "x"}],
           "appeals": {"ethos": 2, "pathos": 1, "logos": 1}, "transitions": [], "transition_quality": 0.4}
    with pytest.raises(Exception):
        RhetoricReport.model_validate(raw)  # ethos > 1 is out of range
    raw["appeals"] = {"ethos": 0.5, "pathos": 0.25, "logos": 0.5}
    report = RhetoricReport.model_validate(raw)
    assert [d.type for d in report.devices] == ["rhetorical_question", "other"]
    assert abs(report.appeals.ethos - 0.4) < 1e-6


def test_extract_json_from_chatty_output():
    assert extract_json('Sure! Here it is: {"a": 1} hope that helps') == {"a": 1}


def test_llm_path_retries_on_invalid_json(monkeypatch):
    good = {"devices": [], "appeals": {"ethos": 0.3, "pathos": 0.3, "logos": 0.4}, "transitions": [],
            "transition_quality": 0.5, "suggestions": [], "summary": "ok"}
    replies = iter(["not json at all", json.dumps(good)])
    calls = []

    def fake_post(url, json=None, timeout=None):  # noqa: A002
        calls.append(json)
        return httpx.Response(200, request=httpx.Request("POST", url),
                              json={"choices": [{"message": {"content": next(replies)}}]})

    monkeypatch.setattr(httpx, "post", fake_post)
    report = llm_rhetoric(SPEECH, "http://llm", "m")
    assert report.summary == "ok" and len(calls) == 2
    assert "Invalid JSON" in calls[1]["messages"][-1]["content"]


def test_content_analyzer_falls_back_when_llm_down(make_ctx, helpers, monkeypatch):
    from pipeline.settings import get_settings

    monkeypatch.setenv("VAKTA_LLM_URL", "http://127.0.0.1:9")
    get_settings.cache_clear()
    ctx = make_ctx()
    words = helpers.fake_words(SPEECH)
    helpers.write_module(ctx, "asr", {"words": words, "text": SPEECH})
    env = ContentAnalyzer().run(ctx)
    assert env["status"] == "ok", env["error"]
    data = env["data"]
    assert data["models"]["rhetoric"] == "heuristic" and "LLM unavailable" in data["note"]
    question = next(d for d in data["rhetoric"]["devices"] if d["type"] == "rhetorical_question")
    assert question["start"] == words[0]["start"]


def test_run_on_asr_stretch_gets_a_short_quote_not_a_wall_of_text():
    from pipeline.content import heuristic_suggestions

    run_on = " ".join(["and then I also like football and debating"] * 15)  # ~120 words, no punctuation
    structure = structure_checks(["Have you ever wondered?", run_on, "So remember, start today."])
    suggestions = heuristic_suggestions(structure, [])
    long = [s for s in suggestions if "sentence break" in s.reason]
    assert long and len(long[0].original.split()) <= 26 and long[0].original.endswith("…")
