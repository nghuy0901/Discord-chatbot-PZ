"""Post-generation groundedness check."""

import asyncio

from rag.groundedness import check_groundedness


SOURCES = [{"content": "Axes degrade with use.", "source": "pz/Axes.md"}]


def judge_returning(payload):
    async def _judge(prompt: str) -> str:
        return payload
    return _judge


def test_grounded_answer_passes():
    res = asyncio.run(check_groundedness(
        "q", "Axes degrade [1].", SOURCES,
        judge=judge_returning('{"grounded": true, "score": 0.9}'),
    ))
    assert res.grounded is True
    assert res.score == 0.9


def test_ungrounded_answer_fails():
    res = asyncio.run(check_groundedness(
        "q", "Axes shoot lasers [1].", SOURCES,
        judge=judge_returning(
            '{"grounded": false, "score": 0.1, "unsupported_claims": ["lasers"]}'
        ),
    ))
    assert res.grounded is False
    assert "lasers" in res.unsupported


def test_low_score_overrides_true_flag():
    res = asyncio.run(check_groundedness(
        "q", "answer", SOURCES,
        judge=judge_returning('{"grounded": true, "score": 0.2}'),
    ))
    assert res.grounded is False


def test_markdown_wrapped_json_parses():
    res = asyncio.run(check_groundedness(
        "q", "answer [1]", SOURCES,
        judge=judge_returning('```json\n{"grounded": true, "score": 0.8}\n```'),
        min_score=0.6,
    ))
    assert res.grounded is True


def test_judge_error_fails_open(monkeypatch):
    monkeypatch.setattr("rag.groundedness.GROUNDEDNESS_FAIL_OPEN", True)
    async def boom(prompt: str) -> str:
        raise RuntimeError("provider down")

    res = asyncio.run(check_groundedness("q", "answer [1]", SOURCES, judge=boom))
    assert res.grounded is True  # fail-open by default
    assert res.error


def test_judge_error_can_fail_closed(monkeypatch):
    monkeypatch.setattr("rag.groundedness.GROUNDEDNESS_FAIL_OPEN", False)

    async def boom(prompt: str) -> str:
        raise RuntimeError("provider down")

    res = asyncio.run(check_groundedness("q", "answer [1]", SOURCES, judge=boom))
    assert res.grounded is False
    assert res.reason == "fail_closed"


def test_no_sources_is_skipped():
    res = asyncio.run(check_groundedness("q", "answer", [], judge=judge_returning("{}")))
    assert res.skipped is True
    assert res.grounded is True


def test_timeout_fails_closed_by_default():
    async def timeout_judge(prompt: str) -> str:
        raise asyncio.TimeoutError()

    res = asyncio.run(check_groundedness("q", "answer [1]", SOURCES, judge=timeout_judge))
    assert res.grounded is False  # fail CLOSED on timeout (H6)
    assert res.reason == "timeout_fail_closed"
    assert res.error == "timeout"


def test_tool_results_become_sources():
    from rag.groundedness import sources_from_tool_results

    srcs = sources_from_tool_results([
        {"name": "get_recipe", "output": {"recipes": [{"name": "Bandage"}]}},
        {"name": "search_items", "output": "raw text"},
    ])
    assert len(srcs) == 2
    assert "Bandage" in srcs[0]["content"]
    assert srcs[0]["source"] == "tool:get_recipe"
    assert srcs[1]["content"] == "raw text"


def test_groundedness_prompt_accepts_only_direct_semantic_equivalence():
    from rag.groundedness import GROUNDEDNESS_PROMPT

    prompt = GROUNDEDNESS_PROMPT.lower()
    assert "guarantees" in prompt and "100%" in prompt
    assert "likely" in prompt and "must not" in prompt
