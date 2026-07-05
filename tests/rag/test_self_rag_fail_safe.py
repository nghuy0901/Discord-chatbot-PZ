import asyncio

import pytest

from rag.self_rag import _parse_batch_response, _parse_individual_response


def test_batch_parse_failure_returns_unknown_low_confidence_grades():
    grades = _parse_batch_response("not json", expected_count=2)

    assert [grade.relevance for grade in grades] == ["UNKNOWN", "UNKNOWN"]
    assert [grade.is_relevant for grade in grades] == [False, False]


def test_individual_parse_failure_returns_unknown_low_confidence_grade():
    grade = _parse_individual_response("not json", index=0)

    assert grade.relevance == "UNKNOWN"
    assert grade.is_relevant is False


@pytest.mark.asyncio
async def test_batch_timeout_returns_unknown_low_confidence_grades(monkeypatch):
    import rag.self_rag as self_rag

    async def fake_wait_for(coro, *_args, **_kwargs):
        coro.close()
        raise asyncio.TimeoutError

    monkeypatch.setattr(self_rag.asyncio, "wait_for", fake_wait_for)

    grades = await self_rag._grade_batch(
        "query",
        [{"content": "irrelevant document"}],
    )

    assert grades[0].relevance == "UNKNOWN"
    assert grades[0].is_relevant is False
