"""End-to-end answer finalization: groundedness override + citation footer."""

import asyncio
from types import SimpleNamespace

from rag.answer_finalize import finalize_rag_answer
from rag.result import RAGDecision, ProvenanceItem


def make_rag_result():
    prov = [
        ProvenanceItem(
            source_id="pz:Axes.md:0:abc",
            source_kind="knowledge_base",
            domain="pz",
            trusted=True,
            rank=1,
            similarity=0.71,
            label="1",
            locator="pz/Axes.md › Weapons > Axes",
            source="pz/Axes.md",
            heading_path="Weapons > Axes",
            excerpt="Axes degrade with use",
        )
    ]
    return SimpleNamespace(
        decision=RAGDecision.ANSWER,
        retrieved_results=[{"content": "Axes degrade with use.", "source": "pz/Axes.md"}],
        provenance=prov,
        metric=SimpleNamespace(
            query_language="vi", rag_decision="answer", decision_reason=""
        ),
    )


def judge_returning(payload):
    async def _judge(prompt: str) -> str:
        return payload
    return _judge


def test_grounded_answer_gets_citation_footer():
    rr = make_rag_result()
    final = asyncio.run(finalize_rag_answer(
        rr, "Rìu để làm gì?", "Rìu bị mòn khi dùng [1].",
        judge=judge_returning('{"grounded": true, "score": 0.9}'),
    ))
    assert final.overridden is False
    assert final.decision is RAGDecision.ANSWER
    assert "Nguồn" in final.text
    assert "Axes.md" in final.text
    assert rr.metric.groundedness_score == 0.9


def test_ungrounded_answer_is_overridden_to_abstain():
    rr = make_rag_result()
    final = asyncio.run(finalize_rag_answer(
        rr, "Rìu để làm gì?", "Rìu bắn laser [1].",
        judge=judge_returning('{"grounded": false, "score": 0.1}'),
    ))
    assert final.overridden is True
    assert final.decision is RAGDecision.ABSTAIN
    assert rr.metric.rag_decision == "abstain"
    assert "đủ" in final.text or "approved" in final.text  # canned refusal


def test_non_answer_decision_is_passthrough():
    rr = make_rag_result()
    rr.decision = RAGDecision.ABSTAIN
    final = asyncio.run(finalize_rag_answer(rr, "q", "text", judge=judge_returning("{}")))
    assert final.text == "text"
    assert final.overridden is False
