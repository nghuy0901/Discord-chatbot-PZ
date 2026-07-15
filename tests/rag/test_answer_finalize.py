"""End-to-end answer finalization: groundedness override + citation footer."""

import asyncio
from types import SimpleNamespace

from rag.answer_finalize import finalize_rag_answer, finalize_tool_answer
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
    assert rr.metric.groundedness_reason == "checked"
    assert rr.metric.groundedness_unsupported_count == 0


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


def test_uncited_factual_line_is_overridden_before_judge():
    rr = make_rag_result()

    async def judge_must_not_run(_prompt):
        raise AssertionError("deterministic citation gate should run first")

    final = asyncio.run(finalize_rag_answer(
        rr,
        "Rìu để làm gì?",
        "Rìu bị mòn khi dùng [1].\nNó cũng bắn được laser.",
        judge=judge_must_not_run,
    ))

    assert final.overridden is True
    assert final.decision is RAGDecision.ABSTAIN
    assert final.groundedness.reason == "citation_coverage_failed"


def test_non_answer_decision_is_passthrough():
    rr = make_rag_result()
    rr.decision = RAGDecision.ABSTAIN
    final = asyncio.run(finalize_rag_answer(rr, "q", "text", judge=judge_returning("{}")))
    assert final.text == "text"
    assert final.overridden is False


def test_conversation_answer_bypasses_rag_finalizer():
    rr = make_rag_result()
    rr.metric.query_intent = "conversation"
    rr.provenance = []
    rr.retrieved_results = []

    async def judge_must_not_run(_prompt):
        raise AssertionError("conversation should not run groundedness")

    final = asyncio.run(finalize_rag_answer(
        rr,
        "Cảm ơn nhé",
        "Không có gì, cần gì cứ gọi mình.",
        judge=judge_must_not_run,
    ))

    assert final.decision is RAGDecision.ANSWER
    assert final.text == "Không có gì, cần gì cứ gọi mình."
    assert rr.metric.groundedness_reason == "not_applicable_conversation"


def test_empty_rag_answer_is_overridden_to_abstain():
    rr = make_rag_result()

    final = asyncio.run(finalize_rag_answer(rr, "Rìu dùng để làm gì?", ""))

    assert final.overridden is True
    assert final.decision is RAGDecision.ABSTAIN
    assert final.text
    assert rr.metric.rag_decision == "abstain"
    assert rr.metric.decision_reason == "empty_generation"


def test_tool_answer_without_tool_results_is_overridden_to_abstain():
    final = asyncio.run(finalize_tool_answer(
        "vũ khí nào có sát thương cao nhất?",
        "Mình vừa tra database: Katanaklinge có damage 200 [KB-1].",
        [],
        language="vi",
    ))
    assert final.overridden is True
    assert final.decision is RAGDecision.ABSTAIN
    assert "Katanaklinge" not in final.text


def test_tool_error_is_not_treated_as_evidence():
    async def judge_must_not_run(_prompt):
        raise AssertionError("tool errors are not evidence")

    final = asyncio.run(finalize_tool_answer(
        "vũ khí nào có sát thương cao nhất?",
        "Battle Axe mạnh nhất.",
        [{"name": "search_items", "output": '{"error":"Invalid column"}'}],
        language="vi",
        judge=judge_must_not_run,
    ))

    assert final.overridden is True
    assert final.decision is RAGDecision.ABSTAIN
    assert "Battle Axe" not in final.text


def test_empty_tool_answer_is_overridden_to_abstain():
    final = asyncio.run(finalize_tool_answer(
        "vũ khí nào mạnh nhất?",
        "",
        [{"name": "search_items", "output": '{"items":[{"name":"Axe"}]}'}],
        language="vi",
    ))

    assert final.overridden is True
    assert final.decision is RAGDecision.ABSTAIN
    assert final.text
