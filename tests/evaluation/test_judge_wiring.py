"""LLM-judge wiring into the release eval path (audit C5/C6)."""

import asyncio

from evaluation.local_judge import JudgeScore
from evaluation.reporting import EvaluationItemResult
from scripts.run_release_eval import (
    _judge_item,
    effective_gate_config,
    summarize,
)


def _item(judge_scores):
    return EvaluationItemResult(
        example_id="x",
        expected_behavior="answer",
        actual_behavior="answer",
        answer="a",
        retrieved_source_ids=[],
        provenance=[],
        deterministic_scores={},
        judge_scores=judge_scores,
        latency_ms=1.0,
    )


class _Example:
    def __init__(self, ground_truth="gt"):
        self.id = "ex1"
        self.question = "q"
        self.ground_truth = ground_truth


def test_judge_item_scores_answered_item():
    async def fake_judge(*, question, answer, ground_truth, contexts):
        assert ground_truth == "gt"
        return JudgeScore(
            correctness=0.9, faithfulness=0.8,
            unsupported_claim=False, critical_error=False, reason="ok",
        )

    scores = asyncio.run(_judge_item(_Example(), "answer", ["ctx"], fake_judge))
    assert scores["correctness"] == 0.9
    assert scores["faithfulness"] == 0.8
    assert scores["unsupported_claim"] is False


def test_judge_item_skips_without_ground_truth_or_judge():
    async def fake_judge(**kwargs):
        return JudgeScore(1, 1, False, False, "")

    assert asyncio.run(_judge_item(_Example(ground_truth=""), "a", [], fake_judge)) == {}
    assert asyncio.run(_judge_item(_Example(), "a", [], None)) == {}


def test_judge_failure_is_swallowed():
    async def boom(**kwargs):
        raise RuntimeError("judge down")

    assert asyncio.run(_judge_item(_Example(), "a", [], boom)) == {}


def test_summarize_aggregates_judge_scores():
    items = [
        _item({"correctness": 0.9, "faithfulness": 0.95, "unsupported_claim": False, "critical_error": False}),
        _item({"correctness": 0.7, "faithfulness": 0.65, "unsupported_claim": True, "critical_error": True}),
    ]
    s = summarize(items)
    assert s["judged_count"] == 2
    assert s["faithfulness"] == round((0.95 + 0.65) / 2, 4)
    assert s["answer_correctness"] == round((0.9 + 0.7) / 2, 4)
    assert s["unsupported_claim_rate"] == 0.5
    assert s["critical_error_count"] == 1


def test_summarize_omits_judge_metrics_when_unjudged():
    s = summarize([_item({})])
    assert "faithfulness" not in s
    assert "judged_count" not in s


def test_judge_metrics_enforced_only_when_calibrated():
    config = {
        "minimums": {"recall_at_5": 0.9},
        "maximums": {},
        "deferred_minimums": {"faithfulness": 0.92, "human_answer_correctness": 0.95},
        "deferred_maximums": {"unsupported_claim_rate": 0.01, "critical_error_count": 0},
    }
    summary = {
        "recall_at_5": 0.95, "faithfulness": 0.5,
        "unsupported_claim_rate": 0.5, "critical_error_count": 2,
    }

    eff_off, promoted_off = effective_gate_config(config, summary, calibrated=False)
    assert "faithfulness" not in eff_off["minimums"]
    assert promoted_off == []

    eff_on, promoted_on = effective_gate_config(config, summary, calibrated=True)
    assert eff_on["minimums"]["faithfulness"] == 0.92
    assert eff_on["maximums"]["unsupported_claim_rate"] == 0.01
    assert eff_on["maximums"]["critical_error_count"] == 0
    # human_answer_correctness is not judge-produced → stays deferred even when calibrated
    assert "human_answer_correctness" not in eff_on["minimums"]
    assert set(promoted_on) == {"faithfulness", "unsupported_claim_rate", "critical_error_count"}
