from evaluation.gates import evaluate_gates


def test_any_hard_gate_failure_blocks_release():
    summary = {
        "recall_at_5": 0.95,
        "source_hit_rate": 0.98,
        "critical_error_count": 1,
        "p95_latency_ms": 1000,
    }
    result = evaluate_gates(
        summary,
        {
            "minimums": {"recall_at_5": 0.92, "source_hit_rate": 0.95},
            "maximums": {"critical_error_count": 0, "p95_latency_ms": 15000},
        },
    )
    assert result.passed is False
    assert "critical_error_count" in result.failures


def test_missing_metric_fails_closed():
    result = evaluate_gates({}, {"minimums": {"recall_at_5": 0.92}})
    assert result.passed is False
    assert result.failures["recall_at_5"]["actual"] is None


def test_baseline_gate_only_references_computed_metrics():
    """The shipped gate config must only enforce metrics the eval actually
    produces — otherwise the now-authoritative gate could never pass (audit C5).
    Un-producible metrics live under deferred_* and are not enforced yet."""
    from scripts.run_release_eval import load_gate_config, summarize
    from evaluation.reporting import EvaluationItemResult

    item = EvaluationItemResult(
        example_id="x",
        expected_behavior="answer",
        actual_behavior="answer",
        answer="a",
        retrieved_source_ids=["s"],
        provenance=[{"trusted": True}],
        deterministic_scores={
            "recall_at_5": 1.0,
            "mrr": 1.0,
            "ndcg_at_5": 1.0,
            "source_hit_rate": 1.0,
            "keyword_coverage": 1.0,
        },
        judge_scores={},
        latency_ms=1.0,
    )
    produced = set(summarize([item]).keys())
    config = load_gate_config()
    gated = set(config.get("minimums", {})) | set(config.get("maximums", {}))
    missing = gated - produced
    assert not missing, f"gate references metrics the eval never computes: {sorted(missing)}"


def test_retrieval_summary_excludes_abstain_and_clarify_rows():
    """Rows with intentionally empty expected_sources must not lower Recall."""
    from evaluation.reporting import EvaluationItemResult
    from scripts.run_release_eval import summarize

    answerable = EvaluationItemResult(
        example_id="answer", expected_behavior="answer", actual_behavior="answer",
        answer="a", retrieved_source_ids=["pz/a.md"], provenance=[],
        deterministic_scores={
            "recall_at_5": 1.0, "mrr": 1.0, "ndcg_at_5": 1.0,
            "source_hit_rate": 1.0, "keyword_coverage": 1.0,
        },
        judge_scores={}, latency_ms=1.0,
    )
    abstain = EvaluationItemResult(
        example_id="abstain", expected_behavior="abstain", actual_behavior="abstain",
        answer="", retrieved_source_ids=[], provenance=[], deterministic_scores={},
        judge_scores={}, latency_ms=1.0,
    )

    summary = summarize([answerable, abstain])

    assert summary["answerable_retrieval_count"] == 1
    assert summary["recall_at_5"] == 1.0
    assert summary["source_hit_rate"] == 1.0
