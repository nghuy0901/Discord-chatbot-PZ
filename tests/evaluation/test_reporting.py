from evaluation.reporting import (
    EvaluationItemResult,
    EvaluationReport,
    stability_rate,
)


def test_report_serializes_items_to_json_dict():
    report = EvaluationReport(
        run_id="run-1",
        dataset_version="public-v1",
        split="development",
        repeat_index=1,
        versions={"prompt": "p1"},
        summary={"recall_at_5": 1.0},
        items=[
            EvaluationItemResult(
                example_id="ex-1",
                expected_behavior="answer",
                actual_behavior="answer",
                answer="ok",
                retrieved_source_ids=["pz/a.md#0"],
                provenance=[{"source_id": "pz/a.md#0"}],
                deterministic_scores={"source_hit_rate": 1.0},
                judge_scores={},
                latency_ms=12.0,
            )
        ],
    )

    data = report.to_dict()

    assert data["run_id"] == "run-1"
    assert data["items"][0]["example_id"] == "ex-1"


def test_stability_rate_compares_behavior_by_example_id():
    first = [
        EvaluationItemResult("ex-1", "answer", "answer", "", [], [], {}, {}, 1.0),
        EvaluationItemResult("ex-2", "abstain", "abstain", "", [], [], {}, {}, 1.0),
    ]
    second = [
        EvaluationItemResult("ex-1", "answer", "answer", "", [], [], {}, {}, 1.0),
        EvaluationItemResult("ex-2", "abstain", "answer", "", [], [], {}, {}, 1.0),
    ]

    assert stability_rate([first, second]) == 0.5
