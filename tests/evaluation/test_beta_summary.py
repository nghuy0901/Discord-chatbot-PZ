from scripts.summarize_beta import summarize_beta


def test_beta_fails_on_confirmed_critical_incident():
    summary = summarize_beta(
        total_queries=300,
        positive_feedback=250,
        negative_feedback=20,
        reviewed_factual_negatives=20,
        confirmed_factual_errors=3,
        critical_incidents=1,
        pii_leaks=0,
        unapproved_source_leaks=0,
        p95_latency_ms=12000,
    )
    assert summary["passed"] is False
    assert "critical_incidents" in summary["failures"]
