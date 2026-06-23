from api.schemas import MetricsSummary


def test_metrics_summary_contains_dashboard_fields():
    summary = MetricsSummary(
        total_queries=10,
        success_rate=0.9,
        error_rate=0.1,
        avg_latency_ms=100,
        p50_latency_ms=80,
        p95_latency_ms=200,
        p99_latency_ms=300,
        retrieval_latency_ms=40,
        llm_latency_ms=60,
        empty_retrieval_rate=0.2,
        average_retrieved_chunks=4.5,
        citation_coverage=0.7,
        cache_hit_rate=0.3,
        total_tokens=1000,
        estimated_cost_usd=0.01,
    )

    data = summary.model_dump()
    assert data["empty_retrieval_rate"] == 0.2
    assert data["citation_coverage"] == 0.7
