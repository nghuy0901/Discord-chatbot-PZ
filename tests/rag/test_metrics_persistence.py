from rag.metrics import RAGMetric


def test_metric_contains_hybrid_and_quality_fields():
    metric = RAGMetric(
        vector_results=3,
        bm25_results=2,
        hybrid_fused_results=4,
        vector_time_ms=11.0,
        bm25_time_ms=7.0,
        self_rag_enabled=True,
        self_rag_graded=5,
        self_rag_relevant=4,
        self_rag_irrelevant=1,
        empty_retrieval=False,
        citation_coverage=0.75,
        prompt_version="abc123",
    )

    data = metric.to_dict()

    assert data["vector_results"] == 3
    assert data["self_rag_relevant"] == 4
    assert data["citation_coverage"] == 0.75
    assert data["prompt_version"] == "abc123"
