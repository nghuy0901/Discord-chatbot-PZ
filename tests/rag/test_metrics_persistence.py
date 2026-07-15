from rag.metrics import RAGMetric
import pytest


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


def test_metric_contains_trust_decision_and_provenance_fields():
    metric = RAGMetric(
        rag_decision="answer",
        decision_reason="trusted evidence met threshold",
        provenance=[{"source_id": "pz:crafting:axe", "domain": "pz"}],
        trusted_source_count=2,
        untrusted_source_count=1,
        evidence_score=0.82,
        groundedness_score=0.91,
        groundedness_reason="checked",
        groundedness_unsupported_count=1,
    )

    data = metric.to_dict()

    assert data["rag_decision"] == "answer"
    assert data["decision_reason"] == "trusted evidence met threshold"
    assert data["provenance"] == [{"source_id": "pz:crafting:axe", "domain": "pz"}]
    assert data["trusted_source_count"] == 2
    assert data["untrusted_source_count"] == 1
    assert data["evidence_score"] == 0.82
    assert data["groundedness_score"] == 0.91
    assert data["groundedness_reason"] == "checked"
    assert data["groundedness_unsupported_count"] == 1


@pytest.mark.asyncio
async def test_db_summary_excludes_conversation_from_groundedness(monkeypatch):
    from rag.metrics import MetricsManager

    captured = {}

    class Connection:
        async def fetchrow(self, query, *args):
            if "FROM rag_metrics" in query:
                captured["query"] = query
                return {}
            return {}

    class Acquire:
        async def __aenter__(self):
            return Connection()

        async def __aexit__(self, *args):
            return None

    class Pool:
        def acquire(self):
            return Acquire()

    async def fake_get_pool():
        return Pool()

    monkeypatch.setattr("rag.db.get_pool", fake_get_pool)
    manager = MetricsManager()
    manager._db_initialized = True

    await manager.get_db_summary()

    assert "groundedness_reason <> 'not_applicable_conversation'" in captured["query"]
