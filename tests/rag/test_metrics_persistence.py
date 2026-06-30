import json

import pytest

from rag.metrics import (
    METRIC_INSERT_COLUMNS,
    REQUIRED_RAG_METRIC_COLUMNS,
    MetricsManager,
    RAGMetric,
)


class FakeConnection:
    def __init__(self):
        self.calls = []

    async def execute(self, sql, *args):
        self.calls.append((sql, args))


class FakeAcquire:
    def __init__(self, connection):
        self.connection = connection

    async def __aenter__(self):
        return self.connection

    async def __aexit__(self, exc_type, exc, tb):
        return False


class FakePool:
    def __init__(self):
        self.connection = FakeConnection()

    def acquire(self):
        return FakeAcquire(self.connection)


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
    provenance = [{"title": "Luật máy chủ", "domain": "server_rules"}]
    metric = RAGMetric(
        rag_decision="answer",
        decision_reason="trusted evidence",
        provenance=provenance,
        trusted_source_count=2,
        untrusted_source_count=1,
        evidence_score=0.8,
    )

    data = metric.to_dict()

    assert data["rag_decision"] == "answer"
    assert data["decision_reason"] == "trusted evidence"
    assert data["provenance"] == provenance
    assert data["trusted_source_count"] == 2
    assert data["untrusted_source_count"] == 1
    assert data["evidence_score"] == 0.8


def test_trust_quality_fields_are_required_and_persisted():
    fields = {
        "rag_decision",
        "decision_reason",
        "provenance",
        "trusted_source_count",
        "untrusted_source_count",
        "evidence_score",
    }

    assert fields <= REQUIRED_RAG_METRIC_COLUMNS
    assert fields <= set(METRIC_INSERT_COLUMNS)


def test_metric_values_serializes_provenance_as_json():
    provenance = [{"title": "Nội quy", "domain": "server_rules"}]
    metric = RAGMetric(provenance=provenance)

    values = MetricsManager()._metric_values(metric, ["provenance"])

    assert json.loads(values[0]) == provenance
    assert "Nội quy" in values[0]


@pytest.mark.asyncio
async def test_persist_metric_passes_provenance_as_json_text(monkeypatch):
    pool = FakePool()

    async def fake_get_pool():
        return pool

    monkeypatch.setattr("rag.db.get_pool", fake_get_pool)
    provenance = [{"title": "Nội quy", "domain": "server_rules"}]

    await MetricsManager()._persist_metric(RAGMetric(provenance=provenance))

    sql, args = pool.connection.calls[0]
    provenance_arg = args[METRIC_INSERT_COLUMNS.index("provenance")]
    assert "INSERT INTO rag_metrics" in sql
    assert isinstance(provenance_arg, str)
    assert json.loads(provenance_arg) == provenance
