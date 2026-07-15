import pytest


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_api_does_not_call_llm_without_evidence(
    async_client,
    monkeypatch,
):
    import api.main as api_main
    from rag.metrics import RAGMetric
    from rag.result import RAGBuildResult, RAGDecision

    called = False

    class CapturingMetrics:
        def __init__(self):
            self.recorded = []

        async def record(self, metric):
            self.recorded.append(metric)

    metrics = CapturingMetrics()

    async def fake_chat_completion(**kwargs):
        nonlocal called
        called = True
        return "invented"

    async def fake_build_rag_result(**kwargs):
        return RAGBuildResult(
            context="",
            domain_prompt="",
            metric=RAGMetric(
                query_id=kwargs["request_context"].query_id,
                request_id=kwargs["request_context"].request_id,
                source=kwargs["request_context"].source,
                channel_id=kwargs["request_context"].channel_id,
                user_id=kwargs["request_context"].user_id,
                original_query=kwargs["query"],
                query_language="vi",
                rag_decision="abstain",
                decision_reason="no_trusted_evidence",
                citation_coverage=1.0,
            ),
            decision=RAGDecision.ABSTAIN,
            decision_reason="no_trusted_evidence",
            evidence_score=0.0,
        )

    monkeypatch.setattr(api_main, "ENABLE_RAG", True)
    monkeypatch.setattr(api_main, "chat_completion", fake_chat_completion)
    monkeypatch.setattr(api_main, "build_rag_result", fake_build_rag_result)
    monkeypatch.setattr(api_main, "get_metrics_manager", lambda: metrics)

    response = await async_client.post(
        "/api/query",
        headers={"X-API-Key": "test-api-key-with-at-least-32chars"},
        json={"query": "Thông tin không có trong KB"},
    )
    assert response.status_code == 200
    assert response.json()["decision"] == "abstain"
    assert called is False
    assert metrics.recorded[0].citation_coverage == 0.0
