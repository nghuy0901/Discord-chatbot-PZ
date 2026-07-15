"""Live smoke tests for the groundedness gate + chunk-level citations.

These drive the real FastAPI app and the real Discord message handler, mocking
only the external boundaries (LLM answer, RAG retrieval, groundedness judge).
"""

import pytest
from types import SimpleNamespace

from rag.metrics import RAGMetric
from rag.result import RAGBuildResult, RAGDecision, ProvenanceItem


TEST_API_KEY = "test-api-key-with-at-least-32chars"


def _provenance():
    return [
        ProvenanceItem(
            source_id="pz:Weapons/Axes.md:0:abc",
            source_kind="knowledge_base",
            domain="pz",
            trusted=True,
            rank=1,
            similarity=0.72,
            label="1",
            locator="pz/Weapons/Axes.md › Weapons > Axes",
            source="pz/Weapons/Axes.md",
            heading_path="Weapons > Axes",
            excerpt="Axes degrade with use",
        )
    ]


def _answer_rag_result(*, query="q", **ctx):
    return RAGBuildResult(
        context="[1] (KB) pz/Weapons/Axes.md › Weapons > Axes\n    Axes degrade with use.",
        domain_prompt="",
        metric=RAGMetric(
            original_query=query,
            query_language="vi",
            query_intent="narrative",
            rag_decision="answer",
            **ctx,
        ),
        decision=RAGDecision.ANSWER,
        decision_reason="trusted_evidence_sufficient",
        evidence_score=0.84,
        provenance=_provenance(),
        retrieved_results=[
            {
                "content": "Axes degrade with use.",
                "source": "pz/Weapons/Axes.md",
                "content_type": "knowledge_base",
                "domain": "pz",
                "trusted": True,
            }
        ],
        primary_domain="pz",
    )


def _conversation_rag_result(*, query="q", **ctx):
    return RAGBuildResult(
        context="",
        domain_prompt="",
        metric=RAGMetric(
            original_query=query,
            query_language="vi",
            query_intent="conversation",
            rag_decision="answer",
            **ctx,
        ),
        decision=RAGDecision.ANSWER,
        decision_reason="conversation_no_rag_required",
        evidence_score=0.0,
        provenance=[],
        retrieved_results=[],
    )


def _judge(payload):
    async def judge(**kwargs):
        return payload
    return judge


# --------------------------------------------------------------------------- #
#  API path
# --------------------------------------------------------------------------- #
@pytest.mark.e2e
@pytest.mark.asyncio
async def test_api_conversation_bypasses_rag_finalizer(async_client, monkeypatch):
    import api.main as api_main

    async def fake_build(**kwargs):
        rc = kwargs["request_context"]
        return _conversation_rag_result(
            query=kwargs["query"], query_id=rc.query_id,
            request_id=rc.request_id, source=rc.source,
            channel_id=rc.channel_id, user_id=rc.user_id,
        )

    async def fake_answer(**kwargs):
        return "Không có gì, cần gì cứ gọi mình."

    monkeypatch.setattr(api_main, "ENABLE_RAG", True)
    monkeypatch.setattr(api_main, "build_rag_result", fake_build)
    monkeypatch.setattr(api_main, "chat_completion", fake_answer)

    resp = await async_client.post(
        "/api/query",
        headers={"X-API-Key": TEST_API_KEY},
        json={"query": "Cảm ơn nhé", "include_sources": True},
    )

    data = resp.json()
    assert data["decision"] == "answer"
    assert data["response"] == "Không có gì, cần gì cứ gọi mình."
    assert data["source_count"] == 0


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_api_empty_generation_never_returns_answer(async_client, monkeypatch):
    import api.main as api_main

    async def fake_build(**kwargs):
        rc = kwargs["request_context"]
        return _answer_rag_result(
            query=kwargs["query"], query_id=rc.query_id,
            request_id=rc.request_id, source=rc.source,
            channel_id=rc.channel_id, user_id=rc.user_id,
        )

    async def fake_answer(**kwargs):
        return ""

    monkeypatch.setattr(api_main, "ENABLE_RAG", True)
    monkeypatch.setattr(api_main, "build_rag_result", fake_build)
    monkeypatch.setattr(api_main, "chat_completion", fake_answer)

    resp = await async_client.post(
        "/api/query",
        headers={"X-API-Key": TEST_API_KEY},
        json={"query": "Rìu dùng để làm gì?"},
    )

    data = resp.json()
    assert data["decision"] == "abstain"
    assert data["response"]


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_api_grounded_answer_returns_chunk_citations(async_client, monkeypatch):
    import api.main as api_main
    import src.ollama_provider as provider

    async def fake_build(**kwargs):
        rc = kwargs["request_context"]
        return _answer_rag_result(
            query=kwargs["query"],
            query_id=rc.query_id,
            request_id=rc.request_id,
            source=rc.source,
            channel_id=rc.channel_id,
            user_id=rc.user_id,
        )

    async def fake_answer(**kwargs):
        return "Rìu bị mòn khi sử dụng [1]."

    monkeypatch.setattr(api_main, "ENABLE_RAG", True)
    monkeypatch.setattr(api_main, "build_rag_result", fake_build)
    monkeypatch.setattr(api_main, "chat_completion", fake_answer)
    monkeypatch.setattr(provider, "chat_completion", _judge('{"grounded": true, "score": 0.95}'))

    resp = await async_client.post(
        "/api/query",
        headers={"X-API-Key": TEST_API_KEY},
        json={"query": "Rìu dùng để làm gì?", "domain": "pz", "include_sources": True},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["decision"] == "answer"
    assert "[1]" in data["response"]
    assert "Nguồn" in data["response"]              # footer attached
    assert data["citations"], "expected chunk-level citations"
    assert data["source_count"] == 1
    cite = data["citations"][0]
    assert cite["label"] == "1"
    assert "Axes.md" in cite["locator"]


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_api_hybrid_uses_vector_path_without_tools(async_client, monkeypatch):
    import api.main as api_main
    import src.ollama_provider as provider

    async def fake_build(**kwargs):
        rc = kwargs["request_context"]
        result = _answer_rag_result(
            query=kwargs["query"], query_id=rc.query_id,
            request_id=rc.request_id, source=rc.source,
            channel_id=rc.channel_id, user_id=rc.user_id,
        )
        result.metric.query_intent = "hybrid"
        return result

    async def fake_answer(**kwargs):
        return "Rìu bị mòn khi sử dụng [1]."

    async def tools_must_not_run(**kwargs):
        raise AssertionError("hybrid narrative queries should not call structured tools")

    monkeypatch.setattr(api_main, "ENABLE_RAG", True)
    monkeypatch.setattr(api_main, "ENABLE_TOOL_CALLING", True)
    monkeypatch.setattr(api_main, "STRUCTURED_TOOLS_AVAILABLE", True)
    monkeypatch.setattr(api_main, "PZ_TOOLS", [{"type": "function"}])
    monkeypatch.setattr(api_main, "build_rag_result", fake_build)
    monkeypatch.setattr(api_main, "chat_with_tools", tools_must_not_run)
    monkeypatch.setattr(api_main, "chat_completion", fake_answer)
    monkeypatch.setattr(provider, "get_last_tool_results", lambda: [])
    monkeypatch.setattr(
        provider, "chat_completion",
        _judge('{"grounded": true, "score": 0.95}'),
    )

    resp = await async_client.post(
        "/api/query",
        headers={"X-API-Key": TEST_API_KEY},
        json={"query": "Rìu dùng để làm gì?", "domain": "pz", "include_sources": True},
    )

    assert resp.status_code == 200
    data = resp.json()
    assert data["decision"] == "answer"
    assert data["source_count"] == 1
    assert data["citations"][0]["label"] == "1"


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_api_failed_tool_falls_back_to_vector_evidence(async_client, monkeypatch):
    import api.main as api_main
    import src.ollama_provider as provider

    async def fake_build(**kwargs):
        rc = kwargs["request_context"]
        result = _answer_rag_result(
            query=kwargs["query"], query_id=rc.query_id,
            request_id=rc.request_id, source=rc.source,
            channel_id=rc.channel_id, user_id=rc.user_id,
        )
        result.metric.query_intent = "analytical"
        return result

    async def fake_answer(**kwargs):
        return "Rìu bị mòn khi sử dụng [1]."

    monkeypatch.setattr(api_main, "ENABLE_RAG", True)
    monkeypatch.setattr(api_main, "ENABLE_TOOL_CALLING", True)
    monkeypatch.setattr(api_main, "STRUCTURED_TOOLS_AVAILABLE", True)
    monkeypatch.setattr(api_main, "PZ_TOOLS", [{"type": "function"}])
    monkeypatch.setattr(api_main, "build_rag_result", fake_build)
    monkeypatch.setattr(api_main, "chat_with_tools", fake_answer)
    monkeypatch.setattr(
        provider,
        "get_last_tool_results",
        lambda: [{"name": "search_items", "output": '{"error":"Invalid column"}'}],
    )
    monkeypatch.setattr(
        provider, "chat_completion",
        _judge('{"grounded": true, "score": 0.95}'),
    )

    resp = await async_client.post(
        "/api/query",
        headers={"X-API-Key": TEST_API_KEY},
        json={"query": "Rìu dùng để làm gì?", "domain": "pz", "include_sources": True},
    )

    data = resp.json()
    assert data["decision"] == "answer"
    assert data["source_count"] == 1


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_api_records_metric_once_after_finalization(async_client, monkeypatch):
    import api.main as api_main
    import src.ollama_provider as provider

    class CapturingMetrics:
        def __init__(self):
            self.recorded = []
            self.updated = []

        async def record(self, metric):
            self.recorded.append(metric)

        async def update_db(self, metric):
            self.updated.append(metric)

    metrics = CapturingMetrics()

    async def fake_build(**kwargs):
        rc = kwargs["request_context"]
        return _answer_rag_result(
            query=kwargs["query"], query_id=rc.query_id,
            request_id=rc.request_id, source=rc.source,
            channel_id=rc.channel_id, user_id=rc.user_id,
        )

    async def fake_answer(**kwargs):
        return "Rìu bị mòn khi sử dụng [1]."

    monkeypatch.setattr(api_main, "ENABLE_RAG", True)
    monkeypatch.setattr(api_main, "build_rag_result", fake_build)
    monkeypatch.setattr(api_main, "chat_completion", fake_answer)
    monkeypatch.setattr(api_main, "get_metrics_manager", lambda: metrics)
    monkeypatch.setattr(
        provider, "chat_completion",
        _judge('{"grounded": true, "score": 0.95}'),
    )

    resp = await async_client.post(
        "/api/query",
        headers={"X-API-Key": TEST_API_KEY},
        json={"query": "Rìu dùng để làm gì?", "domain": "pz"},
    )

    assert resp.status_code == 200
    assert len(metrics.recorded) == 1
    assert metrics.updated == []
    metric = metrics.recorded[0]
    assert metric.rag_decision == "answer"
    assert metric.response_length > 0
    assert metric.citation_coverage == 1.0


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_api_ungrounded_answer_is_refused(async_client, monkeypatch):
    import api.main as api_main
    import src.ollama_provider as provider

    async def fake_build(**kwargs):
        rc = kwargs["request_context"]
        return _answer_rag_result(
            query=kwargs["query"],
            query_id=rc.query_id,
            request_id=rc.request_id,
            source=rc.source,
            channel_id=rc.channel_id,
            user_id=rc.user_id,
        )

    async def fake_answer(**kwargs):
        return "Rìu có thể bắn laser [1]."

    monkeypatch.setattr(api_main, "ENABLE_RAG", True)
    monkeypatch.setattr(api_main, "build_rag_result", fake_build)
    monkeypatch.setattr(api_main, "chat_completion", fake_answer)
    monkeypatch.setattr(provider, "chat_completion", _judge('{"grounded": false, "score": 0.1}'))

    resp = await async_client.post(
        "/api/query",
        headers={"X-API-Key": TEST_API_KEY},
        json={"query": "Rìu có bắn laser không?", "domain": "pz", "include_sources": True},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["decision"] == "abstain"
    assert "laser" not in data["response"].lower()   # hallucination not shown
    assert data["source_count"] == 0
    assert data["citations"] == []


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_api_cache_hit_re_renders_citations(async_client, monkeypatch):
    import api.main as api_main

    cached = {
        "response_text": "Rìu bị mòn khi sử dụng [1].",
        "decision": "answer",
        "language": "vi",
        "provenance": [p.to_dict() for p in _provenance()],
        "prompt_tokens": 3,
        "completion_tokens": 4,
    }

    class DummyCache:
        async def get(self, query, domain):
            return cached

        async def set(self, **kwargs):
            return None

    class StubPre:
        def preprocess(self, query, channel_name=None):
            return query, {
                "domain": "pz",
                "language": "vi",
                "query_intent": SimpleNamespace(value="narrative"),
            }

    monkeypatch.setattr(api_main, "ENABLE_RAG", True)
    monkeypatch.setattr(api_main, "get_preprocessor", lambda: StubPre())
    monkeypatch.setattr(api_main, "get_query_cache", lambda: DummyCache())

    resp = await async_client.post(
        "/api/query",
        headers={"X-API-Key": TEST_API_KEY},
        json={"query": "Rìu dùng để làm gì?", "domain": "pz", "include_sources": True},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["cache_hit"] is True
    assert "Nguồn" in data["response"]
    assert data["citations"][0]["label"] == "1"


# --------------------------------------------------------------------------- #
#  Discord path
# --------------------------------------------------------------------------- #
@pytest.mark.e2e
@pytest.mark.asyncio
async def test_discord_grounded_answer_sends_citation_footer(mock_discord_message, monkeypatch):
    import src.aclient as aclient
    import src.ollama_provider as provider
    import rag.metrics as metrics_module

    class CapturingMetrics:
        def __init__(self):
            self.recorded = []
            self.updated = []

        async def record(self, metric):
            self.recorded.append(metric)

        async def update_db(self, metric):
            self.updated.append(metric)

    metrics = CapturingMetrics()

    async def fake_build_prompt(**kwargs):
        return (
            [{"role": "user", "content": kwargs["user_message"]}],
            0.1,
            "narrative",
            _answer_rag_result(query=kwargs["user_message"]),
        )

    async def fake_answer(**kwargs):
        return "Rìu bị mòn khi sử dụng [1]."

    mock_discord_message.channel  # ensure built
    aclient.discordClient.context_manager.build_prompt = fake_build_prompt
    monkeypatch.setattr(aclient, "chat_completion", fake_answer)
    monkeypatch.setattr(provider, "chat_completion", _judge('{"grounded": true, "score": 0.95}'))
    monkeypatch.setattr(metrics_module, "get_metrics_manager", lambda: metrics)

    await aclient.discordClient._generate_and_send(mock_discord_message, "Rìu dùng để làm gì?")

    sent = mock_discord_message.channel.sent
    assert any("Nguồn" in m.content and "[1]" in m.content for m in sent), \
        f"no citation footer sent: {[m.content for m in sent]}"
    assert len(metrics.recorded) == 1
    assert metrics.updated == []
    assert metrics.recorded[0].response_length > 0


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_discord_ungrounded_answer_is_edited_to_refusal(mock_discord_message, monkeypatch):
    import src.aclient as aclient
    import src.ollama_provider as provider

    async def fake_build_prompt(**kwargs):
        return (
            [{"role": "user", "content": kwargs["user_message"]}],
            0.1,
            "narrative",
            _answer_rag_result(query=kwargs["user_message"]),
        )

    async def fake_answer(**kwargs):
        return "Rìu có thể bắn laser [1]."

    aclient.discordClient.context_manager.build_prompt = fake_build_prompt
    monkeypatch.setattr(aclient, "chat_completion", fake_answer)
    monkeypatch.setattr(provider, "chat_completion", _judge('{"grounded": false, "score": 0.1}'))

    await aclient.discordClient._generate_and_send(mock_discord_message, "Rìu có bắn laser không?")

    sent = mock_discord_message.channel.sent
    assert sent, "expected at least one message"
    assert "laser" not in sent[0].content.lower()        # answer was overridden
    assert not any("Nguồn" in m.content for m in sent)    # no footer for a refusal


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_discord_pre_generation_abstain_resets_citation_metric(
    mock_discord_message, monkeypatch
):
    import src.aclient as aclient
    import rag.metrics as metrics_module

    class CapturingMetrics:
        def __init__(self):
            self.recorded = []

        async def record(self, metric):
            self.recorded.append(metric)

    metric = RAGMetric(
        query_language="vi",
        query_intent="narrative",
        rag_decision="abstain",
        citation_coverage=1.0,
    )
    rag_result = RAGBuildResult(
        context="",
        domain_prompt="",
        metric=metric,
        decision=RAGDecision.ABSTAIN,
        decision_reason="no_trusted_evidence",
        evidence_score=0.0,
    )
    metrics = CapturingMetrics()

    async def fake_build_prompt(**kwargs):
        return ([{"role": "user", "content": kwargs["user_message"]}], 0.1, "narrative", rag_result)

    monkeypatch.setattr(aclient.discordClient.context_manager, "build_prompt", fake_build_prompt)
    monkeypatch.setattr(metrics_module, "get_metrics_manager", lambda: metrics)

    await aclient.discordClient._generate_and_send(mock_discord_message, "Rìu có bay không?")

    assert metrics.recorded[0].citation_coverage == 0.0


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_slash_failed_tool_uses_rag_finalizer_and_records_metric(monkeypatch):
    import src.aclient as aclient
    import rag.answer_finalize as answer_finalize
    import rag.metrics as metrics_module
    import src.ollama_provider as provider
    from rag.answer_finalize import FinalizedAnswer

    class CapturingMetrics:
        def __init__(self):
            self.recorded = []

        async def record(self, metric):
            self.recorded.append(metric)

    metrics = CapturingMetrics()
    rag_result = _answer_rag_result(query="Rìu gây bao nhiêu sát thương?")
    rag_result.metric.query_intent = "analytical"
    called = {"rag": 0}

    async def fake_build_prompt(**kwargs):
        return ([{"role": "user", "content": kwargs["user_message"]}], 0.1, "analytical", rag_result)

    async def fake_chat_with_tools(**kwargs):
        return "Rìu gây sát thương theo bảng chỉ số [1]."

    async def fake_finalize_rag(*args, **kwargs):
        called["rag"] += 1
        return FinalizedAnswer(text=args[2], decision=RAGDecision.ANSWER)

    async def tool_finalizer_must_not_run(*args, **kwargs):
        raise AssertionError("failed tool output is not evidence")

    monkeypatch.setattr(aclient, "ENABLE_TOOL_CALLING", True)
    monkeypatch.setattr(aclient, "STRUCTURED_TOOLS_AVAILABLE", True)
    monkeypatch.setattr(aclient, "PZ_TOOLS", [{"type": "function"}])
    monkeypatch.setattr(aclient.discordClient.context_manager, "build_prompt", fake_build_prompt)
    monkeypatch.setattr(aclient, "chat_with_tools", fake_chat_with_tools)
    monkeypatch.setattr(
        provider,
        "get_last_tool_results",
        lambda: [{"name": "search_items", "output": '{"error":"Invalid column"}'}],
    )
    monkeypatch.setattr(answer_finalize, "finalize_rag_answer", fake_finalize_rag)
    monkeypatch.setattr(answer_finalize, "finalize_tool_answer", tool_finalizer_must_not_run)
    monkeypatch.setattr(metrics_module, "get_metrics_manager", lambda: metrics)

    response = await aclient.discordClient.handle_response("Rìu gây bao nhiêu sát thương?")

    assert response == "Rìu gây sát thương theo bảng chỉ số [1]."
    assert called["rag"] == 1
    assert len(metrics.recorded) == 1
    assert metrics.recorded[0].rag_decision == "answer"
    assert metrics.recorded[0].citation_coverage == 1.0
