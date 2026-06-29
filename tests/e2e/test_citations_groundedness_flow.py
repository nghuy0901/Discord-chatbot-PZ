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


def _judge(payload):
    async def judge(**kwargs):
        return payload
    return judge


# --------------------------------------------------------------------------- #
#  API path
# --------------------------------------------------------------------------- #
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
    cite = data["citations"][0]
    assert cite["label"] == "1"
    assert "Axes.md" in cite["locator"]


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

    await aclient.discordClient._generate_and_send(mock_discord_message, "Rìu dùng để làm gì?")

    sent = mock_discord_message.channel.sent
    assert any("Nguồn" in m.content and "[1]" in m.content for m in sent), \
        f"no citation footer sent: {[m.content for m in sent]}"


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
