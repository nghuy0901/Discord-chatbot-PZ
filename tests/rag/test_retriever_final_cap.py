import pytest


def _trusted_kb(i: int):
    return {
        "content": f"kb result {i}",
        "content_type": "knowledge_base",
        "source_kind": "knowledge_base",
        "domain": "pz",
        "trusted": True,
        "source": f"pz/doc-{i}.md",
        "similarity": 0.8,
        "retrieval_methods": ["vector", "bm25"],
    }


def _trusted_chat(i: int):
    return {
        "content": f"chat result {i}",
        "source_kind": "chat",
        "approval_status": "approved",
        "trusted": True,
        "message_id": f"m-{i}",
        "similarity": 0.7,
        "retrieval_methods": ["vector"],
    }


@pytest.mark.asyncio
async def test_build_rag_result_caps_sources_after_candidate_retrieval(monkeypatch):
    import rag.retriever as retriever

    class FakePreprocessor:
        def preprocess(self, query, channel_name=None):
            return query, {
                "language": "vi",
                "domain": "pz",
                "query_intent": "narrative",
            }

    class FakeRouter:
        def route(self, query, detected_domain=None, channel_name=None):
            return ["pz"]

        def get_domain_prompt(self, domain):
            return ""

    class FakeMetrics:
        async def record(self, metric):
            return None

    async def fake_retrieve_knowledge(**kwargs):
        return [_trusted_kb(i) for i in range(10)], "pz"

    async def fake_retrieve(**kwargs):
        return [_trusted_chat(i) for i in range(10)]

    monkeypatch.setattr(retriever, "KB_TOP_K", 2)
    monkeypatch.setattr(retriever, "SELF_RAG_ENABLED", False)
    monkeypatch.setattr(retriever, "get_preprocessor", lambda: FakePreprocessor())
    monkeypatch.setattr(retriever, "get_metrics_manager", lambda: FakeMetrics())
    monkeypatch.setattr(retriever, "retrieve_knowledge", fake_retrieve_knowledge)
    monkeypatch.setattr(retriever, "retrieve", fake_retrieve)
    monkeypatch.setattr("knowledge.domain_router.get_domain_router", lambda: FakeRouter())

    result = await retriever.build_rag_result("rìu nào tốt?", top_k=3)

    assert result.decision.value == "answer"
    assert len(result.retrieved_results) == 5
    assert sum(1 for item in result.retrieved_results if item["content"].startswith("kb")) == 2
    assert sum(1 for item in result.retrieved_results if item["content"].startswith("chat")) == 3
