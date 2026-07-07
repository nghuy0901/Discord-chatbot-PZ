import pytest


@pytest.mark.asyncio
async def test_retrieve_knowledge_passes_domains_to_hybrid_search(monkeypatch):
    import rag.hybrid_retriever as hybrid_retriever
    import rag.retriever as retriever

    captured = {}

    async def fake_hybrid_search(**kwargs):
        captured.update(kwargs)
        return (
            [{"content": "pz result", "domain": "pz", "source": "pz/doc.md"}],
            {"vector_results": 1, "bm25_results": 0, "fused_results": 1},
        )

    monkeypatch.setattr(retriever, "HYBRID_ENABLED", True)
    monkeypatch.setattr(retriever, "KB_VECTOR_TOP_K", 30, raising=False)
    monkeypatch.setattr(retriever, "KB_BM25_TOP_K", 40, raising=False)
    monkeypatch.setattr(retriever, "KB_FUSED_TOP_K", 40, raising=False)
    monkeypatch.setattr(hybrid_retriever, "hybrid_search", fake_hybrid_search)

    results, primary_domain = await retriever.retrieve_knowledge(
        "generator",
        domains=["pz"],
        top_k=7,
    )

    assert captured["domains"] == ["pz"]
    assert captured["vector_top_k"] == 30
    assert captured["bm25_top_k"] == 40
    assert captured["final_top_k"] == 40
    assert results[0]["domain"] == "pz"
    assert primary_domain == "pz"


@pytest.mark.asyncio
async def test_retrieve_chat_uses_candidate_pool_before_final_cap(monkeypatch):
    import rag.hybrid_retriever as hybrid_retriever
    import rag.retriever as retriever

    captured = {}

    async def fake_hybrid_search(**kwargs):
        captured.update(kwargs)
        return (
            [{"content": "chat result", "trusted": True, "approval_status": "approved"}],
            {"vector_results": 1, "bm25_results": 0, "fused_results": 1},
        )

    monkeypatch.setattr(retriever, "HYBRID_ENABLED", True)
    monkeypatch.setattr(retriever, "RAG_VECTOR_TOP_K", 20, raising=False)
    monkeypatch.setattr(retriever, "RAG_BM25_TOP_K", 20, raising=False)
    monkeypatch.setattr(retriever, "RAG_FUSED_TOP_K", 25, raising=False)
    monkeypatch.setattr(hybrid_retriever, "hybrid_search", fake_hybrid_search)

    await retriever.retrieve("axe", channel_id="c1", top_k=5, include_context=False)

    assert captured["vector_top_k"] == 20
    assert captured["bm25_top_k"] == 20
    assert captured["final_top_k"] == 25


@pytest.mark.asyncio
async def test_hybrid_kb_search_filters_vector_and_bm25_by_domain(monkeypatch):
    from rag.hybrid_retriever import hybrid_search

    vector_domains = []

    class FakeKnowledgeManager:
        def search(self, query, k, score_threshold, domain=None):
            vector_domains.append(domain)
            return [
                {
                    "content": f"{domain} vector result",
                    "domain": domain,
                    "source": f"{domain}/doc.md",
                    "content_type": "knowledge_base",
                    "trusted": True,
                    "similarity": 0.9,
                }
            ]

    class FakeBM25:
        is_ready = True

        def search(self, query, top_k, min_score, filter_dict=None):
            return [
                {
                    "content": "pz bm25 result",
                    "domain": "pz",
                    "source": "pz/bm25.md",
                    "content_type": "knowledge_base",
                    "trusted": True,
                    "bm25_score": 3.0,
                },
                {
                    "content": "server rules bm25 result",
                    "domain": "server_rules",
                    "source": "server_rules/bm25.md",
                    "content_type": "knowledge_base",
                    "trusted": True,
                    "bm25_score": 2.5,
                },
            ]

    monkeypatch.setattr(
        "knowledge.manager.get_knowledge_manager",
        lambda: FakeKnowledgeManager(),
    )
    monkeypatch.setattr("rag.bm25_search.get_kb_bm25", lambda: FakeBM25())

    results, meta = await hybrid_search(
        query="generator",
        search_type="kb",
        domains=["pz"],
        vector_top_k=3,
        bm25_top_k=3,
        final_top_k=5,
    )

    assert vector_domains == ["pz"]
    assert {result["domain"] for result in results} == {"pz"}
    assert meta["vector_results"] == 1
    assert meta["bm25_results"] == 1
