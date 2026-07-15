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
        lexical_query="máy phát điện generator",
        domains=["pz"],
        top_k=7,
    )

    assert captured["domains"] == ["pz"]
    assert captured["query"] == "generator"
    assert captured["bm25_query"] == "máy phát điện generator"
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

    await retriever.retrieve(
        "rìu", lexical_query="rìu axe", channel_id="c1", top_k=5,
        include_context=False,
    )

    assert captured["query"] == "rìu"
    assert captured["bm25_query"] == "rìu axe"
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


@pytest.mark.asyncio
async def test_hybrid_search_uses_separate_vector_and_bm25_queries(monkeypatch):
    from rag.hybrid_retriever import hybrid_search

    captured = {"vector": []}

    class FakeKnowledgeManager:
        def search(self, query, k, score_threshold, domain=None):
            captured["vector"].append(query)
            return []

    class FakeBM25:
        is_ready = True

        def search(self, query, top_k, min_score, filter_dict=None):
            captured["bm25"] = query
            return []

    monkeypatch.setattr(
        "knowledge.manager.get_knowledge_manager", lambda: FakeKnowledgeManager()
    )
    monkeypatch.setattr("rag.bm25_search.get_kb_bm25", lambda: FakeBM25())

    await hybrid_search(
        query="rìu dùng ra sao?",
        bm25_query="rìu dùng ra sao axe",
        search_type="kb",
        domains=["pz"],
    )

    assert captured == {
        "vector": ["rìu dùng ra sao?", "rìu dùng ra sao axe"],
        "bm25": "rìu dùng ra sao axe",
    }


@pytest.mark.asyncio
async def test_augmented_query_vector_hit_survives_final_top_k(monkeypatch):
    from rag.hybrid_retriever import hybrid_search

    class FakeKnowledgeManager:
        def search(self, query, k, score_threshold, domain=None):
            if query == "rìu sát thương bao nhiêu?":
                return [
                    {
                        "doc_id": f"generic-{rank}",
                        "content": f"Generic axe prose {rank}",
                        "domain": "pz",
                        "source": "pz/Weapons/Axes.md",
                        "content_type": "knowledge_base",
                        "trusted": True,
                        "similarity": 0.90 - rank / 100,
                    }
                    for rank in range(7)
                ]
            return [
                {
                    "doc_id": "axe-stats",
                    "content": "Axe damage 1.0-2.5",
                    "domain": "pz",
                    "source": "pz/Items/Axe.md",
                    "content_type": "knowledge_base",
                    "trusted": True,
                    "similarity": 0.95,
                }
            ]

    class FakeBM25:
        is_ready = True

        def search(self, query, top_k, min_score, filter_dict=None):
            return [
                {
                    "doc_id": "axe-stats",
                    "content": "Axe damage 1.0-2.5",
                    "domain": "pz",
                    "source": "pz/Items/Axe.md",
                    "content_type": "knowledge_base",
                    "trusted": True,
                    "bm25_score": 5.0,
                }
            ]

    monkeypatch.setattr(
        "knowledge.manager.get_knowledge_manager", lambda: FakeKnowledgeManager()
    )
    monkeypatch.setattr("rag.bm25_search.get_kb_bm25", lambda: FakeBM25())

    results, _ = await hybrid_search(
        query="rìu sát thương bao nhiêu?",
        bm25_query="rìu sát thương bao nhiêu axe damage",
        search_type="kb",
        domains=["pz"],
        vector_top_k=10,
        bm25_top_k=10,
        final_top_k=7,
    )

    assert any(result["doc_id"] == "axe-stats" for result in results)


def test_record_cap_does_not_drop_distinct_prose_chunks():
    from rag.hybrid_retriever import _cap_per_record

    results = [
        {
            "content": f"Muldraugh prose chunk {index}",
            "content_mode": "prose",
            "source": "pz/Locations/Muldraugh.md",
            "heading_path": "Muldraugh",
            "doc_id": f"doc-{index}",
        }
        for index in range(4)
    ]

    assert _cap_per_record(results, 2) == results


def test_record_cap_still_limits_duplicate_record_items():
    from rag.hybrid_retriever import _cap_per_record

    results = [
        {
            "content": f"Axe record {index}",
            "content_mode": "record_item",
            "source": "pz/Equipment/Tools.md",
            "record_name": "Axe",
            "doc_id": f"doc-{index}",
        }
        for index in range(4)
    ]

    assert len(_cap_per_record(results, 2)) == 2
