from langchain_core.documents import Document


def test_search_similar_filters_metadata_in_python(monkeypatch):
    import rag.db as db

    calls = []

    class FakeStore:
        def similarity_search_with_relevance_scores(self, **kwargs):
            calls.append(kwargs)
            return [
                (
                    Document(
                        page_content="approved",
                        metadata={
                            "approval_status": "approved",
                            "trusted": True,
                            "channel_id": "c1",
                        },
                    ),
                    0.9,
                ),
                (
                    Document(
                        page_content="unapproved",
                        metadata={
                            "approval_status": "unapproved",
                            "trusted": False,
                            "channel_id": "c1",
                        },
                    ),
                    0.95,
                ),
                (
                    Document(
                        page_content="other channel",
                        metadata={
                            "approval_status": "approved",
                            "trusted": True,
                            "channel_id": "c2",
                        },
                    ),
                    0.8,
                ),
            ]

    monkeypatch.setattr(db, "get_vectorstore", lambda: FakeStore())

    results = db.search_similar(
        query="hello",
        k=2,
        filter_dict={
            "approval_status": "approved",
            "trusted": True,
            "channel_id": "c1",
        },
        score_threshold=0.3,
    )

    assert "filter" not in calls[0]
    assert calls[0]["k"] > 2
    assert [item["content"] for item in results] == ["approved"]


def test_knowledge_manager_search_filters_domain_in_python(monkeypatch):
    from knowledge.manager import KnowledgeManager

    calls = []

    class FakeStore:
        def similarity_search_with_relevance_scores(self, **kwargs):
            calls.append(kwargs)
            return [
                (
                    Document(
                        page_content="pz doc",
                        metadata={
                            "domain": "pz",
                            "trusted": True,
                            "content_type": "knowledge_base",
                            "source": "pz/doc.md",
                        },
                    ),
                    0.9,
                ),
                (
                    Document(
                        page_content="rules doc",
                        metadata={
                            "domain": "server_rules",
                            "trusted": True,
                            "content_type": "knowledge_base",
                            "source": "server_rules/doc.md",
                        },
                    ),
                    0.95,
                ),
            ]

    manager = KnowledgeManager()
    monkeypatch.setattr(manager, "_get_vectorstore", lambda: FakeStore())

    results = manager.search(query="book", domain="pz", k=2)

    assert "filter" not in calls[0]
    assert calls[0]["k"] > 2
    assert [item["content"] for item in results] == ["pz doc"]
