import pytest

from rag.result import RAGDecision
from tests.rag.test_query_preprocessor import CONVERSATION_QUERIES


class DummyMetrics:
    async def record(self, metric):
        self.metric = metric


@pytest.mark.asyncio
@pytest.mark.parametrize("query", CONVERSATION_QUERIES)
async def test_conversation_intent_bypasses_evidence_requirement(monkeypatch, query):
    import rag.retriever as retriever

    async def fail_if_retrieved(*args, **kwargs):
        raise AssertionError("conversation turns should not hit retrieval")

    monkeypatch.setattr(retriever, "retrieve", fail_if_retrieved)
    monkeypatch.setattr(retriever, "retrieve_knowledge", fail_if_retrieved)
    monkeypatch.setattr(retriever, "get_metrics_manager", lambda: DummyMetrics())

    result = await retriever.build_rag_result(
        query=query,
        recent_messages=[],
        channel_id="chan",
        channel_name=None,
        user_id="user",
    )

    assert result.decision is RAGDecision.ANSWER
    assert result.decision_reason == "conversation_no_rag_required"
    assert result.context == ""
    assert result.retrieved_results == []
