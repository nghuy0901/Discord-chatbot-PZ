import logging
from types import SimpleNamespace

from rag.responses import decision_response
from rag.result import RAGDecision


def test_abstain_decision_logs_reason_and_context(caplog):
    rag_result = SimpleNamespace(
        decision=RAGDecision.ABSTAIN,
        decision_reason="no_trusted_evidence",
        evidence_score=0.0,
        primary_domain="pz",
        metric=SimpleNamespace(
            query_intent="hybrid",
            query_language="vi",
            detected_domain="pz",
            decision_reason="no_trusted_evidence",
            evidence_score=0.0,
            trusted_source_count=0,
            untrusted_source_count=3,
            query_id="qid-123",
        ),
    )

    with caplog.at_level(logging.WARNING, logger="rag.responses"):
        response = decision_response(
            "vi",
            RAGDecision.ABSTAIN,
            event="discord_pre_generation_decision",
            query="nomnom câu này sao không trả lời được?",
            rag_result=rag_result,
            channel_id="chan-1",
            user_id="user-1",
        )

    assert "Mình chưa có đủ thông tin" in response
    assert "RAG abstain response emitted" in caplog.text
    assert "event=discord_pre_generation_decision" in caplog.text
    assert "reason=no_trusted_evidence" in caplog.text
    assert "query_intent=hybrid" in caplog.text
    assert "detected_domain=pz" in caplog.text
    assert "trusted_source_count=0" in caplog.text
    assert "query_preview=nomnom câu này sao không trả lời được?" in caplog.text
