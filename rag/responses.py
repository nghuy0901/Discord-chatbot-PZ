import logging
import re
from typing import Any, Optional

from rag.result import RAGDecision


logger = logging.getLogger(__name__)


RESPONSES = {
    ("vi", RAGDecision.ABSTAIN): (
        "Mình chưa có đủ thông tin đã được phê duyệt để trả lời chính xác câu này."
    ),
    ("en", RAGDecision.ABSTAIN): (
        "I do not have enough approved information to answer this accurately."
    ),
    ("vi", RAGDecision.CLARIFY): (
        "Bạn có thể nói rõ vật phẩm, địa điểm hoặc quy định nào bạn đang hỏi không?"
    ),
    ("en", RAGDecision.CLARIFY): (
        "Which item, location, or rule are you referring to?"
    ),
}


def _first_non_empty(*values: Any) -> str:
    for value in values:
        if value is not None and value != "":
            return str(value)
    return ""


def _query_preview(query: Optional[str], max_chars: int = 160) -> str:
    if not query:
        return ""
    preview = re.sub(r"\s+", " ", query).strip()
    if len(preview) > max_chars:
        preview = preview[: max_chars - 1] + "..."
    return preview


def _metric_value(rag_result: Any, name: str) -> Any:
    metric = getattr(rag_result, "metric", None)
    if metric is not None and hasattr(metric, name):
        return getattr(metric, name)
    if rag_result is not None and hasattr(rag_result, name):
        return getattr(rag_result, name)
    return None


def _log_decision_response(
    *,
    language: str,
    decision: RAGDecision,
    event: str,
    query: Optional[str],
    rag_result: Any,
    reason: Optional[str],
    channel_id: Optional[str],
    user_id: Optional[str],
) -> None:
    if decision is not RAGDecision.ABSTAIN:
        return

    resolved_reason = _first_non_empty(
        reason,
        _metric_value(rag_result, "decision_reason"),
        "unspecified",
    )
    evidence_score = _metric_value(rag_result, "evidence_score")
    if isinstance(evidence_score, float):
        evidence_score_text = f"{evidence_score:.3f}"
    else:
        evidence_score_text = _first_non_empty(evidence_score, "unknown")

    logger.warning(
        "RAG abstain response emitted: "
        "event=%s reason=%s language=%s query_intent=%s detected_domain=%s "
        "evidence_score=%s trusted_source_count=%s untrusted_source_count=%s "
        "query_id=%s channel_id=%s user_id=%s query_preview=%s",
        event,
        resolved_reason,
        "vi" if language != "en" else "en",
        _first_non_empty(_metric_value(rag_result, "query_intent"), "unknown"),
        _first_non_empty(
            _metric_value(rag_result, "detected_domain"),
            getattr(rag_result, "primary_domain", None),
            "unknown",
        ),
        evidence_score_text,
        _first_non_empty(_metric_value(rag_result, "trusted_source_count"), "unknown"),
        _first_non_empty(_metric_value(rag_result, "untrusted_source_count"), "unknown"),
        _first_non_empty(_metric_value(rag_result, "query_id"), "unknown"),
        channel_id or "unknown",
        user_id or "unknown",
        _query_preview(query) or "unknown",
    )


def decision_response(
    language: str,
    decision: RAGDecision,
    *,
    event: str = "decision_response",
    query: Optional[str] = None,
    rag_result: Any = None,
    reason: Optional[str] = None,
    channel_id: Optional[str] = None,
    user_id: Optional[str] = None,
) -> str:
    key = ("vi" if language == "vi" else "en", decision)
    _log_decision_response(
        language=language,
        decision=decision,
        event=event,
        query=query,
        rag_result=rag_result,
        reason=reason,
        channel_id=channel_id,
        user_id=user_id,
    )
    return RESPONSES[key]
