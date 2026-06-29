from rag.result import RAGDecision


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


def decision_response(language: str, decision: RAGDecision) -> str:
    key = ("vi" if language == "vi" else "en", decision)
    return RESPONSES[key]
