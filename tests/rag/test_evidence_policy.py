from rag.evidence import EvidencePolicy
from rag.result import RAGDecision


def trusted_chunk(**overrides):
    base = {
        "source_id": "pz/Weapons/Axes_Weapon.md#0",
        "source_kind": "knowledge_base",
        "domain": "pz",
        "trusted": True,
        "similarity": 0.72,
        "bm25_score": 1.1,
        "retrieval_methods": ["vector", "bm25"],
        "content_type": "knowledge_base",
    }
    base.update(overrides)
    return base


def test_answers_only_with_strong_trusted_evidence():
    assessment = EvidencePolicy().assess(
        query="Rìu dùng để làm gì?",
        results=[trusted_chunk()],
        recent_messages=[],
    )
    assert assessment.decision is RAGDecision.ANSWER


def test_abstains_without_trusted_evidence():
    assessment = EvidencePolicy().assess(
        query="Rìu dùng để làm gì?",
        results=[],
        recent_messages=[],
    )
    assert assessment.decision is RAGDecision.ABSTAIN


def test_clarifies_referential_query_without_history():
    assessment = EvidencePolicy().assess(
        query="Cái đó cần bao nhiêu?",
        results=[trusted_chunk()],
        recent_messages=[],
    )
    assert assessment.decision is RAGDecision.CLARIFY


def test_unapproved_source_never_enables_answer():
    assessment = EvidencePolicy().assess(
        query="Luật generator là gì?",
        results=[trusted_chunk(trusted=False, approval_status="unapproved")],
        recent_messages=[],
    )
    assert assessment.decision is RAGDecision.ABSTAIN
