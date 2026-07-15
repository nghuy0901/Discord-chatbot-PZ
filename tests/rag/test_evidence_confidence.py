"""Calibrated-confidence behaviour of EvidencePolicy."""

from rag.evidence import EvidencePolicy
from rag.result import RAGDecision


def chunk(**overrides):
    base = {
        "source_id": "pz/Weapons/Axes_Weapon.md#0",
        "source_kind": "knowledge_base",
        "content_type": "knowledge_base",
        "domain": "pz",
        "trusted": True,
        "similarity": 0.45,
        "retrieval_methods": ["vector"],
        "content": "Axe minimum damage maximum damage tree damage",
        "source": "pz/Weapons/Axes_Weapon.md",
    }
    base.update(overrides)
    return base


def test_strong_agreement_answers():
    a = EvidencePolicy().assess(
        query="Rìu dùng để làm gì?",
        results=[chunk(similarity=0.72, retrieval_methods=["vector", "bm25"])],
        recent_messages=[],
    )
    assert a.decision is RAGDecision.ANSWER
    assert a.score >= 0.55


def test_weak_single_source_abstains():
    a = EvidencePolicy().assess(
        query="Rìu dùng để làm gì?",
        results=[chunk(similarity=0.28, retrieval_methods=["vector"])],
        recent_messages=[],
    )
    assert a.decision is RAGDecision.ABSTAIN


def test_borderline_single_source_clarifies():
    a = EvidencePolicy().assess(
        query="Rìu dùng để làm gì?",
        results=[chunk(similarity=0.45, retrieval_methods=["vector"])],
        recent_messages=[],
    )
    assert a.decision is RAGDecision.CLARIFY


def test_analytical_intent_accepts_strong_single_source():
    source = chunk(similarity=0.60, retrieval_methods=["vector"])
    base = EvidencePolicy().assess(
        query="So sánh rìu và búa",
        results=[source],
        recent_messages=[],
        query_intent=None,
    )
    strict = EvidencePolicy().assess(
        query="So sánh rìu và búa",
        results=[source],
        recent_messages=[],
        query_intent="analytical",
    )
    assert base.decision is RAGDecision.ANSWER
    assert strict.decision is RAGDecision.ANSWER


def test_corroboration_raises_confidence():
    one = EvidencePolicy().assess(
        query="q",
        results=[chunk(similarity=0.45)],
        recent_messages=[],
    )
    many = EvidencePolicy().assess(
        query="q",
        results=[
            chunk(similarity=0.45, source=f"pz/source-{index}.md")
            for index in range(4)
        ],
        recent_messages=[],
    )
    assert many.score > one.score


def test_duplicate_chunks_from_same_source_do_not_corroborate():
    one = EvidencePolicy().assess(
        query="axe damage",
        results=[chunk(similarity=0.45)],
        recent_messages=[],
    )
    duplicates = EvidencePolicy().assess(
        query="axe damage",
        results=[chunk(similarity=0.45, chunk_index=index) for index in range(4)],
        recent_messages=[],
    )

    assert duplicates.score == one.score


def test_vector_bm25_agreement_without_query_term_coverage_does_not_answer():
    result = chunk(
        similarity=0.52,
        retrieval_methods=["vector", "bm25"],
        content="Fire weather electricity lore navigation",
    )

    assessment = EvidencePolicy().assess(
        query="rìu sát thương axe damage",
        results=[result],
        recent_messages=[],
    )

    assert assessment.decision is not RAGDecision.ANSWER


def test_untrusted_never_answers():
    a = EvidencePolicy().assess(
        query="q",
        results=[chunk(trusted=False, approval_status="unapproved", similarity=0.95)],
        recent_messages=[],
    )
    assert a.decision is RAGDecision.ABSTAIN
