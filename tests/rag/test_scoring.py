"""Canonical relevance scoring (audit H2)."""

from rag.scoring import relevance_score, has_semantic_score, clamp01


def test_uses_vector_similarity_when_present():
    assert relevance_score({"similarity": 0.72}) == 0.72


def test_clamps_out_of_range_similarity():
    # pgvector relevance can exceed [0,1] or go negative; scoring must clamp.
    assert relevance_score({"similarity": 1.5}) == 1.0
    assert relevance_score({"similarity": -0.3}) == 0.0


def test_agreement_when_both_methods_and_no_similarity():
    assert relevance_score({"retrieval_methods": ["vector", "bm25"]}) == 0.8


def test_lexical_only_is_capped_below_answer_threshold():
    # A BM25-only rank-1 hit should land in CLARIFY territory, not a confident
    # ANSWER (default answer threshold is 0.55).
    score = relevance_score({"retrieval_methods": ["bm25"], "rrf_score": 0.4 / 61})
    assert 0.0 < score <= 0.5


def test_unknown_result_is_zero():
    assert relevance_score({}) == 0.0


def test_explicit_similarity_takes_priority_over_methods():
    assert (
        relevance_score(
            {"similarity": 0.3, "retrieval_methods": ["vector", "bm25"]}
        )
        == 0.3
    )


def test_has_semantic_score():
    assert has_semantic_score({"similarity": 0.1})
    assert not has_semantic_score({"bm25_score": 5.0})


def test_clamp01():
    assert clamp01(2.0) == 1.0
    assert clamp01(-1.0) == 0.0
    assert clamp01(0.5) == 0.5
