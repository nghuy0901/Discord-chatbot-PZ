"""
Canonical relevance scoring for retrieval results.

After hybrid fusion a result dict can carry several incompatible score fields:

- ``similarity``  — pgvector cosine relevance, clamped to [0, 1] (vector arm)
- ``bm25_score``  — raw Okapi BM25 score: unbounded, corpus-relative (lexical arm)
- ``rrf_score``   — Reciprocal-Rank-Fusion score: tiny (~1/(k+rank)), rank-based

Historically the codebase did ``r.get("similarity", r.get("rrf_score", 0))`` in
many places, mixing a [0, 1] cosine value with a ~0.01 RRF value as if they were
on the same scale (audit finding H2). That corrupted the Self-RAG skip
heuristic, the evidence/abstention confidence, the citation percentages, and the
monitoring metrics.

``relevance_score`` is the single, well-defined [0, 1] signal every consumer
(evidence policy, Self-RAG, metrics, citations) should use instead.
"""

import os

# Kept in sync with rag.hybrid_retriever.RRF_K; read independently to avoid a
# circular import between scoring and the retriever.
RRF_K: int = int(os.getenv("RRF_K", "60"))

# A purely lexical (BM25-only) match has no semantic corroboration, so we cap its
# relevance below the default ANSWER threshold (0.55): a bare keyword hit should
# trigger CLARIFY, not a confident answer.
_LEXICAL_ONLY_CEILING: float = float(os.getenv("RAG_LEXICAL_ONLY_CEILING", "0.5"))


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def has_semantic_score(result: dict) -> bool:
    """True if the result carries a real vector similarity (not just lexical)."""
    return result.get("similarity") is not None


def relevance_score(result: dict) -> float:
    """Return a single calibrated relevance in [0, 1] for a result dict.

    Priority:
    1. ``similarity`` (vector cosine, already clamped at the db layer).
    2. lexical+semantic agreement (found by both arms) → strong, fixed signal.
    3. lexical-only → conservative score derived from the RRF rank, capped.
    4. otherwise 0.
    """
    sim = result.get("similarity")
    if sim is not None:
        try:
            return clamp01(float(sim))
        except (TypeError, ValueError):
            pass

    methods = {str(m).lower() for m in (result.get("retrieval_methods") or [])}
    if {"vector", "bm25"} <= methods:
        # Lexical + semantic agreement is a strong, scale-independent signal.
        return 0.8

    rrf = result.get("rrf_score")
    if rrf is not None:
        try:
            # rrf for a rank-1 single-arm hit ≈ weight/(k+1); ×(k+1) lifts it back
            # toward the weight, then we cap it to the lexical-only ceiling.
            return clamp01(min(_LEXICAL_ONLY_CEILING, float(rrf) * (RRF_K + 1)))
        except (TypeError, ValueError):
            pass

    return 0.0
