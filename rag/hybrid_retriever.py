"""
Hybrid Retriever — Combines BM25 lexical search + Vector semantic search
using Reciprocal Rank Fusion (RRF).

Why Hybrid RAG?
- **Vector search** excels at semantic similarity (understanding meaning)
- **BM25 search** excels at exact keyword matching (item names, commands, stats)
- Together they provide superior retrieval quality over either method alone

Fusion Strategy: Reciprocal Rank Fusion (RRF)
    RRF_score(d) = Σ 1 / (k + rank_i(d))
    where k is a constant (default=60) and rank_i(d) is the rank of document d
    in the i-th result list.

RRF is preferred over linear combination because:
- It is rank-based, not score-based, so no normalization needed
- Works well when scores from different systems have different scales
- Simple, effective, and widely used in production search systems
"""

import os
import logging
from typing import List, Dict, Any, Optional, Set, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
HYBRID_ENABLED: bool = os.getenv("HYBRID_RAG_ENABLED", "true").lower() == "true"
RRF_K: int = int(os.getenv("RRF_K", "60"))  # RRF constant, higher = less aggressive
VECTOR_WEIGHT: float = float(os.getenv("VECTOR_WEIGHT", "0.6"))  # relative weight for vector
BM25_WEIGHT: float = float(os.getenv("BM25_WEIGHT", "0.4"))  # relative weight for BM25
HYBRID_TOP_K: int = int(os.getenv("HYBRID_TOP_K", "15"))


# ---------------------------------------------------------------------------
# Reciprocal Rank Fusion
# ---------------------------------------------------------------------------
def reciprocal_rank_fusion(
    result_lists: List[List[Dict[str, Any]]],
    weights: Optional[List[float]] = None,
    k: int = RRF_K,
    id_key: str = "content",
) -> List[Dict[str, Any]]:
    """
    Merge multiple ranked result lists using Reciprocal Rank Fusion (RRF).

    Args:
        result_lists: List of ranked result lists. Each result is a dict.
        weights: Optional per-list weight multipliers. Default is equal weight.
        k: RRF constant (higher = less sensitive to rank differences).
        id_key: Key used to identify unique documents for deduplication.

    Returns:
        A single merged and re-ranked list of results with RRF scores.
    """
    if not result_lists:
        return []

    if weights is None:
        weights = [1.0] * len(result_lists)

    # Document identity → aggregated RRF score + best result dict
    rrf_scores: Dict[str, float] = {}
    best_results: Dict[str, Dict[str, Any]] = {}

    for list_idx, results in enumerate(result_lists):
        weight = weights[list_idx] if list_idx < len(weights) else 1.0

        for rank, result in enumerate(results):
            # Create a content fingerprint for deduplication
            doc_id = _make_doc_id(result, id_key)

            # RRF score contribution
            rrf_contribution = weight / (k + rank + 1)
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + rrf_contribution

            # Keep the result dict with most metadata
            if doc_id not in best_results:
                best_results[doc_id] = result.copy()
            else:
                # Merge metadata from both sources
                existing = best_results[doc_id]
                for key, value in result.items():
                    if key not in existing or existing[key] is None:
                        existing[key] = value

    # Build final results sorted by RRF score
    fused = []
    for doc_id, score in sorted(rrf_scores.items(), key=lambda x: -x[1]):
        result = best_results[doc_id].copy()
        result["rrf_score"] = round(score, 6)

        # Track which methods found this document
        methods = set()
        if result.get("similarity") is not None:
            methods.add("vector")
        if result.get("bm25_score") is not None:
            methods.add("bm25")
        result["retrieval_methods"] = list(methods) if methods else ["unknown"]

        fused.append(result)

    return fused


def _make_doc_id(result: Dict[str, Any], id_key: str = "content") -> str:
    """
    Create a unique identifier for deduplication.
    Uses message_id if available, otherwise falls back to content hash.
    """
    # Prefer explicit IDs
    msg_id = result.get("message_id")
    if msg_id:
        return f"msg:{msg_id}"

    # Fallback: content-based fingerprint
    content = result.get(id_key, "")
    source = result.get("source", "")
    domain = result.get("domain", "")

    # Use first 100 chars of content + source for ID
    fingerprint = f"{domain}:{source}:{content[:100]}"
    return fingerprint


# ---------------------------------------------------------------------------
# Hybrid Search orchestrator
# ---------------------------------------------------------------------------
async def hybrid_search(
    query: str,
    channel_id: Optional[str] = None,
    vector_top_k: int = 15,
    bm25_top_k: int = 15,
    final_top_k: int = HYBRID_TOP_K,
    vector_threshold: float = 0.3,
    bm25_min_score: float = 0.5,
    search_type: str = "chat",  # "chat" or "kb"
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Execute a hybrid search combining vector and BM25 results.

    Args:
        query: Search query text.
        channel_id: Optional channel filter (for chat history).
        vector_top_k: Max results from vector search.
        bm25_top_k: Max results from BM25 search.
        final_top_k: Max final fused results.
        vector_threshold: Min similarity for vector results.
        bm25_min_score: Min BM25 score.
        search_type: "chat" for chat history, "kb" for knowledge base.

    Returns:
        Tuple of (fused_results, search_metadata).
    """
    import time

    metadata = {
        "search_type": search_type,
        "hybrid_enabled": HYBRID_ENABLED,
        "vector_results": 0,
        "bm25_results": 0,
        "fused_results": 0,
        "vector_time_ms": 0,
        "bm25_time_ms": 0,
        "fusion_time_ms": 0,
    }

    # ---- 1. Vector (semantic) search ----
    vector_results = []
    vector_start = time.time()
    try:
        if search_type == "chat":
            from rag.db import search_similar
            filter_dict = {"channel_id": channel_id} if channel_id else None
            vector_results = search_similar(
                query=query,
                k=vector_top_k,
                filter_dict=filter_dict,
                score_threshold=vector_threshold,
            )
            # Tag results
            for r in vector_results:
                r["retrieval_method"] = "vector"
        elif search_type == "kb":
            from knowledge.manager import get_knowledge_manager
            kb = get_knowledge_manager()
            vector_results = kb.search(
                query=query,
                k=vector_top_k,
                score_threshold=vector_threshold,
            )
            for r in vector_results:
                r["retrieval_method"] = "vector"
    except Exception as e:
        logger.warning(f"Hybrid: Vector search failed ({search_type}): {e}")

    metadata["vector_time_ms"] = (time.time() - vector_start) * 1000
    metadata["vector_results"] = len(vector_results)

    # ---- 2. BM25 (lexical) search ----
    bm25_results = []
    bm25_start = time.time()

    if HYBRID_ENABLED:
        try:
            from rag.bm25_search import get_chat_bm25, get_kb_bm25

            if search_type == "chat":
                bm25_idx = get_chat_bm25()
            else:
                bm25_idx = get_kb_bm25()

            if bm25_idx.is_ready:
                filter_dict = {"channel_id": channel_id} if channel_id and search_type == "chat" else None
                bm25_results = bm25_idx.search(
                    query=query,
                    top_k=bm25_top_k,
                    min_score=bm25_min_score,
                    filter_dict=filter_dict,
                )
            else:
                logger.debug(
                    f"Hybrid: BM25 index not ready ({search_type}), "
                    "using vector-only"
                )
        except Exception as e:
            logger.warning(f"Hybrid: BM25 search failed ({search_type}): {e}")

    metadata["bm25_time_ms"] = (time.time() - bm25_start) * 1000
    metadata["bm25_results"] = len(bm25_results)

    # ---- 3. Fuse results ----
    fusion_start = time.time()

    if not HYBRID_ENABLED or not bm25_results:
        # No BM25 results → just return vector results
        fused = vector_results[:final_top_k]
    elif not vector_results:
        # No vector results → just return BM25 results
        fused = bm25_results[:final_top_k]
    else:
        # Fuse with RRF
        fused = reciprocal_rank_fusion(
            result_lists=[vector_results, bm25_results],
            weights=[VECTOR_WEIGHT, BM25_WEIGHT],
            k=RRF_K,
        )
        fused = fused[:final_top_k]

    metadata["fusion_time_ms"] = (time.time() - fusion_start) * 1000
    metadata["fused_results"] = len(fused)

    logger.debug(
        f"Hybrid search ({search_type}): "
        f"vector={metadata['vector_results']}, "
        f"bm25={metadata['bm25_results']}, "
        f"fused={metadata['fused_results']}"
    )

    return fused, metadata
