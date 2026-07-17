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
import hashlib
import logging
import re
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
# Max chunks sharing the same record/heading kept in a KB result set, so that
# near-identical same-name records (e.g. 21 "Glass Bottle" variants) cannot
# flood the top-k and crowd out diverse facts (audit M1).
RAG_MAX_PER_RECORD: int = int(os.getenv("RAG_MAX_PER_RECORD", "2"))


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
    Create a unique identifier for deduplication / cross-arm merging.

    Prefers stable explicit IDs so the vector copy and the BM25 copy of the same
    document merge into one fused entry: message_id → doc_id → full-content hash.
    The previous 100-char content *prefix* could both collide distinct chunks
    (same boilerplate header) and fail to merge identical ones (audit H2/H4).
    """
    msg_id = result.get("message_id")
    if msg_id:
        return f"msg:{msg_id}"

    doc_id = result.get("doc_id")
    if doc_id:
        return f"doc:{doc_id}"

    # Fallback: full-content fingerprint (hash, not a truncated prefix).
    content = result.get(id_key, "") or ""
    source = result.get("source", "")
    domain = result.get("domain", "")
    digest = hashlib.sha256(str(content).encode("utf-8")).hexdigest()[:16]
    return f"{domain}:{source}:{digest}"


def _ensure_methods(results: List[Dict[str, Any]], method: str) -> List[Dict[str, Any]]:
    """Tag single-arm results with a ``retrieval_methods`` list.

    The RRF path sets ``retrieval_methods``; the vector-only / BM25-only
    short-circuits bypass it, so downstream scoring/provenance would see no
    method tag. Ensure it is always present (audit H2).
    """
    for r in results:
        if not r.get("retrieval_methods"):
            r["retrieval_methods"] = [method]
    return results


def _record_key(result: Dict[str, Any]) -> str:
    if result.get("content_mode") not in {"record_item", "recipe"}:
        return ""
    record = result.get("record_name") or result.get("heading_path")
    return f"{result.get('source', '')}:{record}" if record else str(
        result.get("doc_id") or ""
    )


def _cap_per_record(
    results: List[Dict[str, Any]], limit: int
) -> List[Dict[str, Any]]:
    """Keep at most ``limit`` results per record/heading group (audit M1)."""
    if limit <= 0:
        return results
    seen: Dict[str, int] = {}
    capped: List[Dict[str, Any]] = []
    for r in results:
        key = _record_key(r)
        if not key:
            capped.append(r)
            continue
        if seen.get(key, 0) < limit:
            seen[key] = seen.get(key, 0) + 1
            capped.append(r)
    return capped


def _exact_title_word_count(result: Dict[str, Any], query: str) -> int:
    """Return matched title length so the most specific exact title wins."""
    title = result.get("record_name") or str(result.get("heading_path") or "").split(">")[-1]
    title_words = re.findall(r"[^\W_]+", str(title).casefold())
    query_words = re.findall(r"[^\W_]+", str(query).casefold())
    if not title_words or len(title_words) > len(query_words):
        return 0
    width = len(title_words)
    return width if any(
        query_words[index : index + width] == title_words
        for index in range(len(query_words) - width + 1)
    ) else 0


# ---------------------------------------------------------------------------
# Hybrid Search orchestrator
# ---------------------------------------------------------------------------
async def hybrid_search(
    query: str,
    bm25_query: Optional[str] = None,
    channel_id: Optional[str] = None,
    domains: Optional[List[str]] = None,
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
        domains: Optional allowed knowledge domains for KB search.
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
    from rag.trust import filter_trusted_results, trusted_chat_filter

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
    vector_queries = [query]
    if bm25_query and bm25_query.strip().casefold() != query.strip().casefold():
        vector_queries.append(bm25_query)
    vector_start = time.time()
    try:
        if search_type == "chat":
            from rag.db import search_similar
            filter_dict = trusted_chat_filter(channel_id)
            for vector_query in vector_queries:
                vector_results.extend(
                    search_similar(
                        query=vector_query,
                        k=vector_top_k,
                        filter_dict=filter_dict,
                        score_threshold=vector_threshold,
                    )
                )
        elif search_type == "kb":
            from knowledge.manager import get_knowledge_manager
            from rag.trust import trusted_domains_from

            kb = get_knowledge_manager()
            search_domains = trusted_domains_from(domains)
            for vector_query in vector_queries:
                if search_domains:
                    for domain in search_domains:
                        vector_results.extend(
                            kb.search(
                                query=vector_query,
                                domain=domain,
                                k=vector_top_k,
                                score_threshold=vector_threshold,
                            )
                        )
                else:
                    vector_results.extend(
                        kb.search(
                            query=vector_query,
                            k=vector_top_k,
                            score_threshold=vector_threshold,
                        )
                    )
        vector_results = filter_trusted_results(vector_results)
        best_vector_results = {}
        for result in vector_results:
            doc_id = _make_doc_id(result)
            current = best_vector_results.get(doc_id)
            if current is None or result.get("similarity", 0) > current.get(
                "similarity", 0
            ):
                best_vector_results[doc_id] = result
        vector_results = sorted(
            best_vector_results.values(),
            key=lambda item: item.get("similarity", 0),
            reverse=True,
        )[:vector_top_k]
        for result in vector_results:
            result["retrieval_method"] = "vector"
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
                if search_type == "chat":
                    filter_dict = trusted_chat_filter(channel_id)
                elif domains:
                    search_domains = set(domains)
                    filter_dict = None
                else:
                    search_domains = set()
                    filter_dict = None
                bm25_results = bm25_idx.search(
                    query=bm25_query or query,
                    top_k=bm25_top_k,
                    min_score=bm25_min_score,
                    filter_dict=filter_dict,
                )
                if search_type == "kb" and search_domains:
                    bm25_results = [
                        item for item in bm25_results
                        if item.get("domain") in search_domains
                    ]
                bm25_results = filter_trusted_results(bm25_results)
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
        # No BM25 results → vector only. Tag the method so downstream scoring
        # and provenance still see a retrieval_methods list (audit H2).
        fused = _ensure_methods(list(vector_results), "vector")
    elif not vector_results:
        # No vector results → BM25 only (common for VN queries before H3).
        fused = _ensure_methods(list(bm25_results), "bm25")
    else:
        # Fuse with RRF (merges cross-arm copies by stable id, see _make_doc_id).
        fused = reciprocal_rank_fusion(
            result_lists=[vector_results, bm25_results],
            weights=[VECTOR_WEIGHT, BM25_WEIGHT],
            k=RRF_K,
        )

    # KB diversity: cap near-identical same-record chunks before the final cut
    # so the cap cannot silently shrink the result set below final_top_k (M1).
    if search_type == "kb":
        fused.sort(
            key=lambda result: _exact_title_word_count(
                result, bm25_query or query
            ),
            reverse=True,
        )
        fused = _cap_per_record(fused, RAG_MAX_PER_RECORD)
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
