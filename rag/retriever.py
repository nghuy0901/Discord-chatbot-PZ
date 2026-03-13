"""
RAG retriever — Enhanced with query preprocessing (A2), metrics (A3),
similarity threshold + fallback (A4), and knowledge base integration (D2).

Searches both chat history (LangChain PGVector) and static knowledge base,
then merges and formats results for prompt injection.
"""

import os
import time
import logging
from typing import List, Dict, Any, Optional, Tuple

from rag.db import search_similar, get_message_context
from rag.query_preprocessor import get_preprocessor
from rag.metrics import get_metrics_manager, RAGMetric

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
RAG_TOP_K: int = int(os.getenv("RAG_TOP_K", "15"))
RAG_SIMILARITY_THRESHOLD: float = float(os.getenv("RAG_SIMILARITY_THRESHOLD", "0.3"))

# A4: Fallback thresholds
RAG_HIGH_THRESHOLD: float = float(os.getenv("RAG_HIGH_THRESHOLD", "0.5"))
RAG_FALLBACK_THRESHOLD: float = float(os.getenv("RAG_FALLBACK_THRESHOLD", "0.2"))
RAG_FALLBACK_TOP_K: int = int(os.getenv("RAG_FALLBACK_TOP_K", "5"))

# Knowledge base settings
KB_TOP_K: int = int(os.getenv("KB_TOP_K", "5"))
KB_THRESHOLD: float = float(os.getenv("KB_THRESHOLD", "0.35"))


# ---------------------------------------------------------------------------
# Core retrieval with A4 fallback strategy
# ---------------------------------------------------------------------------
async def retrieve(
    query: str,
    channel_id: Optional[str] = None,
    top_k: int = RAG_TOP_K,
    threshold: float = RAG_SIMILARITY_THRESHOLD,
    include_context: bool = True,
) -> List[Dict[str, Any]]:
    """
    Search for relevant historical messages via LangChain PGVector.
    Implements A4 fallback strategy:
    1. Try with normal threshold
    2. If too few results, retry with lower threshold but fewer results
    """
    try:
        filter_dict = {"channel_id": channel_id} if channel_id else None
        results = search_similar(
            query=query,
            k=top_k,
            filter_dict=filter_dict,
            score_threshold=threshold,
        )

        # A4: Fallback strategy — if no results with normal threshold,
        # try with lower threshold but limit to fewer results
        if not results and threshold > RAG_FALLBACK_THRESHOLD:
            logger.debug(
                f"No results at threshold={threshold:.2f}, "
                f"falling back to threshold={RAG_FALLBACK_THRESHOLD:.2f}"
            )
            results = search_similar(
                query=query,
                k=RAG_FALLBACK_TOP_K,
                filter_dict=filter_dict,
                score_threshold=RAG_FALLBACK_THRESHOLD,
            )
            if results:
                logger.debug(f"Fallback found {len(results)} results")

    except Exception as e:
        logger.error(f"RAG search failed: {e}")
        return []

    if include_context and results:
        for result in results:
            msg_id = result.get("message_id")
            if msg_id:
                try:
                    ctx = await get_message_context(msg_id)
                    result["thread_context"] = ctx
                except Exception:
                    result["thread_context"] = []

    return results


# ---------------------------------------------------------------------------
# Prompt formatting
# ---------------------------------------------------------------------------
def format_retrieved_for_prompt(
    results: List[Dict[str, Any]],
    max_chars: int = 3000,
) -> str:
    """Format retrieved messages into a compact block for prompt injection."""
    if not results:
        return ""

    lines: List[str] = ["[Retrieved Chat History — Relevant past conversations]"]
    total_chars = 0

    for i, r in enumerate(results, 1):
        author = r.get("author_name") or r.get("author_id", "unknown")
        ts = r.get("timestamp", "")
        if hasattr(ts, "isoformat"):
            ts = ts.isoformat()
        similarity = r.get("similarity", 0.0)
        content = r.get("content", "")
        msg_id = r.get("message_id", "")

        if len(content) > 500:
            content = content[:497] + "…"

        citation = (
            f"[{i}] ({similarity:.0%} match) @{author} — {ts}\n"
            f"    {content}"
        )

        thread_ctx = r.get("thread_context", [])
        if thread_ctx:
            ctx_lines = []
            for ctx_msg in thread_ctx[:3]:
                ctx_author = ctx_msg.get("author_name") or ctx_msg.get("author_id", "?")
                ctx_content = ctx_msg.get("content", "")[:200]
                edge = ctx_msg.get("edge_type", "related")
                ctx_lines.append(f"    ↳ [{edge}] @{ctx_author}: {ctx_content}")
            citation += "\n" + "\n".join(ctx_lines)

        if total_chars + len(citation) > max_chars:
            lines.append(f"... ({len(results) - i + 1} more results omitted for brevity)")
            break

        lines.append(citation)
        total_chars += len(citation)

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Enhanced RAG context builder with all features
# ---------------------------------------------------------------------------
async def build_rag_context(
    query: str,
    recent_messages: Optional[List[str]] = None,
    channel_id: Optional[str] = None,
    channel_name: Optional[str] = None,
    user_id: Optional[str] = None,
    top_k: int = RAG_TOP_K,
) -> Tuple[str, Optional[str], RAGMetric]:
    """
    High-level helper: build the RAG context block for prompt construction.

    Enhanced with:
    - A2: Query preprocessing (clean Discord formatting, expand abbrevs)
    - A3: Metrics tracking
    - A4: Similarity threshold fallback
    - D2: Knowledge base integration

    Args:
        query: Current user message.
        recent_messages: Recent message strings for context enrichment.
        channel_id: Optional channel scope.
        channel_name: Optional channel name for domain detection.
        user_id: Optional user ID for metrics.
        top_k: Max results.

    Returns:
        Tuple of (formatted_context_string, domain_prompt, metric)
    """
    metric = RAGMetric(
        channel_id=channel_id or "",
        user_id=user_id or "",
        original_query=query[:500],
    )

    start_time = time.time()

    # ---- A2: Preprocess query ----
    preprocessor = get_preprocessor()
    processed_query, query_meta = preprocessor.preprocess(query, channel_name)
    metric.processed_query = processed_query[:500]
    metric.query_language = query_meta.get("language", "unknown")
    metric.detected_domain = query_meta.get("domain")

    # Enrich query with recent context
    search_text = processed_query
    if recent_messages:
        context_snippet = " | ".join(recent_messages[-5:])
        search_text = f"{processed_query} [context: {context_snippet}]"

    retrieval_start = time.time()

    # ---- D2: Search knowledge base ----
    kb_context = ""
    domain_prompt = None
    try:
        from knowledge.domain_router import get_domain_router, format_kb_results_for_prompt

        router = get_domain_router()
        domains_to_search = router.route(
            processed_query,
            detected_domain=metric.detected_domain,
            channel_name=channel_name,
        )

        if domains_to_search:
            kb_results, primary_domain = router.search_knowledge(
                query=processed_query,
                domains=domains_to_search,
                top_k=KB_TOP_K,
                threshold=KB_THRESHOLD,
            )
            metric.kb_results = len(kb_results)
            kb_context = format_kb_results_for_prompt(kb_results)

            if primary_domain:
                domain_prompt = router.get_domain_prompt(primary_domain)
                metric.detected_domain = primary_domain

    except ImportError:
        logger.debug("Knowledge base module not available")
    except Exception as e:
        logger.warning(f"Knowledge base search failed (non-fatal): {e}")

    # ---- Chat history retrieval (with A4 fallback) ----
    chat_results = await retrieve(
        query=search_text,
        channel_id=channel_id,
        top_k=top_k,
    )
    metric.chat_history_results = len(chat_results)

    retrieval_end = time.time()
    metric.retrieval_time_ms = (retrieval_end - retrieval_start) * 1000

    # ---- Compute similarity stats for A3 ----
    all_similarities = [
        r.get("similarity", 0) for r in chat_results if r.get("similarity")
    ]
    if all_similarities:
        metric.avg_similarity = sum(all_similarities) / len(all_similarities)
        metric.max_similarity = max(all_similarities)
        metric.min_similarity = min(all_similarities)
    metric.num_results = len(chat_results) + metric.kb_results

    # ---- Format combined context ----
    chat_context = format_retrieved_for_prompt(chat_results)
    combined_parts = []
    if kb_context:
        combined_parts.append(kb_context)
    if chat_context:
        combined_parts.append(chat_context)

    combined_context = "\n\n".join(combined_parts)

    # ---- A3: Record metric ----
    total_time = (time.time() - start_time) * 1000
    metric.retrieval_time_ms = total_time
    metrics_manager = get_metrics_manager()
    await metrics_manager.record(metric)

    return combined_context, domain_prompt, metric
