"""
RAG retriever — Enhanced with:
- A2: Query preprocessing (clean Discord formatting, expand abbrevs)
- A3: Metrics tracking
- A4: Similarity threshold + fallback
- D2: Knowledge base integration
- 🆕 E1: Hybrid RAG (BM25 lexical + Vector semantic via RRF)
- 🆕 E2: Self-RAG (LLM-based relevance grading)

Pipeline:
    User Query
        → A2: Preprocess (clean, expand, detect language/domain)
        → D2: Knowledge Base search (hybrid: vector + BM25)
        → A4: Chat History search (hybrid: vector + BM25)
        → E1: Reciprocal Rank Fusion (merge BM25 + vector)
        → E2: Self-RAG relevance grading (filter irrelevant results)
        → Format for prompt injection
        → A3: Record metrics
"""

import os
import time
import logging
import re
from typing import List, Dict, Any, Optional, Tuple

from rag.db import search_similar, get_message_context
from rag.query_preprocessor import get_preprocessor
from rag.metrics import get_metrics_manager, RAGMetric
from rag.trust import trusted_chat_filter
from rag.evidence import EvidencePolicy
from rag.result import RAGBuildResult, RAGDecision, ProvenanceItem
from rag.citations import build_citation_context
from src.observability.prompts import current_prompt_cache_version

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
RAG_TOP_K: int = int(os.getenv("RAG_TOP_K", "15"))
RAG_SIMILARITY_THRESHOLD: float = float(os.getenv("RAG_SIMILARITY_THRESHOLD", "0.3"))

# Candidate pool settings. `RAG_TOP_K` / `KB_TOP_K` remain the final caps after
# Self-RAG relevance filtering; these values control the wider pre-rerank pool.
RAG_VECTOR_TOP_K: int = int(os.getenv("RAG_VECTOR_TOP_K", str(RAG_TOP_K)))
RAG_BM25_TOP_K: int = int(os.getenv("RAG_BM25_TOP_K", str(RAG_TOP_K)))
RAG_FUSED_TOP_K: int = int(os.getenv("RAG_FUSED_TOP_K", str(RAG_TOP_K)))

# A4: Fallback thresholds
RAG_HIGH_THRESHOLD: float = float(os.getenv("RAG_HIGH_THRESHOLD", "0.5"))
RAG_FALLBACK_THRESHOLD: float = float(os.getenv("RAG_FALLBACK_THRESHOLD", "0.2"))
RAG_FALLBACK_TOP_K: int = int(os.getenv("RAG_FALLBACK_TOP_K", "5"))

# Knowledge base settings
KB_TOP_K: int = int(os.getenv("KB_TOP_K", "5"))
KB_THRESHOLD: float = float(os.getenv("KB_THRESHOLD", "0.35"))
KB_VECTOR_TOP_K: int = int(os.getenv("KB_VECTOR_TOP_K", str(KB_TOP_K)))
KB_BM25_TOP_K: int = int(os.getenv("KB_BM25_TOP_K", str(KB_TOP_K)))
KB_FUSED_TOP_K: int = int(os.getenv("KB_FUSED_TOP_K", str(KB_TOP_K)))

# E1: Hybrid RAG settings
HYBRID_ENABLED: bool = os.getenv("HYBRID_RAG_ENABLED", "true").lower() == "true"

# E2: Self-RAG settings
SELF_RAG_ENABLED: bool = os.getenv("SELF_RAG_ENABLED", "true").lower() == "true"

UNSAFE_CONTEXT_PATTERNS = [
    "ignore previous instructions",
    "reveal the system prompt",
    "developer message",
    "system message",
]


def sanitize_retrieved_context(text: str) -> str:
    sanitized = text
    for pattern in UNSAFE_CONTEXT_PATTERNS:
        sanitized = re.sub(
            re.escape(pattern),
            "[removed unsafe instruction]",
            sanitized,
            flags=re.IGNORECASE,
        )
    return sanitized


def _sanitize_result_content(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    sanitized_results = []
    for result in results:
        cleaned = result.copy()
        cleaned["content"] = sanitize_retrieved_context(str(cleaned.get("content", "")))
        sanitized_results.append(cleaned)
    return sanitized_results


def _record_hybrid_metadata(
    metric: Optional[RAGMetric],
    search_meta: Dict[str, Any],
) -> None:
    if not metric:
        return
    metric.vector_results += int(search_meta.get("vector_results", 0) or 0)
    metric.bm25_results += int(search_meta.get("bm25_results", 0) or 0)
    metric.hybrid_fused_results += int(search_meta.get("fused_results", 0) or 0)
    metric.vector_time_ms += float(search_meta.get("vector_time_ms", 0.0) or 0.0)
    metric.bm25_time_ms += float(search_meta.get("bm25_time_ms", 0.0) or 0.0)


def _record_self_rag_metadata(metric: RAGMetric, rag_meta: Any) -> None:
    metric.self_rag_enabled = True
    metric.self_rag_graded += int(getattr(rag_meta, "total_graded", 0) or 0)
    metric.self_rag_relevant += int(getattr(rag_meta, "relevant_count", 0) or 0)
    metric.self_rag_relevant += int(getattr(rag_meta, "partial_count", 0) or 0)
    metric.self_rag_irrelevant += int(getattr(rag_meta, "irrelevant_count", 0) or 0)
    metric.self_rag_time_ms += float(getattr(rag_meta, "grading_time_ms", 0.0) or 0.0)
    error = getattr(rag_meta, "error", None)
    if error:
        metric.llm_error = error


def _citation_coverage(results: List[Dict[str, Any]]) -> float:
    if not results:
        return 0.0
    cited = 0
    for result in results:
        if (
            result.get("source")
            or result.get("message_id")
            or result.get("author_name")
            or result.get("author_id")
        ):
            cited += 1
    return cited / len(results)


def _source_id(result: Dict[str, Any]) -> str:
    if result.get("doc_id"):
        return str(result["doc_id"])
    if result.get("message_id"):
        return str(result["message_id"])
    source = result.get("source", "unknown")
    chunk_index = result.get("chunk_index", 0)
    return f"{source}#{chunk_index}"


def _build_provenance(results: List[Dict[str, Any]]) -> List[ProvenanceItem]:
    provenance = []
    for rank, result in enumerate(results, start=1):
        similarity = float(
            result.get("similarity", result.get("rrf_score", 0.0)) or 0.0
        )
        provenance.append(
            ProvenanceItem(
                source_id=_source_id(result),
                source_kind=str(result.get("source_kind") or result.get("content_type") or ""),
                domain=str(result.get("domain") or ""),
                trusted=bool(result.get("trusted")),
                rank=rank,
                similarity=similarity,
            )
        )
    return provenance


# ---------------------------------------------------------------------------
# Core retrieval with A4 fallback + E1 hybrid search
# ---------------------------------------------------------------------------
async def retrieve(
    query: str,
    channel_id: Optional[str] = None,
    top_k: int = RAG_TOP_K,
    threshold: float = RAG_SIMILARITY_THRESHOLD,
    include_context: bool = True,
    metric: Optional[RAGMetric] = None,
) -> List[Dict[str, Any]]:
    """
    Search for relevant historical messages via Hybrid RAG (BM25 + Vector).
    Falls back to vector-only if hybrid is disabled or BM25 index not ready.

    Implements A4 fallback strategy:
    1. Try with normal threshold
    2. If too few results, retry with lower threshold but fewer results
    """
    try:
        if HYBRID_ENABLED:
            # E1: Use hybrid search (vector + BM25 via RRF)
            from rag.hybrid_retriever import hybrid_search
            results, search_meta = await hybrid_search(
                query=query,
                channel_id=channel_id,
                vector_top_k=RAG_VECTOR_TOP_K,
                bm25_top_k=RAG_BM25_TOP_K,
                final_top_k=RAG_FUSED_TOP_K,
                vector_threshold=threshold,
                search_type="chat",
            )
            _record_hybrid_metadata(metric, search_meta)
            logger.debug(
                f"Hybrid chat search: vector={search_meta['vector_results']}, "
                f"bm25={search_meta['bm25_results']}, fused={search_meta['fused_results']}"
            )
        else:
            # Fallback: vector-only search
            filter_dict = trusted_chat_filter(channel_id)
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
            if HYBRID_ENABLED:
                from rag.hybrid_retriever import hybrid_search
                results, fallback_meta = await hybrid_search(
                    query=query,
                    channel_id=channel_id,
                    vector_top_k=RAG_FALLBACK_TOP_K,
                    bm25_top_k=RAG_FALLBACK_TOP_K,
                    final_top_k=RAG_FALLBACK_TOP_K,
                    vector_threshold=RAG_FALLBACK_THRESHOLD,
                    search_type="chat",
                )
                _record_hybrid_metadata(metric, fallback_meta)
            else:
                filter_dict = trusted_chat_filter(channel_id)
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
        if metric:
            metric.retrieval_error = str(e)
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
# Knowledge base retrieval with hybrid search
# ---------------------------------------------------------------------------
async def retrieve_knowledge(
    query: str,
    domains: Optional[List[str]] = None,
    top_k: int = KB_TOP_K,
    threshold: float = KB_THRESHOLD,
    metric: Optional[RAGMetric] = None,
) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    """
    Search knowledge base using hybrid search (vector + BM25).
    Returns (results, primary_domain).
    """
    try:
        from knowledge.domain_router import get_domain_router

        router = get_domain_router()

        if HYBRID_ENABLED:
            # Hybrid KB search
            from rag.hybrid_retriever import hybrid_search

            kb_results, kb_meta = await hybrid_search(
                query=query,
                domains=domains,
                vector_top_k=KB_VECTOR_TOP_K,
                bm25_top_k=KB_BM25_TOP_K,
                final_top_k=KB_FUSED_TOP_K,
                vector_threshold=threshold,
                search_type="kb",
            )
            _record_hybrid_metadata(metric, kb_meta)

            primary_domain = None
            if kb_results:
                primary_domain = kb_results[0].get("domain")

            return kb_results, primary_domain
        else:
            # Fallback: standard domain router search
            if domains is None:
                domains_to_search = router.route(query)
            else:
                domains_to_search = domains

            if domains_to_search:
                return router.search_knowledge(
                    query=query,
                    domains=domains_to_search,
                    top_k=top_k,
                    threshold=threshold,
                )
            return [], None

    except ImportError:
        logger.debug("Knowledge base module not available")
        return [], None
    except Exception as e:
        logger.warning(f"Knowledge base search failed: {e}")
        if metric:
            metric.retrieval_error = str(e)
        return [], None


# ---------------------------------------------------------------------------
# Prompt formatting (enhanced with retrieval method tags)
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
        similarity = r.get("similarity", r.get("rrf_score", 0.0))
        content = sanitize_retrieved_context(r.get("content", ""))

        if len(content) > 500:
            content = content[:497] + "…"

        # E1: Show retrieval method for transparency
        methods = r.get("retrieval_methods", [])
        method_tag = ""
        if methods:
            method_tag = f" [{'+'.join(methods)}]"

        # E2: Show Self-RAG grade if available
        rag_grade = ""
        self_rag_rel = r.get("self_rag_relevance")
        if self_rag_rel:
            self_rag_score = r.get("self_rag_score", 0)
            rag_grade = f" (graded: {self_rag_rel} {self_rag_score:.0%})"

        citation = (
            f"[{i}] ({similarity:.0%} match{method_tag}{rag_grade}) @{author} — {ts}\n"
            f"    {content}"
        )

        thread_ctx = r.get("thread_context", [])
        if thread_ctx:
            ctx_lines = []
            for ctx_msg in thread_ctx[:3]:
                ctx_content = sanitize_retrieved_context(str(ctx_msg.get("content", "")))[:200]
                if not ctx_content.strip():
                    # message_edges rows carry only ids/edge_type — no content.
                    # Skip empty lines instead of emitting "↳ [reply] @?:" noise
                    # into the prompt (audit M5).
                    continue
                ctx_author = ctx_msg.get("author_name") or ctx_msg.get("author_id", "?")
                edge = ctx_msg.get("edge_type", "related")
                ctx_lines.append(f"    ↳ [{edge}] @{ctx_author}: {ctx_content}")
            if ctx_lines:
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
async def build_rag_result(
    query: str,
    recent_messages: Optional[List[str]] = None,
    channel_id: Optional[str] = None,
    channel_name: Optional[str] = None,
    user_id: Optional[str] = None,
    request_context: Optional[Any] = None,
    top_k: int = RAG_TOP_K,
) -> RAGBuildResult:
    """
    High-level helper: build the RAG context block for prompt construction.

    Enhanced pipeline:
    - A2: Query preprocessing (clean Discord formatting, expand abbrevs)
    - A3: Metrics tracking
    - A4: Similarity threshold fallback
    - D2: Knowledge base integration
    - 🆕 E1: Hybrid RAG (BM25 + Vector via Reciprocal Rank Fusion)
    - 🆕 E2: Self-RAG (LLM relevance grading + filtering)

    Args:
        query: Current user message.
        recent_messages: Recent message strings for context enrichment.
        channel_id: Optional channel scope.
        channel_name: Optional channel name for domain detection.
        user_id: Optional user ID for metrics.
        top_k: Max results.

    Returns:
        Structured RAG build result with decision, context, metric, and provenance.
    """
    metric_kwargs = {
        "channel_id": channel_id or "",
        "user_id": user_id or "",
        "request_id": request_context.request_id if request_context else "",
        "source": request_context.source if request_context else "",
        "original_query": query[:500],
        "llm_model": os.getenv("LLM_MODEL", ""),
        "embedding_model": os.getenv("EMBEDDING_MODEL", ""),
        "prompt_version": current_prompt_cache_version(),
        "retrieval_config_version": os.getenv("RETRIEVAL_CONFIG_VERSION", "v1"),
    }
    if request_context:
        metric_kwargs["query_id"] = request_context.query_id
    metric = RAGMetric(**metric_kwargs)

    start_time = time.time()

    # ---- A2: Preprocess query ----
    preprocessor = get_preprocessor()
    processed_query, query_meta = preprocessor.preprocess(query, channel_name)
    metric.processed_query = processed_query[:500]
    metric.query_language = query_meta.get("language", "unknown")
    metric.detected_domain = query_meta.get("domain")

    # Phase 4: Capture query intent for routing
    query_intent = query_meta.get("query_intent")
    if query_intent:
        metric.query_intent = query_intent.value if hasattr(query_intent, 'value') else str(query_intent)

    # Use the cleaned query directly for retrieval. Recent messages still drive
    # clarification detection in EvidencePolicy, but are NOT folded into the
    # embedded / BM25 query — doing so diluted the embedding centroid and
    # injected spurious lexical matches from off-topic chatter (audit M2).
    search_text = processed_query

    retrieval_start = time.time()

    # ---- D2 + E1: Knowledge base hybrid search ----
    kb_context = ""
    domain_prompt = None
    kb_results = []
    try:
        from knowledge.domain_router import get_domain_router, format_kb_results_for_prompt

        router = get_domain_router()
        domains_to_search = router.route(
            processed_query,
            detected_domain=metric.detected_domain,
            channel_name=channel_name,
        )

        if domains_to_search:
            kb_results, primary_domain = await retrieve_knowledge(
                query=processed_query,
                domains=domains_to_search,
                top_k=KB_TOP_K,
                threshold=KB_THRESHOLD,
                metric=metric,
            )
            metric.kb_results = len(kb_results)

            if primary_domain:
                domain_prompt = router.get_domain_prompt(primary_domain)
                metric.detected_domain = primary_domain

    except ImportError:
        logger.debug("Knowledge base module not available")
    except Exception as e:
        logger.warning(f"Knowledge base search failed (non-fatal): {e}")

    # ---- E1: Chat history hybrid retrieval (with A4 fallback) ----
    chat_results = await retrieve(
        query=search_text,
        channel_id=channel_id,
        top_k=top_k,
        metric=metric,
    )
    metric.chat_history_results = len(chat_results)

    retrieval_end = time.time()
    metric.retrieval_time_ms = (retrieval_end - retrieval_start) * 1000

    # ---- E2: Self-RAG relevance grading ----
    self_rag_annotation = ""
    if SELF_RAG_ENABLED and (kb_results or chat_results):
        try:
            from rag.self_rag import grade_relevance, format_self_rag_annotation

            # Grade KB results
            if kb_results:
                kb_results, kb_rag_meta = await grade_relevance(
                    query=processed_query,
                    results=kb_results,
                )
                _record_self_rag_metadata(metric, kb_rag_meta)
                metric.kb_results = len(kb_results)
                logger.debug(
                    f"Self-RAG KB: {kb_rag_meta.relevant_count} relevant, "
                    f"{kb_rag_meta.irrelevant_count} filtered"
                )

            # Grade chat results
            if chat_results:
                chat_results, chat_rag_meta = await grade_relevance(
                    query=processed_query,
                    results=chat_results,
                )
                _record_self_rag_metadata(metric, chat_rag_meta)
                metric.chat_history_results = len(chat_results)
                self_rag_annotation = format_self_rag_annotation(chat_rag_meta)
                logger.debug(
                    f"Self-RAG Chat: {chat_rag_meta.relevant_count} relevant, "
                    f"{chat_rag_meta.irrelevant_count} filtered"
                )

        except ImportError:
            logger.debug("Self-RAG module not available")
        except Exception as e:
            logger.warning(f"Self-RAG grading failed (non-fatal, using unfiltered): {e}")
            metric.llm_error = str(e)

    # Final cap after the wider candidate pool and optional Self-RAG filtering.
    # This keeps the prompt small while still letting retrieval recall more
    # candidates before the relevance filter has a chance to remove noise.
    if kb_results:
        kb_results = kb_results[:KB_TOP_K]
        metric.kb_results = len(kb_results)
    if chat_results:
        chat_results = chat_results[:top_k]
        metric.chat_history_results = len(chat_results)

    # ---- Compute similarity stats for A3 ----
    from rag.scoring import relevance_score

    all_similarities = [relevance_score(r) for r in chat_results]
    all_similarities = [s for s in all_similarities if s > 0]
    if all_similarities:
        metric.avg_similarity = sum(all_similarities) / len(all_similarities)
        metric.max_similarity = max(all_similarities)
        metric.min_similarity = min(all_similarities)
    metric.num_results = len(chat_results) + metric.kb_results
    metric.empty_retrieval = metric.num_results == 0
    metric.citation_coverage = _citation_coverage(kb_results + chat_results)

    all_results = kb_results + chat_results
    assessment = EvidencePolicy().assess(
        query=processed_query,
        results=all_results,
        recent_messages=recent_messages or [],
        query_intent=metric.query_intent,
    )
    trusted_results = assessment.trusted_results
    trusted_kb_results = [
        result for result in trusted_results
        if result.get("content_type") == "knowledge_base"
        or result.get("source_kind") == "knowledge_base"
        or result.get("source_type") == "knowledge_base"
    ]
    trusted_chat_results = [
        result for result in trusted_results
        if result not in trusted_kb_results
    ]

    # Unified, chunk-level citation context + provenance. Markers [n] in the
    # context match the labels in `provenance` so the answer can cite passages.
    citation_context, provenance = build_citation_context(
        trusted_kb_results,
        trusted_chat_results,
        sanitize=sanitize_retrieved_context,
    )

    metric.rag_decision = assessment.decision.value
    metric.decision_reason = assessment.reason
    metric.evidence_score = assessment.score
    metric.trusted_source_count = len(assessment.trusted_results)
    metric.untrusted_source_count = len(assessment.rejected_results)
    metric.provenance = [item.to_dict() for item in provenance]

    # ---- Format combined context (only when we have decided to answer) ----
    combined_parts = []
    if assessment.decision is RAGDecision.ANSWER:
        if self_rag_annotation:
            combined_parts.append(self_rag_annotation)
        if citation_context:
            combined_parts.append(citation_context)

    combined_context = "\n\n".join(combined_parts)

    # ---- A3: Record metric ----
    # retrieval_time_ms already holds the retrieval-phase latency; record the
    # whole-build latency in its own field instead of overwriting it (audit M7).
    metric.total_time_ms = (time.time() - start_time) * 1000
    metrics_manager = get_metrics_manager()
    await metrics_manager.record(metric)

    return RAGBuildResult(
        context=combined_context,
        domain_prompt=domain_prompt or "",
        metric=metric,
        decision=assessment.decision,
        decision_reason=assessment.reason,
        evidence_score=assessment.score,
        provenance=provenance,
        retrieved_results=trusted_results,
        primary_domain=metric.detected_domain,
    )


async def build_rag_context(
    query: str,
    recent_messages: Optional[List[str]] = None,
    channel_id: Optional[str] = None,
    channel_name: Optional[str] = None,
    user_id: Optional[str] = None,
    request_context: Optional[Any] = None,
    top_k: int = RAG_TOP_K,
) -> Tuple[str, Optional[str], RAGMetric]:
    result = await build_rag_result(
        query=query,
        recent_messages=recent_messages,
        channel_id=channel_id,
        channel_name=channel_name,
        user_id=user_id,
        request_context=request_context,
        top_k=top_k,
    )
    return result.context, result.domain_prompt or None, result.metric
