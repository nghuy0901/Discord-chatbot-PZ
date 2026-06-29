import os
import time
import logging
from typing import Optional, Dict, Any
from fastapi import FastAPI, Depends, HTTPException
from fastapi.security.api_key import APIKeyHeader
from pydantic import BaseModel
from api.auth import require_configured_api_key
from api.rate_limit import InMemoryRateLimiter
from api.schemas import MetricsSummary, QueryResponse

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Setup Logging
logger = logging.getLogger("rag_api")
logging.basicConfig(level=logging.INFO)

# Imports from RAG codebase
from rag.db import init_db, close_pool, get_pool
from rag.metrics import get_metrics_manager, RAGMetric
from rag.query_preprocessor import get_preprocessor
from rag.retriever import build_rag_result
from rag.result import RAGDecision
from rag.responses import decision_response
from src.observability.prompts import prompt_version
from src.observability.request_context import RequestContext
from src.ollama_provider import (
    chat_completion,
    chat_with_tools,
    get_last_token_usage,
    health_check as provider_health_check,
)
from utils.cache import get_query_cache
from knowledge.manager import get_knowledge_manager

# Structured tools (PZ tools) if available
try:
    from knowledge.structured.pz_tools import PZ_TOOLS, PZ_TOOL_FUNCTIONS
    STRUCTURED_TOOLS_AVAILABLE = True
except ImportError:
    PZ_TOOLS = []
    PZ_TOOL_FUNCTIONS = {}
    STRUCTURED_TOOLS_AVAILABLE = False

ENABLE_RAG: bool = os.getenv("ENABLE_RAG", "True").lower() == "true"
ENABLE_TOOL_CALLING: bool = os.getenv("ENABLE_TOOL_CALLING", "True").lower() == "true"
API_RATE_LIMIT = int(os.getenv("API_RATE_LIMIT", "60"))
API_RATE_LIMIT_WINDOW_SECONDS = int(os.getenv("API_RATE_LIMIT_WINDOW_SECONDS", "60"))
api_rate_limiter = InMemoryRateLimiter(
    limit=API_RATE_LIMIT,
    window_seconds=API_RATE_LIMIT_WINDOW_SECONDS,
)

# Security configuration
API_KEY_NAME = "X-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

async def get_api_key(api_key: str = Depends(api_key_header)):
    expected_key = require_configured_api_key()
    if not api_key or api_key != expected_key:
        raise HTTPException(status_code=403, detail="Invalid or missing API Key")
    return api_key

# Request body schemas
class QueryRequest(BaseModel):
    query: str
    domain: Optional[str] = None
    user_id: Optional[str] = "api_user"
    channel_id: Optional[str] = "api_channel"
    include_sources: bool = False

# App definition
app = FastAPI(
    title="NomNom RAG REST API",
    description="REST API interface to the Project Zomboid Chatbot RAG System, featuring caching, routing, and observability.",
    version="1.0.0",
)

@app.on_event("startup")
async def startup_event():
    logger.info("Starting up RAG REST API server...")
    # Validate config at boot: hard-fail in production (audit H9), warn in dev.
    from src.startup import validate_runtime_config, is_production

    try:
        validate_runtime_config(require_api=True)
    except Exception as e:
        if is_production():
            raise
        logger.warning(f"Startup config validation (non-fatal in dev): {e}")
    try:
        await init_db()
        await get_metrics_manager().init_db()
        logger.info("Database and metrics storage initialized.")
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Shutting down RAG REST API server...")
    try:
        await close_pool()
        logger.info("Database pools closed.")
    except Exception as e:
        logger.error(f"Error closing DB pool: {e}")

@app.get("/api/health", tags=["Health"])
async def health_check():
    postgres_ok = False
    llm_ok = False
    redis_ok = False

    # Check Postgres
    try:
        pool = await get_pool()
        async with pool.acquire() as conn:
            await conn.execute("SELECT 1;")
            postgres_ok = True
    except Exception as e:
        logger.warning(f"Postgres health check failed: {e}")

    # Check configured LLM provider
    try:
        llm_ok = await provider_health_check()
    except Exception as e:
        logger.warning(f"LLM health check failed: {e}")

    # Check Redis
    try:
        cache = get_query_cache()
        if cache._redis:
            await cache._redis.ping()
            redis_ok = True
    except Exception as e:
        logger.warning(f"Redis health check failed: {e}")

    overall_status = "healthy" if (postgres_ok and llm_ok) else "degraded"

    return {
        "status": overall_status,
        "services": {
            "postgres": "connected" if postgres_ok else "disconnected",
            "llm": "configured" if llm_ok else "not_configured",
            "redis_cache": "connected" if redis_ok else "disconnected (caching disabled)",
        }
    }

@app.post("/api/query", tags=["Query"], response_model=QueryResponse)
async def execute_rag_query(
    payload: QueryRequest,
    api_key: str = Depends(get_api_key),
):
    query_start = time.time()
    query = payload.query
    channel_name = payload.domain
    rate_key = f"{api_key}:{payload.user_id or 'api_user'}"
    if not api_rate_limiter.allow(rate_key):
        raise HTTPException(status_code=429, detail="Rate limit exceeded")
    request_context = RequestContext.new(
        channel_id=payload.channel_id or "api_channel",
        user_id=payload.user_id or "api_user",
        source="api",
    )

    cached_response = None
    detected_domain = None
    query_intent_str = None

    # Preprocess
    if ENABLE_RAG:
        try:
            preprocessor = get_preprocessor()
            _, query_meta = preprocessor.preprocess(query, channel_name)
            detected_domain = query_meta.get("domain")
            query_intent = query_meta.get("query_intent")
            query_intent_str = query_intent.value if hasattr(query_intent, 'value') else str(query_intent)

            # Check cache
            if query_intent_str in ("analytical", "hybrid", "narrative"):
                cache = get_query_cache()
                cached_response = await cache.get(query, detected_domain)
        except Exception as e:
            logger.warning(f"Cache check failed: {e}")

    # Cache hit
    if cached_response:
        response_text = cached_response["response_text"]
        latency = (time.time() - query_start) * 1000
        provenance = cached_response.get("provenance", [])
        # Re-attach the chunk-level citation footer (stored raw in cache).
        if os.getenv("RAG_SHOW_CITATIONS", "true").lower() == "true":
            try:
                from rag.citations import render_sources_footer_from_dicts

                _footer = render_sources_footer_from_dicts(
                    response_text,
                    provenance,
                    language=(cached_response.get("language") or "vi"),
                )
                if _footer:
                    response_text = f"{response_text}\n\n{_footer}"
            except Exception:
                pass

        # Record cache hit metric
        try:
            metrics = get_metrics_manager()
            metric = RAGMetric(
                query_id=request_context.query_id,
                request_id=request_context.request_id,
                source=request_context.source,
                channel_id=payload.channel_id,
                user_id=payload.user_id,
                original_query=query,
                processed_query=query,
                detected_domain=detected_domain,
                query_intent=query_intent_str,
                cache_hit=True,
                response_time_ms=latency,
                response_length=len(response_text),
                prompt_tokens=cached_response.get("prompt_tokens", 0),
                completion_tokens=cached_response.get("completion_tokens", 0),
                total_tokens=cached_response.get("prompt_tokens", 0) + cached_response.get("completion_tokens", 0),
                rag_decision="answer",
                provenance=provenance,
                trusted_source_count=len(provenance),
            )
            await metrics.record(metric)
            # Update metric in db since it has response details
            await metrics.update_db(metric)
        except Exception as e:
            logger.warning(f"Failed to record cache hit metric in API: {e}")

        return {
            "query": query,
            "query_id": request_context.query_id,
            "response": response_text,
            "decision": "answer",
            "cache_hit": True,
            "latency_ms": round(latency, 2),
            "source_count": len(provenance),
            "sources": [
                item.get("source_id", "")
                for item in provenance
                if payload.include_sources and item.get("source_id")
            ],
            "citations": [
                {
                    "label": item.get("label", ""),
                    "locator": item.get("locator", ""),
                    "source": item.get("source", ""),
                    "heading_path": item.get("heading_path", ""),
                    "excerpt": item.get("excerpt", ""),
                    "url": item.get("url", ""),
                    "similarity": item.get("similarity", 0.0),
                }
                for item in provenance
                if payload.include_sources
            ],
            "metrics": {
                "prompt_tokens": cached_response.get("prompt_tokens", 0),
                "completion_tokens": cached_response.get("completion_tokens", 0),
                "total_tokens": cached_response.get("prompt_tokens", 0) + cached_response.get("completion_tokens", 0),
                "estimated_cost_usd": 0.0,
                "detected_domain": detected_domain,
                "query_intent": query_intent_str
            }
        }

    # Cache miss
    try:
        rag_result = None
        rag_context = ""
        domain_prompt = ""
        metric = RAGMetric(
            query_id=request_context.query_id,
            request_id=request_context.request_id,
            source=request_context.source,
            channel_id=payload.channel_id,
            user_id=payload.user_id,
            original_query=query[:500],
        )

        if ENABLE_RAG:
            rag_result = await build_rag_result(
                query=query,
                recent_messages=None,
                channel_id=payload.channel_id,
                channel_name=channel_name,
                user_id=payload.user_id,
                request_context=request_context,
            )
            rag_context = rag_result.context
            domain_prompt_text = rag_result.domain_prompt
            metric = rag_result.metric
            if domain_prompt_text:
                domain_prompt = f"\n# 🎯 Domain-Specific Instructions\n{domain_prompt_text}\n"
            if metric and metric.query_intent:
                query_intent_str = metric.query_intent
            detected_domain = metric.detected_domain

            if rag_result.decision is not RAGDecision.ANSWER:
                response_text = decision_response(
                    rag_result.metric.query_language,
                    rag_result.decision,
                )
                latency = (time.time() - query_start) * 1000
                metric.response_time_ms = latency
                metric.response_length = len(response_text)
                metrics = get_metrics_manager()
                await metrics.update_db(metric)
                return {
                    "query": query,
                    "query_id": request_context.query_id,
                    "response": response_text,
                    "decision": rag_result.decision.value,
                    "cache_hit": False,
                    "latency_ms": round(latency, 2),
                    "source_count": 0,
                    "sources": [],
                    "metrics": {
                        "detected_domain": rag_result.metric.detected_domain,
                        "query_intent": rag_result.metric.query_intent,
                        "evidence_score": rag_result.evidence_score,
                        "decision_reason": rag_result.decision_reason,
                    },
                }

        # Build prompt messages
        from prompts.system_prompt import build_system_prompt
        system_prompt = build_system_prompt(
            rag_context=rag_context,
            domain_prompt=domain_prompt,
        )
        metric.prompt_version = prompt_version(system_prompt)
        prompt_messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query}
        ]

        # Route logic
        use_tools_for_query = (
            ENABLE_TOOL_CALLING
            and STRUCTURED_TOOLS_AVAILABLE
            and PZ_TOOLS
            and query_intent_str in ("analytical", "hybrid", None)
        )

        # Execute LLM call
        if use_tools_for_query:
            response_text = await chat_with_tools(
                messages=prompt_messages,
                tools=PZ_TOOLS,
                tool_functions=PZ_TOOL_FUNCTIONS,
                temperature=0.8,
                request_context=request_context,
            )
        else:
            response_text = await chat_completion(
                messages=prompt_messages,
                temperature=0.8,
                request_context=request_context,
            )

        latency = (time.time() - query_start) * 1000

        # Update metric with final run details
        metric.response_time_ms = latency
        metric.response_length = len(response_text)

        # Retrieve tokens and cost summary
        prompt_tokens = 0
        completion_tokens = 0
        total_tokens = 0
        estimated_cost_usd = 0.0

        token_usage = get_last_token_usage()
        if token_usage:
            prompt_tokens = token_usage.get("prompt_tokens", 0)
            completion_tokens = token_usage.get("completion_tokens", 0)
            total_tokens = token_usage.get("total_tokens", 0)
            estimated_cost_usd = token_usage.get("estimated_cost_usd", 0.0)

            metric.prompt_tokens = prompt_tokens
            metric.completion_tokens = completion_tokens
            metric.total_tokens = total_tokens
            metric.estimated_cost_usd = estimated_cost_usd

        # ---- Groundedness gate + chunk-level citations ----
        # Token usage was already captured above, so the extra judge call inside
        # finalize_rag_answer does not corrupt this turn's accounting.
        # `raw_answer` (no footer) is what gets cached; the footer is re-rendered
        # at display time so the API and Discord paths stay consistent.
        # Tool answers come from the authoritative SQLite DB, which is not in the
        # vector provenance the groundedness gate checks — skip the gate for them.
        decision_value = "answer"
        raw_answer = response_text
        if (
            rag_result
            and rag_result.decision is RAGDecision.ANSWER
            and response_text
            and not use_tools_for_query
        ):
            from rag.answer_finalize import finalize_rag_answer

            finalized = await finalize_rag_answer(rag_result, query, response_text)
            response_text = finalized.text
            decision_value = finalized.decision.value
            metric.response_length = len(response_text)
        elif use_tools_for_query and response_text:
            # C2: verify tool-calling answers against the actual tool outputs,
            # since they are not part of the vector provenance.
            from rag.answer_finalize import finalize_tool_answer
            from src.ollama_provider import get_last_tool_results

            lang = rag_result.metric.query_language if rag_result else "vi"
            finalized = await finalize_tool_answer(
                query, response_text, get_last_tool_results(), language=lang
            )
            response_text = finalized.text
            decision_value = finalized.decision.value
            metric.response_length = len(response_text)

        # Re-save metrics to ensure SQL database write
        metrics = get_metrics_manager()
        await metrics.update_db(metric)

        # Save cache
        provenance = []
        if rag_result:
            provenance = [item.to_dict() for item in rag_result.provenance]

        answered = decision_value == "answer"
        # Vector provenance is only meaningful for non-tool answers.
        cite_ok = answered and not use_tools_for_query
        if (
            rag_result
            and answered
            and query_intent_str in ("analytical", "hybrid", "narrative")
            and response_text
            and provenance
        ):
            try:
                cache = get_query_cache()
                await cache.set(
                    query=query,
                    domain=detected_domain,
                    result={
                        "response_text": raw_answer,
                        "decision": decision_value,
                        "provenance": provenance,
                        "language": rag_result.metric.query_language,
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                    }
                )
            except Exception as ce:
                logger.warning(f"Failed to cache response in API: {ce}")

        return {
            "query": query,
            "query_id": request_context.query_id,
            "response": response_text,
            "decision": decision_value,
            "cache_hit": False,
            "latency_ms": round(latency, 2),
            "source_count": len(provenance) if cite_ok else 0,
            "sources": [
                item.get("source_id", "")
                for item in provenance
                if cite_ok and payload.include_sources and item.get("source_id")
            ],
            "citations": [
                {
                    "label": item.get("label", ""),
                    "locator": item.get("locator", ""),
                    "source": item.get("source", ""),
                    "heading_path": item.get("heading_path", ""),
                    "excerpt": item.get("excerpt", ""),
                    "url": item.get("url", ""),
                    "similarity": item.get("similarity", 0.0),
                }
                for item in provenance
                if cite_ok and payload.include_sources
            ],
            "metrics": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
                "estimated_cost_usd": estimated_cost_usd,
                "detected_domain": detected_domain,
                "query_intent": query_intent_str
            }
        }
    except Exception as e:
        logger.exception(f"Error handling query: {e}")
        # Never leak internal exception text (DSNs, hosts, SQL) to callers (H8).
        raise HTTPException(
            status_code=500,
            detail=f"Internal error (ref: {request_context.request_id[:8]})",
        )

def _build_metrics_summary(raw: Dict[str, Any]) -> MetricsSummary:
    total_queries = int(raw.get("total_queries", raw.get("recent_queries", 0)) or 0)
    errors = int(raw.get("errors", raw.get("total_errors", 0)) or 0)
    error_rate = (errors / total_queries) if total_queries else 0.0
    success_rate = 1.0 - error_rate if total_queries else 0.0

    response_percentiles = raw.get("percentiles", {}).get("response", {})
    avg_latency = float(raw.get("avg_response_time_ms", 0.0) or 0.0)
    retrieval_latency = float(raw.get("avg_retrieval_time_ms", 0.0) or 0.0)

    return MetricsSummary(
        total_queries=total_queries,
        success_rate=round(success_rate, 4),
        error_rate=round(error_rate, 4),
        avg_latency_ms=round(avg_latency, 2),
        p50_latency_ms=float(response_percentiles.get("p50", 0.0) or 0.0),
        p95_latency_ms=float(response_percentiles.get("p95", 0.0) or 0.0),
        p99_latency_ms=float(response_percentiles.get("p99", 0.0) or 0.0),
        retrieval_latency_ms=round(retrieval_latency, 2),
        llm_latency_ms=round(max(avg_latency - retrieval_latency, 0.0), 2),
        empty_retrieval_rate=round(float(raw.get("empty_retrieval_rate", 0.0) or 0.0), 4),
        average_retrieved_chunks=round(float(raw.get("avg_results_per_query", 0.0) or 0.0), 2),
        citation_coverage=round(float(raw.get("citation_coverage", 0.0) or 0.0), 4),
        cache_hit_rate=round(float(raw.get("cache_hit_rate", 0.0) or 0.0), 4),
        total_tokens=int(raw.get("total_tokens", 0) or 0),
        estimated_cost_usd=round(
            float(raw.get("total_estimated_cost_usd", raw.get("estimated_cost_usd", 0.0)) or 0.0),
            6,
        ),
    )


@app.get(
    "/api/metrics",
    tags=["Metrics"],
    dependencies=[Depends(get_api_key)],
    response_model=MetricsSummary,
)
async def get_metrics() -> MetricsSummary:
    try:
        metrics = get_metrics_manager()
        raw = await metrics.get_db_summary()
        return _build_metrics_summary(raw)
    except Exception as e:
        logger.error(f"Error retrieving metrics: {e}")
        raise HTTPException(status_code=500, detail="Internal error retrieving metrics")

@app.post("/api/admin/reload", tags=["Admin"], dependencies=[Depends(get_api_key)])
async def reload_knowledge_base():
    try:
        km = get_knowledge_manager()
        await km.reload()
        return {"status": "success", "message": "Knowledge base reload and cache invalidation completed."}
    except Exception as e:
        logger.error(f"Error reloading knowledge base: {e}")
        raise HTTPException(status_code=500, detail="Internal error reloading knowledge base")
