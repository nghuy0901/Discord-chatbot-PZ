import os
import time
import logging
from typing import Optional, Dict, Any
from fastapi import FastAPI, Depends, HTTPException
from fastapi.security.api_key import APIKeyHeader
from pydantic import BaseModel
from api.auth import require_configured_api_key
from api.rate_limit import InMemoryRateLimiter
from api.schemas import MetricsSummary

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
from rag.retriever import build_rag_context
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
ENABLE_KNOWLEDGE_BASE: bool = os.getenv("ENABLE_KNOWLEDGE_BASE", "True").lower() == "true"
ENABLE_TOOL_CALLING: bool = os.getenv("ENABLE_TOOL_CALLING", "True").lower() == "true"
ENFORCE_RAG_CITATIONS: bool = os.getenv("ENFORCE_RAG_CITATIONS", "True").lower() == "true"
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


def get_embedding_config_error() -> Optional[str]:
    if not ENABLE_RAG:
        return None
    try:
        from src.llm.embedding_factory import build_embedding_config

        build_embedding_config()
        return None
    except Exception as e:
        return str(e)

# Request body schemas
class QueryRequest(BaseModel):
    query: str
    domain: Optional[str] = None
    user_id: Optional[str] = "api_user"
    channel_id: Optional[str] = "api_channel"

# App definition
app = FastAPI(
    title="NomNom RAG REST API",
    description="REST API interface to the Project Zomboid Chatbot RAG System, featuring caching, routing, and observability.",
    version="1.0.0",
)

@app.on_event("startup")
async def startup_event():
    logger.info("Starting up RAG REST API server...")
    try:
        await init_db()
        await get_metrics_manager().init_db()
        logger.info("Database and metrics storage initialized.")
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")
    if ENABLE_KNOWLEDGE_BASE:
        try:
            result = await get_knowledge_manager().load_all()
            logger.info("Knowledge base loaded for API: %s", result)
        except Exception as e:
            logger.error(f"Knowledge base initialization failed: {e}")
    if ENABLE_RAG:
        try:
            from rag.bm25_search import init_bm25_indices

            result = await init_bm25_indices()
            logger.info("BM25 indices initialized for API: %s", result)
        except Exception as e:
            logger.error(f"BM25 initialization failed: {e}")

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
    embedding_ok = False
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

    # Check configured embedding provider
    embedding_error = get_embedding_config_error()
    embedding_ok = embedding_error is None
    if embedding_error:
        logger.warning(f"Embedding health check failed: {embedding_error}")

    # Check Redis
    try:
        cache = get_query_cache()
        if cache._redis:
            await cache._redis.ping()
            redis_ok = True
    except Exception as e:
        logger.warning(f"Redis health check failed: {e}")

    overall_status = "healthy" if (postgres_ok and llm_ok and embedding_ok) else "degraded"

    return {
        "status": overall_status,
        "services": {
            "postgres": "connected" if postgres_ok else "disconnected",
            "llm": "configured" if llm_ok else "not_configured",
            "embedding": "configured" if embedding_ok else "not_configured",
            "redis_cache": "connected" if redis_ok else "disconnected (caching disabled)",
        }
    }

@app.post("/api/query", tags=["Query"])
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
    embedding_error = get_embedding_config_error()
    if embedding_error:
        raise HTTPException(
            status_code=503,
            detail=f"Embedding provider is not configured: {embedding_error}",
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
    if cached_response and ENABLE_RAG and not cached_response.get("citation_validation"):
        logger.info("Ignoring cached RAG response without citation validation metadata.")
        cached_response = None

    if cached_response:
        response_text = cached_response["response_text"]
        latency = (time.time() - query_start) * 1000

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
            "cache_hit": True,
            "latency_ms": round(latency, 2),
            "citation_validation": cached_response.get("citation_validation"),
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
            rag_context, domain_prompt_text, metric = await build_rag_context(
                query=query,
                recent_messages=None,
                channel_id=payload.channel_id,
                channel_name=channel_name,
                user_id=payload.user_id,
                request_context=request_context,
            )
            if domain_prompt_text:
                domain_prompt = f"\n# 🎯 Domain-Specific Instructions\n{domain_prompt_text}\n"
            if metric and metric.query_intent:
                query_intent_str = metric.query_intent

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

        citation_validation = None
        if ENABLE_RAG and ENFORCE_RAG_CITATIONS and rag_context:
            from rag.citations import extract_citation_ids, validate_citations

            evidence_ids = extract_citation_ids(rag_context)
            if evidence_ids:
                citation_result = validate_citations(
                    response_text,
                    evidence_ids,
                    require_citation=True,
                )
                citation_validation = citation_result.to_dict()
                metric.provenance = [
                    {"citation_id": evidence_id} for evidence_id in evidence_ids
                ]
                metric.evidence_score = (
                    len(citation_result.valid_ids) / len(citation_result.citation_ids)
                    if citation_result.citation_ids
                    else 0.0
                )
                if not citation_result.is_valid:
                    metric.rag_decision = "abstain"
                    metric.decision_reason = citation_result.error or "invalid_citation"
                    metric.llm_error = metric.decision_reason
                    await get_metrics_manager().update_db(metric)
                    raise HTTPException(
                        status_code=502,
                        detail=(
                            "LLM response failed citation validation: "
                            f"{metric.decision_reason}"
                        ),
                    )
                metric.rag_decision = "answer"
                metric.decision_reason = "citation_validated"

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

        # Re-save metrics to ensure SQL database write
        metrics = get_metrics_manager()
        await metrics.update_db(metric)

        # Save cache
        if query_intent_str in ("analytical", "hybrid", "narrative") and response_text:
            try:
                cache = get_query_cache()
                await cache.set(
                    query=query,
                    domain=detected_domain,
                    result={
                        "response_text": response_text,
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "citation_validation": citation_validation,
                    }
                )
            except Exception as ce:
                logger.warning(f"Failed to cache response in API: {ce}")

        return {
            "query": query,
            "query_id": request_context.query_id,
            "response": response_text,
            "cache_hit": False,
            "latency_ms": round(latency, 2),
            "citation_validation": citation_validation,
            "metrics": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
                "estimated_cost_usd": estimated_cost_usd,
                "detected_domain": detected_domain,
                "query_intent": query_intent_str
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Error handling query: {e}")
        raise HTTPException(status_code=500, detail=str(e))

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
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/admin/reload", tags=["Admin"], dependencies=[Depends(get_api_key)])
async def reload_knowledge_base():
    try:
        km = get_knowledge_manager()
        await km.reload()
        return {"status": "success", "message": "Knowledge base reload and cache invalidation completed."}
    except Exception as e:
        logger.error(f"Error reloading knowledge base: {e}")
        raise HTTPException(status_code=500, detail=str(e))
