"""
RAG Metrics (A3) — Track and persist performance metrics for the RAG pipeline.

Metrics captured per query:
- Query text (original + preprocessed)
- Retrieval time (ms)
- Number of retrieved results
- Average similarity score
- Response generation time (ms)
- Knowledge domains matched
- User feedback (via reactions)

Storage: PostgreSQL table `rag_metrics` + in-memory rolling window for
real-time /status dashboard data.
"""

import os
import time
import uuid
import logging
import asyncio
from collections import deque
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Any, Deque

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
METRICS_RETENTION_SIZE: int = int(os.getenv("METRICS_RETENTION_SIZE", "500"))

REQUIRED_RAG_METRIC_COLUMNS = {
    "query_id",
    "request_id",
    "source",
    "channel_id",
    "user_id",
    "original_query",
    "processed_query",
    "query_language",
    "detected_domain",
    "query_intent",
    "prompt_version",
    "llm_model",
    "embedding_model",
    "retrieval_config_version",
    "retrieval_time_ms",
    "num_results",
    "avg_similarity",
    "max_similarity",
    "min_similarity",
    "kb_results",
    "chat_history_results",
    "vector_results",
    "bm25_results",
    "hybrid_fused_results",
    "vector_time_ms",
    "bm25_time_ms",
    "self_rag_enabled",
    "self_rag_graded",
    "self_rag_relevant",
    "self_rag_irrelevant",
    "self_rag_time_ms",
    "empty_retrieval",
    "citation_coverage",
    "response_time_ms",
    "response_length",
    "error",
    "retrieval_error",
    "llm_error",
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
    "estimated_cost_usd",
    "cache_hit",
}

REQUIRED_RAG_FEEDBACK_COLUMNS = {
    "query_id",
    "message_id",
    "user_id",
    "feedback_score",
}

METRIC_INSERT_COLUMNS = [
    "query_id",
    "request_id",
    "source",
    "channel_id",
    "user_id",
    "original_query",
    "processed_query",
    "query_language",
    "detected_domain",
    "query_intent",
    "prompt_version",
    "llm_model",
    "embedding_model",
    "retrieval_config_version",
    "retrieval_time_ms",
    "num_results",
    "avg_similarity",
    "max_similarity",
    "min_similarity",
    "kb_results",
    "chat_history_results",
    "vector_results",
    "bm25_results",
    "hybrid_fused_results",
    "vector_time_ms",
    "bm25_time_ms",
    "self_rag_enabled",
    "self_rag_graded",
    "self_rag_relevant",
    "self_rag_irrelevant",
    "self_rag_time_ms",
    "empty_retrieval",
    "citation_coverage",
    "response_time_ms",
    "response_length",
    "error",
    "retrieval_error",
    "llm_error",
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
    "estimated_cost_usd",
    "cache_hit",
]


# ---------------------------------------------------------------------------
# Metric data class
# ---------------------------------------------------------------------------
@dataclass
class RAGMetric:
    """Single RAG query metric record."""
    query_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    timestamp: float = field(default_factory=time.time)
    channel_id: str = ""
    user_id: str = ""
    request_id: str = ""
    source: str = ""

    # Query info
    original_query: str = ""
    processed_query: str = ""
    query_language: str = ""
    detected_domain: Optional[str] = None
    query_intent: Optional[str] = None  # Phase 4: "analytical"/"narrative"/"hybrid"/"conversation"
    prompt_version: str = ""
    llm_model: str = ""
    embedding_model: str = ""
    retrieval_config_version: str = ""

    # Retrieval performance
    retrieval_time_ms: float = 0.0
    num_results: int = 0
    avg_similarity: float = 0.0
    max_similarity: float = 0.0
    min_similarity: float = 0.0

    # Knowledge base results
    kb_results: int = 0
    chat_history_results: int = 0

    # E1: Hybrid RAG metrics
    vector_results: int = 0
    bm25_results: int = 0
    hybrid_fused_results: int = 0
    vector_time_ms: float = 0.0
    bm25_time_ms: float = 0.0

    # E2: Self-RAG metrics
    self_rag_enabled: bool = False
    self_rag_graded: int = 0
    self_rag_relevant: int = 0
    self_rag_irrelevant: int = 0
    self_rag_time_ms: float = 0.0
    empty_retrieval: bool = False
    citation_coverage: float = 0.0

    # Response
    response_time_ms: float = 0.0
    response_length: int = 0

    # Token usage & cost tracking
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    estimated_cost_usd: float = 0.0  # Hypothetical cost at GPT-4o pricing

    # Feedback
    feedback_score: Optional[int] = None  # 1=👍, -1=👎
    feedback_user_id: Optional[str] = None

    # Errors
    error: Optional[str] = None
    retrieval_error: Optional[str] = None
    llm_error: Optional[str] = None

    # Cache
    cache_hit: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Metrics Manager
# ---------------------------------------------------------------------------
class MetricsManager:
    """
    In-memory metrics manager with PostgreSQL persistence.
    Keeps a rolling window of recent metrics for real-time stats.
    """

    def __init__(self, retention_size: int = METRICS_RETENTION_SIZE):
        self._recent: Deque[RAGMetric] = deque(maxlen=retention_size)
        self._total_queries: int = 0
        self._total_errors: int = 0
        self._db_initialized: bool = False

    async def init_db(self) -> None:
        """Verify migration-managed metrics tables exist."""
        try:
            from rag.db import get_pool
            pool = await get_pool()
            async with pool.acquire() as conn:
                await self._verify_columns(conn, "rag_metrics", REQUIRED_RAG_METRIC_COLUMNS)
                await self._verify_columns(conn, "rag_feedback", REQUIRED_RAG_FEEDBACK_COLUMNS)
            self._db_initialized = True
            logger.info("✅ RAG metrics schema verified.")
        except Exception as e:
            self._db_initialized = False
            logger.error(
                "RAG metrics schema verification failed: %s. "
                "Run: python scripts/migrate_db.py",
                e,
            )
            raise

    async def _verify_columns(
        self,
        conn: Any,
        table_name: str,
        required_columns: set[str],
    ) -> None:
        rows = await conn.fetch(
            """
            SELECT column_name
            FROM information_schema.columns
            WHERE table_schema = 'public'
              AND table_name = $1;
            """,
            table_name,
        )
        existing = {row["column_name"] for row in rows}
        missing = sorted(required_columns - existing)
        if missing:
            raise RuntimeError(
                f"{table_name} is missing columns {missing}. "
                "Run: python scripts/migrate_db.py"
            )

    async def record(self, metric: RAGMetric) -> None:
        """Record a metric (in-memory + async DB persist)."""
        self._recent.append(metric)
        self._total_queries += 1
        if metric.error:
            self._total_errors += 1

        # Async DB persist (fire-and-forget)
        if self._db_initialized:
            asyncio.create_task(self._persist_metric(metric))

    async def record_feedback(
        self, message_id: str, query_id: str, user_id: str, score: int
    ) -> None:
        """Record user feedback (👍 = 1, 👎 = -1) for a response."""
        if not self._db_initialized:
            return
        try:
            from rag.db import get_pool
            pool = await get_pool()
            async with pool.acquire() as conn:
                await conn.execute("""
                    INSERT INTO rag_feedback (query_id, message_id, user_id, feedback_score)
                    VALUES ($1, $2, $3, $4)
                    ON CONFLICT (message_id, user_id) DO UPDATE SET feedback_score = $4;
                """, query_id, message_id, user_id, score)
        except Exception as e:
            logger.warning(f"Failed to record feedback: {e}")

    async def _persist_metric(self, metric: RAGMetric) -> None:
        """Persist a single metric to PostgreSQL."""
        try:
            from rag.db import get_pool
            pool = await get_pool()
            async with pool.acquire() as conn:
                placeholders = ", ".join(
                    f"${idx}" for idx in range(1, len(METRIC_INSERT_COLUMNS) + 1)
                )
                columns = ", ".join(METRIC_INSERT_COLUMNS)
                await conn.execute(
                    f"""
                    INSERT INTO rag_metrics ({columns})
                    VALUES ({placeholders})
                    ON CONFLICT (query_id) DO NOTHING;
                    """,
                    *self._metric_values(metric, METRIC_INSERT_COLUMNS),
                )
        except Exception as e:
            logger.debug(f"Metric persist failed: {e}")

    async def update_db(self, metric: RAGMetric) -> None:
        """Update mutable details of an already-recorded metric in DB."""
        if not self._db_initialized:
            return
        try:
            from rag.db import get_pool
            pool = await get_pool()
            async with pool.acquire() as conn:
                update_columns = [c for c in METRIC_INSERT_COLUMNS if c != "query_id"]
                assignments = ", ".join(
                    f"{column} = ${idx}"
                    for idx, column in enumerate(update_columns, start=2)
                )
                await conn.execute(
                    f"""
                    UPDATE rag_metrics SET {assignments}
                    WHERE query_id = $1;
                    """,
                    metric.query_id,
                    *self._metric_values(metric, update_columns),
                )
        except Exception as e:
            logger.debug(f"Metric update failed: {e}")

    def _metric_values(self, metric: RAGMetric, columns: List[str]) -> List[Any]:
        values: List[Any] = []
        for column in columns:
            value = getattr(metric, column)
            if column in {"original_query", "processed_query"} and value:
                value = value[:500]
            values.append(value)
        return values

    # ----- Aggregation helpers -----

    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics from the in-memory rolling window."""
        if not self._recent:
            return {
                "total_queries": self._total_queries,
                "total_errors": self._total_errors,
                "recent_queries": 0,
            }

        recent = list(self._recent)
        retrieval_times = [m.retrieval_time_ms for m in recent if m.retrieval_time_ms > 0]
        similarities = [m.avg_similarity for m in recent if m.avg_similarity > 0]
        response_times = [m.response_time_ms for m in recent if m.response_time_ms > 0]

        return {
            "total_queries": self._total_queries,
            "total_errors": self._total_errors,
            "recent_queries": len(recent),
            "avg_retrieval_time_ms": (
                sum(retrieval_times) / len(retrieval_times) if retrieval_times else 0
            ),
            "avg_similarity": (
                sum(similarities) / len(similarities) if similarities else 0
            ),
            "avg_response_time_ms": (
                sum(response_times) / len(response_times) if response_times else 0
            ),
            "avg_results_per_query": (
                sum(m.num_results for m in recent) / len(recent) if recent else 0
            ),
            "domains_seen": list(set(
                m.detected_domain for m in recent if m.detected_domain
            )),
            "feedback_positive": sum(1 for m in recent if m.feedback_score and m.feedback_score > 0),
            "feedback_negative": sum(1 for m in recent if m.feedback_score and m.feedback_score < 0),
            "percentiles": self.get_percentiles(),
            # Cache stats
            "cache_hits": sum(1 for m in recent if getattr(m, 'cache_hit', False)),
            "cache_hit_rate": round(sum(1 for m in recent if getattr(m, 'cache_hit', False)) / len(recent), 4) if recent else 0,
            # Token usage & cost tracking
            "total_tokens": sum(m.total_tokens for m in recent),
            "total_estimated_cost_usd": round(sum(m.estimated_cost_usd for m in recent), 6),
            "avg_tokens_per_query": round(
                sum(m.total_tokens for m in recent) / len(recent), 0
            ) if recent else 0,
        }

    def get_percentiles(self) -> Dict[str, Any]:
        """Calculate p50, p95, p99 percentiles for retrieval and response latency.

        Uses the in-memory rolling window of recent metrics.
        Returns a dict with 'retrieval' and 'response' sub-dicts,
        each containing p50, p95, p99, min, max, and count.
        """
        if not self._recent:
            return {}

        import statistics

        recent = list(self._recent)
        retrieval_times = sorted(
            [m.retrieval_time_ms for m in recent if m.retrieval_time_ms > 0]
        )
        response_times = sorted(
            [m.response_time_ms for m in recent if m.response_time_ms > 0]
        )

        def _calc(data: list) -> Dict[str, float]:
            if not data:
                return {"p50": 0, "p95": 0, "p99": 0, "min": 0, "max": 0, "count": 0}
            if len(data) < 2:
                val = data[0]
                return {
                    "p50": round(val, 1), "p95": round(val, 1),
                    "p99": round(val, 1), "min": round(val, 1),
                    "max": round(val, 1), "count": 1,
                }
            quantiles = statistics.quantiles(data, n=100)
            return {
                "p50": round(quantiles[49], 1),
                "p95": round(quantiles[94], 1),
                "p99": round(quantiles[98] if len(quantiles) > 98 else quantiles[-1], 1),
                "min": round(min(data), 1),
                "max": round(max(data), 1),
                "count": len(data),
            }

        return {
            "retrieval": _calc(retrieval_times),
            "response": _calc(response_times),
        }

    async def get_db_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get summary from the database for the last N hours."""
        if not self._db_initialized:
            return self.get_summary()
        try:
            from rag.db import get_pool
            pool = await get_pool()
            async with pool.acquire() as conn:
                row = await conn.fetchrow("""
                    SELECT
                        COUNT(*) as total,
                        AVG(retrieval_time_ms) as avg_retrieval,
                        AVG(avg_similarity) as avg_sim,
                        AVG(response_time_ms) as avg_response,
                        AVG(num_results) as avg_results,
                        AVG(CASE WHEN empty_retrieval THEN 1.0 ELSE 0.0 END) as empty_retrieval_rate,
                        AVG(citation_coverage) as citation_coverage,
                        AVG(CASE WHEN cache_hit THEN 1.0 ELSE 0.0 END) as cache_hit_rate,
                        SUM(total_tokens) as total_tokens,
                        SUM(estimated_cost_usd) as total_estimated_cost,
                        COUNT(CASE WHEN error IS NOT NULL OR retrieval_error IS NOT NULL OR llm_error IS NOT NULL THEN 1 END) as errors,
                        PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY retrieval_time_ms) as p50_retrieval,
                        PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY retrieval_time_ms) as p95_retrieval,
                        PERCENTILE_CONT(0.99) WITHIN GROUP (ORDER BY retrieval_time_ms) as p99_retrieval,
                        PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY response_time_ms) as p50_response,
                        PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY response_time_ms) as p95_response,
                        PERCENTILE_CONT(0.99) WITHIN GROUP (ORDER BY response_time_ms) as p99_response
                    FROM rag_metrics
                    WHERE timestamp > NOW() - INTERVAL '%s hours';
                """ % hours)

                feedback_row = await conn.fetchrow("""
                    SELECT
                        COUNT(CASE WHEN feedback_score > 0 THEN 1 END) as positive,
                        COUNT(CASE WHEN feedback_score < 0 THEN 1 END) as negative
                    FROM rag_feedback
                    WHERE timestamp > NOW() - INTERVAL '%s hours';
                """ % hours)

                return {
                    "period_hours": hours,
                    "total_queries": row["total"] if row else 0,
                    "avg_retrieval_time_ms": round(row["avg_retrieval"] or 0, 2),
                    "avg_similarity": round(row["avg_sim"] or 0, 4),
                    "avg_response_time_ms": round(row["avg_response"] or 0, 2),
                    "avg_results_per_query": round(row["avg_results"] or 0, 2),
                    "empty_retrieval_rate": round(row["empty_retrieval_rate"] or 0, 4),
                    "citation_coverage": round(row["citation_coverage"] or 0, 4),
                    "cache_hit_rate": round(row["cache_hit_rate"] or 0, 4),
                    "total_tokens": row["total_tokens"] or 0,
                    "total_estimated_cost_usd": round(row["total_estimated_cost"] or 0, 6),
                    "errors": row["errors"] if row else 0,
                    "feedback_positive": feedback_row["positive"] if feedback_row else 0,
                    "feedback_negative": feedback_row["negative"] if feedback_row else 0,
                    "percentiles": {
                        "retrieval": {
                            "p50": round(row["p50_retrieval"] or 0, 1),
                            "p95": round(row["p95_retrieval"] or 0, 1),
                            "p99": round(row["p99_retrieval"] or 0, 1),
                        },
                        "response": {
                            "p50": round(row["p50_response"] or 0, 1),
                            "p95": round(row["p95_response"] or 0, 1),
                            "p99": round(row["p99_response"] or 0, 1),
                        },
                    },
                }
        except Exception as e:
            logger.warning(f"DB summary query failed: {e}")
            return self.get_summary()


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------
_manager: Optional[MetricsManager] = None


def get_metrics_manager() -> MetricsManager:
    """Return the singleton MetricsManager."""
    global _manager
    if _manager is None:
        _manager = MetricsManager()
    return _manager
