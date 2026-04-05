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

    # Query info
    original_query: str = ""
    processed_query: str = ""
    query_language: str = ""
    detected_domain: Optional[str] = None

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

    # Response
    response_time_ms: float = 0.0
    response_length: int = 0

    # Feedback
    feedback_score: Optional[int] = None  # 1=👍, -1=👎
    feedback_user_id: Optional[str] = None

    # Errors
    error: Optional[str] = None

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
        """Create the rag_metrics and rag_feedback tables if they don't exist."""
        try:
            from rag.db import get_pool
            pool = await get_pool()
            async with pool.acquire() as conn:
                await conn.execute("""
                    CREATE TABLE IF NOT EXISTS rag_metrics (
                        id              BIGSERIAL PRIMARY KEY,
                        query_id        TEXT UNIQUE NOT NULL,
                        timestamp       TIMESTAMPTZ DEFAULT NOW(),
                        channel_id      TEXT,
                        user_id         TEXT,
                        original_query  TEXT,
                        processed_query TEXT,
                        query_language  TEXT,
                        detected_domain TEXT,
                        retrieval_time_ms FLOAT,
                        num_results     INT,
                        avg_similarity  FLOAT,
                        max_similarity  FLOAT,
                        min_similarity  FLOAT,
                        kb_results      INT DEFAULT 0,
                        chat_history_results INT DEFAULT 0,
                        response_time_ms FLOAT,
                        response_length INT,
                        error           TEXT
                    );
                """)
                await conn.execute("""
                    CREATE TABLE IF NOT EXISTS rag_feedback (
                        id              BIGSERIAL PRIMARY KEY,
                        query_id        TEXT REFERENCES rag_metrics(query_id),
                        message_id      TEXT NOT NULL,
                        user_id         TEXT NOT NULL,
                        feedback_score  INT NOT NULL,
                        timestamp       TIMESTAMPTZ DEFAULT NOW(),
                        UNIQUE(message_id, user_id)
                    );
                """)
            self._db_initialized = True
            logger.info("✅ RAG metrics tables initialized.")
        except Exception as e:
            logger.warning(f"⚠️ Metrics DB init failed (non-fatal): {e}")

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
                await conn.execute("""
                    INSERT INTO rag_metrics (
                        query_id, channel_id, user_id,
                        original_query, processed_query,
                        query_language, detected_domain,
                        retrieval_time_ms, num_results,
                        avg_similarity, max_similarity, min_similarity,
                        kb_results, chat_history_results,
                        response_time_ms, response_length, error
                    ) VALUES (
                        $1, $2, $3, $4, $5, $6, $7, $8, $9,
                        $10, $11, $12, $13, $14, $15, $16, $17
                    ) ON CONFLICT (query_id) DO NOTHING;
                """,
                    metric.query_id, metric.channel_id, metric.user_id,
                    metric.original_query[:500], metric.processed_query[:500],
                    metric.query_language, metric.detected_domain,
                    metric.retrieval_time_ms, metric.num_results,
                    metric.avg_similarity, metric.max_similarity, metric.min_similarity,
                    metric.kb_results, metric.chat_history_results,
                    metric.response_time_ms, metric.response_length, metric.error,
                )
        except Exception as e:
            logger.debug(f"Metric persist failed: {e}")

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
                        COUNT(CASE WHEN error IS NOT NULL THEN 1 END) as errors
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
                    "errors": row["errors"] if row else 0,
                    "feedback_positive": feedback_row["positive"] if feedback_row else 0,
                    "feedback_negative": feedback_row["negative"] if feedback_row else 0,
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
