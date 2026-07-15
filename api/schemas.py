from typing import List, Literal

from pydantic import BaseModel


class Citation(BaseModel):
    """A chunk-level citation that a [n] marker in the answer resolves to."""
    label: str = ""
    locator: str = ""
    source: str = ""
    heading_path: str = ""
    excerpt: str = ""
    url: str = ""
    similarity: float = 0.0


class QueryResponse(BaseModel):
    query: str
    query_id: str
    response: str
    decision: Literal["answer", "clarify", "abstain"]
    cache_hit: bool
    latency_ms: float
    source_count: int
    sources: List[str] = []
    citations: List[Citation] = []
    metrics: dict


class MetricsSummary(BaseModel):
    total_queries: int
    success_rate: float
    error_rate: float
    avg_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    retrieval_latency_ms: float
    llm_latency_ms: float
    empty_retrieval_rate: float
    average_retrieved_chunks: float
    citation_coverage: float
    average_groundedness_score: float = 0.0
    groundedness_failure_rate: float = 0.0
    cache_hit_rate: float
    total_tokens: int
    estimated_cost_usd: float
