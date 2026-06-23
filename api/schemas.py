from pydantic import BaseModel


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
    cache_hit_rate: float
    total_tokens: int
    estimated_cost_usd: float
