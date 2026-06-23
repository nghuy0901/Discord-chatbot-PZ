import math
from typing import Iterable, List, Set


def _normalize_sources(sources: Iterable[str]) -> List[str]:
    return [str(source).strip().lower() for source in sources if str(source).strip()]


def recall_at_k(retrieved_sources: List[str], expected_sources: Set[str], k: int) -> float:
    expected = set(_normalize_sources(expected_sources))
    if not expected:
        return 0.0
    retrieved = set(_normalize_sources(retrieved_sources[:k]))
    return len(retrieved & expected) / len(expected)


def precision_at_k(retrieved_sources: List[str], expected_sources: Set[str], k: int) -> float:
    if k <= 0:
        return 0.0
    expected = set(_normalize_sources(expected_sources))
    retrieved = _normalize_sources(retrieved_sources[:k])
    if not retrieved:
        return 0.0
    return len(set(retrieved) & expected) / min(k, len(retrieved))


def mean_reciprocal_rank(retrieved_sources: List[str], expected_sources: Set[str]) -> float:
    expected = set(_normalize_sources(expected_sources))
    for index, source in enumerate(_normalize_sources(retrieved_sources), start=1):
        if source in expected:
            return 1.0 / index
    return 0.0


def ndcg_at_k(retrieved_sources: List[str], expected_sources: Set[str], k: int) -> float:
    expected = set(_normalize_sources(expected_sources))
    retrieved = _normalize_sources(retrieved_sources[:k])
    dcg = 0.0
    for index, source in enumerate(retrieved, start=1):
        relevance = 1.0 if source in expected else 0.0
        dcg += relevance / math.log2(index + 1)
    ideal_hits = min(len(expected), k)
    idcg = sum(1.0 / math.log2(index + 1) for index in range(1, ideal_hits + 1))
    return dcg / idcg if idcg > 0 else 0.0


def source_hit_rate(retrieved_sources: List[str], expected_sources: Set[str]) -> float:
    retrieved = set(_normalize_sources(retrieved_sources))
    expected = set(_normalize_sources(expected_sources))
    return 1.0 if retrieved & expected else 0.0


def keyword_coverage(answer_or_context: str, expected_keywords: List[str]) -> float:
    keywords = [keyword.lower() for keyword in expected_keywords if keyword]
    if not keywords:
        return 0.0
    text = answer_or_context.lower()
    hits = sum(1 for keyword in keywords if keyword in text)
    return hits / len(keywords)
