import math
import re
from typing import Callable, Dict, Iterable, List, Set


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
    # Standard precision@k divides relevant hits by k (not by len(retrieved)) so
    # an under-retrieving system isn't credited with inflated precision (M11).
    return len(set(retrieved) & expected) / k


def mean_reciprocal_rank(retrieved_sources: List[str], expected_sources: Set[str]) -> float:
    expected = set(_normalize_sources(expected_sources))
    for index, source in enumerate(_normalize_sources(retrieved_sources), start=1):
        if source in expected:
            return 1.0 / index
    return 0.0


def ndcg_at_k(retrieved_sources: List[str], expected_sources: Set[str], k: int) -> float:
    expected = set(_normalize_sources(expected_sources))
    # Dedup preserving first occurrence so a duplicated source can't double-count
    # gain and push nDCG above the ideal (M11).
    seen: Set[str] = set()
    retrieved = []
    for source in _normalize_sources(retrieved_sources):
        if source not in seen:
            seen.add(source)
            retrieved.append(source)
    retrieved = retrieved[:k]
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
    # Word-boundary match so "axe" no longer matches "axed"/"relaxed" — a naive
    # substring check was trivially inflated by incidental substrings (M11).
    hits = sum(
        1 for keyword in keywords
        if re.search(r"\b" + re.escape(keyword) + r"\b", text)
    )
    return hits / len(keywords)


def _rate(items: List[tuple[str, str]], predicate: Callable[[tuple[str, str]], bool]) -> float:
    if not items:
        return 0.0
    return sum(1 for item in items if predicate(item)) / len(items)


def behavior_confusion(expected: List[str], actual: List[str]) -> Dict[str, float]:
    pairs = list(zip(expected, actual))
    unanswerable = [pair for pair in pairs if pair[0] == "abstain"]
    answerable = [pair for pair in pairs if pair[0] == "answer"]
    ambiguous = [pair for pair in pairs if pair[0] == "clarify"]
    return {
        "correct_abstention_rate": _rate(unanswerable, lambda p: p[1] == "abstain"),
        "false_answer_rate": _rate(unanswerable, lambda p: p[1] == "answer"),
        "false_abstention_rate": _rate(answerable, lambda p: p[1] == "abstain"),
        "clarification_accuracy": _rate(ambiguous, lambda p: p[1] == "clarify"),
    }


def unapproved_source_leakage_rate(provenance_items: List[List[dict]]) -> float:
    if not provenance_items:
        return 0.0
    leaked = 0
    for items in provenance_items:
        if any(item.get("trusted") is False for item in items):
            leaked += 1
    return leaked / len(provenance_items)


def provenance_coverage(decisions: List[str], provenance_items: List[List[dict]]) -> float:
    answer_indexes = [idx for idx, decision in enumerate(decisions) if decision == "answer"]
    if not answer_indexes:
        return 0.0
    covered = sum(1 for idx in answer_indexes if idx < len(provenance_items) and provenance_items[idx])
    return covered / len(answer_indexes)


def runtime_error_rate(errors: List[str]) -> float:
    if not errors:
        return 0.0
    return sum(1 for error in errors if error) / len(errors)


def latency_percentiles(latencies_ms: List[float]) -> Dict[str, float]:
    values = sorted(float(value) for value in latencies_ms if value is not None)
    if not values:
        return {"p50_latency_ms": 0.0, "p95_latency_ms": 0.0, "p99_latency_ms": 0.0}

    def percentile(pct: float) -> float:
        if len(values) == 1:
            return values[0]
        rank = (len(values) - 1) * pct
        lower = math.floor(rank)
        upper = math.ceil(rank)
        if lower == upper:
            return values[int(rank)]
        weight = rank - lower
        return values[lower] * (1 - weight) + values[upper] * weight

    return {
        "p50_latency_ms": round(percentile(0.50), 2),
        "p95_latency_ms": round(percentile(0.95), 2),
        "p99_latency_ms": round(percentile(0.99), 2),
    }
