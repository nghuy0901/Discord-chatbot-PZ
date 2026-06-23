"""
Self-RAG — LLM-based relevance grading for retrieval results.

Implements the Self-RAG paradigm where the LLM evaluates the relevance of
each retrieved document BEFORE using it in the final response. This reduces
hallucinations and improves answer quality by:

1. **Relevance grading**: Score each retrieved chunk as RELEVANT / IRRELEVANT
2. **Grounded filtering**: Remove retrieval noise before prompt injection
3. **Confidence tagging**: Add relevance metadata for transparency
4. **Adaptive behavior**: If all results are graded irrelevant, fall back
   to general knowledge with explicit disclaimer

The grading uses the configured LLM provider with a specialized prompt.

Performance notes:
- Grading is batched to minimize LLM calls
- Only the top-N results are graded (configurable)
- Grading is skipped if results are few or high-confidence
- Async execution keeps the pipeline responsive
"""

import os
import json
import time
import logging
import asyncio
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SELF_RAG_ENABLED: bool = os.getenv("SELF_RAG_ENABLED", "true").lower() == "true"
SELF_RAG_MAX_GRADE: int = int(os.getenv("SELF_RAG_MAX_GRADE", "10"))
SELF_RAG_MIN_RELEVANT: float = float(os.getenv("SELF_RAG_MIN_RELEVANT", "0.5"))
SELF_RAG_SKIP_THRESHOLD: float = float(os.getenv("SELF_RAG_SKIP_THRESHOLD", "0.7"))
SELF_RAG_TIMEOUT: int = int(os.getenv("SELF_RAG_TIMEOUT", "30"))  # seconds

# Grading mode: "batch" (one LLM call) or "individual" (N calls)
SELF_RAG_MODE: str = os.getenv("SELF_RAG_MODE", "batch")


# ---------------------------------------------------------------------------
# Grading prompt templates
# ---------------------------------------------------------------------------
BATCH_GRADING_PROMPT = """You are a relevance grading assistant. Your task is to evaluate whether each retrieved document is relevant to the user's query.

## User Query
{query}

## Retrieved Documents
{documents}

## Instructions
For each document, evaluate its relevance to the query on this scale:
- **RELEVANT** (score ≥ 0.6): The document contains information that directly helps answer the query
- **PARTIALLY_RELEVANT** (score 0.3-0.6): The document has some related information but is not directly helpful
- **IRRELEVANT** (score < 0.3): The document is not related to the query at all

Respond with ONLY a JSON array. For each document, provide:
- "index": the document number (starting from 1)
- "relevance": "RELEVANT", "PARTIALLY_RELEVANT", or "IRRELEVANT"
- "score": a float between 0.0 and 1.0
- "reason": a brief explanation (max 20 words)

Example response:
[
  {{"index": 1, "relevance": "RELEVANT", "score": 0.85, "reason": "Directly answers the question about crafting"}},
  {{"index": 2, "relevance": "IRRELEVANT", "score": 0.1, "reason": "Discusses unrelated game mechanics"}}
]

IMPORTANT: Return ONLY the JSON array, no other text."""


INDIVIDUAL_GRADING_PROMPT = """You are a relevance grading assistant.

## User Query
{query}

## Document
{document}

## Task
Is this document relevant to answering the user's query?

Respond with ONLY a JSON object:
{{"relevance": "RELEVANT" or "PARTIALLY_RELEVANT" or "IRRELEVANT", "score": 0.0-1.0, "reason": "brief explanation"}}

IMPORTANT: Return ONLY the JSON object, no other text."""


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class GradingResult:
    """Result of Self-RAG relevance grading."""
    index: int
    relevance: str = "UNKNOWN"  # RELEVANT, PARTIALLY_RELEVANT, IRRELEVANT
    score: float = 0.5
    reason: str = ""

    @property
    def is_relevant(self) -> bool:
        return self.relevance in ("RELEVANT", "PARTIALLY_RELEVANT") and self.score >= SELF_RAG_MIN_RELEVANT


@dataclass
class SelfRAGMetadata:
    """Metadata about Self-RAG grading process."""
    enabled: bool = True
    mode: str = "batch"
    total_graded: int = 0
    relevant_count: int = 0
    irrelevant_count: int = 0
    partial_count: int = 0
    skipped: bool = False
    skip_reason: str = ""
    grading_time_ms: float = 0.0
    error: Optional[str] = None
    grades: List[Dict[str, Any]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Grading logic
# ---------------------------------------------------------------------------
async def grade_relevance(
    query: str,
    results: List[Dict[str, Any]],
    max_grade: int = SELF_RAG_MAX_GRADE,
    mode: str = SELF_RAG_MODE,
) -> Tuple[List[Dict[str, Any]], SelfRAGMetadata]:
    """
    Grade the relevance of retrieval results using the LLM.

    Args:
        query: The user's original query.
        results: List of retrieval results to grade.
        max_grade: Maximum number of results to grade.
        mode: "batch" or "individual" grading.

    Returns:
        Tuple of (filtered_results, metadata)
        filtered_results contains only relevant documents, annotated with grades.
    """
    metadata = SelfRAGMetadata(
        enabled=SELF_RAG_ENABLED,
        mode=mode,
    )

    if not SELF_RAG_ENABLED:
        metadata.enabled = False
        metadata.skipped = True
        metadata.skip_reason = "Self-RAG disabled"
        return results, metadata

    if not results:
        metadata.skipped = True
        metadata.skip_reason = "No results to grade"
        return results, metadata

    # Skip grading if results are already high-confidence
    if _should_skip_grading(results):
        metadata.skipped = True
        metadata.skip_reason = "All results already high-confidence"
        return results, metadata

    # Limit the number of documents to grade
    to_grade = results[:max_grade]
    remaining = results[max_grade:]

    start_time = time.time()

    try:
        if mode == "batch":
            grades = await _grade_batch(query, to_grade)
        else:
            grades = await _grade_individual(query, to_grade)

        metadata.grading_time_ms = (time.time() - start_time) * 1000
        metadata.total_graded = len(grades)

        # Apply grades and filter
        filtered = []
        for i, result in enumerate(to_grade):
            if i < len(grades):
                grade = grades[i]
                result["self_rag_relevance"] = grade.relevance
                result["self_rag_score"] = grade.score
                result["self_rag_reason"] = grade.reason

                metadata.grades.append({
                    "index": i + 1,
                    "relevance": grade.relevance,
                    "score": grade.score,
                    "reason": grade.reason,
                })

                if grade.relevance == "RELEVANT":
                    metadata.relevant_count += 1
                elif grade.relevance == "PARTIALLY_RELEVANT":
                    metadata.partial_count += 1
                else:
                    metadata.irrelevant_count += 1

                if grade.is_relevant:
                    filtered.append(result)
            else:
                # Ungraded → include by default
                filtered.append(result)

        # Add remaining (ungraded) results at the end
        filtered.extend(remaining)

        logger.info(
            f"Self-RAG: {metadata.relevant_count} relevant, "
            f"{metadata.partial_count} partial, "
            f"{metadata.irrelevant_count} irrelevant "
            f"({metadata.grading_time_ms:.0f}ms)"
        )

        return filtered, metadata

    except Exception as e:
        metadata.error = str(e)
        metadata.grading_time_ms = (time.time() - start_time) * 1000
        logger.warning(f"Self-RAG grading failed (returning unfiltered): {e}")
        return results, metadata


def _should_skip_grading(results: List[Dict[str, Any]]) -> bool:
    """
    Check if grading can be safely skipped because results are
    already high-confidence.
    """
    if len(results) <= 2:
        return False  # Always grade when few results

    # Check if all results have high similarity scores
    similarities = [
        r.get("similarity", r.get("rrf_score", 0)) for r in results
    ]
    if not similarities:
        return False

    avg_sim = sum(similarities) / len(similarities)
    return avg_sim >= SELF_RAG_SKIP_THRESHOLD


async def _grade_batch(
    query: str,
    results: List[Dict[str, Any]],
) -> List[GradingResult]:
    """
    Grade all results in a single LLM call (efficient but less precise).
    """
    # Format documents for the prompt
    doc_lines = []
    for i, r in enumerate(results, 1):
        content = r.get("content", "")[:500]
        source = r.get("source", r.get("author_name", "unknown"))
        similarity = r.get("similarity", r.get("rrf_score", 0))
        doc_lines.append(
            f"[Document {i}] (similarity: {similarity:.2f}, source: {source})\n{content}"
        )

    documents_text = "\n\n".join(doc_lines)

    prompt = BATCH_GRADING_PROMPT.format(
        query=query,
        documents=documents_text,
    )

    try:
        from src.ollama_provider import chat_completion

        response = await asyncio.wait_for(
            chat_completion(
                messages=[
                    {"role": "system", "content": "You are a precise relevance grading assistant. Output only valid JSON."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,  # Low temperature for consistent grading
            ),
            timeout=SELF_RAG_TIMEOUT,
        )

        grades = _parse_batch_response(response, len(results))
        return grades

    except asyncio.TimeoutError:
        logger.warning(f"Self-RAG batch grading timed out after {SELF_RAG_TIMEOUT}s")
        return [GradingResult(index=i, relevance="RELEVANT", score=0.5, reason="Grading timed out")
                for i in range(len(results))]
    except Exception as e:
        logger.warning(f"Self-RAG batch grading error: {e}")
        return [GradingResult(index=i, relevance="RELEVANT", score=0.5, reason="Grading error")
                for i in range(len(results))]


async def _grade_individual(
    query: str,
    results: List[Dict[str, Any]],
) -> List[GradingResult]:
    """
    Grade each result individually (more precise but slower).
    """
    from src.ollama_provider import chat_completion

    async def _grade_one(idx: int, result: Dict[str, Any]) -> GradingResult:
        content = result.get("content", "")[:500]
        prompt = INDIVIDUAL_GRADING_PROMPT.format(
            query=query,
            document=content,
        )
        try:
            response = await asyncio.wait_for(
                chat_completion(
                    messages=[
                        {"role": "system", "content": "You are a relevance grading assistant. Output only valid JSON."},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.1,
                ),
                timeout=SELF_RAG_TIMEOUT // max(len(results), 1),
            )
            return _parse_individual_response(response, idx)
        except Exception as e:
            logger.debug(f"Individual grading failed for doc {idx}: {e}")
            return GradingResult(index=idx, relevance="RELEVANT", score=0.5, reason="Grading failed")

    # Run all gradings concurrently
    tasks = [_grade_one(i, r) for i, r in enumerate(results)]
    grades = await asyncio.gather(*tasks)
    return list(grades)


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------
def _parse_batch_response(response: str, expected_count: int) -> List[GradingResult]:
    """Parse the batch grading LLM response into GradingResult objects."""
    # Try to extract JSON array from the response
    response = response.strip()

    # Handle markdown code blocks
    if "```" in response:
        # Extract content between code blocks
        import re
        match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", response, re.DOTALL)
        if match:
            response = match.group(1).strip()

    try:
        grades_data = json.loads(response)
    except json.JSONDecodeError:
        # Try to find JSON array in the response
        import re
        match = re.search(r"\[.*\]", response, re.DOTALL)
        if match:
            try:
                grades_data = json.loads(match.group())
            except json.JSONDecodeError:
                logger.warning("Self-RAG: Could not parse batch grading response")
                return [
                    GradingResult(index=i, relevance="RELEVANT", score=0.5, reason="Parse failed")
                    for i in range(expected_count)
                ]
        else:
            return [
                GradingResult(index=i, relevance="RELEVANT", score=0.5, reason="Parse failed")
                for i in range(expected_count)
            ]

    if not isinstance(grades_data, list):
        grades_data = [grades_data]

    results = []
    for i in range(expected_count):
        if i < len(grades_data):
            g = grades_data[i]
            relevance = str(g.get("relevance", "RELEVANT")).upper()
            if relevance not in ("RELEVANT", "PARTIALLY_RELEVANT", "IRRELEVANT"):
                relevance = "RELEVANT"
            score = float(g.get("score", 0.5))
            score = max(0.0, min(1.0, score))
            reason = str(g.get("reason", ""))[:200]
            results.append(GradingResult(
                index=i,
                relevance=relevance,
                score=score,
                reason=reason,
            ))
        else:
            results.append(GradingResult(
                index=i,
                relevance="RELEVANT",
                score=0.5,
                reason="Not graded (missing in response)",
            ))

    return results


def _parse_individual_response(response: str, index: int) -> GradingResult:
    """Parse an individual grading LLM response."""
    response = response.strip()

    # Handle markdown code blocks
    if "```" in response:
        import re
        match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", response, re.DOTALL)
        if match:
            response = match.group(1).strip()

    try:
        data = json.loads(response)
    except json.JSONDecodeError:
        import re
        match = re.search(r"\{.*\}", response, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group())
            except json.JSONDecodeError:
                return GradingResult(index=index, relevance="RELEVANT", score=0.5, reason="Parse failed")
        else:
            return GradingResult(index=index, relevance="RELEVANT", score=0.5, reason="Parse failed")

    relevance = str(data.get("relevance", "RELEVANT")).upper()
    if relevance not in ("RELEVANT", "PARTIALLY_RELEVANT", "IRRELEVANT"):
        relevance = "RELEVANT"

    score = float(data.get("score", 0.5))
    score = max(0.0, min(1.0, score))
    reason = str(data.get("reason", ""))[:200]

    return GradingResult(
        index=index,
        relevance=relevance,
        score=score,
        reason=reason,
    )


# ---------------------------------------------------------------------------
# Format Self-RAG annotations for prompt
# ---------------------------------------------------------------------------
def format_self_rag_annotation(metadata: SelfRAGMetadata) -> str:
    """
    Create a brief annotation about Self-RAG grading for the system prompt.
    This helps the LLM understand the confidence level of retrieved data.
    """
    if not metadata.enabled or metadata.skipped:
        return ""

    total = metadata.total_graded
    if total == 0:
        return ""

    relevant_pct = (
        (metadata.relevant_count + metadata.partial_count) / total * 100
        if total > 0 else 0
    )

    lines = [
        f"[Self-RAG Quality Assessment: {relevant_pct:.0f}% of retrieved documents graded as relevant]"
    ]

    if metadata.irrelevant_count > 0:
        lines.append(
            f"[{metadata.irrelevant_count} irrelevant documents were filtered out]"
        )

    if relevant_pct < 50:
        lines.append(
            "[⚠️ Low retrieval confidence — retrieved data may not fully answer the query. "
            "Use your general knowledge as supplement and clearly indicate when doing so.]"
        )

    return "\n".join(lines)
