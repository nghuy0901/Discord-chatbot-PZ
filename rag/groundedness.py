"""
Groundedness — post-generation faithfulness check.

After the LLM produces an answer, this verifies that the answer is actually
supported by the trusted retrieved chunks. If it is not (the model drifted into
unsupported / hallucinated territory), the caller can override the decision to
ABSTAIN instead of sending an ungrounded answer.

Design notes:
- Gated by ``RAG_GROUNDEDNESS_ENABLED`` (default on).
- Uses an injectable ``judge`` coroutine so it is unit-testable without an LLM;
  in production it defaults to the configured chat provider.
- **Fail-open**: on timeout / parse / provider error it returns ``grounded=True``
  (configurable via ``RAG_GROUNDEDNESS_FAIL_OPEN``) so infrastructure hiccups
  degrade to "answer" rather than blanket refusal.
"""

import os
import re
import json
import time
import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, List, Optional, Sequence

logger = logging.getLogger(__name__)

GROUNDEDNESS_ENABLED: bool = (
    os.getenv("RAG_GROUNDEDNESS_ENABLED", "true").lower() == "true"
)
GROUNDEDNESS_MIN_SCORE: float = float(os.getenv("RAG_GROUNDEDNESS_MIN_SCORE", "0.6"))
GROUNDEDNESS_TIMEOUT: int = int(os.getenv("RAG_GROUNDEDNESS_TIMEOUT", "20"))
GROUNDEDNESS_FAIL_OPEN: bool = (
    os.getenv("RAG_GROUNDEDNESS_FAIL_OPEN", "true").lower() == "true"
)
# A judge TIMEOUT is the dangerous failure mode — it happens precisely when the
# system is under load. Fail CLOSED on timeout by default so we abstain rather
# than ship an unverified (possibly hallucinated) answer (audit H6).
GROUNDEDNESS_TIMEOUT_FAIL_OPEN: bool = (
    os.getenv("RAG_GROUNDEDNESS_TIMEOUT_FAIL_OPEN", "false").lower() == "true"
)
GROUNDEDNESS_MAX_SOURCES: int = int(os.getenv("RAG_GROUNDEDNESS_MAX_SOURCES", "8"))
GROUNDEDNESS_SOURCE_CHARS: int = int(os.getenv("RAG_GROUNDEDNESS_SOURCE_CHARS", "600"))

Judge = Callable[[str], Awaitable[str]]

GROUNDEDNESS_PROMPT = """You are a strict fact-checking assistant. Decide whether the ANSWER is fully supported by the SOURCES. An answer is grounded only if every factual claim it makes can be verified from the sources. General conversational filler is fine, but specific facts must be supported.

Apply these rules strictly:
- A claim carrying marker [n] must be explicitly supported by source [n], not merely by some other source.
- Missing, blank, "-", or unspecified source fields are NOT evidence that a property is absent or that the opposite is true.
- Do not accept plausible game knowledge, common sense, or an inference as support.
- If even one material factual claim is not explicit in its cited source, set grounded=false and list it.

## User Query
{query}

## Sources
{sources}

## Answer
{answer}

Respond with ONLY a JSON object:
{{"grounded": true or false, "score": 0.0-1.0, "unsupported_claims": ["..."]}}

- "score" is the fraction of the answer's factual content supported by the sources.
- List any claims that are NOT supported by the sources in "unsupported_claims".
IMPORTANT: Return ONLY the JSON object, no other text."""


@dataclass
class GroundednessResult:
    grounded: bool = True
    score: float = 1.0
    unsupported: List[str] = field(default_factory=list)
    skipped: bool = False
    reason: str = ""
    error: Optional[str] = None
    latency_ms: float = 0.0


def _format_sources(sources: Sequence[Dict[str, Any]]) -> str:
    lines: List[str] = []
    for i, src in enumerate(sources[:GROUNDEDNESS_MAX_SOURCES], start=1):
        content = str(src.get("content", ""))[:GROUNDEDNESS_SOURCE_CHARS]
        locator = (
            src.get("source")
            or src.get("heading_path")
            or src.get("author_name")
            or src.get("message_id")
            or "source"
        )
        lines.append(f"[{i}] ({locator})\n{content}")
    return "\n\n".join(lines)


def sources_from_tool_results(
    tool_results: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Turn executed tool-call results into groundedness sources.

    Lets a tool-calling answer be verified against the *actual* tool output
    (audit C2), since tool answers are not part of the vector provenance. Each
    input item is ``{"name": <tool>, "output": <json/str>}``.
    """
    sources: List[Dict[str, Any]] = []
    for i, tr in enumerate(tool_results or [], start=1):
        name = str(tr.get("name") or f"tool_{i}")
        output = tr.get("output")
        content = (
            output
            if isinstance(output, str)
            else json.dumps(output, ensure_ascii=False, default=str)
        )
        try:
            parsed = json.loads(content)
        except (json.JSONDecodeError, TypeError):
            parsed = None
        if isinstance(parsed, dict) and parsed.get("error"):
            continue
        sources.append(
            {"content": content, "source": f"tool:{name}", "source_kind": "tool"}
        )
    return sources


async def _default_judge(prompt: str) -> str:
    from src.ollama_provider import chat_completion

    return await chat_completion(
        messages=[
            {
                "role": "system",
                "content": "You are a precise fact-checking assistant. Output only valid JSON.",
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0.0,
    )


def _parse(response: str) -> Dict[str, Any]:
    text = (response or "").strip()
    if "```" in text:
        match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
        if match:
            text = match.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            return json.loads(match.group())
        raise


async def check_groundedness(
    query: str,
    answer: str,
    sources: Sequence[Dict[str, Any]],
    *,
    judge: Optional[Judge] = None,
    min_score: float = GROUNDEDNESS_MIN_SCORE,
) -> GroundednessResult:
    """Verify ``answer`` is supported by ``sources``.

    Returns a :class:`GroundednessResult`; ``grounded`` is ``False`` only when the
    judge confidently reports insufficient support. Errors fail open (or closed,
    per ``RAG_GROUNDEDNESS_FAIL_OPEN``).
    """
    if not GROUNDEDNESS_ENABLED:
        return GroundednessResult(skipped=True, reason="disabled")
    if not answer or not answer.strip():
        return GroundednessResult(skipped=True, reason="empty_answer")
    if not sources:
        # Nothing to ground against — let the evidence policy own this case.
        return GroundednessResult(skipped=True, reason="no_sources")

    prompt = GROUNDEDNESS_PROMPT.format(
        query=query,
        sources=_format_sources(sources),
        answer=answer[:4000],
    )
    run_judge = judge or _default_judge
    start = time.time()

    try:
        response = await asyncio.wait_for(
            run_judge(prompt), timeout=GROUNDEDNESS_TIMEOUT
        )
        data = _parse(response)
        score = max(0.0, min(1.0, float(data.get("score", 0.0))))
        unsupported = [str(c) for c in (data.get("unsupported_claims") or [])][:10]
        explicit = data.get("grounded")
        grounded = (
            bool(explicit) and score >= min_score
            if explicit is not None
            else score >= min_score
        )
        return GroundednessResult(
            grounded=grounded,
            score=score,
            unsupported=unsupported,
            reason="checked",
            latency_ms=(time.time() - start) * 1000,
        )
    except asyncio.TimeoutError:
        # Timeout → fail closed by default (H6): under load the judge times out
        # exactly when verification matters most; abstaining beats shipping an
        # unverified answer.
        logger.warning(
            f"Groundedness check timed out after {GROUNDEDNESS_TIMEOUT}s; "
            f"fail_open={GROUNDEDNESS_TIMEOUT_FAIL_OPEN}"
        )
        return GroundednessResult(
            grounded=GROUNDEDNESS_TIMEOUT_FAIL_OPEN,
            score=1.0 if GROUNDEDNESS_TIMEOUT_FAIL_OPEN else 0.0,
            reason="timeout_fail_open" if GROUNDEDNESS_TIMEOUT_FAIL_OPEN else "timeout_fail_closed",
            error="timeout",
            latency_ms=(time.time() - start) * 1000,
        )
    except Exception as exc:  # provider or parse failure
        logger.warning(f"Groundedness check failed ({exc}); fail_open={GROUNDEDNESS_FAIL_OPEN}")
        return GroundednessResult(
            grounded=GROUNDEDNESS_FAIL_OPEN,
            score=1.0 if GROUNDEDNESS_FAIL_OPEN else 0.0,
            reason="fail_open" if GROUNDEDNESS_FAIL_OPEN else "fail_closed",
            error=str(exc),
            latency_ms=(time.time() - start) * 1000,
        )
