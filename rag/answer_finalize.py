"""
Answer finalization — the single place where a generated RAG answer is made
safe and citable before it reaches the user.

Two responsibilities:

1. **Groundedness gate** — verify the answer is supported by the trusted sources.
   If not, override the decision to ABSTAIN and return the canned refusal so an
   ungrounded (hallucinated) answer is never shown.
2. **Citations** — append a user-facing "Sources" footer mapping the [n] markers
   the answer used back to specific chunks/passages.

Both consumers (Discord bot, FastAPI) call :func:`finalize_rag_answer` after
generation so the behavior stays consistent.
"""

import os
import logging
from dataclasses import dataclass
from typing import Any, Optional

from rag.result import RAGDecision
from rag.responses import decision_response
from rag.citations import render_sources_footer, validate_factual_line_citations
from rag.groundedness import (
    check_groundedness,
    sources_from_tool_results,
    GroundednessResult,
    Judge,
)

logger = logging.getLogger(__name__)

SHOW_CITATIONS: bool = os.getenv("RAG_SHOW_CITATIONS", "true").lower() == "true"


@dataclass
class FinalizedAnswer:
    text: str                       # answer (+footer) or the refusal message
    footer: str = ""                # the sources footer, if any
    overridden: bool = False        # True when groundedness forced an ABSTAIN
    decision: RAGDecision = RAGDecision.ANSWER
    groundedness: Optional[GroundednessResult] = None


def _language(rag_result: Any) -> str:
    lang = getattr(getattr(rag_result, "metric", None), "query_language", "") or ""
    return "vi" if lang != "en" else "en"


async def finalize_rag_answer(
    rag_result: Any,
    query: str,
    answer_text: str,
    *,
    judge: Optional[Judge] = None,
) -> FinalizedAnswer:
    """Apply the groundedness gate and attach citations to ``answer_text``.

    Only meaningful for answered RAG turns; for any other case it returns the
    answer unchanged.
    """
    if (
        rag_result is None
        or getattr(rag_result, "decision", None) is not RAGDecision.ANSWER
    ):
        return FinalizedAnswer(text=answer_text, decision=RAGDecision.ANSWER)

    language = _language(rag_result)
    metric = getattr(rag_result, "metric", None)

    if not answer_text or not answer_text.strip():
        if metric is not None:
            metric.rag_decision = RAGDecision.ABSTAIN.value
            metric.decision_reason = "empty_generation"
            metric.groundedness_score = 0.0
            metric.groundedness_reason = "empty_generation"
        return FinalizedAnswer(
            text=decision_response(
                language,
                RAGDecision.ABSTAIN,
                event="finalize_rag_empty_generation",
                query=query,
                rag_result=rag_result,
                reason="empty_generation",
            ),
            overridden=True,
            decision=RAGDecision.ABSTAIN,
        )

    if getattr(metric, "query_intent", "") == "conversation":
        metric.groundedness_reason = "not_applicable_conversation"
        return FinalizedAnswer(
            text=answer_text,
            decision=RAGDecision.ANSWER,
            groundedness=GroundednessResult(
                skipped=True,
                reason="not_applicable_conversation",
            ),
        )

    sources = list(getattr(rag_result, "retrieved_results", []) or [])

    # ---- 1. Deterministic citation coverage + groundedness gate ----
    citation_ok, uncited_lines = validate_factual_line_citations(
        answer_text,
        list(getattr(rag_result, "provenance", []) or []),
    )
    if not citation_ok:
        grounded = GroundednessResult(
            grounded=False,
            score=0.0,
            unsupported=uncited_lines,
            reason="citation_coverage_failed",
        )
    else:
        grounded = await check_groundedness(query, answer_text, sources, judge=judge)
    if metric is not None:
        try:
            metric.groundedness_score = grounded.score
            metric.groundedness_reason = grounded.reason
            metric.groundedness_unsupported_count = len(grounded.unsupported)
            metric.groundedness_error = grounded.error
        except Exception:
            pass

    if not grounded.grounded and not grounded.skipped:
        if metric is not None:
            metric.rag_decision = RAGDecision.ABSTAIN.value
            metric.decision_reason = "ungrounded_answer"
        refusal = decision_response(
            language,
            RAGDecision.ABSTAIN,
            event="finalize_rag_ungrounded_answer",
            query=query,
            rag_result=rag_result,
            reason="ungrounded_answer",
        )
        logger.info(
            f"Groundedness override → ABSTAIN (score={grounded.score:.2f}, "
            f"unsupported={len(grounded.unsupported)})"
        )
        return FinalizedAnswer(
            text=refusal,
            overridden=True,
            decision=RAGDecision.ABSTAIN,
            groundedness=grounded,
        )

    # ---- 2. Citations ----
    footer = ""
    if SHOW_CITATIONS:
        footer = render_sources_footer(
            answer_text,
            list(getattr(rag_result, "provenance", []) or []),
            language=language,
        )

    text = f"{answer_text}\n\n{footer}" if footer else answer_text
    return FinalizedAnswer(
        text=text,
        footer=footer,
        decision=RAGDecision.ANSWER,
        groundedness=grounded,
    )


async def finalize_tool_answer(
    query: str,
    answer_text: str,
    tool_results: Any,
    *,
    language: str = "vi",
    judge: Optional[Judge] = None,
) -> FinalizedAnswer:
    """Verify a tool-calling answer against the actual tool outputs (audit C2).

    Tool answers are not part of the vector provenance, so the normal
    groundedness gate skips them. This grounds the answer against the tool
    results instead; if the model drifted beyond what the tools returned, the
    decision is overridden to ABSTAIN with the canned refusal.
    """
    if not answer_text or not answer_text.strip():
        return FinalizedAnswer(
            text=decision_response(
                language,
                RAGDecision.ABSTAIN,
                event="finalize_tool_empty_generation",
                query=query,
                reason="empty_generation",
            ),
            overridden=True,
            decision=RAGDecision.ABSTAIN,
        )

    sources = sources_from_tool_results(tool_results or [])
    if not sources:
        # Tool-routed answers are only safe when at least one tool result was
        # actually captured. Otherwise the model may claim it queried the DB
        # while answering from unsupported priors.
        return FinalizedAnswer(
            text=decision_response(
                language,
                RAGDecision.ABSTAIN,
                event="finalize_tool_missing_tool_results",
                query=query,
                reason="tool_routed_answer_without_tool_results",
            ),
            overridden=True,
            decision=RAGDecision.ABSTAIN,
        )

    grounded = await check_groundedness(query, answer_text, sources, judge=judge)
    if not grounded.grounded and not grounded.skipped:
        return FinalizedAnswer(
            text=decision_response(
                language,
                RAGDecision.ABSTAIN,
                event="finalize_tool_ungrounded_answer",
                query=query,
                reason="tool_answer_ungrounded",
            ),
            overridden=True,
            decision=RAGDecision.ABSTAIN,
            groundedness=grounded,
        )
    return FinalizedAnswer(text=answer_text, groundedness=grounded)
