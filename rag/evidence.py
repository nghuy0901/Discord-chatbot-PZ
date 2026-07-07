import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

from rag.result import RAGDecision
from rag.scoring import relevance_score
from rag.trust import filter_trusted_results, is_trusted_result


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


@dataclass(frozen=True)
class EvidenceAssessment:
    decision: RAGDecision
    reason: str
    score: float
    trusted_results: List[Dict[str, Any]]
    rejected_results: List[Dict[str, Any]]


class EvidencePolicy:
    """Decide whether the retrieved evidence is strong enough to answer.

    Instead of a binary "good enough / not good enough" threshold, this computes
    a *calibrated confidence* in [0, 1] from several independent signals:

    - vector / fused similarity of the best trusted result
    - agreement between lexical (BM25) and semantic (vector) retrieval
    - Self-RAG relevance grade
    - corroboration: how many independent trusted sources support the query
    - separation: the margin between the best and second-best source

    The confidence is then mapped to one of three decisions with thresholds that
    can be tuned per query intent (analytical queries demand stronger evidence
    than narrative ones). Trust filtering always happens first, so an unapproved
    source can never raise the confidence.
    """

    def __init__(
        self,
        min_self_rag_score: float | None = None,
        answer_min_confidence: float | None = None,
        clarify_min_confidence: float | None = None,
    ):
        self.min_self_rag_score = min_self_rag_score or float(
            os.getenv("RAG_EVIDENCE_MIN_SELF_RAG_SCORE", "0.70")
        )
        # Decision thresholds on the calibrated confidence score.
        self.answer_min_confidence = (
            answer_min_confidence
            if answer_min_confidence is not None
            else float(os.getenv("RAG_ANSWER_MIN_CONFIDENCE", "0.55"))
        )
        self.clarify_min_confidence = (
            clarify_min_confidence
            if clarify_min_confidence is not None
            else float(os.getenv("RAG_CLARIFY_MIN_CONFIDENCE", "0.35"))
        )
        # Floors / weights for the confidence model.
        self.agreement_floor = float(os.getenv("RAG_EVIDENCE_AGREEMENT_FLOOR", "0.80"))
        self.self_rag_floor = float(os.getenv("RAG_EVIDENCE_SELF_RAG_FLOOR", "0.75"))
        self.support_floor = float(os.getenv("RAG_EVIDENCE_SUPPORT_FLOOR", "0.40"))
        self.corroboration_weight = float(
            os.getenv("RAG_EVIDENCE_CORROBORATION_WEIGHT", "0.10")
        )
        self.separation_weight = float(
            os.getenv("RAG_EVIDENCE_SEPARATION_WEIGHT", "0.05")
        )
        self.analytical_confidence_bonus = float(
            os.getenv("RAG_ANALYTICAL_CONFIDENCE_BONUS", "0.05")
        )
        self.conversation_confidence_discount = float(
            os.getenv("RAG_CONVERSATION_CONFIDENCE_DISCOUNT", "0.10")
        )

    def assess(
        self,
        *,
        query: str,
        results: Sequence[Dict[str, Any]],
        recent_messages: Sequence[str],
        query_intent: Optional[str] = None,
    ) -> EvidenceAssessment:
        trusted = filter_trusted_results(results)
        rejected = [result for result in results if not is_trusted_result(result)]

        if self._needs_clarification(query, recent_messages):
            return EvidenceAssessment(
                RAGDecision.CLARIFY,
                "referential_or_underspecified_query",
                0.0,
                trusted,
                rejected,
            )

        if not trusted:
            return EvidenceAssessment(
                RAGDecision.ABSTAIN,
                "no_trusted_evidence",
                0.0,
                trusted,
                rejected,
            )

        confidence = self._confidence(trusted)
        answer_min, clarify_min = self._thresholds(query_intent)

        if confidence >= answer_min:
            return EvidenceAssessment(
                RAGDecision.ANSWER,
                "trusted_evidence_sufficient",
                confidence,
                trusted,
                rejected,
            )
        if confidence >= clarify_min:
            return EvidenceAssessment(
                RAGDecision.CLARIFY,
                "trusted_evidence_borderline",
                confidence,
                trusted,
                rejected,
            )
        return EvidenceAssessment(
            RAGDecision.ABSTAIN,
            "trusted_evidence_below_threshold",
            confidence,
            trusted,
            rejected,
        )

    # ------------------------------------------------------------------ #
    #  Confidence model
    # ------------------------------------------------------------------ #
    def _confidence(self, trusted: Sequence[Dict[str, Any]]) -> float:
        """Aggregate per-source strengths into a single confidence in [0, 1]."""
        strengths = sorted(
            (self._result_strength(item) for item in trusted), reverse=True
        )
        if not strengths:
            return 0.0

        top1 = strengths[0]
        top2 = strengths[1] if len(strengths) > 1 else 0.0

        # Corroboration: independent trusted sources that clear the support floor.
        support = sum(1 for s in strengths if s >= self.support_floor)
        corroboration = (
            min(max(support - 1, 0), 3) / 3.0 * self.corroboration_weight
        )

        # Separation: a clear winner is more trustworthy than a tie.
        separation = max(top1 - top2, 0.0) * self.separation_weight

        return _clamp(top1 + corroboration + separation)

    def _result_strength(self, result: Dict[str, Any]) -> float:
        """Strength of a single trusted source in [0, 1]."""
        similarity = relevance_score(result)
        methods = {str(m).lower() for m in (result.get("retrieval_methods") or [])}
        self_rag_score = _clamp(float(result.get("self_rag_score", 0.0) or 0.0))
        self_rag_relevance = str(result.get("self_rag_relevance") or "").lower()

        strength = similarity

        # Lexical + semantic agreement is a strong, scale-independent signal.
        if {"vector", "bm25"} <= methods:
            strength = max(strength, self.agreement_floor)

        # Self-RAG explicitly graded this chunk as supporting the query.
        if (
            self_rag_relevance in {"relevant", "partially_relevant"}
            and self_rag_score >= self.min_self_rag_score
        ):
            strength = max(strength, self_rag_score, self.self_rag_floor)
        elif self_rag_score:
            strength = max(strength, self_rag_score * 0.8)

        return _clamp(strength)

    def _thresholds(self, query_intent: Optional[str]) -> tuple[float, float]:
        """Intent-aware decision thresholds.

        Analytical questions (stats, recipes, comparisons) need exact, strongly
        supported facts, so they require higher confidence before answering.
        """
        answer_min = self.answer_min_confidence
        clarify_min = self.clarify_min_confidence
        intent = (query_intent or "").lower()
        if intent == "analytical":
            answer_min = _clamp(answer_min + self.analytical_confidence_bonus)
        elif intent == "conversation":
            answer_min = _clamp(answer_min - self.conversation_confidence_discount)
        clarify_min = min(clarify_min, answer_min)
        return answer_min, clarify_min

    @staticmethod
    def _needs_clarification(query: str, recent_messages: Sequence[str]) -> bool:
        if recent_messages:
            return False
        normalized = query.lower().strip()
        referential = re.search(
            r"\b(cái đó|nó|thứ đó|chỗ đó|that one|it|there)\b",
            normalized,
        )
        return bool(referential)
