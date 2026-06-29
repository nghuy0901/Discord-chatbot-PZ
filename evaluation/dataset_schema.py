from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Iterable, List, Optional


class ExpectedBehavior(str, Enum):
    ANSWER = "answer"
    ABSTAIN = "abstain"
    CLARIFY = "clarify"


REQUIRED_DATASET_FIELDS = {
    "id",
    "dataset_version",
    "question",
    "language",
    "category",
    "split",
    "expected_behavior",
    "ground_truth",
    "expected_sources",
    "expected_context_keywords",
    "critical",
    "approved_by",
    "approved_at",
}


@dataclass(frozen=True)
class GoldenExample:
    id: str
    dataset_version: str
    question: str
    language: str
    category: str
    split: str
    expected_behavior: ExpectedBehavior
    ground_truth: str
    expected_sources: List[str]
    expected_context_keywords: List[str]
    critical: bool
    approved_by: str
    approved_at: str

    # Backward-compatible fields used by existing scripts.
    @property
    def domain(self) -> Optional[str]:
        if self.expected_sources:
            source = self.expected_sources[0]
            if "/" in source:
                return source.split("/", 1)[0]
        if self.category == "server_rules":
            return "server_rules"
        return None

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "GoldenExample":
        missing = REQUIRED_DATASET_FIELDS - set(raw)
        if missing:
            raise ValueError(f"Missing required dataset fields: {sorted(missing)}")
        try:
            behavior = ExpectedBehavior(str(raw["expected_behavior"]))
        except ValueError as exc:
            raise ValueError(f"Invalid expected_behavior: {raw['expected_behavior']}") from exc
        return cls(
            id=str(raw["id"]),
            dataset_version=str(raw["dataset_version"]),
            question=str(raw["question"]),
            language=str(raw["language"]),
            category=str(raw["category"]),
            split=str(raw["split"]),
            expected_behavior=behavior,
            ground_truth=str(raw.get("ground_truth") or ""),
            expected_sources=list(raw.get("expected_sources") or []),
            expected_context_keywords=list(raw.get("expected_context_keywords") or []),
            critical=bool(raw["critical"]),
            approved_by=str(raw.get("approved_by") or ""),
            approved_at=str(raw.get("approved_at") or ""),
        )


def validate_example(raw: Dict[str, Any]) -> GoldenExample:
    item = GoldenExample.from_dict(raw)
    errors = _validate_item(item)
    if errors:
        raise ValueError("; ".join(errors))
    return item


def validate_dataset(
    rows: Iterable[Dict[str, Any]],
    *,
    require_release_quota: bool = False,
) -> List[str]:
    errors: List[str] = []
    items: List[GoldenExample] = []
    seen_ids: set[str] = set()

    for row_number, raw in enumerate(rows, start=1):
        try:
            item = GoldenExample.from_dict(raw)
        except ValueError as exc:
            errors.append(f"row {row_number}: {exc}")
            continue

        if item.id in seen_ids:
            errors.append(f"duplicate id: {item.id}")
        seen_ids.add(item.id)

        for item_error in _validate_item(item):
            errors.append(f"{item.id}: {item_error}")
        items.append(item)

    if require_release_quota and not errors:
        errors.extend(_validate_release_quota(items))

    return errors


def _validate_item(item: GoldenExample) -> List[str]:
    errors: List[str] = []
    if item.language not in {"vi", "en"}:
        errors.append("language must be vi or en")
    if item.split not in {"development", "holdout"}:
        errors.append("split must be development or holdout")
    if not item.approved_by:
        errors.append("approved_by is required")
    if not item.approved_at:
        errors.append("approved_at is required")
    if item.expected_behavior is ExpectedBehavior.ANSWER:
        if not item.ground_truth.strip():
            errors.append("answer examples require ground_truth")
        if not item.expected_sources:
            errors.append("answer examples require expected_sources")
    if item.expected_behavior is ExpectedBehavior.ABSTAIN and item.expected_sources:
        errors.append("abstain examples require empty expected_sources")
    return errors


def _validate_release_quota(items: List[GoldenExample]) -> List[str]:
    errors: List[str] = []
    if len(items) != 300:
        errors.append(f"release dataset must contain exactly 300 rows, got {len(items)}")

    category_counts = Counter(item.category for item in items)
    expected_categories = {
        "answerable": 180,
        "unanswerable": 60,
        "ambiguous": 30,
        "adversarial": 30,
    }
    for category, expected in expected_categories.items():
        actual = category_counts.get(category, 0)
        if actual != expected:
            errors.append(f"category {category} must have {expected} rows, got {actual}")

    behavior_counts = Counter(item.expected_behavior.value for item in items)
    expected_behaviors = {"answer": 180, "abstain": 84, "clarify": 36}
    for behavior, expected in expected_behaviors.items():
        actual = behavior_counts.get(behavior, 0)
        if actual != expected:
            errors.append(f"behavior {behavior} must have {expected} rows, got {actual}")

    split_counts = Counter(item.split for item in items)
    if split_counts.get("development", 0) != 210:
        errors.append(
            "split development must have 210 rows, "
            f"got {split_counts.get('development', 0)}"
        )
    if split_counts.get("holdout", 0) != 90:
        errors.append(
            "split holdout must have 90 rows, "
            f"got {split_counts.get('holdout', 0)}"
        )

    vi_ratio = (
        sum(1 for item in items if item.language == "vi") / len(items)
        if items
        else 0.0
    )
    if not 0.75 <= vi_ratio <= 0.85:
        errors.append(f"Vietnamese ratio must be between 0.75 and 0.85, got {vi_ratio:.2f}")

    for item in items:
        if item.category == "answerable" and item.expected_behavior is not ExpectedBehavior.ANSWER:
            errors.append(f"{item.id}: answerable rows must have answer behavior")
        if item.category == "unanswerable" and item.expected_behavior is not ExpectedBehavior.ABSTAIN:
            errors.append(f"{item.id}: unanswerable rows must have abstain behavior")
        if item.category == "ambiguous" and item.expected_behavior is not ExpectedBehavior.CLARIFY:
            errors.append(f"{item.id}: ambiguous rows must have clarify behavior")

    adversarial_behaviors = Counter(
        item.expected_behavior.value for item in items if item.category == "adversarial"
    )
    if adversarial_behaviors.get("abstain", 0) != 24:
        errors.append(
            "adversarial rows must include 24 abstain examples, "
            f"got {adversarial_behaviors.get('abstain', 0)}"
        )
    if adversarial_behaviors.get("clarify", 0) != 6:
        errors.append(
            "adversarial rows must include 6 clarify examples, "
            f"got {adversarial_behaviors.get('clarify', 0)}"
        )

    return errors
