from dataclasses import dataclass
from typing import List, Optional


@dataclass
class GoldenExample:
    question: str
    ground_truth: str
    domain: Optional[str]
    expected_sources: List[str]
    expected_context_keywords: List[str]


REQUIRED_DATASET_FIELDS = {
    "question",
    "ground_truth",
    "domain",
    "expected_sources",
    "expected_context_keywords",
}


def validate_example(raw: dict) -> GoldenExample:
    missing = REQUIRED_DATASET_FIELDS - set(raw)
    if missing:
        raise ValueError(f"Missing required dataset fields: {sorted(missing)}")
    return GoldenExample(
        question=str(raw["question"]),
        ground_truth=str(raw["ground_truth"]),
        domain=raw.get("domain"),
        expected_sources=list(raw.get("expected_sources") or []),
        expected_context_keywords=list(raw.get("expected_context_keywords") or []),
    )
