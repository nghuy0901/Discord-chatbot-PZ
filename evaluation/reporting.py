from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List


@dataclass
class EvaluationItemResult:
    example_id: str
    expected_behavior: str
    actual_behavior: str
    answer: str
    retrieved_source_ids: List[str]
    provenance: List[Dict[str, Any]]
    deterministic_scores: Dict[str, float]
    judge_scores: Dict[str, float]
    latency_ms: float
    error: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class EvaluationReport:
    run_id: str
    dataset_version: str
    split: str
    repeat_index: int
    versions: Dict[str, str]
    summary: Dict[str, float]
    items: List[EvaluationItemResult]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "dataset_version": self.dataset_version,
            "split": self.split,
            "repeat_index": self.repeat_index,
            "versions": self.versions,
            "summary": self.summary,
            "items": [item.to_dict() for item in self.items],
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)

    def write_json(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(self.to_json(), encoding="utf-8")


def stability_rate(repeated_items: List[List[EvaluationItemResult]]) -> float:
    if len(repeated_items) < 2:
        return 1.0
    baseline = {item.example_id: item.actual_behavior for item in repeated_items[0]}
    if not baseline:
        return 0.0

    stable = 0
    total = 0
    for example_id, expected_behavior in baseline.items():
        total += 1
        if all(
            run_item.actual_behavior == expected_behavior
            for run in repeated_items[1:]
            for run_item in run
            if run_item.example_id == example_id
        ):
            stable += 1
    return stable / total if total else 0.0
