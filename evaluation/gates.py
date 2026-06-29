from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict


@dataclass(frozen=True)
class GateResult:
    passed: bool
    failures: Dict[str, Dict[str, Any]]
    checked: Dict[str, Dict[str, Any]]


def evaluate_gates(summary: Dict[str, float], config: Dict[str, Any]) -> GateResult:
    failures: Dict[str, Dict[str, Any]] = {}
    checked: Dict[str, Dict[str, Any]] = {}

    for name, threshold in config.get("minimums", {}).items():
        actual = summary.get(name)
        checked[name] = {"actual": actual, "required": threshold, "operator": ">="}
        if actual is None or actual < threshold:
            failures[name] = {"actual": actual, "required": threshold}

    for name, threshold in config.get("maximums", {}).items():
        actual = summary.get(name)
        checked[name] = {"actual": actual, "required": threshold, "operator": "<="}
        if actual is None or actual > threshold:
            failures[name] = {"actual": actual, "required": threshold}

    return GateResult(passed=not failures, failures=failures, checked=checked)
