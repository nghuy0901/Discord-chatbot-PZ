"""Build a schema-valid eval dataset (GoldenExample JSONL) from reviewed candidates.

Authoring pipeline:
  1. scripts/export_eval_candidates.py inventory  --output <inv.json>
  2. scripts/export_eval_candidates.py draft --inventory <inv.json> --output <draft.jsonl>
  3. <human review> — for each row a reviewer fills:
       review_status="approved", expected_behavior, category, language,
       ground_truth (answer rows), expected_sources (answer rows),
       expected_context_keywords, critical, split, reviewer
  4. scripts/build_eval_dataset.py --candidates <draft.jsonl> --output <release_qa.jsonl>
  5. scripts/validate_eval_dataset.py --dataset <release_qa.jsonl> [--require-release-quota]

Only rows with review_status == "approved" are emitted; the rest are skipped.
The produced file is validated against the GoldenExample schema before writing.
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from evaluation.dataset_schema import validate_dataset


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"line {line_number}: invalid JSON: {exc}") from exc
    return rows


def _coalesce(row: Dict[str, Any], *keys: str, default: Any = None) -> Any:
    for key in keys:
        value = row.get(key)
        if value not in (None, "", []):
            return value
    return default


def build_rows(
    candidates: List[Dict[str, Any]],
    *,
    dataset_version: str,
    default_reviewer: str,
    approved_at: str,
    default_split: str,
) -> "tuple[List[Dict[str, Any]], int]":
    """Map approved review rows → GoldenExample dicts. Returns (rows, skipped)."""
    rows: List[Dict[str, Any]] = []
    skipped = 0
    counters: Dict[str, int] = {}
    for row in candidates:
        if str(row.get("review_status", "")).lower() != "approved":
            skipped += 1
            continue
        behavior = str(_coalesce(row, "expected_behavior", default="") or "")
        category = str(_coalesce(row, "category", "candidate_category", default="") or "")
        key = f"{category or 'item'}-{behavior or 'x'}"
        counters[key] = counters.get(key, 0) + 1
        identifier = str(_coalesce(row, "id", default="") or f"{key}-{counters[key]:03d}")
        reviewer = str(
            _coalesce(row, "reviewer", "approved_by", default=default_reviewer)
            or default_reviewer
        )
        rows.append(
            {
                "id": identifier,
                "dataset_version": dataset_version,
                "question": str(row.get("question", "")),
                "language": str(_coalesce(row, "language", default="vi")),
                "category": category,
                "split": str(_coalesce(row, "split", default=default_split)),
                "expected_behavior": behavior,
                "ground_truth": str(
                    _coalesce(row, "ground_truth", "draft_ground_truth", default="") or ""
                ),
                "expected_sources": list(
                    _coalesce(row, "expected_sources", "candidate_source_ids", default=[]) or []
                ),
                "expected_context_keywords": list(row.get("expected_context_keywords") or []),
                "critical": bool(row.get("critical", False)),
                "approved_by": reviewer,
                "approved_at": str(_coalesce(row, "approved_at", default=approved_at)),
            }
        )
    return rows, skipped


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build a schema-valid eval dataset from reviewed candidates."
    )
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--dataset-version", default="public-v1")
    parser.add_argument("--reviewer", default="maintainer")
    parser.add_argument(
        "--default-split", default="development", choices=["development", "holdout"]
    )
    parser.add_argument("--approved-at", default=None)
    parser.add_argument("--require-release-quota", action="store_true")
    args = parser.parse_args()

    approved_at = args.approved_at or datetime.now(tz=timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )

    try:
        candidates = load_jsonl(Path(args.candidates))
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1

    rows, skipped = build_rows(
        candidates,
        dataset_version=args.dataset_version,
        default_reviewer=args.reviewer,
        approved_at=approved_at,
        default_split=args.default_split,
    )

    errors = validate_dataset(rows, require_release_quota=args.require_release_quota)
    if errors:
        for error in errors:
            print(error, file=sys.stderr)
        return 1

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"built {len(rows)} dataset rows ({skipped} non-approved skipped) → {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
