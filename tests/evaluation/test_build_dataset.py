"""scripts/build_eval_dataset.build_rows — reviewed candidates → schema rows."""

import json
from pathlib import Path

from evaluation.dataset_schema import validate_dataset
from scripts.build_eval_dataset import build_rows


def test_build_rows_filters_unapproved_and_validates():
    candidates = [
        {
            "question": "Búa dùng làm gì?",
            "review_status": "approved",
            "expected_behavior": "answer",
            "category": "answerable",
            "language": "vi",
            "ground_truth": "Đóng đinh và làm vũ khí.",
            "expected_sources": ["pz/Equipment/Tools.md"],
            "expected_context_keywords": ["búa"],
            "reviewer": "alice",
            "split": "development",
        },
        {"question": "skip", "review_status": "unreviewed", "candidate_category": "answerable"},
    ]
    rows, skipped = build_rows(
        candidates,
        dataset_version="v9",
        default_reviewer="bot",
        approved_at="2026-06-29T00:00:00Z",
        default_split="development",
    )
    assert skipped == 1
    assert len(rows) == 1
    row = rows[0]
    assert row["dataset_version"] == "v9"
    assert row["approved_by"] == "alice"
    assert row["expected_sources"] == ["pz/Equipment/Tools.md"]
    assert row["id"]
    assert validate_dataset(rows, require_release_quota=False) == []


def test_build_rows_falls_back_to_draft_fields():
    candidates = [
        {
            "question": "q",
            "review_status": "approved",
            "expected_behavior": "answer",
            "candidate_category": "answerable",
            "language": "vi",
            "draft_ground_truth": "GT",
            "candidate_source_ids": ["pz/Equipment/Tools.md"],
        }
    ]
    rows, _ = build_rows(
        candidates,
        dataset_version="v1",
        default_reviewer="rev",
        approved_at="2026-06-29T00:00:00Z",
        default_split="holdout",
    )
    assert rows[0]["ground_truth"] == "GT"
    assert rows[0]["expected_sources"] == ["pz/Equipment/Tools.md"]
    assert rows[0]["approved_by"] == "rev"
    assert rows[0]["split"] == "holdout"


def test_seed_dataset_is_schema_valid():
    rows = [
        json.loads(line)
        for line in Path("evaluation/data/release_qa.v1.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    assert len(rows) >= 20
    assert validate_dataset(rows, require_release_quota=False) == []
