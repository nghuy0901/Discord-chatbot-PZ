import json
from collections import Counter, defaultdict
from pathlib import Path

from evaluation.dataset_schema import (
    ExpectedBehavior,
    GoldenExample,
    validate_dataset,
)


def example(i: int, behavior: str = "answer", split: str = "development"):
    return {
        "id": f"pz-{i:03d}",
        "dataset_version": "public-v1",
        "question": "Rìu có tác dụng gì?",
        "language": "vi",
        "category": "answerable",
        "split": split,
        "expected_behavior": behavior,
        "ground_truth": "Rìu là vũ khí và công cụ chặt cây.",
        "expected_sources": ["pz/Weapons/Axes_Weapon.md"],
        "expected_context_keywords": ["rìu", "chặt cây"],
        "critical": False,
        "approved_by": "admin1",
        "approved_at": "2026-06-25T00:00:00Z",
    }


def test_unanswerable_example_allows_empty_ground_truth_and_sources():
    raw = example(1, behavior="abstain")
    raw["category"] = "unanswerable"
    raw["ground_truth"] = ""
    raw["expected_sources"] = []
    item = GoldenExample.from_dict(raw)
    assert item.expected_behavior is ExpectedBehavior.ABSTAIN


def test_dataset_rejects_duplicate_ids():
    rows = [example(1), example(1)]
    errors = validate_dataset(rows, require_release_quota=False)
    assert any("duplicate id" in error for error in errors)


def _release_v2_rows():
    path = Path("evaluation/data/release_qa.v2.jsonl")
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def test_release_v2_has_valid_300_row_quota_and_unique_questions():
    rows = _release_v2_rows()

    assert len(rows) == 300
    assert validate_dataset(rows, require_release_quota=True) == []
    assert len({row["question"].casefold() for row in rows}) == 300
    assert {row["approved_by"] for row in rows} == {"codex-draft"}


def test_release_v2_answer_sources_exist_and_fact_groups_do_not_leak_splits():
    rows = _release_v2_rows()
    group_splits = defaultdict(set)

    for row in rows:
        if row["expected_behavior"] == "answer":
            assert row.get("fact_group")
            group_splits[row["fact_group"]].add(row["split"])
            for source in row["expected_sources"]:
                assert (Path("knowledge/docs") / source).is_file(), source

    assert len(group_splits) == 60
    assert all(len(splits) == 1 for splits in group_splits.values())


def test_release_v2_summary_matches_dataset():
    rows = _release_v2_rows()
    summary = json.loads(
        Path("evaluation/data/release_qa.v2.summary.json").read_text(encoding="utf-8")
    )

    assert summary["rows"] == 300
    assert summary["categories"] == dict(Counter(row["category"] for row in rows))
    assert summary["behaviors"] == dict(
        Counter(row["expected_behavior"] for row in rows)
    )
    assert summary["splits"] == dict(Counter(row["split"] for row in rows))
    assert summary["languages"] == dict(Counter(row["language"] for row in rows))
