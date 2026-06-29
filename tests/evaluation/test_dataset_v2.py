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
