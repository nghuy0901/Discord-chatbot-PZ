from evaluation.retrieval_metrics import (
    keyword_coverage,
    mean_reciprocal_rank,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    source_hit_rate,
)


def test_recall_and_precision_at_k():
    retrieved = ["a.md", "b.md", "c.md"]
    expected = {"b.md", "d.md"}
    assert recall_at_k(retrieved, expected, k=3) == 0.5
    assert precision_at_k(retrieved, expected, k=3) == 1 / 3


def test_mrr_uses_first_relevant_rank():
    retrieved = ["a.md", "b.md", "c.md"]
    expected = {"b.md"}
    assert mean_reciprocal_rank(retrieved, expected) == 0.5


def test_ndcg_at_k_rewards_early_relevant_results():
    retrieved = ["b.md", "a.md", "c.md"]
    expected = {"b.md", "c.md"}
    score = ndcg_at_k(retrieved, expected, k=3)
    assert 0.0 < score <= 1.0


def test_source_hit_and_keyword_coverage():
    assert source_hit_rate(["a.md"], {"a.md"}) == 1.0
    assert keyword_coverage("Axe requires carpentry skill", ["axe", "skill"]) == 1.0
