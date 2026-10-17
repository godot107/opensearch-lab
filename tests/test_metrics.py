import math

from oslab.metrics import mrr_at_k, ndcg_at_k, recall_at_k, score_run


def test_recall_counts_relevant_found_in_top_k():
    assert recall_at_k(["a", "b", "c"], {"a": 1, "z": 1}, k=3) == 0.5
    assert recall_at_k(["b", "c", "a"], {"a": 1}, k=2) == 0.0


def test_mrr_is_reciprocal_rank_of_first_hit():
    assert mrr_at_k(["x", "a", "b"], {"a": 1, "b": 1}, k=10) == 0.5
    assert mrr_at_k(["x", "y"], {"a": 1}, k=10) == 0.0


def test_ndcg_perfect_and_graded():
    assert ndcg_at_k(["a", "b"], {"a": 2, "b": 1}, k=10) == 1.0
    swapped = ndcg_at_k(["b", "a"], {"a": 2, "b": 1}, k=10)
    expected = (1 + 2 / math.log2(3)) / (2 + 1 / math.log2(3))
    assert math.isclose(swapped, expected)


def test_score_run_counts_missing_queries_as_zero():
    qrels = {"q1": {"a": 1}, "q2": {"b": 1}}
    out = score_run({"q1": ["a"]}, qrels, k=10)
    assert out["recall@10"] == 0.5 and out["mrr@10"] == 0.5 and out["n_queries"] == 2
