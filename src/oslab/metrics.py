"""Retrieval metrics over BEIR qrels: recall@k, MRR@k, nDCG@k (graded), latency percentiles."""
import math

import numpy as np


def recall_at_k(ranked, relevant, k):
    if not relevant:
        return 0.0
    return len(set(ranked[:k]) & set(relevant)) / len(relevant)


def mrr_at_k(ranked, relevant, k):
    for i, doc in enumerate(ranked[:k], 1):
        if doc in relevant:
            return 1.0 / i
    return 0.0


def ndcg_at_k(ranked, relevant, k):
    """relevant: {doc_id: grade}. Linear gain, as in BEIR's pytrec_eval setup."""
    dcg = sum(relevant.get(d, 0) / math.log2(i + 1) for i, d in enumerate(ranked[:k], 1))
    ideal = sorted(relevant.values(), reverse=True)[:k]
    idcg = sum(g / math.log2(i + 1) for i, g in enumerate(ideal, 1))
    return dcg / idcg if idcg else 0.0


def score_run(run, qrels, k):
    """run: {qid: [doc ids, best first]}. Averages over every query in qrels;
    a query with no results scores 0, it isn't dropped."""
    per = {"recall": [], "mrr": [], "ndcg": []}
    for qid, rel in qrels.items():
        ranked = run.get(qid, [])
        per["recall"].append(recall_at_k(ranked, rel, k))
        per["mrr"].append(mrr_at_k(ranked, rel, k))
        per["ndcg"].append(ndcg_at_k(ranked, rel, k))
    return {f"{m}@{k}": float(np.mean(v)) for m, v in per.items()} | {"n_queries": len(qrels)}


def latency_summary(seconds):
    a = np.asarray(seconds) * 1000
    return {"p50_ms": float(np.percentile(a, 50)), "p95_ms": float(np.percentile(a, 95)),
            "mean_ms": float(a.mean())}
