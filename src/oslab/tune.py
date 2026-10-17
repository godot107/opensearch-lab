"""Lesson 1 tuning: pick BM25 settings on the TRAIN questions, then score the TEST questions once.

Grid:
- index-time: analyzer (english = stemming + stopwords, standard = neither) and BM25's k1 / b
  (k1: how fast repeated words stop adding score; b: how much long documents are penalised).
  Each combination needs its own index, because similarity and analyzers are fixed at creation.
- query-time: multi_match type and title boost.

Choosing among ~400 settings on 809 questions can overfit a little, so the test score is the honest
number, and `paired_bootstrap` says whether the gain over the untuned baseline is bigger than noise.
"""
import itertools
import json
from pathlib import Path

import numpy as np
from opensearchpy import helpers

from oslab import beir, metrics, search, stack, store

ANALYZERS = ("english", "standard")
K1 = (0.6, 0.9, 1.2, 1.5)
B = (0.5, 0.75, 0.9, 1.0)
TYPES = ("best_fields", "most_fields", "cross_fields", "best_fields_tb")  # tb = tie_breaker 0.3
TITLE_BOOST = (0.5, 1.0, 1.5, 2.0, 3.0)


def index_name(analyzer, k1, b):
    return f"tune-{analyzer}-k{k1}-b{b}".replace(".", "_")


def build_indices(client=None):
    client = client or stack.opensearch()
    docs = list(store.iter_docs())
    for analyzer, k1, b in itertools.product(ANALYZERS, K1, B):
        name = index_name(analyzer, k1, b)
        if client.indices.exists(index=name) and client.count(index=name)["count"] == len(docs):
            continue
        if client.indices.exists(index=name):
            client.indices.delete(index=name)
        client.indices.create(index=name, body={
            "settings": {"number_of_shards": 1, "number_of_replicas": 0,
                         "similarity": {"default": {"type": "BM25", "k1": k1, "b": b}}},
            "mappings": {"dynamic": "strict", "properties": {
                "title": {"type": "text", "analyzer": analyzer},
                "body": {"type": "text", "analyzer": analyzer}}}})
        helpers.bulk(client, ({"_index": name, "_id": i, "_source": {"title": t, "body": bd}}
                              for i, t, bd in docs), chunk_size=1000)
        client.indices.refresh(index=name)
    return len(ANALYZERS) * len(K1) * len(B)


def configs():
    for analyzer, k1, b, typ, boost in itertools.product(ANALYZERS, K1, B, TYPES, TITLE_BOOST):
        cfg = {"index": index_name(analyzer, k1, b), "analyzer": analyzer, "k1": k1, "b": b,
               "type": typ.removesuffix("_tb"), "title_boost": boost,
               "tie_breaker": 0.3 if typ.endswith("_tb") else 0.0}
        yield cfg


def per_query_ndcg(run, qrels, k=10):
    return np.array([metrics.ndcg_at_k(run.get(q, []), rel, k) for q, rel in qrels.items()])


def paired_bootstrap(a, b, n=10_000, seed=0):
    """Mean difference b - a over the same queries, with a 95% interval from resampling queries."""
    rng = np.random.default_rng(seed)
    d = b - a
    boots = d[rng.integers(0, len(d), (n, len(d)))].mean(axis=1)
    return {"mean_diff": float(d.mean()), "ci95": [float(np.percentile(boots, 2.5)),
                                                    float(np.percentile(boots, 97.5))],
            "p_le_0": float((boots <= 0).mean())}


def sweep(ds_dir="data/beir/scifact", out_dir="results"):
    client = stack.opensearch()
    build_indices(client)
    _, train_q, train_rel = beir.load(ds_dir, "train")
    rows = []
    for cfg in configs():
        run = search.os_bm25_many(train_q, cfg, client=client)
        rows.append(cfg | metrics.score_run(run, train_rel, 10))
    rows.sort(key=lambda r: -r["ndcg@10"])
    Path(out_dir).mkdir(exist_ok=True)
    Path(out_dir, "tune-train.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))

    best = {k: rows[0][k] for k in ("index", "analyzer", "k1", "b", "type", "title_boost", "tie_breaker")}
    base_train = metrics.score_run(search.os_bm25_many(train_q, search.BASELINE, client=client),
                                   train_rel, 10)
    _, test_q, test_rel = beir.load(ds_dir, "test")
    base_run = search.os_bm25_many(test_q, search.BASELINE, client=client)
    best_run = search.os_bm25_many(test_q, best, client=client)
    out = {"config": best, "n_configs": len(rows),
           "train": {"baseline_ndcg@10": base_train["ndcg@10"], "tuned_ndcg@10": rows[0]["ndcg@10"]},
           "test": {"baseline_ndcg@10": metrics.score_run(base_run, test_rel, 10)["ndcg@10"],
                    "tuned_ndcg@10": metrics.score_run(best_run, test_rel, 10)["ndcg@10"],
                    "tuned_minus_baseline": paired_bootstrap(per_query_ndcg(base_run, test_rel),
                                                             per_query_ndcg(best_run, test_rel))}}
    Path(out_dir, "tune-best.json").write_text(json.dumps(out, indent=2) + "\n")
    return out, rows
