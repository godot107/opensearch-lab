"""Lesson 3: vector and hybrid search on the same SciFact questions and labels as lesson 1.

Vectors are bge-large (1024 numbers per text, normalised), exported from rag-bedrock-compare by
scripts/export_bge.py. Backends:

- exact-vector:   brute force over every document in numpy. The known answer for the
                  approximate (HNSW) indexes below: they should find the same neighbours.
- os-vector:      OpenSearch k-NN field (HNSW graph, faiss engine).
- pg-vector:      pgvector HNSW index on the same Postgres rows.
- os-hybrid:      one OpenSearch query that runs BM25 and k-NN, rescales each list's scores to 0-1
                  (min-max) and blends them with a weight picked on the TRAIN questions.
- os-hybrid-rrf:  same two lists merged by rank (reciprocal rank fusion): no scores, no weight.
"""
import json
from functools import lru_cache
from pathlib import Path

import numpy as np
from opensearchpy import helpers

from oslab import beir, metrics, search, stack, store

VEC_FILE = Path("data/vectors/scifact-bge.npz")
INDEX = "scifact-hybrid"
DIM = 1024
WEIGHTS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)   # share of the vector score
TUNED_FILE = Path("results/hybrid-best.json")
DEPTH = 100   # candidates each sub-query contributes before blending
BACKENDS = ("exact-vector", "os-vector", "pg-vector", "os-hybrid", "os-hybrid-rrf")


@lru_cache(maxsize=1)
def vectors():
    z = np.load(VEC_FILE)
    return {"doc_ids": [str(i) for i in z["doc_ids"]], "doc_vecs": z["doc_vecs"],
            "query": dict(zip((str(i) for i in z["query_ids"]), z["query_vecs"]))}


@lru_cache(maxsize=1)
def text_to_qid():
    out = {}
    for line in open(Path("data/beir/scifact/queries.jsonl")):
        q = json.loads(line)
        out[q["text"]] = str(q["_id"])
    return out


def qvec(text):
    return vectors()["query"][text_to_qid()[text]]


# ---------- indexes ----------

def build_index(client=None):
    """BM25 fields with the tuned lesson-1 settings, plus a k-NN vector field."""
    client = client or stack.opensearch()
    cfg = search.tuned_config()
    v = vectors()
    if client.indices.exists(index=INDEX):
        client.indices.delete(index=INDEX)
    client.indices.create(index=INDEX, body={
        "settings": {"index": {"knn": True, "number_of_shards": 1, "number_of_replicas": 0},
                     "similarity": {"default": {"type": "BM25", "k1": cfg["k1"], "b": cfg["b"]}}},
        "mappings": {"dynamic": "strict", "properties": {
            "title": {"type": "text", "analyzer": cfg["analyzer"]},
            "body": {"type": "text", "analyzer": cfg["analyzer"]},
            "embedding": {"type": "knn_vector", "dimension": DIM, "space_type": "innerproduct",
                          "method": {"name": "hnsw", "engine": "faiss",
                                     "parameters": {"m": 16, "ef_construction": 128}}}}}})
    vec = dict(zip(v["doc_ids"], v["doc_vecs"]))
    helpers.bulk(client, ({"_index": INDEX, "_id": i,
                           "_source": {"title": t, "body": b, "embedding": vec[i].tolist()}}
                          for i, t, b in store.iter_docs()), chunk_size=200)
    client.indices.refresh(index=INDEX)
    return client.count(index=INDEX)["count"]


def load_pgvector(conn=None):
    conn = conn or stack.postgres()
    v = vectors()
    conn.execute("create extension if not exists vector")
    conn.execute(f"alter table docs add column if not exists embedding vector({DIM})")
    with conn.cursor() as cur:
        cur.executemany("update docs set embedding = %s::vector where id = %s",
                        [(str(vec.tolist()), i) for i, vec in zip(v["doc_ids"], v["doc_vecs"])])
    conn.execute("create index if not exists docs_embedding on docs "
                 "using hnsw (embedding vector_ip_ops) with (m = 16, ef_construction = 128)")
    return conn.execute("select count(*) from docs where embedding is not null").fetchone()[0]


def create_pipelines(client=None):
    client = client or stack.opensearch()
    tr = client.transport.perform_request
    for w in WEIGHTS:
        tr("PUT", f"/_search/pipeline/{pipeline_name(w)}", body={
            "description": f"min-max normalise, then {1 - w:.1f} x BM25 + {w:.1f} x vector",
            "phase_results_processors": [{"normalization-processor": {
                "normalization": {"technique": "min_max"},
                "combination": {"technique": "arithmetic_mean",
                                "parameters": {"weights": [round(1 - w, 2), round(w, 2)]}}}}]})
    tr("PUT", "/_search/pipeline/hybrid-rrf", body={
        "description": "reciprocal rank fusion of BM25 and vector lists (rank_constant 60)",
        "phase_results_processors": [{"score-ranker-processor": {
            "combination": {"technique": "rrf", "rank_constant": 60}}}]})


def pipeline_name(w):
    return f"hybrid-minmax-w{int(round(w * 100)):02d}"


# ---------- queries ----------

def bm25_clause(text):
    cfg = search.tuned_config()
    return search.bm25_body(text, DEPTH, **cfg)["query"]


def knn_body(vec, k=10):
    return {"size": k, "_source": False,
            "query": {"knn": {"embedding": {"vector": vec.tolist(), "k": max(k, DEPTH)}}}}


def hybrid_body(text, vec, k=10):
    return {"size": k, "_source": False,
            "query": {"hybrid": {"pagination_depth": DEPTH, "queries": [
                bm25_clause(text), {"knn": {"embedding": {"vector": vec.tolist(), "k": DEPTH}}}]}}}


def search_many(bodies, client=None, pipeline=None, workers=8):
    """[(qid, body)] -> {qid: [ids]}. One request per query, 8 at a time: _msearch can't take a
    named search pipeline (neither as a URL parameter nor in its per-search header)."""
    from concurrent.futures import ThreadPoolExecutor
    client = client or stack.opensearch()
    params = {"search_pipeline": pipeline} if pipeline else {}

    def one(item):
        q, b = item
        return q, [h["_id"] for h in client.search(index=INDEX, body=b, params=params)["hits"]["hits"]]

    with ThreadPoolExecutor(workers) as ex:
        return dict(ex.map(one, bodies))


def exact(vec, k=10):
    v = vectors()
    s = v["doc_vecs"] @ vec
    top = np.argpartition(-s, k)[:k]
    return [v["doc_ids"][j] for j in top[np.argsort(-s[top])]]


PG_KNN = "select id from docs order by embedding <#> %s::vector limit %s"  # <#> = negative inner product


def tune_weight(ds_dir="data/beir/scifact", client=None):
    """Pick the BM25/vector blend on the TRAIN questions only."""
    client = client or stack.opensearch()
    _, q, rel = beir.load(ds_dir, "train")
    bodies = [(qid, hybrid_body(t, qvec(t))) for qid, t in q.items()]
    scores = {w: metrics.score_run(search_many(bodies, client, pipeline_name(w)), rel, 10)["ndcg@10"]
              for w in WEIGHTS}
    best = max(scores, key=scores.get)
    TUNED_FILE.parent.mkdir(exist_ok=True)
    TUNED_FILE.write_text(json.dumps({"vector_weight": best, "train_ndcg@10": scores}, indent=2) + "\n")
    return best, scores


def tuned_weight():
    if not TUNED_FILE.exists():
        raise SystemExit("no tuned hybrid weight: run `python -m oslab hybrid` first")
    return json.loads(TUNED_FILE.read_text())["vector_weight"]


def retriever(backend):
    if backend == "exact-vector":
        return lambda text, k: exact(qvec(text), k)
    if backend == "pg-vector":
        conn = stack.postgres()
        conn.execute("set hnsw.ef_search = 100")
        return lambda text, k: [r[0] for r in conn.execute(
            PG_KNN, (str(qvec(text).tolist()), k)).fetchall()]
    client = stack.opensearch()
    if backend == "os-vector":
        return lambda text, k: [h["_id"] for h in client.search(
            index=INDEX, body=knn_body(qvec(text), k))["hits"]["hits"]]
    pipeline = "hybrid-rrf" if backend == "os-hybrid-rrf" else pipeline_name(tuned_weight())
    return lambda text, k: [h["_id"] for h in client.search(
        index=INDEX, body=hybrid_body(text, qvec(text), k), params={"search_pipeline": pipeline})["hits"]["hits"]]


def ann_recall(backend, queries, k=10):
    """Share of the exact top-k that the approximate index also returned (1.0 = no loss)."""
    fn = retriever(backend)
    return float(np.mean([len(set(fn(t, k)) & set(exact(qvec(t), k))) / k for t in queries.values()]))
