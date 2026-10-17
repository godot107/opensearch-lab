"""Keyword retrievers for lesson 1. Each returns doc ids, best first.

- os-bm25:       OpenSearch BM25 over title and body, untuned starting point (best_fields, title^2).
- os-bm25-tuned: the settings `oslab tune` picked on the TRAIN split (results/tune-best.json).
- pg-and:        Postgres websearch_to_tsquery: every word must match. The default people
                 reach for, and the wrong one for long natural-language queries.
- pg-or:         Postgres any-word match, ranked by ts_rank_cd. The fair baseline.
"""
import json
from pathlib import Path

from oslab import stack, sync

BACKENDS = ("os-bm25", "os-bm25-tuned", "pg-and", "pg-or")
BASELINE = {"index": sync.DOCS_INDEX, "type": "best_fields", "title_boost": 2.0, "tie_breaker": 0.0}
TUNED_FILE = Path("results/tune-best.json")


def bm25_body(query, k=10, type="best_fields", title_boost=2.0, tie_breaker=0.0, **_):
    mm = {"query": query, "type": type, "fields": [f"title^{title_boost}", "body"]}
    if type == "best_fields" and tie_breaker:
        mm["tie_breaker"] = tie_breaker
    return {"size": k, "_source": False, "query": {"multi_match": mm}}


def os_bm25(query, k=10, client=None, index=sync.DOCS_INDEX, **cfg):
    client = client or stack.opensearch()
    hits = client.search(index=index, body=bm25_body(query, k, **cfg))["hits"]["hits"]
    return [h["_id"] for h in hits]


def os_bm25_many(queries, cfg, k=10, client=None):
    """{qid: text} -> {qid: [ids]} via _msearch, 100 queries per round trip (for tuning sweeps)."""
    client = client or stack.opensearch()
    qids, out = list(queries), {}
    for i in range(0, len(qids), 100):
        chunk = qids[i:i + 100]
        lines = []
        for q in chunk:
            lines += [{"index": cfg["index"]}, bm25_body(queries[q], k, **cfg)]
        for q, r in zip(chunk, client.msearch(body=lines)["responses"]):
            if "error" in r:
                raise RuntimeError(r["error"])
            out[q] = [h["_id"] for h in r["hits"]["hits"]]
    return out


def tuned_config():
    if not TUNED_FILE.exists():
        raise SystemExit("no tuned config: run `python -m oslab tune` first")
    return json.loads(TUNED_FILE.read_text())["config"]


# Turn plainto_tsquery's AND (&) into OR (|). The query text is passed as a bind
# parameter and only the resulting tsquery is rewritten, so this is injection-safe.
PG_OR = """
with q as (select replace(plainto_tsquery('english', %(q)s)::text, ' & ', ' | ') as t)
select id from docs, q
where q.t <> '' and tsv @@ to_tsquery('english', q.t)
order by ts_rank_cd(tsv, to_tsquery('english', q.t)) desc, id
limit %(k)s
"""

PG_AND = """
select id from docs, websearch_to_tsquery('english', %(q)s) q
where tsv @@ q
order by ts_rank_cd(tsv, q) desc, id
limit %(k)s
"""


def pg(query, k=10, conn=None, mode="or"):
    conn = conn or stack.postgres()
    sql = PG_OR if mode == "or" else PG_AND
    return [r[0] for r in conn.execute(sql, {"q": query, "k": k}).fetchall()]


def retriever(backend):
    if backend.startswith("os-"):
        client = stack.opensearch()
        cfg = tuned_config() if backend == "os-bm25-tuned" else BASELINE
        return lambda q, k: os_bm25(q, k, client=client, **cfg)
    if backend.startswith("pg-"):
        conn = stack.postgres()
        return lambda q, k: pg(q, k, conn=conn, mode=backend[3:])
    from oslab import hybrid  # lesson 3 backends
    return hybrid.retriever(backend)
