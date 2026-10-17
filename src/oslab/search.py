"""Keyword retrievers for lesson 1. Each returns doc ids, best first.

- os-bm25:  OpenSearch BM25 over title and body (title weighted 2x, an arbitrary
            starting point that lesson 1 tunes).
- pg-and:   Postgres websearch_to_tsquery: every word must match. The default people
            reach for, and the wrong one for long natural-language queries.
- pg-or:    Postgres any-word match, ranked by ts_rank_cd. The fair baseline.
"""
from oslab import stack, sync

BACKENDS = ("os-bm25", "pg-and", "pg-or")


def os_bm25(query, k=10, client=None, index=sync.DOCS_INDEX, title_boost=2.0):
    client = client or stack.opensearch()
    body = {"size": k, "_source": False,
            "query": {"multi_match": {"query": query, "type": "best_fields",
                                      "fields": [f"title^{title_boost}", "body"]}}}
    return [h["_id"] for h in client.search(index=index, body=body)["hits"]["hits"]]


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
    os_client = stack.opensearch() if backend.startswith("os-") else None
    pg_conn = stack.postgres() if backend.startswith("pg-") else None
    return {
        "os-bm25": lambda q, k: os_bm25(q, k, client=os_client),
        "pg-and": lambda q, k: pg(q, k, conn=pg_conn, mode="and"),
        "pg-or": lambda q, k: pg(q, k, conn=pg_conn, mode="or"),
    }[backend]
