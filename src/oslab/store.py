"""Postgres is the source of truth for documents; OpenSearch is built from it.

The table also carries Postgres's own full-text index (a generated tsvector),
so lesson 1 can ask "do I even need OpenSearch?" on the same rows.
"""
from oslab import stack

SCHEMA = """
create table if not exists docs (
    id         text primary key,
    title      text not null default '',
    body       text not null,
    updated_at timestamptz not null default now(),
    tsv        tsvector generated always as (
        setweight(to_tsvector('english', title), 'A') ||
        setweight(to_tsvector('english', body), 'B')) stored
);
create index if not exists docs_tsv on docs using gin (tsv);
"""


def load(corpus, conn=None):
    """Upsert BEIR docs ({id: {title, text}}). Idempotent: re-running changes nothing."""
    conn = conn or stack.postgres()
    conn.execute(SCHEMA)
    rows = [(i, (d.get("title") or "").strip(), d["text"].strip()) for i, d in corpus.items()]
    with conn.cursor() as cur:
        cur.executemany(
            """insert into docs (id, title, body) values (%s, %s, %s)
               on conflict (id) do update set title = excluded.title, body = excluded.body,
                 updated_at = now()
               where (docs.title, docs.body) is distinct from (excluded.title, excluded.body)""",
            rows)
    return conn.execute("select count(*) from docs").fetchone()[0]


def iter_docs(conn=None, batch=500):
    conn = conn or stack.postgres()
    with conn.cursor() as cur:
        cur.execute("select id, title, body from docs order by id")
        while rows := cur.fetchmany(batch):
            yield from rows
