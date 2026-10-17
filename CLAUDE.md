# opensearch-lab

Learn OpenSearch by measuring it on its three main use cases: full-text search, log analytics,
vector/hybrid search (+ key-value DBs). Trello #102 https://trello.com/c/2YogZt3j, from #4 item 9. Started 2026-10-17. Portfolio piece; docs stay plain-language for general readers.
Plan: docs/ROADMAP.md. Open work: TODO.md.

## Run

```bash
docker compose up -d                 # OpenSearch 3.1 :9200, Dashboards :5601, Postgres(pgvector) :5432
source .venv/bin/activate            # own venv, CPU only
python -m oslab load && python -m oslab sync && python -m oslab eval && python -m oslab report
pytest -q                            # tests/test_stack.py skips when the stack is down
```

## Decisions

- **Postgres is the source of truth**; OpenSearch indices are rebuilt from it (`oslab.sync.rebuild`).
  Chosen over DynamoDB because this is a learning lab: Postgres also gives the full-text (tsvector)
  and pgvector baselines each lesson compares OpenSearch against.
- **Explicit `strict` mappings**, never dynamic. Single shard, so BM25 IDF is exact on a small corpus.
- **Same tokenizer on both sides** (english analyzer / `to_tsvector('english')`) so lesson 1 compares
  ranking, not stemming.
- **Every lesson is scored against a known answer**: BEIR qrels for search, injected incidents for logs.
- **Security plugin is off and ports bind to 127.0.0.1.** Local only. Any AWS run uses a small managed
  domain torn down the same session; never OpenSearch Serverless (idle minimum, see CloudBurn).
- Image version 3.1.0 is pinned to one Amazon OpenSearch Service supports.

## Data

- `data/beir/scifact` is a symlink to rag-bedrock-compare's copy (`python -m oslab fetch` downloads
  it instead if the link is missing). Lesson 3 should reuse that project's bge-large vectors from its
  Chroma store rather than re-embed (GPU hold).
- `metrics.py` and `beir.py` are copied from rag-bedrock-compare so scores are comparable.

## Gotchas

- `websearch_to_tsquery` ANDs every word: 274/300 SciFact queries return nothing. `search.PG_OR`
  rewrites the tsquery to OR; the query text stays a bind parameter.
- Published BEIR BM25 on SciFact is ~0.665 nDCG@10; ours is 0.610 untuned (best_fields, title^2).
