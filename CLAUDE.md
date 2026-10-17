# opensearch-lab

Learn OpenSearch by measuring it on its three main use cases: full-text search, log analytics,
vector/hybrid search (+ key-value DBs). Trello #102 https://trello.com/c/2YogZt3j, from #4 item 9. Started 2026-10-17. Portfolio piece; docs stay plain-language for general readers.
Plan: docs/ROADMAP.md. Open work: TODO.md.

## Run

```bash
docker compose up -d                 # OpenSearch 3.1 :9200, Dashboards :5601, Postgres(pgvector) :5432
source .venv/bin/activate            # own venv, CPU only
python -m oslab load && python -m oslab sync && python -m oslab tune && python -m oslab hybrid
python -m oslab eval && python -m oslab report        # lessons 1 + 3
python -m oslab logs; docker compose --profile logs up -d fluent-bit; python -m oslab detect   # lesson 2
python -m oslab kv                                    # lesson 4
python notebooks/build.py && (execute each notebook with nbconvert --execute --inplace)
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

- `_msearch` can't take a named search pipeline (URL param or header line): hybrid uses parallel
  single searches (`hybrid.search_many`).
- Alerting and AD plugins: their config indices don't exist until the first monitor/detector, so
  the first `_search` 404s (handled). AD results: `entity` is nested; aggregate on it and you get
  nothing silently: page through hits instead (`detect.detector_results`).
- AD (Random Cut Forest) is stochastic: re-runs of the same detector differ (1-min run caught 8/8 at
  grade>0.4 once, 6/8 the next time). Results in results/ come from logs/ad_runs.json's runs.
- AD historical analysis: ~2 min for 2 weeks x 6 endpoints at 1-min interval.
- DynamoDB Local: `ddb().meta.client` is the resource's client and takes plain Python values, not
  `{"S": ...}` typed values. Timings are the emulator's; only read units transfer to AWS.
- Backend dispatch is by exact name (an `os-`/`pg-` prefix check once routed lesson-3 backends to
  lesson-1 code and produced identical scores: check for that symptom).

- `websearch_to_tsquery` ANDs every word: 274/300 SciFact queries return nothing. `search.PG_OR`
  rewrites the tsquery to OR; the query text stays a bind parameter.
- Published BEIR BM25 on SciFact is ~0.665 nDCG@10; ours is 0.610 untuned (best_fields, title^2).
