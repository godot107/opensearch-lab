# Roadmap

Three lessons, one per thing OpenSearch is actually used for. Each one runs on the local Docker
stack, is scored against a known answer, and ends with "when would I *not* use OpenSearch for this?"

## Lesson 1: Full-text search (started)

**Question:** How much better is a real search engine than the database I already have?

- Data: BEIR SciFact (5,183 paper abstracts, 300 test questions with expert relevance labels),
  the same set used in rag-bedrock-compare, so lesson 3 can be compared with that project.
- Postgres holds the documents; `oslab sync` rebuilds the OpenSearch index from it.
- Baseline (2026-10-17): OpenSearch BM25 nDCG@10 0.610 vs Postgres any-word 0.368 vs Postgres
  all-words 0.072 (no results at all for 274 of 300 queries).
- Next:
  - Tune toward the published BEIR BM25 score (~0.665): `most_fields` vs `best_fields`, title
    boost, standard vs english analyzer. Pick settings on the **train** split, then report test once.
  - Show the search-app features Postgres lacks or makes awkward: highlighting, typo tolerance
    (`fuzziness`), synonyms, `search_as_you_type` autocomplete, and explain why a doc ranked where it did.
  - A short look at analyzers with `_analyze`: what "english" does to a sentence.
- Skills shown: mappings, analyzers, BM25, relevance evaluation, a source-of-truth → index sync.

## Lesson 2: Log analytics

**Question:** Can I find an incident in a pile of logs, and get alerted to it, before a user complains?

- Data: a synthetic log generator for a fake web shop with **incidents injected at known times**
  (error spike, slow endpoint, credential-stuffing burst). Known answer = the injected schedule.
  Optional real data afterwards: Loghub HDFS (has labeled anomalies).
- Ingest with Fluent Bit (or Data Prepper) into daily indices behind a data stream.
- Index State Management policy: hot → delete after N days (what keeps log clusters affordable).
- Dashboards: errors over time, p95 latency by endpoint, top IPs by failed logins.
- Alerting monitor + Anomaly Detection plugin: score how many injected incidents each catches,
  how fast, and how many false alarms.
- Skills shown: data streams, ISM, aggregations, Dashboards, alerting, anomaly detection.

## Lesson 3: Vector and hybrid search

**Question:** Does meaning-based search beat keyword search, and does combining them beat both?

- Same SciFact data and labels as lesson 1. Reuse rag-bedrock-compare's bge-large embeddings
  (already computed, in its Chroma store), so nothing needs the GPU. Query embeddings for 300
  questions are cheap on CPU.
- `knn_vector` field (HNSW), then a search pipeline with normalization + score combination
  for hybrid. Compare BM25, vector, hybrid, and pgvector on the same Postgres rows.
- Compare against rag-bedrock-compare's local-bge result (nDCG@10 0.744).
- Skills shown: k-NN indexes, search pipelines, hybrid scoring, vector DB trade-offs.

## Later / optional

- One AWS run of a lesson on a small managed domain, torn down the same session.
  **Never OpenSearch Serverless** for a lab (idle minimum is hundreds of $/month).
- Blog post and executed walkthrough notebook per lesson.
