# TODO

## Lesson 1: full-text search
- [x] Docker stack, Postgres source of truth, sync to OpenSearch, BM25 vs Postgres baseline
- [ ] Tune BM25 on the train split (fields type, title boost, analyzer); report test once
- [ ] Search-app features: highlighting, fuzziness, synonyms, autocomplete, `explain`
- [ ] Executed walkthrough notebook (notebooks/01_full_text.ipynb)

## Lesson 2: log analytics
- [ ] Synthetic log generator with injected incidents (known schedule = ground truth)
- [ ] Fluent Bit service in docker-compose, data stream + ISM retention policy
- [ ] Dashboards saved objects (export to repo), alerting monitor, anomaly detector
- [ ] Score detection: incidents caught, time to detect, false alarms

## Lesson 3: vector / hybrid
- [ ] Export bge-large vectors from rag-bedrock-compare's Chroma store
- [ ] knn_vector index, hybrid search pipeline; compare BM25 / vector / hybrid / pgvector

## Lesson 4: key-value (proposed)
- [ ] DynamoDB Local service; access-pattern table design; timings vs Postgres; change feed -> OpenSearch

## Wrap-up
- [ ] Blog post, public/private decision, push to github.com/godot107/opensearch-lab
