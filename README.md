# opensearch-lab

**What is OpenSearch good for, and when is the database you already have good enough?**

OpenSearch is a search engine you run yourself (an open-source fork of Elasticsearch). Instead of
storing rows like a normal database, it builds an *inverted index*, like the index at the back of a
book that lists every page a word appears on. That makes "find the documents that best match these
words, ranked, out of millions" fast.

This project learns OpenSearch by testing it on the three jobs it's most used for, each scored
against a known right answer:

1. **Full-text search**: a search box over documents (started)
2. **Log analytics**: finding and alerting on problems in application logs (planned)
3. **Vector / hybrid search**: matching by meaning, the search step behind AI chat assistants (planned)

Everything runs locally in Docker for $0. Details: [docs/ROADMAP.md](docs/ROADMAP.md).

## First result: a search engine vs. Postgres

The test set is **SciFact**: 5,183 scientific paper abstracts and 300 questions, each with expert
labels saying which abstracts answer it. Every method returns its top 10 abstracts per question.

| Method | nDCG@10 | Recall@10 | Typical time per search |
|---|---|---|---|
| OpenSearch (BM25 ranking) | **0.610** | **0.750** | 4 ms |
| Postgres full-text, any word may match | 0.368 | 0.514 | 39 ms |
| Postgres full-text, every word must match | 0.072 | 0.073 | 11 ms |

How to read it:
- **Recall@10**: the share of the right answers that showed up in the top 10.
- **nDCG@10**: also rewards putting the right answers near the *top*. 1.0 is perfect.
- The "every word must match" setting is Postgres's usual default for search boxes. On real
  questions it found **nothing at all** for 274 of the 300, because no abstract contains every word
  of a long question.
- Allowing any word to match fixes that, but Postgres's ranking is much weaker than BM25. BM25
  rewards rare words (like "mesenchymal") more than common ones and doesn't over-reward long documents.

The documents live in **Postgres** (the source of truth), and the OpenSearch index is rebuilt from
it. That's the normal set-up: a search index is a fast, rebuildable copy, not where the real data lives.

## Run it

```bash
docker compose up -d                       # OpenSearch :9200, Dashboards :5601, Postgres :5432
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt && pip install -e .
python -m oslab fetch                      # download SciFact -> data/beir/scifact
python -m oslab load                       # documents -> Postgres
python -m oslab sync                       # Postgres -> OpenSearch index
python -m oslab eval && python -m oslab report
pytest -q                                  # stack tests skip if Docker is down
docker compose down                        # add -v to also delete the data
```

Needs Docker and about 3 GB of free RAM. On Linux, OpenSearch also needs
`sudo sysctl -w vm.max_map_count=262144` if your system's value is lower.
