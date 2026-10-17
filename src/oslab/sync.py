"""Rebuild an OpenSearch index from Postgres.

Explicit mapping, never dynamic: OpenSearch guesses field types from the first
document it sees, and a wrong guess can only be fixed by reindexing.
The 'english' analyzer lowercases, drops stopwords and stems ("cells" -> "cell"),
matching what Postgres's to_tsvector('english', ...) does, so lesson 1 compares
ranking functions rather than tokenizers.
"""
from opensearchpy import helpers

from oslab import stack, store

DOCS_INDEX = "scifact-docs"

DOCS_MAPPING = {
    "settings": {"number_of_shards": 1, "number_of_replicas": 0},
    "mappings": {
        "dynamic": "strict",
        "properties": {
            "title": {"type": "text", "analyzer": "english"},
            "body": {"type": "text", "analyzer": "english"},
        },
    },
}


def rebuild(index=DOCS_INDEX, os_client=None, pg_conn=None):
    """Drop and rebuild the index. Single shard: BM25's IDF is computed per shard,
    so one shard keeps scores exact on a corpus this small."""
    os_client = os_client or stack.opensearch()
    if os_client.indices.exists(index=index):
        os_client.indices.delete(index=index)
    os_client.indices.create(index=index, body=DOCS_MAPPING)
    actions = ({"_index": index, "_id": i, "_source": {"title": t, "body": b}}
               for i, t, b in store.iter_docs(pg_conn))
    ok, errors = helpers.bulk(os_client, actions, chunk_size=1000, raise_on_error=False)
    if errors:
        raise RuntimeError(f"{len(errors)} docs failed to index, first: {errors[0]}")
    os_client.indices.refresh(index=index)  # make writes searchable now, not in ~1 s
    return os_client.count(index=index)["count"]
