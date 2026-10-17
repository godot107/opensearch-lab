"""Lesson 1, part 2: search-box features beyond ranking.

- typo tolerance: `fuzziness: AUTO` lets a word match others within 1-2 edits. Measured on the test
  questions with one seeded typo each, against the same expert labels.
- highlighting: return the matching snippets with the words marked.
- synonyms: a synonym_graph filter at search time, so "heart attack" also finds "myocardial infarction".
- autocomplete: a search_as_you_type field over titles.
- explain: why a document got its score.
"""
import random
import re

from opensearchpy import helpers

from oslab import search, stack, store

FEATURE_INDEX = "scifact-features"

SYNONYMS = [
    "heart attack, myocardial infarction",
    "cancer, tumor, tumour, neoplasm, carcinoma",
    "kidney, renal",
    "liver, hepatic",
    "stroke, cerebrovascular accident",
    "high blood pressure, hypertension",
]


def build_feature_index(client=None):
    """Tuned BM25 settings plus a synonym-aware search analyzer and an autocomplete field."""
    client = client or stack.opensearch()
    cfg = search.tuned_config()
    if client.indices.exists(index=FEATURE_INDEX):
        client.indices.delete(index=FEATURE_INDEX)
    client.indices.create(index=FEATURE_INDEX, body={
        "settings": {
            "number_of_shards": 1, "number_of_replicas": 0,
            "similarity": {"default": {"type": "BM25", "k1": cfg["k1"], "b": cfg["b"]}},
            "analysis": {
                "filter": {
                    "en_stop": {"type": "stop", "stopwords": "_english_"},
                    "en_stem": {"type": "stemmer", "language": "english"},
                    "en_possessive": {"type": "stemmer", "language": "possessive_english"},
                    "med_synonyms": {"type": "synonym_graph", "synonyms": SYNONYMS},
                },
                "analyzer": {
                    # Same steps as the built-in "english" analyzer, so scores stay comparable.
                    "en_index": {"tokenizer": "standard",
                                 "filter": ["en_possessive", "lowercase", "en_stop", "en_stem"]},
                    # Synonyms only at search time: change the list without reindexing.
                    "en_search_syn": {"tokenizer": "standard",
                                      "filter": ["en_possessive", "lowercase", "med_synonyms",
                                                 "en_stop", "en_stem"]},
                },
            },
        },
        "mappings": {"dynamic": "strict", "properties": {
            "title": {"type": "text", "analyzer": "en_index", "search_analyzer": "en_index",
                      "fields": {"syn": {"type": "text", "analyzer": "en_index",
                                         "search_analyzer": "en_search_syn"},
                                 "suggest": {"type": "search_as_you_type"}}},
            "body": {"type": "text", "analyzer": "en_index",
                     "fields": {"syn": {"type": "text", "analyzer": "en_index",
                                        "search_analyzer": "en_search_syn"}}},
        }},
    })
    helpers.bulk(client, ({"_index": FEATURE_INDEX, "_id": i, "_source": {"title": t, "body": b}}
                          for i, t, b in store.iter_docs()), chunk_size=1000)
    client.indices.refresh(index=FEATURE_INDEX)
    return client.count(index=FEATURE_INDEX)["count"]


def add_typo(text, rng):
    """One realistic typo (swap, drop, double or replace a letter) in one word of 5+ letters."""
    words = text.split()
    idx = [i for i, w in enumerate(words) if re.fullmatch(r"[A-Za-z]{5,}", w)]
    if not idx:
        return text
    i = rng.choice(idx)
    w, j = words[i], rng.randrange(1, len(words[i]) - 1)
    kind = rng.choice(["swap", "drop", "double", "replace"])
    if kind == "swap":
        w = w[:j] + w[j + 1] + w[j] + w[j + 2:]
    elif kind == "drop":
        w = w[:j] + w[j + 1:]
    elif kind == "double":
        w = w[:j] + w[j] + w[j:]
    else:
        w = w[:j] + rng.choice("aeiourstln".replace(w[j].lower(), "")) + w[j + 1:]
    words[i] = w
    return " ".join(words)


def typo_queries(queries, seed=0):
    rng = random.Random(seed)
    return {q: add_typo(t, rng) for q, t in queries.items()}


def query(text, k=10, fuzzy=False, synonyms=False, highlight=False):
    cfg = search.tuned_config()
    sfx = ".syn" if synonyms else ""
    mm = {"query": text, "type": cfg["type"],
          "fields": [f"title{sfx}^{cfg['title_boost']}", f"body{sfx}"]}
    if fuzzy:
        # prefix_length 1: the first letter must be right (typos there are rare, and it keeps it fast).
        mm |= {"fuzziness": "AUTO", "prefix_length": 1}
    body = {"size": k, "_source": highlight, "query": {"multi_match": mm}}
    if highlight:
        body["highlight"] = {"fields": {f"body{sfx}": {"fragment_size": 120, "number_of_fragments": 2},
                                        f"title{sfx}": {"number_of_fragments": 0}},
                             "pre_tags": ["**"], "post_tags": ["**"]}
    return body


def run_many(queries, k=10, client=None, **opts):
    client = client or stack.opensearch()
    qids, out = list(queries), {}
    for i in range(0, len(qids), 100):
        chunk = qids[i:i + 100]
        lines = []
        for q in chunk:
            lines += [{"index": FEATURE_INDEX}, query(queries[q], k, **opts)]
        for q, r in zip(chunk, client.msearch(body=lines)["responses"]):
            out[q] = [h["_id"] for h in r["hits"]["hits"]]
    return out


def autocomplete(prefix, k=5, client=None):
    client = client or stack.opensearch()
    body = {"size": k, "_source": ["title"], "query": {"multi_match": {
        "query": prefix, "type": "bool_prefix",
        "fields": ["title.suggest", "title.suggest._2gram", "title.suggest._3gram"]}}}
    return [h["_source"]["title"] for h in client.search(index=FEATURE_INDEX, body=body)["hits"]["hits"]]


def explain(text, doc_id, client=None):
    client = client or stack.opensearch()
    return client.explain(index=FEATURE_INDEX, id=doc_id, body={"query": query(text)["query"]})
