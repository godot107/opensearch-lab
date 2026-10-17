"""Export bge-large vectors for lesson 3, using rag-bedrock-compare's environment (chromadb, torch).

    ../rag-bedrock-compare/.venv/bin/python scripts/export_bge.py

- Documents: copied from rag-bedrock-compare's Chroma store (already embedded there; nothing re-runs).
- Queries: all 1,109 SciFact questions (train + test) embedded with the same model and BGE's
  query prefix. Short texts, about a minute on CPU (the GPU is not needed).
Writes data/vectors/scifact-bge.npz: doc_ids, doc_vecs, query_ids, query_vecs (float32, normalised).
"""
import json
import sys
from pathlib import Path

import numpy as np

RBC = Path(__file__).resolve().parents[2] / "rag-bedrock-compare"
OUT = Path(__file__).resolve().parents[1] / "data" / "vectors" / "scifact-bge.npz"
PREFIX = "Represent this sentence for searching relevant passages: "


def main():
    import chromadb
    col = chromadb.PersistentClient(path=str(RBC / "data" / "chroma")).get_collection("scifact-bge")
    got = col.get(include=["embeddings"], limit=col.count())
    doc_ids, doc_vecs = got["ids"], np.asarray(got["embeddings"], dtype=np.float32)

    import os
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("BAAI/bge-large-en-v1.5", device=sys.argv[1] if len(sys.argv) > 1 else "cpu",
                                cache_folder="/mnt/ssd/hf-cache")
    queries = [json.loads(l) for l in open(RBC / "data" / "beir" / "scifact" / "queries.jsonl")]
    q_ids = [str(q["_id"]) for q in queries]
    q_vecs = model.encode([PREFIX + q["text"] for q in queries], batch_size=32,
                          normalize_embeddings=True, show_progress_bar=True).astype(np.float32)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez(OUT, doc_ids=np.array(doc_ids), doc_vecs=doc_vecs, query_ids=np.array(q_ids), query_vecs=q_vecs)
    print(OUT, doc_vecs.shape, q_vecs.shape)


if __name__ == "__main__":
    main()
