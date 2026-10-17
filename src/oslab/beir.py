"""Download and load a BEIR dataset: corpus, queries, and graded qrels.

The zip is untrusted input: members are extracted only if they resolve inside the
target directory, and nothing in it is executed.
"""
import csv
import json
import urllib.request
import zipfile
from pathlib import Path

BEIR_URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{name}.zip"


def fetch(name, data_dir):
    data_dir = Path(data_dir)
    target = data_dir / "beir"
    if (target / name / "corpus.jsonl").exists():
        return target / name
    target.mkdir(parents=True, exist_ok=True)
    zpath = target / f"{name}.zip"
    print(f"Downloading {BEIR_URL.format(name=name)}")
    urllib.request.urlretrieve(BEIR_URL.format(name=name), zpath)
    root = target.resolve()
    with zipfile.ZipFile(zpath) as zf:
        for member in zf.namelist():
            if not (root / member).resolve().is_relative_to(root):
                raise ValueError(f"unsafe path in zip: {member}")
        zf.extractall(root)
    zpath.unlink()
    return target / name


def doc_text(doc):
    """The exact text both backends embed: title, then body."""
    title = (doc.get("title") or "").strip()
    return f"{title}\n\n{doc['text'].strip()}" if title else doc["text"].strip()


def load(ds_dir, split="test"):
    ds_dir = Path(ds_dir)
    corpus = {}
    with open(ds_dir / "corpus.jsonl") as f:
        for line in f:
            d = json.loads(line)
            corpus[str(d["_id"])] = d
    qrels = {}
    with open(ds_dir / "qrels" / f"{split}.tsv") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            score = int(row["score"])
            if score > 0:
                qrels.setdefault(str(row["query-id"]), {})[str(row["corpus-id"])] = score
    queries = {}
    with open(ds_dir / "queries.jsonl") as f:
        for line in f:
            q = json.loads(line)
            if str(q["_id"]) in qrels:
                queries[str(q["_id"])] = q["text"]
    return corpus, queries, qrels
