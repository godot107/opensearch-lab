"""Score a retriever on a BEIR split and save results/<dataset>-<backend>.json."""
import json
import time
from pathlib import Path

from oslab import metrics, search


def run(backend, queries, qrels, k=10, out_dir="results", dataset="scifact"):
    fn = search.retriever(backend)
    fn(next(iter(queries.values())), k)  # warm-up: first query pays connection/cache costs
    ranked, lat = {}, []
    for qid, text in queries.items():
        t0 = time.perf_counter()
        ranked[qid] = fn(text, k)
        lat.append(time.perf_counter() - t0)
    res = {"dataset": dataset, "backend": backend, "k": k}
    res |= metrics.score_run(ranked, qrels, k) | metrics.latency_summary(lat)
    res["empty_results"] = sum(1 for q in qrels if not ranked.get(q))
    Path(out_dir).mkdir(exist_ok=True)
    Path(out_dir, f"{dataset}-{backend}.json").write_text(json.dumps(res, indent=2) + "\n")
    return res
