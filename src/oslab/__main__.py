"""python -m oslab fetch | load | sync | tune | eval [--backend B] | report"""
import argparse
import json
from pathlib import Path

from oslab import beir, evaluate, search, stack, store, sync, tune

DATASET = "scifact"


def main(argv=None):
    p = argparse.ArgumentParser(prog="oslab")
    p.add_argument("cmd", choices=["fetch", "load", "sync", "tune", "eval", "report"])
    p.add_argument("--backend", choices=search.BACKENDS + ("all",), default="all")
    a = p.parse_args(argv)

    if a.cmd != "fetch" and a.cmd != "report" and not stack.is_up():
        raise SystemExit("stack is down: run `docker compose up -d` first")
    ds_dir = Path("data/beir") / DATASET
    if a.cmd == "fetch":
        print(beir.fetch(DATASET, "data"))
    elif a.cmd == "load":
        corpus, _, _ = beir.load(ds_dir)
        print(f"postgres docs: {store.load(corpus)}")
    elif a.cmd == "sync":
        print(f"opensearch {sync.DOCS_INDEX}: {sync.rebuild()} docs")
    elif a.cmd == "tune":
        out, rows = tune.sweep()
        print(json.dumps(out, indent=2))
    elif a.cmd == "eval":
        _, queries, qrels = beir.load(ds_dir)
        for b in search.BACKENDS if a.backend == "all" else (a.backend,):
            print(json.dumps(evaluate.run(b, queries, qrels)))
    elif a.cmd == "report":
        rows = [json.loads(f.read_text()) for f in sorted(Path("results").glob(f"{DATASET}-*.json"))]
        print("| backend | nDCG@10 | recall@10 | MRR@10 | p50 ms | empty |\n|---|---|---|---|---|---|")
        for r in sorted(rows, key=lambda r: -r["ndcg@10"]):
            print(f"| {r['backend']} | {r['ndcg@10']:.3f} | {r['recall@10']:.3f} | {r['mrr@10']:.3f} "
                  f"| {r['p50_ms']:.1f} | {r['empty_results']} |")


if __name__ == "__main__":
    main()
