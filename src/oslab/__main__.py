"""python -m oslab <cmd>

lesson 1: fetch | load | sync | tune | eval | report
lesson 2: logs (generate + template/ISM; then start Fluent Bit) | detect (rules + anomaly detection)
lesson 3: hybrid (then eval)
lesson 4: kv
"""
import argparse
import json
from pathlib import Path

from oslab import beir, evaluate, hybrid, search, stack, store, sync, tune

DATASET = "scifact"


def main(argv=None):
    p = argparse.ArgumentParser(prog="oslab")
    p.add_argument("cmd", choices=["fetch", "load", "sync", "tune", "hybrid", "eval", "report", "logs", "detect", "kv"])
    p.add_argument("--backend", choices=search.BACKENDS + hybrid.BACKENDS + ("all",), default="all")
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
    elif a.cmd == "hybrid":
        print(f"{hybrid.INDEX}: {hybrid.build_index()} docs; pgvector rows: {hybrid.load_pgvector()}")
        hybrid.create_pipelines()
        print(json.dumps(hybrid.tune_weight()[1], indent=2))
    elif a.cmd == "eval":
        _, queries, qrels = beir.load(ds_dir)
        for b in search.BACKENDS + hybrid.BACKENDS if a.backend == "all" else (a.backend,):
            print(json.dumps(evaluate.run(b, queries, qrels)))
    elif a.cmd == "logs":
        from oslab import logpipe, logs
        print(f"{logs.generate()} lines -> logs/shop.log")
        logpipe.setup()
        print("next: docker compose --profile logs up -d fluent-bit")
    elif a.cmd == "detect":
        lesson2()
    elif a.cmd == "kv":
        lesson4()
    elif a.cmd == "report":
        rows = [json.loads(f.read_text()) for f in sorted(Path("results").glob(f"{DATASET}-*.json"))]
        print("| backend | nDCG@10 | recall@10 | MRR@10 | p50 ms | empty |\n|---|---|---|---|---|---|")
        for r in sorted(rows, key=lambda r: -r["ndcg@10"]):
            print(f"| {r['backend']} | {r['ndcg@10']:.3f} | {r['recall@10']:.3f} | {r['mrr@10']:.3f} "
                  f"| {r['p50_ms']:.1f} | {r['empty_results']} |")


def lesson2():
    import pandas as pd
    from oslab import detect, logpipe, logs
    n = logpipe.wait_for(sum(1 for _ in open("logs/shop.log")))
    print(f"{n} log documents indexed")
    inc = detect.incidents_from_file()
    f = detect.rolling_features(detect.minute_metrics())
    th = detect.calibrate(f)
    alerts = detect.rule_alerts(f, th)
    per, s = detect.score(detect.all_rule_minutes(alerts), inc)
    detect.save("rules", s, per)
    detect.create_monitor(th["error_rate_high"])
    print("rules", s)
    test_start = pd.Timestamp(logs.TEST_START)
    curve, runs = [], {}
    for iv in (1, 5):
        det, task = detect.run_detector(iv)
        runs[iv] = {"detector_id": det, "task_id": task["task_id"]}
        res = detect.detector_results(det, task["task_id"])
        res.to_csv(f"results/ad-{iv}m-anomalies.csv", index=False)
        for g in (0, 0.2, 0.4, 0.6, 0.8, 0.9, 0.95):
            _, sg = detect.score(detect.grade_minutes(res, g), inc)
            clean = sorted(set(res[(res.grade > g) & (res.t < test_start)].t))
            curve.append({"interval_min": iv, "min_grade": g,
                          "clean_week_false_alarms": len(detect.episodes(clean)), **sg})
        per_ad, s_ad = detect.score(detect.grade_minutes(res, 0.8), inc)
        detect.save(f"ad-{iv}m", s_ad | {"min_grade": 0.8}, per_ad)
        print(f"anomaly detection {iv}m", s_ad)
        if iv == 5:
            both = sorted(set(detect.all_rule_minutes(alerts)) | set(detect.grade_minutes(res, 0.8)))
            per_b, s_b = detect.score(both, inc)
            detect.save("rules+ad-5m", s_b, per_b)
            print("rules + anomaly detection 5m", s_b)
    pd.DataFrame(curve).to_csv("results/ad-threshold-curve.csv", index=False)
    Path("logs/ad_runs.json").write_text(json.dumps(runs))


def lesson4():
    from collections import Counter

    import numpy as np
    from oslab import kv
    if not kv.is_up():
        raise SystemExit("DynamoDB Local is down: docker compose up -d dynamodb")
    kv.create_tables()
    print("docs", kv.load_docs())
    views = kv.synthetic_views()
    print("views", kv.load_views(views))
    pg = kv.pg_patterns()
    popular = Counter(v["doc"] for v in views).most_common(1)[0][0]
    doc, user = "4983", "u0042"
    rows = {"get_doc": (kv.timed(lambda: kv.get_doc(doc)), kv.timed(lambda: pg["get_doc"](doc))[1]),
            "user_recent": (kv.timed(lambda: kv.user_recent(user)), kv.timed(lambda: pg["user_recent"](user))[1]),
            "readers_scan": (kv.timed(lambda: kv.readers_by_scan(popular), reps=5),
                             kv.timed(lambda: pg["readers"](popular), reps=5)[1])}
    kv.add_doc_index()
    rows["readers_gsi"] = (kv.timed(lambda: kv.readers_by_gsi(popular)), None)
    patterns = [{"pattern": k, "items": r[0], "read_units": r[1], "dynamodb_ms": round(ms, 2),
                 "postgres_ms": pgms and round(pgms, 2), "usd_per_million": round(kv.usd(r[1]), 2)}
                for k, ((r, ms), pgms) in rows.items()]
    Path("results/kv-patterns.json").write_text(json.dumps(
        {"table": kv.table_stats(), "popular_doc": popular, "patterns": patterns,
         "price_verified": False}, indent=2) + "\n")
    for r in patterns:
        print(r)

    os_ = stack.opensearch()
    if os_.indices.exists(index=kv.SEARCH_INDEX):
        os_.indices.delete(index=kv.SEARCH_INDEX)
    sync_ = kv.StreamSync(os_)
    sync_.poll()
    initial = kv.consistent(os_)
    t = kv.ddb().Table(kv.DOCS)
    keep = {i: t.get_item(Key={"id": i})["Item"] for i in ("4983", "31715818")}
    t.update_item(Key={"id": "4983"}, UpdateExpression="set title = :t",
                  ExpressionAttributeValues={":t": "EDITED title for the sync test"})
    t.delete_item(Key={"id": "31715818"})
    sync_.poll()
    after = kv.consistent(os_)
    for item in keep.values():
        t.put_item(Item=item)
    sync_.poll()
    lag = sync_.lag_ms(20)
    res = {"initial": initial, "after_edit_delete": after, "restored": kv.consistent(os_),
           "lag_ms": {"p50": float(np.median(lag)), "p95": float(np.percentile(lag, 95)),
                      "max": float(max(lag)), "poll_every_ms": 200}}
    Path("results/kv-sync.json").write_text(json.dumps(res, indent=2) + "\n")
    print(json.dumps(res))


if __name__ == "__main__":
    main()
