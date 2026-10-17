"""Lesson 2 detectors, all scored against the injected incident schedule.

1. Static threshold rules: what an Alerting monitor checks every minute over the last 5 minutes.
   Thresholds come from the clean week: "the worst minute of a normal week, plus 10%", so a normal
   week never fires. Replayed here over week 2 with the same aggregations a monitor would run.
2. A seasonal traffic rule: requests vs the normal amount for that time of day (learned on week 1).
3. The Anomaly Detection plugin (Random Cut Forest), one model per endpoint, run as a historical
   analysis over both weeks. Unsupervised: it is never told what an incident looks like.

Scoring: alert minutes within 5 min of each other merge into one alert *episode*. An incident counts
as caught if an episode overlaps [start, end + 10 min]; time to detect = first alert - start.
Episodes that overlap no incident are false alarms.
"""
import json
import time
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from oslab import logpipe, logs, stack

WINDOW = 5      # minutes per rule evaluation
MARGIN = 1.10
DETECTOR = "shop-endpoints"


# ---------- per-minute metrics straight from OpenSearch aggregations ----------

def minute_metrics(client=None):
    client = client or stack.opensearch()
    frames = []
    for wk in range(2):
        lo = logs.START + timedelta(days=7 * wk)
        body = {"size": 0, "query": {"range": {"@timestamp": {
                    "gte": lo.isoformat(), "lt": (lo + timedelta(days=7)).isoformat()}}},
                "aggs": {"m": {"date_histogram": {"field": "@timestamp", "fixed_interval": "1m",
                                                  "min_doc_count": 0},
                               "aggs": {"errors": {"sum": {"field": "is_5xx"}},
                                        "login_fail": {"sum": {"field": "is_login_fail"}},
                                        "search": {"filter": {"term": {"endpoint": "/search"}},
                                                   "aggs": {"lat": {"sum": {"field": "latency_ms"}}}}}}}}
        res = client.search(index=logpipe.PATTERN, body=body, request_timeout=120)
        frames.append(pd.DataFrame([{
            "t": pd.Timestamp(b["key_as_string"]), "requests": b["doc_count"],
            "errors": b["errors"]["value"], "login_fail": b["login_fail"]["value"],
            "search_n": b["search"]["doc_count"], "search_lat": b["search"]["lat"]["value"]}
            for b in res["aggregations"]["m"]["buckets"]]))
    df = pd.concat(frames).set_index("t").sort_index()
    df.index = df.index.tz_convert("UTC") if df.index.tz else df.index.tz_localize("UTC")
    full = pd.date_range(logs.START, periods=logs.MINUTES, freq="1min", tz="UTC")
    return df.reindex(full, fill_value=0)


def rolling_features(m):
    r = m.rolling(WINDOW, min_periods=WINDOW).sum()
    f = pd.DataFrame(index=m.index)
    f["error_rate"] = r["errors"] / r["requests"].clip(lower=1)
    f["login_fail"] = r["login_fail"]
    f["search_ms"] = r["search_lat"] / r["search_n"].clip(lower=1)
    f["requests"] = r["requests"]
    # Seasonal expectation: average 5-min volume at this minute of day in week 1, smoothed over 30 min.
    w1 = f.loc[f.index < logs.TEST_START, "requests"]
    prof = w1.groupby(w1.index.hour * 60 + w1.index.minute).mean()
    prof = pd.concat([prof.iloc[-15:], prof, prof.iloc[:15]]).rolling(31, center=True).mean().iloc[15:-15]
    f["traffic_ratio"] = f["requests"] / prof.reindex(f.index.hour * 60 + f.index.minute).to_numpy()
    return f.dropna()


RULES = {  # feature, direction
    "error_rate_high": ("error_rate", "above"),
    "login_fail_high": ("login_fail", "above"),
    "search_slow": ("search_ms", "above"),
    "traffic_low_static": ("requests", "below"),
    "traffic_low_seasonal": ("traffic_ratio", "below"),
}


def calibrate(f):
    """Thresholds from the clean week only."""
    w1 = f[f.index < logs.TEST_START]
    th = {}
    for name, (col, d) in RULES.items():
        th[name] = float(w1[col].max() * MARGIN) if d == "above" else float(w1[col].min() / MARGIN)
    return th


def rule_alerts(f, th):
    w2 = f[f.index >= logs.TEST_START]
    out = {}
    for name, (col, d) in RULES.items():
        out[name] = (w2[col] > th[name]) if d == "above" else (w2[col] < th[name])
    return pd.DataFrame(out)


# ---------- scoring ----------

def episodes(alert_minutes, gap=5):
    """Sorted alert timestamps -> [(start, end)] merging alerts <= gap minutes apart."""
    eps = []
    for t in sorted(alert_minutes):
        if eps and t - eps[-1][1] <= pd.Timedelta(minutes=gap):
            eps[-1][1] = t
        else:
            eps.append([t, t])
    return [tuple(e) for e in eps]


def score(alert_minutes, incidents, grace=10):
    eps = episodes(alert_minutes)
    rows, used = [], set()
    for inc in incidents:
        lo, hi = pd.Timestamp(inc.start), pd.Timestamp(inc.end) + pd.Timedelta(minutes=grace)
        hits = [i for i, (a, b) in enumerate(eps) if a <= hi and b >= lo]
        used |= set(hits)
        first = min((max(eps[i][0], lo) for i in hits), default=None)
        rows.append({"kind": inc.kind, "severity": inc.severity, "start": inc.start,
                     "caught": bool(hits),
                     "minutes_to_detect": None if first is None else (first - lo).total_seconds() / 60})
    per = pd.DataFrame(rows)
    return per, {"caught": int(per.caught.sum()), "of": len(per),
                 "caught_big": int(per[per.severity == "big"].caught.sum()),
                 "caught_subtle": int(per[per.severity == "subtle"].caught.sum()),
                 "median_minutes_to_detect": float(per.minutes_to_detect.median()) if per.caught.any() else None,
                 "false_alarms": len(eps) - len(used)}


# ---------- Alerting plugin: the same rule as a real monitor ----------

def error_rate_monitor(threshold):
    """Query-level monitor for the error-rate rule. Created disabled: the logs are historical, so a
    live monitor (which always looks at the last 5 minutes before *now*) would never see them."""
    return {
        "type": "monitor", "name": "shop error rate", "monitor_type": "query_level_monitor",
        "enabled": False, "schedule": {"period": {"interval": 1, "unit": "MINUTES"}},
        "inputs": [{"search": {"indices": [logpipe.PATTERN], "query": {
            "size": 0, "query": {"range": {"@timestamp": {"gte": "{{period_end}}||-5m",
                                                          "lte": "{{period_end}}"}}},
            "aggs": {"errors": {"sum": {"field": "is_5xx"}}}}}}],
        "triggers": [{"name": "error rate above normal", "severity": "2", "condition": {"script": {
            "source": ("def n = ctx.results[0].hits.total.value; "
                       f"return n > 0 && ctx.results[0].aggregations.errors.value / n > {threshold:.5f}"),
            "lang": "painless"}}, "actions": []}],
    }


def create_monitor(threshold, client=None):
    client = client or stack.opensearch()
    try:
        found = client.transport.perform_request("POST", "/_plugins/_alerting/monitors/_search", body={
            "query": {"match_phrase": {"monitor.name": "shop error rate"}}})["hits"]["hits"]
    except Exception as e:  # the plugin's config index only exists after the first monitor
        if getattr(e, "status_code", None) != 404:
            raise
        found = []
    for h in found:
        client.transport.perform_request("DELETE", f"/_plugins/_alerting/monitors/{h['_id']}")
    return client.transport.perform_request("POST", "/_plugins/_alerting/monitors",
                                            body=error_rate_monitor(threshold))["_id"]


# ---------- Anomaly Detection plugin ----------

AD = "/_plugins/_anomaly_detection/detectors"


def detector_body(interval=1):
    feat = lambda name, agg: {"feature_name": name, "feature_enabled": True,  # noqa: E731
                              "aggregation_query": {name: agg}}
    return {
        "name": f"{DETECTOR}-{interval}m", "description": "Per-endpoint traffic, errors, latency and failed logins",
        "time_field": "@timestamp", "indices": [logpipe.PATTERN],
        "category_field": ["endpoint"],
        "feature_attributes": [feat("requests", {"value_count": {"field": "status"}}),
                               feat("errors", {"sum": {"field": "is_5xx"}}),
                               feat("latency", {"avg": {"field": "latency_ms"}}),
                               feat("login_fail", {"sum": {"field": "is_login_fail"}})],
        "detection_interval": {"period": {"interval": interval, "unit": "Minutes"}},
        "window_delay": {"period": {"interval": 1, "unit": "Minutes"}},
        "shingle_size": 8,
    }


def run_detector(interval=1, client=None, poll=15, timeout=3600):
    """(Re)create the detector, run historical analysis over both weeks, wait for it to finish."""
    client = client or stack.opensearch()
    tr = client.transport.perform_request
    try:
        found = tr("POST", f"{AD}/_search", body={"query": {"term": {"name.keyword": f"{DETECTOR}-{interval}m"}}})["hits"]["hits"]
    except Exception as e:  # the plugin's index only exists after the first detector
        if getattr(e, "status_code", None) != 404:
            raise
        found = []
    for h in found:
        try:
            tr("POST", f"{AD}/{h['_id']}/_stop?historical=true")
        except Exception:
            pass
        tr("DELETE", f"{AD}/{h['_id']}")
    det_id = tr("POST", AD, body=detector_body(interval))["_id"]
    start_ms = int(logs.START.timestamp() * 1000)
    end_ms = int((logs.START + timedelta(minutes=logs.MINUTES)).timestamp() * 1000)
    tr("POST", f"{AD}/{det_id}/_start", body={"start_time": start_ms, "end_time": end_ms})
    t0 = time.time()
    while time.time() - t0 < timeout:
        task = tr("GET", f"{AD}/{det_id}?task=true").get("historical_analysis_task", {})
        state = task.get("state")
        if state in ("FINISHED", "FAILED", "STOPPED"):
            if state != "FINISHED":
                raise RuntimeError(f"historical analysis {state}: {task.get('error')}")
            return det_id, task
        time.sleep(poll)
    raise TimeoutError("historical analysis did not finish")


def detector_results(det_id, task_id, client=None):
    """Every (minute, endpoint) the model graded as anomalous, with its grade and confidence.
    Pages through the hits: `entity` is a nested field, so a plain terms aggregation on it
    silently returns nothing."""
    client = client or stack.opensearch()
    rows, after = [], None
    while True:
        body = {"size": 5000, "_source": ["data_start_time", "entity", "anomaly_grade", "confidence"],
                "sort": [{"data_start_time": "asc"}, {"model_id": "asc"}],
                "query": {"bool": {"filter": [
                    {"term": {"detector_id": det_id}}, {"term": {"task_id": task_id}},
                    {"range": {"anomaly_grade": {"gt": 0}}}]}}}
        if after:
            body["search_after"] = after
        hits = client.transport.perform_request(
            "POST", f"{AD}/results/_search", body=body)["hits"]["hits"]
        rows += [{"t": pd.Timestamp(h["_source"]["data_start_time"], unit="ms", tz="UTC"),
                  "endpoint": h["_source"]["entity"][0]["value"],
                  "grade": h["_source"]["anomaly_grade"], "confidence": h["_source"]["confidence"]}
                 for h in hits]
        if len(hits) < 5000:
            break
        after = hits[-1]["sort"]
    return pd.DataFrame(rows, columns=["t", "endpoint", "grade", "confidence"])


def save(name, summary, per, out_dir="results"):
    Path(out_dir).mkdir(exist_ok=True)
    per = per.assign(start=per.start.astype(str))
    Path(out_dir, f"logs-{name}.json").write_text(json.dumps(
        {"detector": name, **summary, "incidents": per.to_dict("records")}, indent=2, default=str) + "\n")


def incidents_from_file(path="logs/shop.incidents.json"):
    from datetime import datetime
    return [logs.Incident(d["kind"], d["severity"], datetime.fromisoformat(d["start"]), d["minutes"])
            for d in json.loads(Path(path).read_text())]


def all_rule_minutes(alerts):
    return list(alerts.index[alerts.any(axis=1)])


def grade_minutes(res, min_grade=0.0, test_only=True):
    r = res[res.grade > min_grade]
    if test_only:
        r = r[r.t >= pd.Timestamp(logs.TEST_START)]
    return sorted(set(r.t))


def per_rule_table(alerts, incidents):
    """Which rule caught which incident: rows = incidents, columns = rules."""
    out = {}
    for name in alerts:
        per, s = score(list(alerts.index[alerts[name]]), incidents)
        out[name] = list(per.caught) + [s["false_alarms"]]
    idx = [f"{i.kind} ({i.severity})" for i in incidents] + ["false alarms"]
    return pd.DataFrame(out, index=idx)


def summarize_np(x):
    return {"n": int(len(x)), "median": float(np.median(x)) if len(x) else None}
