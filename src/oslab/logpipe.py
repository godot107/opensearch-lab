"""Lesson 2 index set-up: what a log cluster needs before the first line arrives.

- An index template, so every new daily index logs-shop-* gets the right field types
  (keyword for things you filter and group by, numbers for things you average).
- An ISM (Index State Management) policy: hot for 7 days, read-only and shrunk to one segment,
  deleted after 30. Retention is what keeps a log cluster's disk, and bill, from growing forever.
"""
import time

from oslab import stack

TEMPLATE = "logs-shop"
POLICY = "logs-shop-retention"
PATTERN = "logs-shop-*"

MAPPING = {
    "dynamic": "strict",
    "properties": {
        "@timestamp": {"type": "date"},
        "service": {"type": "keyword"},
        "endpoint": {"type": "keyword"},
        "method": {"type": "keyword"},
        "status": {"type": "short"},
        "latency_ms": {"type": "float"},
        "client_ip": {"type": "ip"},
        "user": {"type": "keyword"},
        "level": {"type": "keyword"},
        "is_5xx": {"type": "byte"},
        "is_login_fail": {"type": "byte"},
    },
}

ISM = {"policy": {
    "description": "Shop logs: hot 7 days, then read-only and force-merged, deleted at 30 days.",
    "default_state": "hot",
    "states": [
        {"name": "hot", "actions": [],
         "transitions": [{"state_name": "warm", "conditions": {"min_index_age": "7d"}}]},
        {"name": "warm", "actions": [{"read_only": {}}, {"force_merge": {"max_num_segments": 1}}],
         "transitions": [{"state_name": "delete", "conditions": {"min_index_age": "30d"}}]},
        {"name": "delete", "actions": [{"delete": {}}], "transitions": []},
    ],
    "ism_template": [{"index_patterns": [PATTERN], "priority": 100}],
}}


def setup(client=None):
    client = client or stack.opensearch()
    client.indices.put_index_template(name=TEMPLATE, body={
        "index_patterns": [PATTERN], "priority": 100,
        "template": {"settings": {"number_of_shards": 1, "number_of_replicas": 0,
                                  "refresh_interval": "5s"},
                     "mappings": MAPPING}})
    path = f"/_plugins/_ism/policies/{POLICY}"
    try:
        cur = client.transport.perform_request("GET", path)
        client.transport.perform_request(
            "PUT", f"{path}?if_seq_no={cur['_seq_no']}&if_primary_term={cur['_primary_term']}", body=ISM)
    except Exception as e:
        if getattr(e, "status_code", None) != 404:
            raise
        client.transport.perform_request("PUT", path, body=ISM)


def count(client=None):
    client = client or stack.opensearch()
    if not client.indices.exists(index=PATTERN):
        return 0
    client.indices.refresh(index=PATTERN)
    return client.count(index=PATTERN)["count"]


def wait_for(expected, timeout=1800, client=None):
    """Block until Fluent Bit has delivered `expected` docs (or stops making progress)."""
    t0, last, stalled = time.time(), -1, 0
    while (n := count(client)) < expected and time.time() - t0 < timeout:
        stalled = stalled + 1 if n == last else 0
        if stalled >= 12:
            raise RuntimeError(f"ingestion stalled at {n}/{expected}: check `docker compose logs fluent-bit`")
        last = n
        time.sleep(10)
    return n
