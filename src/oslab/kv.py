"""Lesson 4: a key-value database (DynamoDB, run locally) next to Postgres and OpenSearch.

A key-value store answers exactly the questions its keys were designed for, very fast, at a cost that
grows with the data *returned*, not the data *stored*. Anything else means reading the whole table.

Tables:
- kv-docs:  partition key `id`. The SciFact abstracts again. Stream enabled (change feed).
- kv-views: partition key `user`, sort key `viewed_at` ("what did this user read, newest first").
            Synthetic reading history. A global secondary index `by-doc` (partition key `doc`) is
            added later to show how to fix an access pattern the table wasn't designed for.

Read cost is counted in read capacity units (RCUs): one RCU = one strongly consistent read of up to
4 KB. DynamoDB Local reports them exactly as AWS would, which is what makes the cost lesson possible.
"""
import os
import time

import boto3
import numpy as np
from boto3.dynamodb.conditions import Key
from opensearchpy import helpers

from oslab import stack, store

DDB_URL = os.environ.get("OSLAB_DDB_URL", "http://localhost:8000")
DOCS, VIEWS, GSI = "kv-docs", "kv-views", "by-doc"
SEARCH_INDEX = "kv-docs-search"
# On-demand list prices, us-east-1, NOT verified against the AWS pricing page this session.
USD_PER_MILLION_READ_UNITS = 0.125


def ddb():
    # DynamoDB Local accepts any credentials; these dummy values never reach AWS.
    return boto3.resource("dynamodb", endpoint_url=DDB_URL, region_name="us-east-1",
                          aws_access_key_id="local", aws_secret_access_key="local")


def client():
    return ddb().meta.client


def streams():
    return boto3.client("dynamodbstreams", endpoint_url=DDB_URL, region_name="us-east-1",
                        aws_access_key_id="local", aws_secret_access_key="local")


def is_up():
    try:
        client().list_tables()
        return True
    except Exception:
        return False


# ---------- tables ----------

def create_tables():
    c = client()
    for name in (DOCS, VIEWS):
        if name in c.list_tables()["TableNames"]:
            c.delete_table(TableName=name)
    c.create_table(TableName=DOCS, BillingMode="PAY_PER_REQUEST",
                   AttributeDefinitions=[{"AttributeName": "id", "AttributeType": "S"}],
                   KeySchema=[{"AttributeName": "id", "KeyType": "HASH"}],
                   StreamSpecification={"StreamEnabled": True, "StreamViewType": "NEW_AND_OLD_IMAGES"})
    c.create_table(TableName=VIEWS, BillingMode="PAY_PER_REQUEST",
                   AttributeDefinitions=[{"AttributeName": "user", "AttributeType": "S"},
                                         {"AttributeName": "viewed_at", "AttributeType": "S"}],
                   KeySchema=[{"AttributeName": "user", "KeyType": "HASH"},
                              {"AttributeName": "viewed_at", "KeyType": "RANGE"}])
    for name in (DOCS, VIEWS):
        c.get_waiter("table_exists").wait(TableName=name)


def load_docs():
    t = ddb().Table(DOCS)
    n = 0
    with t.batch_writer() as w:
        for i, title, body in store.iter_docs():
            w.put_item(Item={"id": i, "title": title, "body": body})
            n += 1
    return n


def synthetic_views(n_users=1000, per_user=50, seed=0):
    """Reading history: each user reads `per_user` abstracts, popular ones more often (Zipf-like)."""
    rng = np.random.default_rng(seed)
    ids = [r[0] for r in store.iter_docs()]
    popularity = 1 / np.arange(1, len(ids) + 1) ** 0.8
    popularity /= popularity.sum()
    order = rng.permutation(len(ids))
    t0 = 1_790_000_000
    out = []
    for u in range(n_users):
        docs = rng.choice(len(ids), per_user, p=popularity)
        times = np.sort(rng.integers(t0, t0 + 30 * 86400, per_user))
        for j, (d, ts) in enumerate(zip(docs, times)):
            iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(int(ts)))
            out.append({"user": f"u{u:04d}", "viewed_at": f"{iso}#{j:02d}", "doc": ids[order[d]],
                        "seconds": int(rng.integers(5, 600))})
    return out


def load_views(views):
    t = ddb().Table(VIEWS)
    with t.batch_writer() as w:
        for v in views:
            w.put_item(Item=v)
    conn = stack.postgres()
    conn.execute("drop table if exists views")
    conn.execute("create table views (\"user\" text, viewed_at text, doc text, seconds int, "
                 "primary key (\"user\", viewed_at))")
    with conn.cursor() as cur, cur.copy('copy views ("user", viewed_at, doc, seconds) from stdin') as cp:
        for v in views:
            cp.write_row((v["user"], v["viewed_at"], v["doc"], v["seconds"]))
    conn.execute("analyze views")
    return len(views)


def add_doc_index():
    """The fix for 'who read document X?': a global secondary index keyed by doc."""
    c = client()
    desc = c.describe_table(TableName=VIEWS)["Table"]
    if any(g["IndexName"] == GSI for g in desc.get("GlobalSecondaryIndexes", [])):
        return
    c.update_table(TableName=VIEWS,
                   AttributeDefinitions=[{"AttributeName": "doc", "AttributeType": "S"},
                                         {"AttributeName": "viewed_at", "AttributeType": "S"}],
                   GlobalSecondaryIndexUpdates=[{"Create": {
                       "IndexName": GSI,
                       "KeySchema": [{"AttributeName": "doc", "KeyType": "HASH"},
                                     {"AttributeName": "viewed_at", "KeyType": "RANGE"}],
                       "Projection": {"ProjectionType": "KEYS_ONLY"}}}])
    while True:
        g = [g for g in c.describe_table(TableName=VIEWS)["Table"].get("GlobalSecondaryIndexes", [])
             if g["IndexName"] == GSI]
        if g and g[0]["IndexStatus"] == "ACTIVE":
            return
        time.sleep(0.5)


# ---------- access patterns: (DynamoDB call, Postgres equivalent), timed with read cost ----------

def timed(fn, reps=50):
    fn()  # warm-up
    out, t = [], []
    for _ in range(reps):
        t0 = time.perf_counter()
        out = fn()
        t.append(time.perf_counter() - t0)
    return out, float(np.median(t) * 1000)


def get_doc(doc_id):
    r = ddb().Table(DOCS).get_item(Key={"id": doc_id}, ReturnConsumedCapacity="TOTAL")
    return 1, r["ConsumedCapacity"]["CapacityUnits"]


def user_recent(user, k=20):
    r = ddb().Table(VIEWS).query(KeyConditionExpression=Key("user").eq(user), ScanIndexForward=False,
                                 Limit=k, ReturnConsumedCapacity="TOTAL")
    return len(r["Items"]), r["ConsumedCapacity"]["CapacityUnits"]


def readers_by_scan(doc_id):
    """No key for this question: read every item in the table and filter afterwards."""
    t, kwargs, n, cu = ddb().Table(VIEWS), {}, 0, 0.0
    while True:
        r = t.scan(FilterExpression=Key("doc").eq(doc_id), ReturnConsumedCapacity="TOTAL", **kwargs)
        n += len(r["Items"])
        cu += r["ConsumedCapacity"]["CapacityUnits"]
        if "LastEvaluatedKey" not in r:
            return n, cu
        kwargs["ExclusiveStartKey"] = r["LastEvaluatedKey"]


def readers_by_gsi(doc_id):
    t, kwargs, n, cu = ddb().Table(VIEWS), {}, 0, 0.0
    while True:
        r = t.query(IndexName=GSI, KeyConditionExpression=Key("doc").eq(doc_id),
                    ReturnConsumedCapacity="TOTAL", **kwargs)
        n += len(r["Items"])
        cu += r["ConsumedCapacity"]["CapacityUnits"]
        if "LastEvaluatedKey" not in r:
            return n, cu
        kwargs["ExclusiveStartKey"] = r["LastEvaluatedKey"]


def pg_patterns(conn=None):
    conn = conn or stack.postgres()
    return {
        "get_doc": lambda d: conn.execute("select * from docs where id = %s", (d,)).fetchall(),
        "user_recent": lambda u: conn.execute(
            'select * from views where "user" = %s order by viewed_at desc limit 20', (u,)).fetchall(),
        "readers": lambda d: conn.execute("select * from views where doc = %s", (d,)).fetchall(),
    }


def table_stats():
    d = client().describe_table(TableName=VIEWS)["Table"]
    return {"items": d["ItemCount"], "bytes": d["TableSizeBytes"]}


def usd(units, times=1_000_000):
    """Dollar cost of running a read `times` times (default: a million requests)."""
    return units * times / 1e6 * USD_PER_MILLION_READ_UNITS


# ---------- change feed: DynamoDB Streams -> OpenSearch ----------

class StreamSync:
    """Poll the kv-docs stream and apply each change to an OpenSearch index.
    On AWS this loop is a Lambda function triggered by the stream (or the managed zero-ETL
    DynamoDB -> OpenSearch integration); locally a poller does the same job."""

    def __init__(self, os_client=None):
        self.os = os_client or stack.opensearch()
        self.st = streams()
        arn = client().describe_table(TableName=DOCS)["Table"]["LatestStreamArn"]
        shards = self.st.describe_stream(StreamArn=arn)["StreamDescription"]["Shards"]
        self.iters = [self.st.get_shard_iterator(StreamArn=arn, ShardId=s["ShardId"],
                                                 ShardIteratorType="TRIM_HORIZON")["ShardIterator"]
                      for s in shards]
        if not self.os.indices.exists(index=SEARCH_INDEX):
            self.os.indices.create(index=SEARCH_INDEX, body={
                "settings": {"number_of_shards": 1, "number_of_replicas": 0},
                "mappings": {"dynamic": "strict", "properties": {
                    "title": {"type": "text", "analyzer": "english"},
                    "body": {"type": "text", "analyzer": "english"}}}})

    def poll(self, refresh=True):
        """Apply every waiting change. Returns how many were applied."""
        actions, nxt = [], []
        for it in self.iters:
            while it:
                r = self.st.get_records(ShardIterator=it, Limit=1000)
                for rec in r["Records"]:
                    keys = rec["dynamodb"]["Keys"]["id"]["S"]
                    if rec["eventName"] == "REMOVE":
                        actions.append({"_op_type": "delete", "_index": SEARCH_INDEX, "_id": keys})
                    else:
                        img = rec["dynamodb"]["NewImage"]
                        actions.append({"_op_type": "index", "_index": SEARCH_INDEX, "_id": keys,
                                        "_source": {"title": img["title"]["S"], "body": img["body"]["S"]}})
                it = r.get("NextShardIterator")
                if not r["Records"]:
                    break
            nxt.append(it)
        self.iters = [i for i in nxt if i]
        if actions:
            helpers.bulk(self.os, actions, raise_on_error=False)
            if refresh:
                self.os.indices.refresh(index=SEARCH_INDEX)
        return len(actions)

    def lag_ms(self, trials=20, poll_every=0.2):
        """Write a doc to DynamoDB, then time how long until OpenSearch search finds it, with the
        poller checking every `poll_every` s (Lambda stream triggers typically fire within ~1 s)."""
        t, os_ = ddb().Table(DOCS), self.os
        out = []
        for k in range(trials):
            token = f"zzlagprobe{k}x{int(time.time() * 1000)}"
            doc_id = f"probe-{k}"
            t0 = time.perf_counter()
            t.put_item(Item={"id": doc_id, "title": token, "body": "lag probe"})
            while True:
                self.poll()
                hits = os_.search(index=SEARCH_INDEX, body={"query": {"match": {"title": token}}})
                if hits["hits"]["total"]["value"]:
                    break
                time.sleep(poll_every)
            out.append((time.perf_counter() - t0) * 1000)
            t.delete_item(Key={"id": doc_id})
        self.poll()
        return out


def consistent(os_client=None):
    """Known answer for the sync: OpenSearch holds exactly the DynamoDB items, same text."""
    os_client = os_client or stack.opensearch()
    items, kwargs = {}, {}
    t = ddb().Table(DOCS)
    while True:
        r = t.scan(**kwargs)
        items.update({i["id"]: (i["title"], i["body"]) for i in r["Items"]})
        if "LastEvaluatedKey" not in r:
            break
        kwargs["ExclusiveStartKey"] = r["LastEvaluatedKey"]
    indexed = {h["_id"]: (h["_source"]["title"], h["_source"]["body"])
               for h in helpers.scan(os_client, index=SEARCH_INDEX, query={"query": {"match_all": {}}})}
    return {"dynamodb": len(items), "opensearch": len(indexed), "identical": items == indexed}


