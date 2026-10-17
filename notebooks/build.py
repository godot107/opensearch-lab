"""Build the lesson notebooks, then execute them so outputs are embedded:

    .venv/bin/python notebooks/build.py            # all four (or: build.py 01 03)
    for n in notebooks/0*.ipynb; do .venv/bin/jupyter nbconvert --to notebook --execute --inplace $n; done

Needs the docker compose stack up and the lesson commands already run (README). The notebooks read
saved results for the slow steps (the 640-setting sweep, the anomaly-detector runs) and re-run the
fast ones live. Written for a general reader: short code cells, each after plain-language text.
"""
import sys
from pathlib import Path

import nbformat as nbf

HERE = Path(__file__).parent


def _nb():
    cells = []
    md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip()))  # noqa: E731
    code = lambda s: cells.append(nbf.v4.new_code_cell(s.strip()))  # noqa: E731
    return cells, md, code


SETUP = """
%matplotlib inline
import os, json
os.chdir('..')  # run from the project root so data/ and results/ resolve
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from oslab import plot, stack
plot.style()
pd.set_option('display.width', 160)
assert stack.is_up(), 'start the stack first: docker compose up -d'
"""


def lesson_01():
    c, md, code = _nb()
    md("""
# 1 · Full-text search: a search engine vs. the database you already have

**The question:** Postgres can already search text. How much better is a real search engine, and
what does it give you besides better ranking?

**The test:** SciFact, 5,183 scientific abstracts and expert-labelled questions: 809 for *tuning*
(train) and 300 kept aside for the *final score* (test). Each method returns its top 10 abstracts.

**How the score works:** *nDCG@10* rewards putting the right abstracts near the top of the 10
(1.0 = perfect). *Recall@10* is the share of right abstracts that made the top 10 at all.
""")
    code(SETUP + "\nfrom oslab import search, tune, features, beir, metrics")
    md("""
## 1 · What the search engine does to text first: analyzers

Before indexing, every text goes through an *analyzer*. The `english` analyzer lowercases, drops
filler words ("the", "of") and *stems* words to a root, so "cells" and "cell" match. Postgres's
`to_tsvector('english', ...)` does the same, so any difference below is about **ranking**.
""")
    code("""
os_ = stack.opensearch()
text = 'Mesenchymal stem cells are repairing the damaged cartilage'
for analyzer in ('standard', 'english'):
    toks = [t['token'] for t in os_.indices.analyze(body={'analyzer': analyzer, 'text': text})['tokens']]
    print(f'{analyzer:9s}', toks)
print('postgres ', stack.postgres().execute("select to_tsvector('english', %s)", (text,)).fetchone()[0])
""")
    md("""
## 2 · First result: OpenSearch vs Postgres, out of the box

Three ways to answer each test question:

- **OpenSearch BM25**: BM25 is the standard ranking formula. Rare words ("mesenchymal") count more
  than common ones, and a word repeated 20 times doesn't count 20 times as much.
- **Postgres, every word must match**: what `websearch_to_tsquery` does: the usual choice for a search box.
- **Postgres, any word may match**, ranked by Postgres's `ts_rank_cd`.
""")
    code("""
rows = [json.load(open(f'results/scifact-{b}.json')) for b in ('os-bm25', 'pg-or', 'pg-and')]
pd.DataFrame(rows)[['backend', 'ndcg@10', 'recall@10', 'p50_ms', 'empty_results']].round(3)
""")
    md("""
**Reading it:** requiring every word is a trap for real questions: **274 of 300 questions return
nothing**, because no abstract contains every word of a long question. Allowing any word fixes that,
but Postgres's ranking is far behind BM25.
""")
    md("""
## 3 · Tuning, done honestly

BM25 has settings, and so does the query. We tried **640 combinations** on the 809 *train* questions:

- analyzer: `english` vs `standard` (no stemming)
- `k1` (how fast repeated words stop adding score) and `b` (how much long abstracts are penalised)
- how title and body scores combine (`best_fields` = best single field, `most_fields` = add them up,
  `cross_fields` = treat them as one big field) and how much the title counts

The winner was then scored **once** on the 300 test questions. (The sweep takes ~3 minutes:
`python -m oslab tune`; here we load its saved output.)
""")
    code("""
sweep = pd.read_json('results/tune-train.jsonl', lines=True)
sweep['query_type'] = sweep['type'].str.replace('_fields', '') + np.where(sweep['tie_breaker'] > 0, '+tb', '')
sweep['k1'] = sweep['k1'].round(2)
best = json.load(open('results/tune-best.json'))
print('chosen on train:', {k: best['config'][k] for k in ('analyzer', 'k1', 'b', 'type', 'title_boost')})
print('test nDCG@10: untuned %.3f -> tuned %.3f' % (best['test']['baseline_ndcg@10'], best['test']['tuned_ndcg@10']))
d = best['test']['tuned_minus_baseline']
print('gain %.3f, 95%% interval [%.3f, %.3f]' % (d['mean_diff'], *d['ci95']))
""")
    md("""
The gain's *95% interval* comes from a **paired bootstrap**: resample the 300 test questions many
times and recompute the gain each time. The interval stays well above zero, so the gain is real,
not luck in which questions happened to be in the test set.

**Which setting mattered?** For each setting, the best score reachable with it:
""")
    code("""
eng = sweep[sweep.analyzer == 'english'].groupby(['query_type', 'title_boost'])['ndcg@10'].max().unstack(0)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 3.4), gridspec_kw={'width_ratios': [2.2, 1]})
for k, col in enumerate(['best', 'best+tb', 'cross', 'most']):
    ax1.plot(eng.index, eng[col], marker='o', markersize=5, color=plot.SERIES[k], label=col)
ax1.scatter([2.0], [best['train']['baseline_ndcg@10']], s=90, facecolors='none', edgecolors=plot.INK, zorder=3)
ax1.annotate('untuned start', (2.0, best['train']['baseline_ndcg@10']), xytext=(2.1, 0.585), color=plot.MUTED)
ax1.set_xlabel('title weight (body = 1)'); ax1.set_ylabel('best train nDCG@10')
ax1.set_title('Title weight x how fields combine'); ax1.legend(title='fields', ncol=2, loc='lower left')
an = sweep.groupby('analyzer')['ndcg@10'].max()
ax2.bar(an.index, an.values, color=plot.SERIES[0], width=0.5); ax2.set_ylim(0.6, 0.71)
ax2.set_title('Analyzer')
plt.tight_layout()
""")
    md("""
**What tuning taught:** the biggest lever was the **title weight**. The untuned query (circled)
counted titles double, so a title match could drown out a body that actually answered the question.
Lowering it to 0.5-1x helps whichever way the fields are combined; `cross_fields` (treat title and
body as one field) is the most forgiving of a heavy title. Stemming (`english`)
added about 0.03. `k1` and `b`, the settings textbooks discuss most, moved the best score by less
than 0.002.

For reference, the published BEIR benchmark reports BM25 at about 0.665 on this test set. The
tuned 0.690 is a little above it, the untuned 0.610 below it.
""")
    md("""
## 4 · Search-box features beyond ranking

### Typos
We gave every test question one realistic typo ("inductiae" for "inductive") and re-scored, with and
without *fuzzy matching* (`fuzziness: AUTO`: a word may match others 1-2 letters different).
""")
    code("""
features.build_feature_index()
_, q, rel = beir.load('data/beir/scifact', 'test')
typo_q = features.typo_queries(q)
print('example:', list(typo_q.values())[0])
res = {}
for name, qs in (('clean questions', q), ('one typo each', typo_q)):
    for fuzzy in (False, True):
        res[(name, 'fuzzy on' if fuzzy else 'fuzzy off')] = metrics.score_run(features.run_many(qs, fuzzy=fuzzy), rel, 10)['ndcg@10']
pd.Series(res).unstack().round(3)
""")
    md("""
One typo per question costs about 0.04 nDCG. Fuzzy matching wins most of it back, and costs
nothing on correctly spelled questions.

### Highlighting: show *why* a result matched
""")
    code("""
body = features.query('vitamin D deficiency and bone fractures', k=2, highlight=True)
for h in stack.opensearch().search(index=features.FEATURE_INDEX, body=body)['hits']['hits']:
    print(h['_source']['title']); [print('   ...', f, '...') for f in h['highlight'].get('body', [])]
""")
    md("""
### Autocomplete (search-as-you-type), and synonyms
""")
    code("""
for prefix in ('vitamin d def', 'stem ce', 'myocard'):
    print(f'{prefix!r:16} ->', features.autocomplete(prefix, k=3))
syn = metrics.score_run(features.run_many(q, synonyms=True), rel, 10)['ndcg@10']
plain = metrics.score_run(features.run_many(q), rel, 10)['ndcg@10']
print(f'\\nsynonyms on: {syn:.4f}   off: {plain:.4f}')
""")
    md("""
**Synonyms made no measurable difference.** Our small medical list ("heart attack" = "myocardial
infarction") rarely appears in these questions. Synonym lists help when they are built from what
real users type and fail to find, not from a guess.

### Explain: why a document got its score
""")
    code("""
qtext = 'Mesenchymal stem cells repair cartilage'
top = features.run_many({'x': qtext}, k=1)['x'][0]
e = features.explain(qtext, top)['explanation']
print(round(e['value'], 2), e['description'])
for d in e['details'][:4]:
    print('  ', round(d['value'], 2), d['description'][:90])
""")
    md("""
## When *not* to use OpenSearch for this

- **Small data and simple needs:** Postgres's any-word search, or a trigram index for typos, may
  be good enough, and it's one less system to run and keep in sync.
- **As the only copy of the data:** no multi-document transactions, and writes take about 1 second
  to become searchable. Here Postgres is the source of truth and `oslab sync` rebuilds the index.
- **Lookups by exact id:** use the database.

**Industry uses:** product and catalog search, documentation and help-centre search, legal and
patent discovery, internal "search everything" portals.
""")
    return c


def lesson_02():
    c, md, code = _nb()
    md("""
# 2 · Log analytics: find the incident before a customer does

**The question:** a web shop writes a log line for every request. Can we spot problems
automatically, quickly, without paging someone every hour for nothing?

**The test:** two simulated weeks of logs (1.2 million lines). **Week 1 is clean** and only used to
set alert thresholds. **Week 2 has 8 incidents injected at known times**: a big and a subtle
version of an error spike, a slow search page, a credential-stuffing attack (bots trying stolen
passwords) and an outage. The schedule is the answer key.

**Scoring:** an incident is *caught* if an alert fires between its start and 10 minutes after it ends.
Alerts that match no incident are *false alarms*: the thing that makes on-call people ignore alerts.
""")
    code(SETUP + """
from oslab import logs, logpipe, detect
incidents = detect.incidents_from_file()
pd.DataFrame([{'kind': i.kind, 'severity': i.severity, 'start (UTC)': i.start.strftime('%a %H:%M'), 'minutes': i.minutes} for i in incidents])
""")
    md("""
## 1 · The pipeline

The generator writes JSON lines to `logs/shop.log`. **Fluent Bit**, a small log shipper that
normally runs next to every server, tails the file and sends batches to OpenSearch, one index per
day (`logs-shop-2026.10.05`, ...). Daily indices make deleting old data cheap: drop a whole index.
""")
    code("""
print(open('infra/fluent-bit/fluent-bit.conf').read().split('[OUTPUT]')[1])
print('documents in OpenSearch:', logpipe.count(), '   lines in the file:', sum(1 for _ in open('logs/shop.log')))
""")
    md("""
Every line arrived, once. `Generate_ID` makes each document's id a hash of its content, so
replaying the file can't create duplicates.

**Retention:** an *Index State Management* policy moves each daily index to read-only after 7 days
and deletes it after 30. Without one, a log cluster's disk and bill only ever grow.
""")
    code("""
p = stack.opensearch().transport.perform_request('GET', f'/_plugins/_ism/policies/{logpipe.POLICY}')['policy']
for s in p['states']:
    print(f"{s['name']:7s} actions={[k for a in s['actions'] for k in a if k != 'retry']} then -> {[(t['state_name'], t['conditions']) for t in s['transitions']]}")
""")
    md("""
## 2 · Looking at week 2

Per-minute numbers come straight from OpenSearch *aggregations* (counts and sums per time bucket,
computed inside the cluster). Red bands are the injected incidents.
""")
    code("""
m = detect.minute_metrics()
f = detect.rolling_features(m)
w2 = f[f.index >= logs.TEST_START]
panels = [('requests', 'Requests per 5 min'), ('error_rate', 'Server error rate'),
          ('search_ms', 'Search page latency (ms)'), ('login_fail', 'Failed logins per 5 min')]
fig, axes = plt.subplots(4, 1, figsize=(12, 8), sharex=True)
for ax, (col, title) in zip(axes, panels):
    ax.plot(w2.index, w2[col], color=plot.SERIES[0], linewidth=0.8)
    for i in incidents:
        ax.axvspan(i.start, i.end, color=plot.INCIDENT, alpha=0.25, linewidth=0)
    ax.set_title(title)
plt.tight_layout()
""")
    md("""
Some incidents jump out (the big credential-stuffing attack). Others hide in the daily rhythm:
traffic swings 7x between 04:00 and 18:00, so a 50% drop at midday is still more traffic than a
normal night.

## 3 · Detector A: threshold rules

What most teams start with, and what an OpenSearch *Alerting monitor* checks every minute: a
number over the last 5 minutes crosses a line. Each line is set from the clean week only:
"the worst 5 minutes of a normal week, plus 10%". So by construction, week 1 never fires.
""")
    code("""
th = detect.calibrate(f)
alerts = detect.rule_alerts(f, th)
print({k: round(v, 3) for k, v in th.items()})
detect.per_rule_table(alerts, incidents).astype(int)
""")
    code("""
per, s = detect.score(detect.all_rule_minutes(alerts), incidents)
print(s)
""")
    md("""
**Rules catch 7 of 8 with zero false alarms.** Two lessons:

- **The fixed traffic floor misses the subtle outage.** Only the *seasonal* rule (traffic vs the
  normal amount *for that time of day*) catches it.
- **The subtle error spike is missed by every rule.** Checkout errors rose to 8%, but checkout is
  8% of traffic, so the *shop-wide* error rate barely moves. A rule per endpoint would catch it.

The error-rate rule also exists as a real Alerting monitor (created disabled: our logs are in the
past, and a live monitor only looks at the last 5 minutes before *now*):
""")
    code("""
mon = detect.error_rate_monitor(th['error_rate_high'])
print(json.dumps(mon['inputs'][0]['search']['query'], indent=1)[:400])
print('trigger:', mon['triggers'][0]['condition']['script']['source'])
""")
    md("""
## 4 · Detector B: the Anomaly Detection plugin

OpenSearch's built-in anomaly detection uses **Random Cut Forest**, a model that learns what normal
looks like and scores how surprising each new point is (the *anomaly grade*, 0 to 1). We gave it one
model per endpoint, watching request count, errors, average latency and failed logins, and **never
told it what an incident looks like**. It ran over both weeks as a *historical analysis*, at
1-minute and 5-minute intervals.
""")
    code("""
curve = pd.read_csv('results/ad-threshold-curve.csv')
curve[curve.min_grade.isin([0, 0.4, 0.8])][['interval_min', 'min_grade', 'caught', 'caught_big', 'caught_subtle', 'false_alarms', 'clean_week_false_alarms']]
""")
    code("""
fig, ax = plt.subplots(figsize=(7, 3.6))
for k, (iv, g) in enumerate(curve.groupby('interval_min')):
    ax.plot(g.false_alarms, g.caught, marker='o', markersize=5, color=plot.SERIES[k], label=f'anomaly detection, {iv}-min')
ax.scatter([s['false_alarms']], [s['caught']], s=70, color=plot.SERIES[2], zorder=3, label='threshold rules')
ax.set_xlabel('false alarms in the test week'); ax.set_ylabel('incidents caught (of 8)')
ax.set_ylim(0, 8.5); ax.legend(loc='lower right'); ax.set_title('Catching incidents vs crying wolf')
plt.tight_layout()
""")
    md("""
**Reading it:** at 1-minute resolution the model raises 65-170 false alarms a week. Per-endpoint
counts per minute are tiny (a handful of requests at night), so ordinary noise looks surprising.
At 5 minutes it is much quieter (about 10 a week at grade ≥ 0.8) but catches 6 of 8. Random Cut
Forest is random: re-running the same detector gives somewhat different results.

**Where the two disagree** (5-minute detector, grade ≥ 0.8):
""")
    code("""
ad = json.load(open('results/logs-ad-5m.json'))
rules = json.load(open('results/logs-rules.json'))
cmp = pd.DataFrame({'incident': [f"{i['kind']} ({i['severity']})" for i in rules['incidents']],
                    'rules': [i['caught'] for i in rules['incidents']],
                    'anomaly detection': [i['caught'] for i in ad['incidents']]})
cmp['either'] = cmp['rules'] | cmp['anomaly detection']
cmp
""")
    md("""
**The two are complementary.** Anomaly detection, modelling each endpoint separately, catches the
subtle checkout error spike that every shop-wide rule missed (late, after about 50 minutes). Rules
catch the known patterns fast and silently. Together they cover all 8.

**What a real team would do:** rules for the failure modes you know (errors, latency, traffic vs
the usual for this hour, login failures), each per endpoint; anomaly detection at a coarse interval
as a safety net for the ones you don't know, routed to a dashboard or a low-priority channel rather
than a pager.

## When *not* to use OpenSearch for this

- **Metrics only, no need to search raw lines:** a time-series database (Prometheus, CloudWatch
  metrics) is cheaper per data point.
- **Low volume:** CloudWatch Logs Insights or a managed log service avoids running a cluster.
- **Keep-forever archives:** keep raw logs in S3 and query them occasionally (Athena); keep only
  recent days in OpenSearch.

**Industry uses:** application troubleshooting, security monitoring (SIEM), audit trails, tracing
slow requests across services.
""")
    return c


def lesson_03():
    c, md, code = _nb()
    md("""
# 3 · Vector and hybrid search: matching by meaning

**The question:** keyword search needs the same words. Does matching by *meaning* do better, and does
blending both beat either one?

**How vectors work:** a language model (here *bge-large*) turns each text into 1,024 numbers so that
texts with similar meaning get similar numbers. Search = find the abstracts whose numbers are closest
to the question's. This is the search step behind AI assistants that "chat with your documents" (RAG).

**The test:** the same SciFact questions and expert labels as lesson 1. The abstract vectors were
computed in the rag-bedrock-compare project; this lesson reuses them.
""")
    code(SETUP + "\nfrom oslab import hybrid, search, beir, metrics, tune")
    md("""
## 1 · First check the plumbing: approximate vs exact

Comparing a question against every abstract is exact but slow at millions of documents. Search
engines use **HNSW**, a graph that jumps quickly toward the closest vectors, accepting that it may
occasionally miss one. Known answer: brute force in numpy. How many of the exact top 10 does each
index find?
""")
    code("""
_, q, rel = beir.load('data/beir/scifact', 'test')
{b: hybrid.ann_recall(b, q) for b in ('os-vector', 'pg-vector')}
""")
    md("""
Both find exactly the same top 10 (1.0). At 5,000 abstracts that's expected; at millions, the
HNSW settings (`m`, `ef_search`) trade speed for this number, and it's worth re-measuring.

The *exact* vector run reproduces rag-bedrock-compare's 0.744 nDCG@10 (here 0.746: the question
vectors were computed on CPU instead of GPU, a tiny numeric difference).

## 2 · Hybrid: blending keyword and meaning scores

OpenSearch can run BM25 and vector search in one query, then blend. The two scores are on different
scales, so a *search pipeline* first rescales each list to 0-1 (*min-max*), then takes a weighted
average. The weight was picked on the train questions only:
""")
    code("""
hb = json.load(open('results/hybrid-best.json'))
w = pd.Series(hb['train_ndcg@10']); w.index = w.index.astype(float)
fig, ax = plt.subplots(figsize=(6.5, 3))
ax.plot(w.index, w.values, marker='o', markersize=5, color=plot.SERIES[0])
ax.set_xlabel('share of the vector score (0 = pure BM25, 1 = pure vector)'); ax.set_ylabel('train nDCG@10')
ax.set_title(f"Best blend on train: {hb['vector_weight']:.0%} vector")
plt.tight_layout()
""")
    md("""
The other option, **reciprocal rank fusion (RRF)**, ignores scores and blends by *rank* (1st, 2nd, ...),
so there's no weight to tune.

## 3 · The test result
""")
    code("""
names = ['os-bm25', 'os-bm25-tuned', 'os-vector', 'pg-vector', 'os-hybrid', 'os-hybrid-rrf']
t = pd.DataFrame([json.load(open(f'results/scifact-{b}.json')) for b in names])
t[['backend', 'ndcg@10', 'recall@10', 'mrr@10', 'p50_ms']].round(3).sort_values('ndcg@10', ascending=False)
""")
    code("""
cmp = json.load(open('results/hybrid-compare.json'))
for k in ('vector_vs_bm25', 'hybrid_vs_vector', 'rrf_vs_vector', 'hybrid_vs_bm25'):
    d = cmp[k]; print(f"{k:17s} gain {d['mean_diff']:+.3f}   95% interval [{d['ci95'][0]:+.3f}, {d['ci95'][1]:+.3f}]")
""")
    md("""
**Reading it:**
- **Meaning beats keywords** here: vector search is +0.057 over even the *tuned* BM25.
- **Hybrid adds almost nothing on top** (+0.006, and the interval includes zero). Many articles claim
  hybrid always wins; on this data the evidence doesn't support it. It tends to help more when
  questions contain exact codes, names or rare terms that embedding models blur.
- OpenSearch's vector index and pgvector return identical results. The difference is operational:
  pgvector keeps vectors next to the rows they describe; OpenSearch adds BM25, hybrid pipelines and
  scale-out in the same engine.

## 4 · Where keyword and meaning disagree
""")
    code("""
corpus, _, _ = beir.load('data/beir/scifact', 'test')
bm, vec = search.retriever('os-bm25-tuned'), search.retriever('exact-vector')
diff = []
for qid, text in q.items():
    gold = set(rel[qid])
    b_hit, v_hit = bool(gold & set(bm(text, 10))), bool(gold & set(vec(text, 10)))
    if v_hit and not b_hit:
        diff.append((qid, text))
print(len(diff), 'questions where only vector search found a right abstract; first two:')
for qid, text in diff[:2]:
    gid = next(iter(rel[qid]))
    print('\\nQ:', text); print('A:', corpus[gid]['title'])
""")
    md("""
These are typically paraphrases: the right abstract says the same thing in different words, which
BM25 can't see.

## When *not* to use OpenSearch for this

- **Already on Postgres, under a few million vectors:** pgvector gave identical results here, with no
  second system to keep in sync.
- **Serverless and pay-per-query:** Amazon S3 Vectors (see rag-bedrock-compare) has no idle cost.
  Avoid OpenSearch *Serverless* for small projects: its minimum charge is several hundred dollars a month.

**Industry uses:** retrieval for AI assistants (RAG), "more like this" recommendations, semantic
product search, duplicate and near-duplicate detection.
""")
    return c


def lesson_04():
    c, md, code = _nb()
    md("""
# 4 · Key-value databases: fast for the questions you planned

**The question:** when is a key-value store like DynamoDB the right home for data, and what does it
cost you?

**What a key-value store is:** every item has a *key*; give the key, get the item back, in a few
milliseconds no matter how big the table gets. A second part of the key (the *sort key*) keeps items
in order under the first, so "this user's items, newest first" is also one fast request. Any other
question means reading the whole table.

**Setup:** DynamoDB Local (AWS's official emulator) in Docker, holding the SciFact abstracts again
plus a synthetic reading history: 1,000 users x 50 views, popular abstracts read more often.
""")
    code(SETUP + "\nfrom oslab import kv")
    md("""
## 1 · Designing keys from the questions

| Table | Partition key | Sort key | Question it answers fast |
|---|---|---|---|
| `kv-docs` | `id` | | "give me abstract 4983" |
| `kv-views` | `user` | `viewed_at` | "what did user u0042 read lately?" |

In Postgres you design tables around the *data* and add indexes later. In DynamoDB you design keys
around the *questions*, up front.
""")
    code("""
print('views table:', kv.table_stats())
kv.ddb().Table(kv.VIEWS).get_item(Key={'user': 'u0042', 'viewed_at': kv.ddb().Table(kv.VIEWS).query(
    KeyConditionExpression=kv.Key('user').eq('u0042'), Limit=1)['Items'][0]['viewed_at']})['Item']
""")
    md("""
## 2 · Each question's cost

DynamoDB bills by **read capacity units**: one unit is roughly one 4 KB read. The emulator reports
them exactly as AWS would, so we can count the cost of each question (the timings are a local
emulator's and don't transfer to AWS; the units do).

The question the table was *not* designed for, "who read abstract X?", has no key. The only way is
a **Scan**: read every item and filter afterwards. The fix is a **global secondary index (GSI)**, a
second copy of the keys organised by `doc`.
""")
    code("""
p = json.load(open('results/kv-patterns.json'))
labels = {'get_doc': 'get one abstract by id', 'user_recent': "a user's last 20 reads",
          'readers_scan': 'who read abstract X? (scan)', 'readers_gsi': 'who read abstract X? (GSI)'}
t = pd.DataFrame(p['patterns']).assign(question=lambda d: d.pattern.map(labels))
t[['question', 'items', 'read_units', 'usd_per_million', 'postgres_ms']]
""")
    code("""
fig, ax = plt.subplots(figsize=(7, 2.8))
ax.barh(t.question[::-1], t.read_units[::-1], color=plot.SERIES[0], height=0.55)
ax.set_xscale('log'); ax.set_xlabel('read capacity units per request (log scale)')
ax.set_title('One question the keys were not designed for costs 750x more')
for y, v in enumerate(t.read_units[::-1]):
    ax.text(v * 1.15, y, f'{v:g}', va='center', color=plot.MUTED)
plt.tight_layout()
""")
    md("""
**Reading it:** a keyed read costs 0.5 units whatever the table size. The scan cost 376 units for a
3 MB table, and **grows with the table**: at 3 GB it would be about 1,000x more again. The GSI brings
the same question down to 13 units, but it is a second copy of the keys: more storage, more write
cost, and it is updated *eventually* (a moment after the main table).

The dollar column uses the on-demand list price of $0.125 per million read
units (us-east-1, not re-verified this session).

Postgres answers all of these, including "who read X?", with one SQL query, and an index can be added
later without redesigning anything. That flexibility is why it's the default; DynamoDB's trade is
predictable speed and cost at any scale for the questions you planned, with no server to run.

## 3 · Keeping a search index in sync: the change feed

DynamoDB can't search text. The standard pattern: turn on **DynamoDB Streams** (a log of every insert,
update and delete), and have a small function apply each change to OpenSearch. On AWS that function
is a Lambda; locally, `kv.StreamSync` polls the stream.

**Known answer:** after syncing, OpenSearch must hold exactly the same items with the same text.
""")
    code("""
sync = json.load(open('results/kv-sync.json'))
print('after the initial load:      ', sync['initial'])
print('after one edit and one delete:', sync['after_edit_delete'])
print('write -> searchable delay (ms):', {k: round(v) for k, v in sync['lag_ms'].items()})
""")
    md("""
The copy was exact, and edits and deletes flowed through. The delay here (about 20 ms) is a local
poller's; on AWS, the stream-to-Lambda hop typically adds up to about a second. Either way the search
index is **eventually consistent**: a user who saves and immediately searches may not see their change.
Applications either accept that or read the item back by key.

## Choosing between the three

| Need | Best fit |
|---|---|
| Relational data, ad-hoc questions, transactions | **Postgres** |
| Known access patterns, any scale, no server, pay per request | **DynamoDB** |
| Ranked text search, log analytics, vector/hybrid search | **OpenSearch**, fed from one of the above |

**Industry uses of key-value stores:** shopping carts, user sessions and profiles, game state,
device telemetry by device and time, feature stores for ML models.
""")
    return c


LESSONS = {"01": ("01_full_text_search", lesson_01), "02": ("02_log_analytics", lesson_02),
           "03": ("03_vector_hybrid_search", lesson_03), "04": ("04_key_value", lesson_04)}


if __name__ == "__main__":
    for key in sys.argv[1:] or LESSONS:
        name, fn = LESSONS[key]
        nb = nbf.v4.new_notebook(cells=fn())
        nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
        nbf.write(nb, HERE / f"{name}.ipynb")
        print(HERE / f"{name}.ipynb")
