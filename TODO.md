# TODO

All four lessons are built, scored and in executed notebooks (2026-10-17).

## Open
- [ ] Proofread notebooks + README; blog post (suggested spine: lesson 2 "rules vs anomaly detection",
      or lesson 1's "Postgres finds nothing for 274/300 questions")
- [ ] Public/private decision, then push to github.com/godot107/opensearch-lab
- [ ] Postgres any-word latency measured 39 ms on the first run and ~75-80 ms on every later run;
      not caused by the vector column or table bloat (both tested). Unexplained; README says 40-80 ms.
- [ ] Optional: per-endpoint threshold rules in lesson 2 (would likely catch the subtle checkout spike)
- [ ] Optional: Dashboards saved objects exported to the repo (index pattern + a log dashboard)
- [ ] Optional: Loghub HDFS (real labelled logs) as a second lesson-2 dataset
- [ ] Optional: one lesson on a small managed AWS domain, torn down the same session (never Serverless)

## Not done as planned
- Lesson 2 used daily indices + Fluent Bit Logstash_Format rather than a data stream (simpler with
  Fluent Bit 4.0; ISM retention still applies via ism_template).
- Lesson 3's ANN check is trivially 1.0 at 5k docs; a larger corpus is needed to see HNSW trade-offs.
