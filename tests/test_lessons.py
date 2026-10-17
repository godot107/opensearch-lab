"""Known-answer checks for the lesson helpers that don't need the docker stack."""
import random
from datetime import timedelta

import numpy as np
import pandas as pd

from oslab import detect, features, logs, tune


def test_schedule_is_in_week_two_and_spread_out():
    inc = logs.schedule()
    assert len(inc) == 8
    assert all(logs.TEST_START + timedelta(hours=12) <= i.start for i in inc)
    assert all(i.end <= logs.TEST_START + timedelta(days=7) for i in inc)
    starts = sorted(i.start for i in inc)
    assert all(b - a >= timedelta(hours=6) for a, b in zip(starts, starts[1:]))
    assert {(i.kind, i.severity) for i in inc} == {(k, s) for k in
            ("error_spike", "slow_search", "cred_stuffing", "outage") for s in ("big", "subtle")}


def test_effects_only_inside_the_incident():
    inc = [logs.Incident("outage", "big", logs.TEST_START, 20)]
    assert logs.effects(inc, logs.TEST_START)["traffic"] == 0.1
    assert logs.effects(inc, logs.TEST_START + timedelta(minutes=20))["traffic"] == 1.0


def T(m):
    return pd.Timestamp(logs.TEST_START) + pd.Timedelta(minutes=m)


def test_episodes_merge_close_alerts():
    assert detect.episodes([T(0), T(3), T(8), T(20)]) == [(T(0), T(8)), (T(20), T(20))]


def test_score_counts_catches_delay_and_false_alarms():
    inc = [logs.Incident("outage", "big", T(100).to_pydatetime(), 20),
           logs.Incident("error_spike", "subtle", T(500).to_pydatetime(), 30)]
    per, s = detect.score([T(103), T(104), T(129), T(300)], inc)   # 129 = within the 10-min grace
    assert list(per.caught) == [True, False]
    assert per.minutes_to_detect.iloc[0] == 3
    assert s["false_alarms"] == 1 and s["caught"] == 1


def test_add_typo_changes_exactly_one_long_word():
    rng = random.Random(1)
    for text in ["Mesenchymal stem cells repair cartilage", "Vitamin deficiency raises fracture risk"]:
        out = features.add_typo(text, rng)
        diff = [(a, b) for a, b in zip(text.split(), out.split()) if a != b]
        assert len(diff) == 1 and len(diff[0][0]) >= 5
    assert features.add_typo("a b c", rng) == "a b c"


def test_paired_bootstrap_finds_a_real_shift_and_no_fake_one():
    rng = np.random.default_rng(0)
    a = rng.uniform(0, 1, 300)
    assert tune.paired_bootstrap(a, a + 0.05)["ci95"][0] > 0
    lo, hi = tune.paired_bootstrap(a, a + rng.normal(0, 0.1, 300))["ci95"]
    assert lo < 0 < hi
