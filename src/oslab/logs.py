"""Lesson 2: synthetic web-shop logs with incidents injected at known times.

Two simulated weeks. Week 1 is clean and only used to calibrate alert thresholds ("a normal week
never fires"). Week 2 contains 8 incidents: a big and a subtle version of four kinds. The schedule
is the known answer every detector is scored against.

The generator writes one JSON line per request to logs/shop.log; Fluent Bit (docker compose
--profile logs) tails that file into daily indices logs-shop-YYYY.MM.DD.
"""
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

START = datetime(2026, 9, 28, tzinfo=timezone.utc)   # Monday; week 1 = calibration, week 2 = test
MINUTES = 14 * 24 * 60
TEST_START = START + timedelta(days=7)
BASE_RPM = 60                                         # mean requests per minute

ENDPOINTS = {  # share of traffic, median latency ms, latency spread (lognormal sigma)
    "/": (0.20, 40, 0.4), "/search": (0.25, 120, 0.5), "/product": (0.25, 60, 0.4),
    "/cart": (0.12, 50, 0.4), "/checkout": (0.08, 300, 0.5), "/login": (0.10, 80, 0.4),
}
BASE_5XX = 0.004
BASE_LOGIN_FAIL = 0.06


@dataclass
class Incident:
    kind: str        # error_spike | slow_search | cred_stuffing | outage
    severity: str    # big | subtle
    start: datetime
    minutes: int

    @property
    def end(self):
        return self.start + timedelta(minutes=self.minutes)

    def as_json(self):
        return asdict(self) | {"start": self.start.isoformat(), "end": self.end.isoformat()}


def schedule(seed=7):
    """8 incidents in week 2: random day and hour, at least 6 h apart, never in the first 12 h."""
    rng = np.random.default_rng(seed)
    spec = [("error_spike", "big", 25), ("error_spike", "subtle", 60),
            ("slow_search", "big", 40), ("slow_search", "subtle", 60),
            ("cred_stuffing", "big", 15), ("cred_stuffing", "subtle", 30),
            ("outage", "big", 20), ("outage", "subtle", 30)]
    starts = []
    while len(starts) < len(spec):
        m = int(rng.integers(12 * 60, 7 * 24 * 60 - 120))
        if all(abs(m - s) >= 6 * 60 for s in starts):
            starts.append(m)
    rng.shuffle(spec)
    return sorted((Incident(k, s, TEST_START + timedelta(minutes=m), d)
                   for (k, s, d), m in zip(spec, starts)), key=lambda i: i.start)


def daily_shape(minute_of_day):
    """Quiet at 04:00 UTC, busiest around 18:00: 0.25x to 1.75x the mean."""
    return 1 + 0.75 * np.sin(2 * np.pi * (minute_of_day - 11 * 60) / 1440)


def effects(incidents, t):
    """Multipliers active at minute t for each incident kind."""
    e = {"traffic": 1.0, "checkout_5xx": None, "search_latency": 1.0, "stuffing": (0, 0)}
    for inc in incidents:
        if not inc.start <= t < inc.end:
            continue
        big = inc.severity == "big"
        if inc.kind == "error_spike":
            e["checkout_5xx"] = 0.35 if big else 0.08
        elif inc.kind == "slow_search":
            e["search_latency"] = 6.0 if big else 2.0
        elif inc.kind == "cred_stuffing":
            e["stuffing"] = (40, 10) if big else (6, 3)   # attacking IPs, attempts/min each
        elif inc.kind == "outage":
            e["traffic"] = 0.1 if big else 0.5
    return e


def generate(path="logs/shop.log", seed=0, incidents=None):
    """Write the two weeks of logs. Returns the number of lines."""
    incidents = schedule() if incidents is None else incidents
    rng = np.random.default_rng(seed)
    names = list(ENDPOINTS)
    share = np.array([ENDPOINTS[n][0] for n in names])
    users = [f"u{i:05d}" for i in range(5000)]
    user_ips = {u: f"10.{rng.integers(0, 256)}.{rng.integers(0, 256)}.{rng.integers(1, 255)}" for u in users}
    attacker_ips = [f"203.0.113.{i}" for i in range(1, 60)]   # TEST-NET-3 documentation range
    Path(path).parent.mkdir(exist_ok=True)
    n = 0
    with open(path, "w") as f:
        for m in range(MINUTES):
            t0 = START + timedelta(minutes=m)
            e = effects(incidents, t0)
            lam = BASE_RPM * daily_shape(m % 1440) * e["traffic"]
            count = rng.poisson(lam)
            ep = rng.choice(len(names), count, p=share)
            secs = np.sort(rng.uniform(0, 60, count))
            events = []
            for k, s in zip(ep, secs):
                name = names[k]
                _, med, sig = ENDPOINTS[name]
                lat = rng.lognormal(np.log(med), sig)
                if name == "/search":
                    lat *= e["search_latency"]
                p5 = e["checkout_5xx"] if (name == "/checkout" and e["checkout_5xx"]) else BASE_5XX
                user = users[rng.integers(len(users))]
                ip, status = user_ips[user], 200
                if rng.random() < p5:
                    status = int(rng.choice([500, 502, 503]))
                elif name == "/login" and rng.random() < BASE_LOGIN_FAIL:
                    status = 401
                elif rng.random() < 0.02:
                    status = 404
                events.append((s, name, status, lat, user, ip))
            n_ips, per_ip = e["stuffing"]
            for ip in attacker_ips[:n_ips]:
                for s in rng.uniform(0, 60, rng.poisson(per_ip)):
                    events.append((s, "/login", 401, rng.lognormal(np.log(80), 0.4),
                                   users[rng.integers(len(users))], ip))
            for s, name, status, lat, user, ip in sorted(events):
                ts = t0 + timedelta(seconds=float(s))
                rec = {"@timestamp": ts.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z",
                       "service": "shop-web", "endpoint": name, "method": "POST" if name in
                       ("/login", "/checkout", "/cart") else "GET", "status": status,
                       "latency_ms": round(float(lat), 1), "client_ip": ip, "user": user,
                       "level": "error" if status >= 500 else "warn" if status >= 400 else "info",
                       "is_5xx": int(status >= 500), "is_login_fail": int(name == "/login" and status == 401)}
                f.write(json.dumps(rec) + "\n")
                n += 1
    Path(path).with_suffix(".incidents.json").write_text(
        json.dumps([i.as_json() for i in incidents], indent=2) + "\n")
    return n
