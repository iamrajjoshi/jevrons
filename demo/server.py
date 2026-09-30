"""Jevrons demo server: runs a drawn digit through a trained net and streams every neuron as it resolves.

  uv run python demo/server.py              # exact (visitors), mock and replay (URL flags); http://127.0.0.1:8765
  uv run python demo/server.py --live       # also allow live Jev calls (add --journal to keep them in demo/runs/journal.jsonl)
  uv run python demo/server.py --check      # self-check: exact == jevrons.net.forward, replay == journals, live limits

Live calls go to whatever backends jevrons.jev picks; set JEVRONS_BACKENDS=vercel to pin one.
Models come from demo/models.json; an entry whose weights file is missing is listed as unavailable.
Replay and mock data are cut from the stage 7 and 8a training journals by demo/make_replay.py.
A live draw is only ever answered by live Jev. Past --max-live it waits in line; a 429, 5xx or timeout backs off
and retries; it ends early only on the daily cap, a key or billing failure, or its 5-minute deadline.
"""

import argparse
import hashlib
import json
import math
import os
import random
import select
import socket
import threading
import time
import uuid
from collections import deque
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from functools import cache
from itertools import groupby
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np

from jevrons.digits import _load
from jevrons.jev import BACKENDS, USD_PER_INPUT_TOKEN, Jev
from jevrons.net import StepNeuron, decide, forward
from jevrons.stage8g import QUESTIONS
from jevrons.states import SCALAR_Q_ALT, neuron_question, neuron_state, r2

try:
    import sentry_sdk
except ImportError:  # local runs and CI don't need it
    sentry_sdk = None


# ---------- telemetry ----------
# Sentry, only where SENTRY_DSN is set (the Fly app): errors, plus one span per draw tagged with its network, source
# and outcome, which is the usage count. Nothing from the request is attached: no drawing, no IP.

def init_telemetry():
    if not (sentry_sdk and os.environ.get("SENTRY_DSN")):
        return
    from sentry_sdk.integrations.stdlib import StdlibIntegration
    sentry_sdk.init(dsn=os.environ["SENTRY_DSN"], environment=os.environ.get("SENTRY_ENVIRONMENT", "production"),
                    release=os.environ.get("FLY_IMAGE_REF"), send_default_pii=False, traces_sample_rate=1.0,
                    disabled_integrations=[StdlibIntegration()])  # no span per Jev call: 74 a draw is noise


def report(e: BaseException | str, **tags):
    """An error or warning to Sentry, if it's on. TypeSafe rate limits group into one issue, not thousands."""
    if not (sentry_sdk and sentry_sdk.is_initialized()):
        return
    with sentry_sdk.new_scope() as scope:
        for k, v in tags.items():
            scope.set_tag(k, v)
        if "429" in str(e):
            scope.fingerprint = ["typesafe-rate-limited"]
        if isinstance(e, BaseException):
            sentry_sdk.capture_exception(e)
        else:
            sentry_sdk.capture_message(e, level="warning")


class _NoSpan:
    def set_tag(self, *a):
        pass

    def finish(self):
        pass


def draw_span(model: str, source: str):
    if not (sentry_sdk and sentry_sdk.is_initialized()):
        return _NoSpan()
    span = sentry_sdk.start_transaction(op="draw", name=f"draw {model} {source}")
    span.set_tag("model", model)
    span.set_tag("source", source)
    return span

DEMO = Path(__file__).resolve().parent
ROOT = DEMO.parent
JOURNAL = DEMO / "runs" / "journal.jsonl"
SOURCES = ("exact", "mock", "replay", "live")
ARGS = argparse.Namespace(live=False, daily_usd=9.0, per_ip_hour=0, max_live=0, journal=False, calls_per_min=2200,
                          upstream=None)
DRAW_USD = 0.0025  # a stage 7 or 8a draw bills about $0.0017-0.0019; budget a little more so the daily cap isn't overshot


# ---------- live budget, queue, pacing and upstream health ----------

class LiveRefused(Exception):
    """How a live draw ends early. The message is safe to show the visitor."""


class Cancelled(Exception):
    """The visitor went away; the draw's remaining calls are dropped."""


BUDGET = "Today's live budget is used up. Exact math still works, or try again tomorrow."
UNAVAILABLE = "Live Jev isn't available right now. Try exact math."
STILL_BUSY = "TypeSafe is still under heavy load. Try again in a few minutes."
DEADLINE_S = 300  # a live draw that hasn't finished 5 minutes after it arrived (queue included) ends
TICK_S = 0.25  # how often a waiting draw looks at its status and whether the visitor is still there
KEEPALIVE_S = 10  # an SSE comment at least this often, so proxies (Cloudflare cuts at 100 s idle, Fly) keep the stream
BACKOFF_S = (1, 2, 4, 8, 10)  # a call that hit a 429, 5xx or timeout waits this long before its next try, then every 10 s

SPEND = DEMO / "runs" / "spend.json"
_budget_lock = threading.Lock()
_ip_draws: dict[str, list[float]] = {}


def _spent_today() -> tuple[str, float]:
    today = time.strftime("%Y-%m-%d", time.gmtime())
    try:
        d = json.loads(SPEND.read_text())
    except (OSError, ValueError):
        d = {}
    return today, float(d.get(today, 0.0))


def budget_left() -> bool:
    with _budget_lock:
        return _spent_today()[1] + DRAW_USD <= ARGS.daily_usd


def reserve_live(ip: str):
    """Refuse a live draw over the daily dollar cap or the visitor's hourly limit, before any call."""
    if not budget_left():
        raise LiveRefused(BUDGET)
    with _budget_lock:
        if ARGS.per_ip_hour:  # visitors are only tracked when the limit is on, and only for the last hour
            now = time.time()
            for k in list(_ip_draws):
                if not (kept := [t for t in _ip_draws[k] if now - t < 3600]):
                    del _ip_draws[k]
                else:
                    _ip_draws[k] = kept
            if len(_ip_draws.get(ip, [])) >= ARGS.per_ip_hour:
                raise LiveRefused(f"That's {ARGS.per_ip_hour} live draws this hour, the limit per visitor. Exact math still works.")
            _ip_draws.setdefault(ip, []).append(now)


def charge_live(usd: float):
    with _budget_lock:
        today, spent = _spent_today()
        try:
            d = json.loads(SPEND.read_text())
        except (OSError, ValueError):
            d = {}
        d[today] = round(spent + usd, 6)
        SPEND.write_text(json.dumps(dict(sorted(d.items())[-7:])))  # the last 7 UTC days; the cap only reads today


def public_error(e: Exception) -> str:
    """What a visitor sees when a live draw ends early. Billing, key and backend failures are never described."""
    return str(e) if isinstance(e, LiveRefused) else UNAVAILABLE


class Admission:
    """At most `n` live draws run at once (0: no limit); the rest wait in line, first come first served. A waiting
    draw holds only its connection's thread and an unstarted generator."""

    def __init__(self, n: int):
        self.n, self.running, self.line, self.cv = n, 0, [], threading.Condition()

    def join(self) -> object:
        with self.cv:
            self.line.append(t := object())
            return t

    def admit(self, t, timeout: float) -> int | None:
        """None once `t` may run, else how many draws are ahead of it, after waiting up to `timeout`."""
        with self.cv:
            for _ in range(2):
                if self.line[0] is t and (not self.n or self.running < self.n):
                    self.line.pop(0)
                    self.running += 1
                    self.cv.notify_all()
                    return None
                if not timeout:
                    break
                self.cv.wait(timeout)
            return self.line.index(t)

    def leave(self, t):
        with self.cv:
            if t in self.line:
                self.line.remove(t)
                self.cv.notify_all()

    def release(self):
        with self.cv:
            self.running -= 1
            self.cv.notify_all()


RECOVER_S = 60.0  # after a 429 the pacer climbs back to the full rate over this long


class Pacer:
    """The Jev call rate all live draws share (GCRA). An idle demo sends a burst of up to 128 calls at once, so one
    draw's layer goes out together; past that, calls are spaced to `per_min`. A 429 halves the rate (once per 2 s
    at most, never below a sixteenth) and it climbs back linearly, so a busy Jev gets fewer calls, not none."""

    def __init__(self, per_min: float, burst: int = 128):
        self.max, self.burst, self.tat, self.cut, self.lock = per_min, burst, 0.0, None, threading.Lock()

    def rate(self, now: float) -> float:
        if self.cut is None:
            return self.max
        t, r = self.cut
        return min(self.max, r + (self.max - r) * (now - t) / RECOVER_S)

    def slow_down(self):
        with self.lock:
            now = time.monotonic()
            if self.max and not (self.cut and now - self.cut[0] < 2):
                self.cut = (now, max(self.max / 16, self.rate(now) / 2))

    def reserve(self, commit: bool = True) -> float:
        """Seconds until this call may go; a slot is claimed if `commit`, or if it may go now."""
        if not self.max:
            return 0.0
        with self.lock:
            now = time.monotonic()
            r = self.rate(now)
            step = 60 / r
            tat = max(self.tat, now)
            wait = max(0.0, tat - now - (max(8, self.burst * r / self.max) - 1) * step)
            if commit or not wait:
                self.tat = tat + step
            return wait


# Upstream health, for the banner: degraded after 3 transient failures (429, 5xx, timeouts) within a minute; clear
# after 20 successes in a row or 2 minutes without a failure.
_health = {"errs": deque(maxlen=3), "last": 0.0, "ok": 0, "on": False}


def upstream(ok: bool):
    h, now = _health, time.monotonic()
    if ok:
        h["ok"] += 1
        if h["on"] and h["ok"] >= 20:
            h["on"] = False
            print("upstream healthy again", flush=True)
        return
    h["errs"].append(now)
    h["last"], h["ok"] = now, 0
    if not h["on"] and len(h["errs"]) == 3 and now - h["errs"][0] < 60:
        h["on"] = True
        print("upstream degraded: live calls are hitting rate limits or errors", flush=True)
        report("TypeSafe upstream degraded: live calls are hitting rate limits or errors", upstream="typesafe")


def degraded() -> bool:
    if _health["on"] and time.monotonic() - _health["last"] > 120:
        _health["on"] = False
    return _health["on"]


# TypeSafe's own status page, for the banner's wording. Fetched on a background thread at most once a minute, so a
# slow or failing status page never touches a draw; None means unknown.
STATUS_URL = os.environ.get("JEVRONS_STATUS_URL", "https://status.typesafe.ai/index.json")
_ts = {"at": -math.inf, "value": None, "lock": threading.Lock()}


def fetch_status() -> dict | None:
    """{"state": "operational" or what isn't, "report": the title of an unresolved status report, or None}."""
    import urllib.request
    try:
        with urllib.request.urlopen(STATUS_URL, timeout=5) as r:
            d = json.load(r)
        inc = d.get("included") or []
        api = next((i["attributes"].get("status") for i in inc if i.get("type") == "status_page_resource"
                    and i["attributes"].get("public_name") == "api.typesafe.ai"), "operational")
        report_title = next((i["attributes"].get("title") for i in inc if i.get("type") == "status_report"
                       and i["attributes"].get("aggregate_state") != "resolved"), None)
        state = d["data"]["attributes"]["aggregate_state"]
        return {"state": state if state != "operational" else api, "report": report_title}
    except Exception as e:
        report(e, where="status-page")
        return None


def typesafe_status() -> dict | None:
    if time.monotonic() - _ts["at"] >= 60 and STATUS_URL and _ts["lock"].acquire(blocking=False):
        _ts["at"] = time.monotonic()
        def refresh():
            try:
                _ts["value"] = fetch_status()
            finally:
                _ts["lock"].release()
        threading.Thread(target=refresh, daemon=True).start()
    return _ts["value"]


def health() -> dict:
    """What the banner reads: our own calls' signal and TypeSafe's word."""
    return {"degraded": ARGS.live and degraded(), "typesafe": typesafe_status() if ARGS.live else None}


class Draw:
    """One live draw's shared state: the visitor's cancel flag, its deadline, and how many of its calls are
    waiting, and why ("busy": backing off after a failure, "pace": held by the shared rate)."""

    def __init__(self):
        self.cancel, self.deadline = threading.Event(), time.monotonic() + DEADLINE_S
        self.waits, self.lock = {"busy": 0, "pace": 0}, threading.Lock()

    def check(self):
        if self.cancel.is_set():
            raise Cancelled()
        if time.monotonic() > self.deadline:
            raise LiveRefused(STILL_BUSY)

    def sleep(self, s: float, why: str | None = None):
        with self.lock:
            if why:
                self.waits[why] += 1
        try:
            left = self.deadline - time.monotonic()
            if self.cancel.wait(max(0.0, min(s, left))):
                raise Cancelled()
            if s > left:
                raise LiveRefused(STILL_BUSY)
        finally:
            with self.lock:
                if why:
                    self.waits[why] -= 1

    def reason(self) -> str | None:
        return "busy" if self.waits["busy"] else "pace" if self.waits["pace"] else None


QUEUE = Admission(0)  # replaced at startup from --max-live
PACER = Pacer(0)  # replaced at startup from --calls-per-min


# ---------- models ----------

def registry() -> list[dict]:
    models = json.loads((DEMO / "models.json").read_text())
    for m in models:
        m["available"] = (ROOT / m["weights"]).exists()
    return models


def model(name: str) -> dict:
    m = next((m for m in registry() if m["name"] == name), None)
    if m is None or not m["available"]:
        raise ValueError(f"unknown or unavailable model {name!r}")
    return m


@cache
def params(path: str) -> list[tuple[np.ndarray, np.ndarray]]:
    d = np.load(ROOT / path)
    a = [d[f"arr_{i}"] for i in range(len(d.files))]
    return list(zip(a[::2], a[1::2]))


_test_lock = threading.Lock()


def test_set() -> tuple[np.ndarray, np.ndarray]:
    with _test_lock:  # a burst on a cold server would otherwise load the 60 MB test set once per request
        return _test_set()


@cache
def _test_set() -> tuple[np.ndarray, np.ndarray]:
    x = _load("t10k-images-idx3-ubyte.gz", 16).reshape(-1, 784) / 255.0
    return x, _load("t10k-labels-idx1-ubyte.gz", 8).astype(int)


E_QUESTIONS = {k: QUESTIONS[k] for k in ("v1", "choice")}


def neuron_request(fmt: str, x: np.ndarray, w: np.ndarray, b: float):
    """(state, questions, true z), exactly as the training run sent it."""
    if fmt == "scalar":  # Variant A: code adds up z, Jev only judges the sign
        z = r2(x @ w + b)
        return {"z": z}, SCALAR_Q_ALT, z
    keep = np.round(x, 2) != 0  # zero inputs are dropped, as in JevNeuron
    if fmt == "sparse-e":  # stage 8a's SparseENeuron: arm E, sending only terms whose input and weight are both nonzero
        keep &= np.round(w, 2) != 0
        products = [r2(v) for v in x[keep] * w[keep]] + [r2(b)]
        return {"products": products}, E_QUESTIONS, sum(products)
    if fmt == "e":  # stage 8g arm E, as CalibratedJevNeuron sends it: yes/no and a choice on one folded state
        products = [r2(v) for v in x[keep] * w[keep]] + [r2(b)]
        return {"products": products}, E_QUESTIONS, sum(products)
    state, z = neuron_state(fmt, x[keep], w[keep], b)
    return state, neuron_question(fmt), z


def activation(answers: dict) -> float:
    """Mean over the questions asked; a choice counts its "above" probability (CalibratedJevNeuron)."""
    return float(np.mean([a["above"] if isinstance(a, dict) else a for a in answers.values()]))


def terms(state: dict) -> list[float]:
    return state.get("products") or [state["z"]]


# ---------- neuron sources: (state, questions, z, tag, rng) -> {answers, backend, latency_ms, tokens} ----------

def exact_source(state, questions, z, tag, rng):
    return {"answers": {k: 1.0 if z > 0 else 0.0 for k in questions}, "backend": "local step", "latency_ms": 0,
            "tokens": 0}


@cache
def e_table(path: str) -> dict:
    return json.loads((ROOT / path).read_text())


def mock_source(state, questions, z, tag, rng):
    """Arm E neurons: resample a recorded test call of the same network family (stage 7 or 8a) with the same layer
    and a similar z / spread, so the mock inherits Jev's real leans and grading. Token count from a fit to the
    stage 7 journal (381 + 5.37 per number). Other neurons: MockJevNeuron (stage 1 fit: z + 0.77 * spread * N(0,1) > 0),
    latency and tokens fitted to the stage 5 journal (median 0.25 s, ~310 + 5.3 tokens per term)."""
    t = terms(state)
    spread = 0.0 if "z" in state else math.sqrt(sum(v * v for v in t))  # scalar Jev is a near-perfect step
    if "choice" in questions:
        tab = e_table(model(tag["model"])["mock"])
        b = int(np.digitize(z / (spread + 1e-6), tab["bins"]))
        have = [int(k.split(":")[1]) for k in tab["samples"] if k.startswith(f"{tag['layer']}:")]
        pool = tab["samples"][f"{tag['layer']}:{min(have, key=lambda h: abs(h - b))}"]
        v1, above, latency = pool[rng.integers(len(pool))]
        latency = min(latency, 2.0)  # the journal's tail is rate-limit retries at 128 threads
        answers, tokens = {"v1": v1, "choice": {"above": above, "below": round(1 - above, 2)}}, 381 + 5.37 * len(t)
    else:
        latency = min(2.0, rng.lognormal(math.log(0.25), 0.35))
        fires = z + 0.77 * spread * rng.standard_normal() > 0
        answers, tokens = {k: 0.97 if fires else 0.03 for k in questions}, 310 + 5.3 * len(t)
    time.sleep(latency)
    return {"answers": answers, "backend": "simulated", "latency_ms": round(latency * 1000), "tokens": round(tokens)}


def _key(state, questions) -> str:
    return hashlib.sha1(json.dumps([state, questions], sort_keys=True).encode()).hexdigest()


def records(path: Path):
    return (json.loads(line) for line in path.read_text().splitlines() if line.strip()) if path.exists() else ()


@cache
def replay_index(name: str) -> dict[str, dict]:
    """The committed demo live runs (demo/runs/journal.jsonl), plus the model's cut of its training journal, read once
    per model: draws journaled with --journal replay after a restart. Later records win, so a state asked twice
    replays the answer the rest of that run was built on."""
    m = model(name)
    slim = lambda r: {k: r[k] for k in ("answers", "served", "latency_s")} | {"usage": {"input_tokens": r["usage"].get("input_tokens", 0)}}
    index = {_key(r["state"], r["questions"]): slim(r) for r in records(DEMO / "runs" / "journal.jsonl") if r["tag"].get("model", name) == name}
    if "replay" in m:
        index.update((r["key"], slim(r)) for r in records(ROOT / m["replay"]))
    return index  # only what replay_source reads: the states themselves stay out of memory


def replay_source(index):
    def source(state, questions, z, tag, rng):
        r = index.get(_key(state, questions))
        if r is None:  # never asked: fall back to the mock, which labels itself "simulated"
            return mock_source(state, questions, z, tag, rng)
        wait = r["latency_s"]  # keep the recorded arrival order, but squeeze rate-limit retries (15 s -> 2.9 s)
        time.sleep(wait if wait < 1.5 else 1.5 + 0.1 * (wait - 1.5))
        return {"answers": r["answers"], "backend": f"recorded {r['served']}",
                "latency_ms": round(r["latency_s"] * 1000), "tokens": r["usage"].get("input_tokens", 0)}
    return source


@cache
def recorded_digits(path: str) -> list[int]:
    return sorted({r["mnist"] for r in records(ROOT / path)})


@cache
def recorded_predictions(name: str) -> dict[int, str]:
    """What the model predicts on each of its recorded test digits, on the recorded answers (no waiting): the same
    requests and lookup as replay_source, layer by layer."""
    m, index, x = model(name), replay_index(name), test_set()[0]
    out = {}
    for i in recorded_digits(m["replay"]) if "replay" in m else ():
        a = x[i]
        for W, b in params(m["weights"]):
            recs = [index.get(_key(*neuron_request(m["format"], a, W[:, j], b[j])[:2])) for j in range(W.shape[1])]
            if None in recs:
                break
            a = np.array([activation(r["answers"]) for r in recs])
        else:
            out[i] = m["labels"][int(decide(a[None])[0])]
    return out


@cache
def picks() -> dict[str, dict]:
    """Per model, its recorded test digits and its disagreements with its twin: digit indices only, so it's published
    while the replay and mock files (raw Jev answers) stay local. --check rebuilds it when those files are present."""
    p = DEMO / "runs" / "recorded.json"
    return json.loads(p.read_text()) if p.exists() else {}


def make_picks() -> dict[str, dict]:
    return {m["name"]: {"digits": recorded_digits(m["replay"]), "disagree": disagreements(m["name"])}
            for m in registry() if m["available"] and "replay" in m and (ROOT / m["replay"]).exists()}


def disagreements(name: str) -> list[int]:
    """Recorded test digits that the model reads right and its twin reads wrong, both on recorded answers
    (falls back to any digit they read differently if there are none)."""
    mine, theirs = recorded_predictions(name), recorded_predictions(model(name)["twin"])
    y = test_set()[1]
    both = sorted(i for i in mine.keys() & theirs.keys() if mine[i] != theirs[i])
    return [i for i in both if mine[i] == str(y[i])] or both


_captured = threading.local()
_jev = None
_jev_lock = threading.Lock()


def live_jev() -> Jev:
    """One Jev client per process. Each backend's post is wrapped so the calling thread can read back the
    usage and latency that Jev journals (Jev.ask returns only the answers)."""
    global _jev
    with _jev_lock:
        if _jev is None:
            if ARGS.upstream:  # dev: every backend goes to a local stub, and no real key is sent to it
                for b in BACKENDS.values():
                    b.key_var = None
            # the daily cap is enforced per draw in reserve_live; without --journal nothing is written to disk
            _jev = Jev(JOURNAL if ARGS.journal else None, max_usd=None, wait_rounds=1)
            for b in _jev.backends:
                if ARGS.upstream:
                    b.endpoint = ARGS.upstream
                # PACER spaces the calls all draws share; the backend's own pacing would only delay a layer's burst
                b.per_min = max(b.per_min, 30000)
                b.attempts = 1  # live_source retries, with backoff that stops when the visitor leaves
                def post(state, questions, _orig=b.post, _name=b.name):
                    t0 = time.monotonic()
                    resp, attempts = _orig(state, questions)
                    _captured.last = (_name, resp, time.monotonic() - t0)
                    return resp, attempts
                b.post = post
        return _jev


# Jev's latency has a long tail: in 64-call bursts the median call took 0.3-0.5 s, but a few per draw hung for
# 15-25 s or came back 504, and a layer waits for its slowest call. So a call that isn't back by each HEDGE_S mark
# is sent again, if the shared rate has room, and the first answer wins.
HEDGE_S = (1.5, 4.0, 10.0)
# ponytail: a losing attempt keeps its thread until the gateway answers or times out; fine at --max-live 16.
_attempts = ThreadPoolExecutor(128)


def transient(e: Exception) -> bool:
    """429, 5xx, timeouts and network errors pass; a 4xx (key, credits, a bad request) or the wrong model doesn't."""
    return "HTTP 4" not in str(e) and " served " not in str(e)


def hedged(state, questions, tag, draw):
    def attempt():
        draw.check()  # an attempt still queued in the pool when the visitor leaves is never sent
        answers = live_jev().ask(state, questions, tag)
        return answers, _captured.last  # read on the thread that made the call

    t0, futs = time.monotonic(), [_attempts.submit(attempt)]
    while True:
        mark = HEDGE_S[len(futs) - 1] if len(futs) <= len(HEDGE_S) else math.inf
        left = mark - (time.monotonic() - t0)  # <= 0: a hedge is due but the shared rate had no room; ask again next tick
        wait([f for f in futs if not f.done()] or futs, timeout=left if 0 < left < TICK_S else TICK_S,
             return_when=FIRST_COMPLETED)
        won = next((f for f in futs if f.done() and f.exception() is None), None)
        if won:
            answers, (name, resp, latency) = won.result()
            tokens = (resp.get("usage") or {}).get("input_tokens", 0)
            return {"answers": answers, "backend": f"{name} {resp.get('model', '?')}",
                    "latency_ms": round((time.monotonic() - t0) * 1000), "hedged": len(futs) - 1,
                    "tokens": tokens * len(futs), "billed": True}  # every attempt sent is billed
        if all(f.done() for f in futs):
            raise futs[-1].exception()
        draw.check()
        if time.monotonic() - t0 >= mark and PACER.reserve(commit=False) == 0:
            futs.append(_attempts.submit(attempt))


def live_source(state, questions, z, tag, rng, draw):
    """Only live Jev answers a live neuron. A 429, 5xx or timeout backs off and tries again (BACKOFF_S, with
    jitter) until the draw's deadline; a key or billing failure ends the draw."""
    for n in range(10**6):
        draw.check()
        if (w := PACER.reserve()) > 0:
            draw.sleep(w, "pace" if w > 1 else None)  # the page is only told about a wait it would notice
        try:
            out = hedged(state, questions, tag, draw)
        except (Cancelled, LiveRefused):
            raise
        except Exception as e:
            if not transient(e):
                raise LiveRefused(UNAVAILABLE) from e
            upstream(ok=False)
            if "429" in str(e):
                PACER.slow_down()
            draw.sleep(BACKOFF_S[min(n, len(BACKOFF_S) - 1)] * random.uniform(0.75, 1.25), "busy")
            continue
        upstream(ok=True)
        return out


# ---------- inference ----------

def run(name: str, source: str, pixels, seed: int = 0, draw: Draw | None = None):
    """Yield one event per neuron in the order answers arrive, layer by layer, then the result. A live draw passes
    its Draw, and then also yields None every TICK_S while nothing arrives, so the caller can keep the stream alive."""
    m = model(name)
    if source == "live" and not ARGS.live:
        raise ValueError("live source is off; start the server with --live")
    if source == "live" and draw is None:
        raise ValueError("a live draw needs a Draw")
    ask = {"exact": exact_source, "mock": mock_source, "live": lambda *a: live_source(*a, draw=draw)}.get(source) \
        or (lambda *a: replay_source(replay_index(name))(*a))  # the replay index is read on first use
    run_id = uuid.uuid4().hex[:8]
    a = np.clip(np.asarray(pixels, float), 0, 1)
    calls = tokens = 0
    yield {"type": "start", "run": run_id, "model": name, "source": source, "labels": m["labels"],
           "layers": [W.shape[1] for W, _ in params(m["weights"])]}
    t0 = time.monotonic()  # after start: a live draw may wait in line between the two
    for layer, (W, b) in enumerate(params(m["weights"])):
        def one(j, x=a, W=W, b=b, layer=layer):
            state, questions, z = neuron_request(m["format"], x, W[:, j], b[j])
            tag = {"demo": run_id, "model": name, "layer": layer, "j": j, "z": z}
            ans = ask(state, questions, z, tag, np.random.default_rng([seed, layer, j]))
            p = activation(ans["answers"])
            return {"type": "neuron", "layer": layer, "j": j, "n_terms": len(terms(state)), "z": round(z, 4),
                    "terms": terms(state), "questions": questions, "p": p, "disagree": (p >= 0.5) != (z > 0), **ans}

        yield {"type": "layer", "layer": layer, "n": W.shape[1], "t_ms": round((time.monotonic() - t0) * 1000)}
        acts = np.zeros(W.shape[1])
        # one thread per neuron at most; exact answers are instant, so they need none of their own
        ex = ThreadPoolExecutor(1 if source == "exact" else W.shape[1])
        try:
            pending = {ex.submit(one, j) for j in range(W.shape[1])}
            while pending:
                done, pending = wait(pending, timeout=TICK_S if draw else None, return_when=FIRST_COMPLETED)
                if not done:
                    yield None
                for f in done:
                    ev = f.result()
                    acts[ev["j"]] = ev["p"]
                    calls += 1
                    tokens += ev["tokens"]
                    yield {**ev, "t_ms": round((time.monotonic() - t0) * 1000)}
        finally:  # a draw that ends early (the visitor left, a failure) drops its queued neurons and doesn't wait
            ex.shutdown(wait=False, cancel_futures=True)
        a = acts
    yield {"type": "result", "outputs": a.tolist(), "prediction": m["labels"][int(decide(a[None])[0])],
           "no_fire": bool((a < 0.5).all()) if len(a) > 1 else False, "calls": calls, "tokens": tokens,
           "usd": tokens * USD_PER_INPUT_TOKEN, "billed": source == "live" and tokens > 0,
           "elapsed_ms": round((time.monotonic() - t0) * 1000)}


# ---------- http ----------

class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *a, **k):
        super().__init__(*a, directory=str(DEMO / "web"), **k)

    def log_message(self, fmt, *a):
        pass

    def send_json(self, obj, code=200):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        url = urlparse(self.path)
        q = {k: v[0] for k, v in parse_qs(url.query).items()}
        try:
            if url.path == "/api/models":
                return self.send_json({"models": registry(), "live": ARGS.live, "daily_usd": ARGS.daily_usd} | health())
            if url.path == "/api/status":  # polled by open pages every 30 s while live, for the banner
                return self.send_json(health())
            if url.path == "/api/weights":
                (W1, b1), (W2, b2) = params(model(q.get("name", ""))["weights"])
                return self.send_json({"W1": np.round(W1.T, 3).tolist(), "b1": b1.round(3).tolist(),
                                       "W2": np.round(W2, 3).tolist(), "b2": b2.round(3).tolist()})
            if url.path == "/api/disagree":  # recorded test digits this model and its twin read differently
                name = q.get("model", "")
                return self.send_json({"model": name, "twin": model(name)["twin"],
                                       "digits": picks().get(name, {}).get("disagree", [])})
            if url.path == "/api/digit":
                x, y = test_set()
                labels = [int(v) for v in q.get("labels", "").split(",") if v != ""] or list(range(10))
                pool = np.where(np.isin(y, labels))[0]
                if "recorded" in q:  # digits whose every call was recorded for this model
                    rec = [i for i in picks().get(model(q["recorded"])["name"], {}).get("digits", []) if y[i] in labels]
                    pool = np.array(rec) if rec else pool
                i = int(q["index"]) if "index" in q else int(np.random.default_rng().choice(pool))
                return self.send_json({"index": i, "label": int(y[i]), "pixels": x[i].round(3).tolist()})
        except (ValueError, KeyError, IndexError) as e:
            return self.send_json({"error": str(e)}, 400)
        return super().do_GET()

    def client_ip(self) -> str:
        # ponytail: trusts the proxy's client-IP header; run behind a proxy that sets it (Fly does), or a visitor
        # can spoof it and dodge the per-visitor limit. The daily dollar cap holds either way.
        fwd = self.headers.get("Fly-Client-IP") or self.headers.get("X-Forwarded-For", "").split(",")[0].strip()
        return fwd or self.client_address[0]

    def gone(self) -> bool:
        """The visitor closed the connection: the socket reads as ended (it sends nothing more after the body)."""
        try:
            return bool(select.select([self.connection], [], [], 0)[0]) and not self.connection.recv(1, socket.MSG_PEEK)
        except OSError:
            return True

    def do_POST(self):
        if urlparse(self.path).path != "/api/run":
            return self.send_json({"error": "not found"}, 404)
        try:
            req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
            # a test digit is sent by index so the states match the training journal to the last digit
            pixels = test_set()[0][int(req["mnist"])] if "mnist" in req else [float(v) for v in req["pixels"]]
            if len(pixels) != 784 or req.get("source") not in SOURCES:
                raise ValueError("need 784 pixels and a known source")
            live = req["source"] == "live"
            draw = Draw() if live else None
            events = run(str(req["model"]), req["source"], pixels, int(req.get("seed", 0)), draw)
            first = next(events)  # validation errors surface here, before the stream starts
        except (ValueError, KeyError, TypeError, json.JSONDecodeError) as e:
            return self.send_json({"error": str(e)}, 400)
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        last, shown = [0.0], {"health": None, "wait": (None, None)}
        span, outcome = draw_span(str(req["model"]), req["source"]), "error"

        def send(ev=None):  # None: a keepalive comment, if the stream has been quiet for KEEPALIVE_S
            if ev is None and time.monotonic() - last[0] < KEEPALIVE_S:
                return
            self.wfile.write(b": keepalive\n\n" if ev is None else f"data: {json.dumps(ev)}\n\n".encode())
            self.wfile.flush()
            last[0] = time.monotonic()

        def status(wait):  # (reason, ahead): tell the page when it changes, and whether Jev's upstream is degraded
            if self.gone():
                raise ConnectionResetError
            if (h := health()) != shown["health"]:
                shown["health"] = h
                send({"type": "degraded", "on": h["degraded"], "typesafe": h["typesafe"]})
            if wait != shown["wait"]:
                shown["wait"] = wait
                send({"type": "waiting", "reason": wait[0]} | ({"ahead": wait[1]} if wait[1] is not None else {}))
            send()

        usd, done, admitted = 0.0, False, False
        try:
            send(first)
            if live:
                status((None, None))
                reserve_live(self.client_ip())
                ticket, timeout = QUEUE.join(), 0
                try:
                    while (ahead := QUEUE.admit(ticket, timeout)) is not None:
                        draw.check()
                        status(("queue", ahead))
                        timeout = TICK_S
                    admitted = True
                finally:
                    if not admitted:
                        QUEUE.leave(ticket)
                if not budget_left():  # spent while this draw waited in line
                    raise LiveRefused(BUDGET)
            for ev in events:
                if live:
                    status((draw.reason(), None))
                if ev is None:
                    continue
                if ev["type"] == "neuron" and ev.get("billed"):
                    usd += ev["tokens"] * USD_PER_INPUT_TOKEN
                done = ev["type"] == "result"
                send(ev)
            outcome = "done" if done else outcome
        except (BrokenPipeError, ConnectionResetError, Cancelled):
            outcome = "left"  # the visitor left: the finally below stops the draw's remaining calls
        except Exception as e:
            outcome = "refused" if isinstance(e, LiveRefused) and not e.__cause__ else "error"
            if not isinstance(e, LiveRefused) or e.__cause__:
                print(f"run failed: {e!r}" + (f" from {e.__cause__!r}" if e.__cause__ else ""), flush=True)
                report(e.__cause__ or e, model=str(req["model"]), source=req["source"])
            try:
                send({"type": "error", "message": public_error(e) if live else str(e)})
            except OSError:
                pass
        finally:
            if live:
                draw.cancel.set()
            events.close()
            if admitted:
                # a failed or abandoned draw can still bill calls that were in flight: charge it a full draw
                charge_live(usd if done else max(usd, DRAW_USD))
                QUEUE.release()
            span.set_tag("outcome", outcome)
            span.finish()


def check():
    """Streamed exact inference must match jevrons.net.forward with StepNeuron on real test digits. Compared on
    output activations (predictions can differ on random tie breaks). The demo's true z uses the two-decimal
    values actually sent, so a sum within rounding of zero may flip: allow 4% of digits to differ (exact-trained
    swap networks keep output sums close to zero)."""
    x, y = test_set()
    for m in registry():
        if not m["available"]:
            print(f"{m['name']:>14}: weights not there yet")
            continue
        idx = np.where(np.isin(y, [int(v) for v in m["labels"]]))[0][:200]
        ref = forward(params(m["weights"]), x[idx], StepNeuron())[0][-1]
        res = [list(run(m["name"], "exact", x[i]))[-1] for i in idx]
        same = np.mean([np.array_equal(r["outputs"], o) for r, o in zip(res, ref)])
        acc = np.mean([r["prediction"] == str(t) for r, t in zip(res, y[idx])])
        print(f"{m['name']:>14}: outputs match forward on {same:.1%} of {len(idx)} test digits, exact-neuron accuracy {acc:.3f}")
        assert same >= 0.96, m["name"]
    check_sparse_mask()
    # Replay of recorded test digits must hit the training journal on every call and land on its recorded outputs:
    # the demo builds each state (which terms, which rounding) exactly as the training run did.
    time.sleep = lambda s: None
    from jevrons.digits import test_subset
    for name, result, n, seed in (("s7-jev", "runs/stage7/result-jev.json", 2000, 7),
                                  ("s8a-jev", "runs/stage8a/result-jev.json", 1000, 0),
                                  ("s8a-swap", "runs/stage8a/result-swap.json", 1000, 0)):
        if not (ROOT / model(name)["replay"]).exists():
            print(f"{name:>14}: replay files aren't published (raw Jev answers); replay check skipped")
            continue
        te = list(test_subset(y, n, seed=seed))
        recorded = json.loads((ROOT / result).read_text())["test"]["outputs"]
        digits = recorded_digits(model(name)["replay"])
        for i in digits:
            evs = list(run(name, "replay", x[i]))
            assert all(e["backend"].startswith("recorded") for e in evs if e["type"] == "neuron"), (name, i)
            assert np.allclose(np.round(evs[-1]["outputs"], 2), recorded[te.index(i)], atol=0.006), (name, i)
        print(f"{name:>14}: replay of {len(digits)} recorded test digits matches its training journal call for call")
    if all((ROOT / m["replay"]).exists() for m in registry() if "replay" in m):
        want = make_picks()
        if want != picks():
            (DEMO / "runs" / "recorded.json").write_text(json.dumps(want) + "\n")
            picks.cache_clear()
        print("   picks index: demo/runs/recorded.json matches the replay files")
    check_live_limits(x[0])


def check_sparse_mask():
    """Stage 8a's connections are the weights' zero pattern (net.fit holds absent weights at 0), and a hidden call
    sends exactly the connected, nonzero-weight terms whose pixel is nonzero, plus the bias."""
    from jevrons.stage8a import KIND, K, H, active_pixels, connections, data
    mask = connections(KIND, K, H, active_pixels(data()[0])) == 1
    x = test_set()[0][0]
    for name in ("s8a-jev", "s8a-swap"):
        (W1, b1), _ = params(model(name)["weights"])
        assert np.array_equal(W1 != 0, mask), name
        for j in range(H):
            state, _, _ = neuron_request("sparse-e", x, W1[:, j], b1[j])
            want = (np.round(x, 2) != 0) & mask[:, j] & (np.round(W1[:, j], 2) != 0)
            assert len(state["products"]) == want.sum() + 1, (name, j)
    print(f"      s8a mask: W1 != 0 is stage8a.connections() for both arms; hidden calls send connected, nonzero terms only")


def stub_upstream(behave, port: int = 0, status=lambda: (200, {"data": {"attributes": {"aggregate_state": "operational"}}})):
    """A local stand-in for Jev's API, for the check and for trying the demo offline ($0). `behave(n)` gets the
    call's number (1, 2, ...) and returns (HTTP code, seconds to take). A 200 answers every question
    confidently yes (above) and reports the model and upstream the request asked for."""
    count, lock = [0], threading.Lock()

    class Stub(SimpleHTTPRequestHandler):
        def do_POST(self):
            req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            with lock:
                count[0] += 1
                code, delay = behave(count[0])
            threading.Event().wait(delay)  # not time.sleep, which check() turns off for replay
            body = {"detail": "The upstream provider is currently experiencing high demand" if code == 429 else "stub"}
            if code == 200:
                body = {"model": req["model"], "usage": {"input_tokens": 500},
                        "answers": {k: {"noul": 0.8} if q["type"] == "noul" else {"probabilities": {"above": 0.7, "below": 0.3}}
                                    for k, q in req["questions"].items()}}
                if "providerOptions" in req:
                    body["provider_metadata"] = {"gateway": {"routing": {"finalProvider": "typesafe-ai"}}}
            data = json.dumps(body).encode()
            self.send_response(code)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):  # a status page, shaped like status.typesafe.ai/index.json
            code, body = status()
            data = json.dumps(body).encode()
            self.send_response(code)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *a):
            pass

    srv = ThreadingHTTPServer(("127.0.0.1", port), Stub)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, count


def check_live_limits(pixels):
    """The live path against a local stub (nothing billed, no real key sent): a draw only ever gets live answers.
    429s back off and retry until they succeed, with the page told it's waiting and that the upstream is degraded;
    a 402 ends the draw with a generic message; the daily cap refuses before any call; draws past --max-live wait
    in line and a visitor who leaves frees the slot and stops the draw's calls."""
    global JOURNAL, SPEND, QUEUE, PACER, KEEPALIVE_S, STATUS_URL, _jev
    import tempfile
    import urllib.request
    tmp = Path(tempfile.mkdtemp())
    JOURNAL, SPEND = tmp / "journal.jsonl", tmp / "spend.json"
    behave = {"f": lambda n: (200, 0)}
    ok = {"data": {"attributes": {"aggregate_state": "operational"}}, "included": [
        {"type": "status_page_resource", "attributes": {"public_name": "api.typesafe.ai", "status": "operational"}},
        {"type": "status_report", "attributes": {"title": "Elevated API latency", "aggregate_state": "resolved"}}]}
    page = {"v": (200, ok)}
    stub, calls = stub_upstream(lambda n: behave["f"](n), status=lambda: page["v"])
    STATUS_URL = f"http://127.0.0.1:{stub.server_port}/index.json"  # never the real status page
    assert fetch_status() == {"state": "operational", "report": None}
    page["v"] = (200, {"data": {"attributes": {"aggregate_state": "degraded"}}, "included": ok["included"] + [
        {"type": "status_report", "attributes": {"title": "Degradation in API Traffic", "aggregate_state": "degraded"}}]})
    assert fetch_status() == {"state": "degraded", "report": "Degradation in API Traffic"}
    page["v"] = (500, {})
    assert fetch_status() is None  # unknown: the banner goes on our own calls alone
    page["v"] = (200, ok)
    srv = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    ARGS.live, ARGS.daily_usd, ARGS.upstream = True, 1.0, f"http://127.0.0.1:{stub.server_port}/"
    _jev, QUEUE, PACER = None, Admission(0), Pacer(2200)
    body = json.dumps({"model": "s8a-jev", "source": "live", "pixels": list(map(float, pixels))}).encode()

    def draw():
        req = urllib.request.Request(f"http://127.0.0.1:{srv.server_port}/api/run", body)
        text = urllib.request.urlopen(req).read().decode()
        return [json.loads(line[6:]) for line in text.splitlines() if line.startswith("data: ")], text.count(": keepalive")

    def finished(evs):  # every neuron answered by live Jev, and billed
        neurons = [e for e in evs if e["type"] == "neuron"]
        assert len(neurons) == 74 and evs[-1]["type"] == "result" and evs[-1]["billed"], evs[-1]
        assert all(e["billed"] and not e["backend"].startswith(("simulated", "recorded")) for e in neurons)

    def opened():  # a draw on a raw connection, so the test can hang up on it
        c = socket.create_connection(("127.0.0.1", srv.server_port))
        c.sendall(b"POST /api/run HTTP/1.1\r\nHost: x\r\nContent-Length: %d\r\n\r\n%s" % (len(body), body))
        return c, c.makefile("rb")

    def until(ok, s=3.0):
        t = time.monotonic() + s
        while not ok() and time.monotonic() < t:
            threading.Event().wait(0.02)
        return ok()

    behave["f"] = lambda n: (429 if n <= 64 else 200, 0)  # the whole hidden layer is refused once, then Jev recovers
    evs, _ = draw()
    finished(evs)
    assert {"type": "waiting", "reason": "busy"} in evs, [e for e in evs if e["type"] == "waiting"]
    flags = [k for k, _ in groupby(e["on"] for e in evs if e["type"] == "degraded")]
    # on after the 429s, off after 20 successes; a slow runner can land a late 429 after that and flip it once more
    assert flags[:3] == [False, True, False] and flags[-1] is False and not degraded() and PACER.cut, flags
    status = json.load(urllib.request.urlopen(f"http://127.0.0.1:{srv.server_port}/api/status"))
    assert status == {"degraded": False, "typesafe": {"state": "operational", "report": None}}, status
    behave["f"] = lambda n: (402, 0)
    evs, _ = draw()
    assert evs[-1] == {"type": "error", "message": UNAVAILABLE} and not any(e["type"] == "neuron" for e in evs), evs[-1]
    # let that draw wind down fully: its charge is written before its slot is released, and calls already sent may
    # still land. On a slow runner (CI) both can trail the error event.
    assert until(lambda: QUEUE.running == 0 and _attempts._work_queue.empty(), 10.0), "402 draw never released its slot"
    n0 = -1
    while n0 != calls[0]:
        n0 = calls[0]
        threading.Event().wait(0.5)
    behave["f"] = lambda n: (200, 0)
    SPEND.write_text(json.dumps({time.strftime("%Y-%m-%d", time.gmtime()): 1.0}))
    evs, _ = draw()
    assert evs[-1] == {"type": "error", "message": BUDGET} and calls[0] == n0 and _spent_today()[1] == 1.0, evs[-1]
    SPEND.write_text("{}")
    ARGS.per_ip_hour = 1
    _ip_draws["203.0.113.9"] = [time.time() - 7200]  # a visitor from two hours ago is forgotten at the next draw
    finished(draw()[0])
    assert list(_ip_draws) == ["127.0.0.1"], _ip_draws
    assert "per visitor" in draw()[0][-1]["message"]
    ARGS.per_ip_hour = 0

    QUEUE, PACER, KEEPALIVE_S = Admission(1), Pacer(2200), 0.3
    behave["f"] = lambda n: (200, 0.6)
    first = threading.Thread(target=lambda: finished(draw()[0]))
    first.start()
    assert until(lambda: QUEUE.running == 1)
    evs, keepalives = draw()  # waits for the first draw, then runs
    first.join()
    finished(evs)
    waits = [e for e in evs if e["type"] == "waiting"]
    # it waited in line first, then ran; a slow runner may also show pacing or a retry while it runs
    assert waits[0] == {"type": "waiting", "reason": "queue", "ahead": 0} and keepalives, waits
    assert all(w.get("reason") in (None, "queue", "busy", "pace") for w in waits) and waits[-1]["reason"] is None, waits

    behave["f"] = lambda n: (200, 1.0)
    base = calls[0]
    a, ra = opened()
    assert until(lambda: QUEUE.running == 1)
    assert until(lambda: calls[0] - base >= 64, 10.0), "the hidden layer never went out"  # all 64 sent, 1 s each
    b, rb = opened()
    while b"queue" not in rb.readline():
        pass
    rb.close(), b.close()
    assert until(lambda: not QUEUE.line) and QUEUE.running == 1, "a visitor who left is still in line"
    n0 = calls[0]
    ra.close(), a.close()  # leave mid-draw, while the hidden layer is out
    assert until(lambda: QUEUE.running == 0), "a visitor who left still holds a slot"
    threading.Event().wait(1.5)
    assert calls[0] == n0, f"{calls[0] - n0} calls sent after the visitor left"  # the output layer never goes out
    assert not JOURNAL.exists() and not list(tmp.glob("*failures*")), "live calls were journaled without --journal"
    SPEND.write_text(json.dumps({f"2026-01-{d:02d}": 1.0 for d in range(1, 11)}))
    charge_live(0.0)
    assert len(json.loads(SPEND.read_text())) == 7, SPEND.read_text()
    for s in (stub, srv):
        s.shutdown()
    print("          live: only live answers; 429s back off and finish live (waiting + degraded on/off told), a 402 ends\n"
          "                the draw plainly, the daily cap and per-visitor limit refuse before any call, max-live queues,\n"
          "                leaving frees the slot and stops calls; TypeSafe's status page read (operational, a report, down);\n"
          "                nothing journaled without --journal, spend.json keeps 7 days")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--live", action="store_true", help="allow the live source (real Jev calls, billed)")
    ap.add_argument("--daily-usd", type=float, default=9.0, help="live spend cap per UTC day, about 5,000 draws at $9 (demo/runs/spend.json)")
    ap.add_argument("--per-ip-hour", type=int, default=0, help="live draws per visitor per hour (0 = no limit)")
    ap.add_argument("--max-live", type=int, default=0, help="live draws running at once; the rest wait in line (0 = no limit)")
    ap.add_argument("--calls-per-min", type=float, default=2200, help="Jev calls a minute across all live draws, halved after a 429 (0 = no limit)")
    ap.add_argument("--journal", action="store_true", help="keep live calls in demo/runs/journal.jsonl (off: nothing written)")
    ap.add_argument("--upstream", help=argparse.SUPPRESS)  # dev: send every live call to this URL (a local stub), keyless
    ap.add_argument("--stub", help=argparse.SUPPRESS)  # dev: serve only a fake Jev on --port, e.g. busy=30,delay=0.5,code=200
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--check", action="store_true")
    ARGS = ap.parse_args()
    if ARGS.check:
        check()
    elif ARGS.stub is not None:  # 429 for the first `busy` seconds, then `code` after `delay` seconds a call
        spec, t0 = {"busy": 0, "delay": 0, "code": 200} | {k: float(v) for k, v in (kv.split("=") for kv in ARGS.stub.split(",") if kv)}, time.monotonic()
        ThreadingHTTPServer.request_queue_size = 1024
        stub_upstream(lambda n: (429, 0) if time.monotonic() - t0 < spec["busy"] else (int(spec["code"]), spec["delay"]), ARGS.port)
        print(f"fake Jev at http://127.0.0.1:{ARGS.port}/  {spec}", flush=True)
        threading.Event().wait()
    else:
        init_telemetry()
        QUEUE, PACER = Admission(ARGS.max_live), Pacer(ARGS.calls_per_min)
        if ARGS.upstream:  # stub calls cost nothing: keep them off the real spend counter
            import tempfile
            SPEND = Path(tempfile.mkdtemp()) / "spend.json"
        print(f"http://{ARGS.host}:{ARGS.port}  live={'on' if ARGS.live else 'off'}  daily cap ${ARGS.daily_usd}"
              + (f"  upstream {ARGS.upstream}" if ARGS.upstream else ""), flush=True)
        ThreadingHTTPServer.request_queue_size = 128  # the default backlog of 5 drops connections in a burst of draws
        ThreadingHTTPServer((ARGS.host, ARGS.port), Handler).serve_forever()
