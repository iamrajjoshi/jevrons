"""Jevrons demo server: runs a drawn digit through a trained net and streams every neuron as it resolves.

  uv run python demo/server.py              # exact (visitors), mock and replay (URL flags); http://127.0.0.1:8765
  uv run python demo/server.py --live       # also allow live Jev calls (add --journal to keep them in demo/runs/journal.jsonl)
  uv run python demo/server.py --check      # self-check: exact == jevrons.net.forward, replay == journals, live limits

Live calls go to whatever backends jevrons.jev picks; set JEVRONS_BACKENDS=vercel to pin one.
Models come from demo/models.json; an entry whose weights file is missing is listed as unavailable.
Replay and mock data are cut from the stage 7 and 8a training journals by demo/make_replay.py.
A live draw that is refused (budget, limits) or fails mid-run finishes on recorded answers where the exact
state was recorded and simulated ones elsewhere, and says so per neuron.
"""

import argparse
import hashlib
import json
import math
import threading
import time
import uuid
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, as_completed, wait
from functools import cache
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np

from jevrons.digits import _load
from jevrons.jev import USD_PER_INPUT_TOKEN, Jev
from jevrons.net import StepNeuron, decide, forward
from jevrons.stage8g import QUESTIONS
from jevrons.states import SCALAR_Q_ALT, neuron_question, neuron_state, r2

DEMO = Path(__file__).resolve().parent
ROOT = DEMO.parent
JOURNAL = DEMO / "runs" / "journal.jsonl"
SOURCES = ("exact", "mock", "replay", "live")
ARGS = argparse.Namespace(live=False, daily_usd=9.0, per_ip_hour=0, max_live=0, journal=False)
DRAW_USD = 0.0025  # a stage 7 or 8a draw bills about $0.0017-0.0019; budget a little more so the daily cap isn't overshot


# ---------- live budget and limits ----------

class LiveRefused(Exception):
    """A live draw the server won't start. The message is safe to show the visitor."""


SPEND = DEMO / "runs" / "spend.json"
_budget_lock = threading.Lock()
_live_slots = None  # set at startup when --max-live is on
_ip_draws: dict[str, list[float]] = {}


def _spent_today() -> tuple[str, float]:
    today = time.strftime("%Y-%m-%d", time.gmtime())
    try:
        d = json.loads(SPEND.read_text())
    except (OSError, ValueError):
        d = {}
    return today, float(d.get(today, 0.0))


def reserve_live(ip: str):
    """Refuse a live draw over the daily dollar cap, over the visitor's hourly limit, or when too many draws are
    already running. Returns the day the draw is charged to."""
    with _budget_lock:
        today, spent = _spent_today()
        if spent + DRAW_USD > ARGS.daily_usd:
            raise LiveRefused("Today's live budget is used up, so this draw uses recorded and simulated answers.")
        now = time.time()
        if ARGS.per_ip_hour:  # visitors are only tracked when the limit is on, and only for the last hour
            for k in list(_ip_draws):
                if not (kept := [t for t in _ip_draws[k] if now - t < 3600]):
                    del _ip_draws[k]
                else:
                    _ip_draws[k] = kept
            if len(_ip_draws.get(ip, [])) >= ARGS.per_ip_hour:
                raise LiveRefused(f"That's {ARGS.per_ip_hour} live draws this hour, the limit per visitor, so this one uses recorded and simulated answers.")
        if _live_slots and not _live_slots.acquire(blocking=False):
            raise LiveRefused("Too many live draws are running right now, so this one uses recorded and simulated answers.")
        if ARGS.per_ip_hour:
            _ip_draws.setdefault(ip, []).append(now)
        return today


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
    """What a visitor sees when a live draw fails. Rate limits are said plainly; billing, key and other
    backend failures are not described."""
    msg = str(e)
    if isinstance(e, LiveRefused):
        return msg
    if "429" in msg:
        return "Jev is rate-limiting the demo right now; the rest of this draw uses recorded and simulated answers."
    return "Something went wrong with the live calls; the rest of this draw uses recorded and simulated answers."


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


@cache
def test_set() -> tuple[np.ndarray, np.ndarray]:
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
            # the daily cap is enforced per draw in reserve_live; without --journal nothing is written to disk
            _jev = Jev(JOURNAL if ARGS.journal else None, max_usd=None, wait_rounds=1)
            for b in _jev.backends:
                # Training paces calls at 3,000/min; a draw is one burst of 64, and bursts of 64 at once drew no 429s
                # (2026-09-28). Pacing here only delays the last calls of the burst and trips the hedges.
                b.per_min = max(b.per_min, 30000)
                def post(state, questions, _orig=b.post, _name=b.name):
                    t0 = time.monotonic()
                    resp, attempts = _orig(state, questions)
                    _captured.last = (_name, resp, time.monotonic() - t0)
                    return resp, attempts
                b.post = post
        return _jev


# Jev's latency has a long tail: in 64-call bursts the median call took 0.3-0.5 s, but a few per draw hung for
# 15-25 s or came back 504, and a layer waits for its slowest call. So a call that isn't back by each HEDGE_S mark
# is sent again and the first answer wins; after GIVE_UP_S that one neuron uses a recorded or simulated answer.
HEDGE_S, GIVE_UP_S = (1.5, 4.0), 20.0
# ponytail: a losing attempt keeps its thread until the gateway answers or times out; fine at --max-live 16.
_attempts = ThreadPoolExecutor(256)


class SlowCall(Exception):
    """A live call that didn't answer within GIVE_UP_S, hedges included."""


def live_source(state, questions, z, tag, rng):
    def attempt():
        answers = live_jev().ask(state, questions, tag)
        return answers, _captured.last  # read on the thread that made the call

    t0, futs = time.monotonic(), [_attempts.submit(attempt)]
    while True:
        limit = HEDGE_S[len(futs) - 1] if len(futs) <= len(HEDGE_S) else GIVE_UP_S
        wait([f for f in futs if not f.done()] or futs, timeout=max(0.0, limit - (time.monotonic() - t0)),
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
        if time.monotonic() - t0 >= limit:
            if len(futs) > len(HEDGE_S):
                raise SlowCall(f"no answer in {GIVE_UP_S:.0f} s")
            futs.append(_attempts.submit(attempt))


# ---------- inference ----------

def run(name: str, source: str, pixels, seed: int = 0, refused: str | None = None):
    """Yield one event per neuron in the order answers arrive, layer by layer, then the result. A live draw that
    was refused (`refused` is the reason) or whose calls fail runs the rest on replay, which falls back to the mock."""
    m = model(name)
    if source == "live" and not ARGS.live:
        raise ValueError("live source is off; start the server with --live")
    def fallback(*a):  # the replay index is read on first use, so exact-only servers never load it
        return replay_source(replay_index(name))(*a)

    failed = {"why": refused, "told": False}

    def live_or_fallback(*a):
        if failed["why"] is None:
            try:
                return live_source(*a)
            except SlowCall:  # only this neuron falls back; the rest of the draw stays live
                return fallback(*a)
            except Exception as e:  # rate limit, billing or backend failure: finish the draw without live calls
                if failed["why"] is None:
                    print(f"live call failed, finishing on fallbacks: {e!r}", flush=True)
                failed["why"] = failed["why"] or public_error(e)
        return fallback(*a)

    ask = {"exact": exact_source, "mock": mock_source, "live": live_or_fallback}.get(source) or fallback
    run_id = uuid.uuid4().hex[:8]
    a = np.clip(np.asarray(pixels, float), 0, 1)
    t0 = time.monotonic()
    calls = tokens = 0
    yield {"type": "start", "run": run_id, "model": name, "source": source, "labels": m["labels"],
           "layers": [W.shape[1] for W, _ in params(m["weights"])]}
    if refused:
        failed["told"] = True
        yield {"type": "notice", "message": refused}
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
        with ThreadPoolExecutor(1 if source == "exact" else W.shape[1]) as ex:
            for f in as_completed([ex.submit(one, j) for j in range(W.shape[1])]):
                ev = f.result()
                if failed["why"] and not failed["told"]:
                    failed["told"] = True
                    yield {"type": "notice", "message": failed["why"]}
                acts[ev["j"]] = ev["p"]
                calls += 1
                tokens += ev["tokens"] if ev.get("billed") or source != "live" else 0
                yield {**ev, "t_ms": round((time.monotonic() - t0) * 1000)}
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
                return self.send_json({"models": registry(), "live": ARGS.live, "daily_usd": ARGS.daily_usd})
            if url.path == "/api/weights":
                (W1, b1), (W2, b2) = params(model(q.get("name", ""))["weights"])
                return self.send_json({"W1": np.round(W1.T, 3).tolist(), "b1": b1.round(3).tolist(),
                                       "W2": np.round(W2, 3).tolist(), "b2": b2.round(3).tolist()})
            if url.path == "/api/disagree":  # recorded test digits this model and its twin read differently
                name = q.get("model", "")
                return self.send_json({"model": name, "twin": model(name)["twin"], "digits": disagreements(name)})
            if url.path == "/api/digit":
                x, y = test_set()
                labels = [int(v) for v in q.get("labels", "").split(",") if v != ""] or list(range(10))
                pool = np.where(np.isin(y, labels))[0]
                if "recorded" in q:  # digits whose every call was recorded for this model
                    m = model(q["recorded"])
                    rec = [i for i in recorded_digits(m["replay"]) if y[i] in labels] if "replay" in m else []
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

    def do_POST(self):
        if urlparse(self.path).path != "/api/run":
            return self.send_json({"error": "not found"}, 404)
        try:
            req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
            # a test digit is sent by index so the states match the training journal to the last digit
            pixels = test_set()[0][int(req["mnist"])] if "mnist" in req else [float(v) for v in req["pixels"]]
            if len(pixels) != 784 or req.get("source") not in SOURCES:
                raise ValueError("need 784 pixels and a known source")
            live, refused = req["source"] == "live", None
            if live and not ARGS.live:
                raise ValueError("live source is off; start the server with --live")
            if live:
                try:
                    reserve_live(self.client_ip())
                except LiveRefused as e:
                    live, refused = False, str(e)  # nothing reserved, nothing billed: the draw runs on fallbacks
        except (ValueError, KeyError, TypeError, json.JSONDecodeError) as e:
            return self.send_json({"error": str(e)}, 400)
        usd, done = 0.0, False
        try:
            events = run(str(req["model"]), req["source"], pixels, int(req.get("seed", 0)), refused)
            try:
                first = next(events)  # validation errors surface here, before the stream starts
            except ValueError as e:
                return self.send_json({"error": str(e)}, 400)
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            try:
                self.wfile.write(f"data: {json.dumps(first)}\n\n".encode())
                try:
                    for ev in events:
                        if ev["type"] == "neuron" and ev.get("billed"):
                            usd += ev["tokens"] * USD_PER_INPUT_TOKEN
                        done = ev["type"] == "result"
                        self.wfile.write(f"data: {json.dumps(ev)}\n\n".encode())
                        self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError):
                    raise
                except Exception as e:  # anything the per-call fallback didn't catch
                    print(f"run failed: {e!r}", flush=True)
                    msg = public_error(e) if live else str(e)
                    self.wfile.write(f"data: {json.dumps({'type': 'error', 'message': msg})}\n\n".encode())
            except (BrokenPipeError, ConnectionResetError):
                pass  # browser went away; in-flight neurons finish, nothing more is sent
        finally:
            if live:
                # a failed or abandoned draw can still bill calls that were in flight: charge it a full draw
                charge_live(usd if done else max(usd, DRAW_USD))
                if _live_slots:
                    _live_slots.release()


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
        te = list(test_subset(y, n, seed=seed))
        recorded = json.loads((ROOT / result).read_text())["test"]["outputs"]
        digits = recorded_digits(model(name)["replay"])
        for i in digits:
            evs = list(run(name, "replay", x[i]))
            assert all(e["backend"].startswith("recorded") for e in evs if e["type"] == "neuron"), (name, i)
            assert np.allclose(np.round(evs[-1]["outputs"], 2), recorded[te.index(i)], atol=0.006), (name, i)
        print(f"{name:>14}: replay of {len(digits)} recorded test digits matches its training journal call for call")
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


def check_live_limits(pixels):
    """Live path against a local stub backend (nothing billed): a 429 is told to the visitor as a rate limit, a 402
    as a generic error, and the draw finishes on recorded or simulated answers; the per-visitor limit and the daily
    cap refuse live before any call and the draw runs on those fallbacks, unbilled."""
    global JOURNAL, SPEND
    import tempfile
    import urllib.request
    tmp = Path(tempfile.mkdtemp())
    JOURNAL, SPEND = tmp / "journal.jsonl", tmp / "spend.json"
    code = {"v": 429}

    class Stub(SimpleHTTPRequestHandler):
        def do_POST(self):
            self.rfile.read(int(self.headers["Content-Length"]))
            self.send_response(code["v"])
            self.end_headers()
            self.wfile.write(b'{"detail": "stub"}')

        def log_message(self, *a):
            pass

    servers = [ThreadingHTTPServer(("127.0.0.1", 0), h) for h in (Stub, Handler)]
    for srv in servers:
        threading.Thread(target=srv.serve_forever, daemon=True).start()
    for b in live_jev().backends:
        b.endpoint = f"http://127.0.0.1:{servers[0].server_port}/"
    ARGS.live, ARGS.daily_usd, ARGS.per_ip_hour = True, 1.0, 3

    def draw():
        body = json.dumps({"model": "s8a-jev", "source": "live", "pixels": list(map(float, pixels))}).encode()
        req = urllib.request.Request(f"http://127.0.0.1:{servers[1].server_port}/api/run", body)
        evs = [json.loads(line[6:]) for line in urllib.request.urlopen(req).read().decode().splitlines() if line.startswith("data: ")]
        neurons = [e for e in evs if e["type"] == "neuron"]
        assert len(neurons) == 74 and evs[-1]["type"] == "result" and not evs[-1]["billed"]
        assert all(e["backend"] == "simulated" or e["backend"].startswith("recorded") for e in neurons)
        return next(e["message"] for e in evs if e["type"] == "notice")

    msg = draw()
    assert "rate-limiting" in msg, msg
    code["v"] = 402
    msg = draw()
    assert msg.startswith("Something went wrong") and "402" not in msg and "credit" not in msg, msg
    _ip_draws["203.0.113.9"] = [time.time() - 7200]  # a visitor from two hours ago is forgotten at the next draw
    draw()  # third draw this hour from this visitor: still allowed live
    assert "203.0.113.9" not in _ip_draws and list(_ip_draws) == ["127.0.0.1"], _ip_draws
    msg = draw()
    assert "per visitor" in msg, msg
    assert not JOURNAL.exists() and not list(tmp.glob("*failures*")), "live calls were journaled without --journal"
    SPEND.write_text(json.dumps({f"2026-01-{d:02d}": 1.0 for d in range(1, 11)}))
    charge_live(0.0)
    assert len(json.loads(SPEND.read_text())) == 7, SPEND.read_text()
    ARGS.per_ip_hour = 100
    SPEND.write_text(json.dumps({time.strftime("%Y-%m-%d", time.gmtime()): 1.0}))
    msg = draw()
    assert "budget" in msg and _spent_today()[1] == 1.0, msg
    for srv in servers:
        srv.shutdown()
    print("          live: rate limit, payment failure, per-visitor limit and daily cap fall back to recorded/simulated answers, unbilled;\n"
          "                nothing journaled without --journal, stale visitors pruned, spend.json keeps 7 days")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--live", action="store_true", help="allow the live source (real Jev calls, billed)")
    ap.add_argument("--daily-usd", type=float, default=9.0, help="live spend cap per UTC day, about 5,000 draws at $9 (demo/runs/spend.json)")
    ap.add_argument("--per-ip-hour", type=int, default=0, help="live draws per visitor per hour (0 = no limit)")
    ap.add_argument("--max-live", type=int, default=0, help="live draws running at once (0 = no limit)")
    ap.add_argument("--journal", action="store_true", help="keep live calls in demo/runs/journal.jsonl (off: nothing written)")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--check", action="store_true")
    ARGS = ap.parse_args()
    if ARGS.check:
        check()
    else:
        _live_slots = threading.BoundedSemaphore(ARGS.max_live) if ARGS.max_live else None
        print(f"http://{ARGS.host}:{ARGS.port}  live={'on' if ARGS.live else 'off'}  daily cap ${ARGS.daily_usd}", flush=True)
        ThreadingHTTPServer.request_queue_size = 128  # the default backlog of 5 drops connections in a burst of draws
        ThreadingHTTPServer((ARGS.host, ARGS.port), Handler).serve_forever()
