"""Jevrons demo server: runs a drawn digit through a trained net and streams every neuron as it resolves.

  uv run python demo/server.py              # exact, mock and replay sources; http://127.0.0.1:8765
  uv run python demo/server.py --live       # also allow live Jev calls (journaled to demo/runs/journal.jsonl)
  uv run python demo/server.py --check      # self-check: streamed exact inference == jevrons.net.forward

Live calls go to whatever backends jevrons.jev picks; set JEVRONS_BACKENDS=vercel to pin one.
Models come from demo/models.json; an entry whose weights file is missing is listed as unavailable.
Stage 7's replay and mock data are cut from its training journal by demo/make_replay.py.
"""

import argparse
import hashlib
import json
import math
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
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
ARGS = argparse.Namespace(live=False, daily_usd=2.0, per_ip_hour=20, max_live=4)
DRAW_USD = 0.0025  # a stage 7 draw bills about $0.0017; budget a little more so the daily cap isn't overshot


# ---------- live budget and limits ----------

class LiveRefused(Exception):
    """A live draw the server won't start. The message is safe to show the visitor."""


SPEND = DEMO / "runs" / "spend.json"
_budget_lock = threading.Lock()
_live_slots = threading.BoundedSemaphore(ARGS.max_live)
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
            raise LiveRefused("Today's live budget is used up. Replay shows real recorded Jev answers.")
        now = time.time()
        recent = [t for t in _ip_draws.get(ip, []) if now - t < 3600]
        if len(recent) >= ARGS.per_ip_hour:
            raise LiveRefused(f"That's {ARGS.per_ip_hour} live draws this hour, the limit per visitor. Replay still works.")
        if not _live_slots.acquire(blocking=False):
            raise LiveRefused("Too many live draws are running right now. Try again in a few seconds.")
        _ip_draws[ip] = recent + [now]
        return today


def charge_live(usd: float):
    with _budget_lock:
        today, spent = _spent_today()
        try:
            d = json.loads(SPEND.read_text())
        except (OSError, ValueError):
            d = {}
        d[today] = round(spent + usd, 6)
        SPEND.write_text(json.dumps(d))


def public_error(e: Exception) -> str:
    """What a visitor sees when a live draw fails. Rate limits are said plainly; billing, key and other
    backend failures are not described."""
    msg = str(e)
    if isinstance(e, LiveRefused):
        return msg
    if "429" in msg:
        return "Jev is rate-limiting the demo right now. Try again in a minute, or use replay."
    return "Something went wrong with the live call. Replay still works."


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
def e_table() -> dict:
    return json.loads((DEMO / "runs/stage7-mock.json").read_text())


def mock_source(state, questions, z, tag, rng):
    """Arm E neurons: resample a recorded stage 7 test call with the same layer and a similar z / spread,
    so the mock inherits Jev's real leans and grading. Token count from a fit to that journal
    (381 + 5.37 per number). Other neurons: MockJevNeuron (stage 1 fit: z + 0.77 * spread * N(0,1) > 0),
    latency and tokens fitted to the stage 5 journal (median 0.25 s, ~310 + 5.3 tokens per term)."""
    t = terms(state)
    spread = 0.0 if "z" in state else math.sqrt(sum(v * v for v in t))  # scalar Jev is a near-perfect step
    if "choice" in questions:
        tab = e_table()
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
    return {"answers": answers, "backend": "mock", "latency_ms": round(latency * 1000), "tokens": round(tokens)}


def _key(state, questions) -> str:
    return hashlib.sha1(json.dumps([state, questions], sort_keys=True).encode()).hexdigest()


def records(path: Path):
    return (json.loads(line) for line in path.read_text().splitlines() if line.strip()) if path.exists() else ()


def replay_index(m: dict) -> dict[str, dict]:
    """Demo live runs, plus the model's cut of its training journal. Later records win, so a state asked
    twice replays the answer the rest of that run was built on."""
    index = {_key(r["state"], r["questions"]): r for r in records(JOURNAL)}
    if "replay" in m:
        index.update((r["key"], r) for r in records(ROOT / m["replay"]))
    return index


def replay_source(index):
    def source(state, questions, z, tag, rng):
        r = index.get(_key(state, questions))
        if r is None:  # never asked live: fall back to the mock, and say so
            return {**mock_source(state, questions, z, tag, rng), "backend": "mock (not recorded)"}
        wait = r["latency_s"]  # keep the recorded arrival order, but squeeze rate-limit retries (15 s -> 2.9 s)
        time.sleep(wait if wait < 1.5 else 1.5 + 0.1 * (wait - 1.5))
        return {"answers": r["answers"], "backend": f"replay {r['served']}",
                "latency_ms": round(r["latency_s"] * 1000), "tokens": r["usage"].get("input_tokens", 0)}
    return source


@cache
def recorded_digits(path: str) -> list[int]:
    return sorted({r["mnist"] for r in records(ROOT / path)})


_captured = threading.local()
_jev = None
_jev_lock = threading.Lock()


def live_jev() -> Jev:
    """One Jev client per process. Each backend's post is wrapped so the calling thread can read back the
    usage and latency that Jev journals (Jev.ask returns only the answers)."""
    global _jev
    with _jev_lock:
        if _jev is None:
            _jev = Jev(JOURNAL, max_usd=None, wait_rounds=1)  # the daily cap is enforced per draw in reserve_live
            for b in _jev.backends:
                def post(state, questions, _orig=b.post, _name=b.name):
                    t0 = time.monotonic()
                    resp, attempts = _orig(state, questions)
                    _captured.last = (_name, resp, time.monotonic() - t0)
                    return resp, attempts
                b.post = post
        return _jev


def live_source(state, questions, z, tag, rng):
    answers = live_jev().ask(state, questions, tag)
    name, resp, latency = _captured.last
    served = resp.get("model", "?")
    return {"answers": answers, "backend": f"{name} {served}", "latency_ms": round(latency * 1000),
            "tokens": (resp.get("usage") or {}).get("input_tokens", 0)}


# ---------- inference ----------

def run(name: str, source: str, pixels, seed: int = 0):
    """Yield one event per neuron in the order answers arrive, layer by layer, then the result."""
    m = model(name)
    if source == "live" and not ARGS.live:
        raise ValueError("live source is off; start the server with --live")
    ask = {"exact": exact_source, "mock": mock_source, "live": live_source}.get(source) or replay_source(replay_index(m))
    run_id = uuid.uuid4().hex[:8]
    a = np.clip(np.asarray(pixels, float), 0, 1)
    t0 = time.monotonic()
    calls = tokens = 0
    yield {"type": "start", "run": run_id, "model": name, "source": source, "labels": m["labels"],
           "layers": [W.shape[1] for W, _ in params(m["weights"])]}
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
        with ThreadPoolExecutor(64) as ex:
            for f in as_completed([ex.submit(one, j) for j in range(W.shape[1])]):
                ev = f.result()
                acts[ev["j"]] = ev["p"]
                calls += 1
                tokens += ev["tokens"]
                yield {**ev, "t_ms": round((time.monotonic() - t0) * 1000)}
        a = acts
    yield {"type": "result", "outputs": a.tolist(), "prediction": m["labels"][int(decide(a[None])[0])],
           "no_fire": bool((a < 0.5).all()) if len(a) > 1 else False, "calls": calls, "tokens": tokens,
           "usd": tokens * USD_PER_INPUT_TOKEN, "billed": source == "live",
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
            live = req["source"] == "live"
            if live and not ARGS.live:
                raise ValueError("live source is off; start the server with --live")
            if live:
                reserve_live(self.client_ip())
        except LiveRefused as e:
            return self.send_json({"error": str(e)}, 429)
        except (ValueError, KeyError, TypeError, json.JSONDecodeError) as e:
            return self.send_json({"error": str(e)}, 400)
        usd, done = 0.0, False
        try:
            events = run(str(req["model"]), req["source"], pixels, int(req.get("seed", 0)))
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
                        if ev["type"] == "neuron":
                            usd += ev["tokens"] * USD_PER_INPUT_TOKEN
                        done = ev["type"] == "result"
                        self.wfile.write(f"data: {json.dumps(ev)}\n\n".encode())
                        self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError):
                    raise
                except Exception as e:  # rate limit, billing or backend failure mid-run
                    print(f"run failed: {e!r}", flush=True)
                    msg = public_error(e) if live else str(e)
                    self.wfile.write(f"data: {json.dumps({'type': 'error', 'message': msg})}\n\n".encode())
            except (BrokenPipeError, ConnectionResetError):
                pass  # browser went away; in-flight neurons finish, nothing more is sent
        finally:
            if live:
                # a failed or abandoned draw can still bill calls that were in flight: charge it a full draw
                charge_live(usd if done else max(usd, DRAW_USD))
                _live_slots.release()


def check():
    """Streamed exact inference must match jevrons.net.forward with StepNeuron on real test digits. Compared on
    output activations (predictions can differ on random tie breaks). The demo's true z uses the two-decimal
    values actually sent, so a sum within rounding of zero may flip: allow 4% of digits to differ (exact-trained
    swap networks keep output sums close to zero; stage 6's seed 0 control differs on 3%)."""
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
    # Replay of a recorded stage 7 test digit must hit the journal on every call and land on its recorded outputs.
    time.sleep = lambda s: None
    from jevrons.digits import test_subset
    te = list(test_subset(y, 2000, seed=7))
    recorded = json.loads((ROOT / "runs/stage7/result-jev.json").read_text())["test"]["outputs"]
    for i in recorded_digits(model("s7-jev")["replay"])[:12]:
        evs = list(run("s7-jev", "replay", x[i]))
        assert all(e["backend"].startswith("replay") for e in evs if e["type"] == "neuron"), i
        assert np.allclose(np.round(evs[-1]["outputs"], 2), recorded[te.index(i)], atol=0.006), i
    print("        s7-jev: replay of 12 recorded test digits matches the stage 7 journal call for call")
    check_live_limits(x[0])


def check_live_limits(pixels):
    """Live path against a local stub backend (nothing billed): a 429 tells the visitor it's a rate limit, a 402
    shows only a generic error, and the daily cap and per-visitor limit refuse the draw before any call."""
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
        body = json.dumps({"model": "s7-jev", "source": "live", "pixels": list(map(float, pixels))}).encode()
        req = urllib.request.Request(f"http://127.0.0.1:{servers[1].server_port}/api/run", body)
        try:
            text = urllib.request.urlopen(req).read().decode()
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read())["error"]
        err = [json.loads(line[6:]) for line in text.splitlines() if line.startswith("data: ")][-1]
        return 200, err.get("message", "")

    status, msg = draw()
    assert status == 200 and "rate-limiting" in msg, msg
    code["v"] = 402
    status, msg = draw()
    assert status == 200 and msg.startswith("Something went wrong") and "402" not in msg and "credit" not in msg, msg
    status, msg = draw()  # third draw this hour from this visitor: still allowed
    status, msg = draw()
    assert status == 429 and "per visitor" in msg, msg
    ARGS.per_ip_hour = 100
    SPEND.write_text(json.dumps({time.strftime("%Y-%m-%d", time.gmtime()): 1.0}))
    status, msg = draw()
    assert status == 429 and "budget" in msg, msg
    for srv in servers:
        srv.shutdown()
    print("          live: rate limit, payment failure, per-visitor limit and daily cap all handled")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--live", action="store_true", help="allow the live source (real Jev calls, billed)")
    ap.add_argument("--daily-usd", type=float, default=2.0, help="live spend cap per UTC day (demo/runs/spend.json)")
    ap.add_argument("--per-ip-hour", type=int, default=20, help="live draws per visitor per hour")
    ap.add_argument("--max-live", type=int, default=4, help="live draws running at once")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--check", action="store_true")
    ARGS = ap.parse_args()
    if ARGS.check:
        check()
    else:
        _live_slots = threading.BoundedSemaphore(ARGS.max_live)
        print(f"http://{ARGS.host}:{ARGS.port}  live={'on' if ARGS.live else 'off'}  daily cap ${ARGS.daily_usd}", flush=True)
        ThreadingHTTPServer((ARGS.host, ARGS.port), Handler).serve_forever()
