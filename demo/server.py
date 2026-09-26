"""Jevrons demo server: runs a drawn digit through a trained net and streams every neuron as it resolves.

  uv run python demo/server.py              # exact, mock and replay sources; http://127.0.0.1:8765
  uv run python demo/server.py --live       # also allow live Jev calls (journaled to demo/runs/journal.jsonl)
  uv run python demo/server.py --check      # self-check: streamed exact inference == jevrons.net.forward

Models come from demo/models.json; an entry whose weights file is missing is listed as unavailable.
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
from jevrons.states import SCALAR_Q_ALT, neuron_question, neuron_state, r2

DEMO = Path(__file__).resolve().parent
ROOT = DEMO.parent
JOURNAL = DEMO / "runs" / "journal.jsonl"
SOURCES = ("exact", "mock", "replay", "live")
ARGS = argparse.Namespace(live=False, max_usd=0.05)


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


def neuron_request(fmt: str, x: np.ndarray, w: np.ndarray, b: float):
    """(state, questions, answer key, true z), exactly as jevrons.net sends it."""
    if fmt == "scalar":  # Variant A: code adds up z, Jev only judges the sign
        z = r2(x @ w + b)
        return {"z": z}, SCALAR_Q_ALT, "pos", z
    keep = np.round(x, 2) != 0  # zero inputs are dropped, as in JevNeuron
    state, z = neuron_state(fmt, x[keep], w[keep], b)
    return state, neuron_question(fmt), "fires", z


def terms(state: dict) -> list[float]:
    return state.get("products") or [state["z"]]


# ---------- neuron sources: (state, questions, key, z, tag, rng) -> {p, backend, latency_ms, tokens} ----------

def exact_source(state, questions, key, z, tag, rng):
    return {"p": 1.0 if z > 0 else 0.0, "backend": "local step", "latency_ms": 0, "tokens": 0}


def mock_source(state, questions, key, z, tag, rng):
    """MockJevNeuron per call (stage 1 fit: z + 0.77 * spread of terms * N(0,1) > 0), with fake latency
    and token count fitted to the stage 5 journal (median 0.25 s, ~310 + 5.3 tokens per term)."""
    t = terms(state)
    spread = 0.0 if "z" in state else math.sqrt(sum(v * v for v in t))  # scalar Jev is a near-perfect step
    latency = min(2.0, rng.lognormal(math.log(0.25), 0.35))
    time.sleep(latency)
    fires = z + 0.77 * spread * rng.standard_normal() > 0
    return {"p": 0.97 if fires else 0.03, "backend": "mock", "latency_ms": round(latency * 1000),
            "tokens": round(310 + 5.3 * len(t))}


def _key(state, questions) -> str:
    return hashlib.sha1(json.dumps([state, questions], sort_keys=True).encode()).hexdigest()


def replay_index() -> dict[str, dict]:
    if not JOURNAL.exists():
        return {}
    records = (json.loads(line) for line in JOURNAL.read_text().splitlines() if line.strip())
    return {_key(r["state"], r["questions"]): r for r in records}


def replay_source(index):
    def source(state, questions, key, z, tag, rng):
        r = index.get(_key(state, questions))
        if r is None:  # never asked live: fall back to the mock, and say so
            return {**mock_source(state, questions, key, z, tag, rng), "backend": "mock (not recorded)"}
        time.sleep(min(r["latency_s"], 3))  # keep the recorded arrival order; cap the retry outliers
        return {"p": r["answers"][key], "backend": f"replay {r['served']}", "latency_ms": round(r["latency_s"] * 1000),
                "tokens": r["usage"].get("input_tokens", 0)}
    return source


_captured = threading.local()
_jev = None
_jev_lock = threading.Lock()


def live_jev() -> Jev:
    """One Jev client per process. Each backend's post is wrapped so the calling thread can read back the
    usage and latency that Jev journals (Jev.ask returns only the answers)."""
    global _jev
    with _jev_lock:
        if _jev is None:
            _jev = Jev(JOURNAL, max_usd=ARGS.max_usd)
            for b in _jev.backends:
                def post(state, questions, _orig=b.post, _name=b.name):
                    t0 = time.monotonic()
                    resp, attempts = _orig(state, questions)
                    _captured.last = (_name, resp, time.monotonic() - t0)
                    return resp, attempts
                b.post = post
        return _jev


def live_source(state, questions, key, z, tag, rng):
    jev = live_jev()
    p = jev.ask(state, questions, tag)[key]
    name, resp, latency = _captured.last
    served = resp.get("model", "?")
    return {"p": p, "backend": f"{name} {served}", "latency_ms": round(latency * 1000),
            "tokens": (resp.get("usage") or {}).get("input_tokens", 0)}


# ---------- inference ----------

def run(name: str, source: str, pixels, seed: int = 0):
    """Yield one event per neuron in the order answers arrive, layer by layer, then the result."""
    m = model(name)
    if source == "live" and not ARGS.live:
        raise ValueError("live source is off; start the server with --live")
    ask = {"exact": exact_source, "mock": mock_source, "live": live_source}.get(source) or replay_source(replay_index())
    run_id = uuid.uuid4().hex[:8]
    a = np.clip(np.asarray(pixels, float), 0, 1)
    t0 = time.monotonic()
    calls = tokens = 0
    yield {"type": "start", "run": run_id, "model": name, "source": source, "labels": m["labels"],
           "layers": [W.shape[1] for W, _ in params(m["weights"])]}
    for layer, (W, b) in enumerate(params(m["weights"])):
        def one(j, x=a, W=W, b=b, layer=layer):
            state, questions, key, z = neuron_request(m["format"], x, W[:, j], b[j])
            tag = {"demo": run_id, "model": name, "layer": layer, "j": j, "z": z}
            ans = ask(state, questions, key, z, tag, np.random.default_rng([seed, layer, j]))
            return {"type": "neuron", "layer": layer, "j": j, "n_terms": len(terms(state)), "z": round(z, 4),
                    "terms": terms(state), "disagree": (ans["p"] >= 0.5) != (z > 0), **ans}

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
                return self.send_json({"models": registry(), "live": ARGS.live, "max_usd": ARGS.max_usd})
            if url.path == "/api/weights":
                (W1, b1), (W2, b2) = params(model(q.get("name", ""))["weights"])
                return self.send_json({"W1": np.round(W1.T, 3).tolist(), "b1": b1.round(3).tolist(),
                                       "W2": np.round(W2, 3).tolist(), "b2": b2.round(3).tolist()})
            if url.path == "/api/digit":
                x, y = test_set()
                labels = [int(v) for v in q.get("labels", "").split(",") if v != ""] or list(range(10))
                pool = np.where(np.isin(y, labels))[0]
                i = int(q["index"]) if "index" in q else int(np.random.default_rng().choice(pool))
                return self.send_json({"index": i, "label": int(y[i]), "pixels": x[i].round(3).tolist()})
        except (ValueError, KeyError, IndexError) as e:
            return self.send_json({"error": str(e)}, 400)
        return super().do_GET()

    def do_POST(self):
        if urlparse(self.path).path != "/api/run":
            return self.send_json({"error": "not found"}, 404)
        try:
            req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
            pixels = [float(v) for v in req["pixels"]]
            if len(pixels) != 784 or req.get("source") not in SOURCES:
                raise ValueError("need 784 pixels and a known source")
            events = run(str(req["model"]), req["source"], pixels, int(req.get("seed", 0)))
            first = next(events)  # validation errors surface here, before the stream starts
        except (ValueError, KeyError, TypeError, json.JSONDecodeError) as e:
            return self.send_json({"error": str(e)}, 400)
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        try:
            self.wfile.write(f"data: {json.dumps(first)}\n\n".encode())
            try:
                for ev in events:
                    self.wfile.write(f"data: {json.dumps(ev)}\n\n".encode())
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                raise
            except Exception as e:  # e.g. spend cap or a backend outage mid-run
                self.wfile.write(f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n".encode())
        except (BrokenPipeError, ConnectionResetError):
            pass  # browser went away; in-flight neurons finish, nothing more is sent


def check():
    """Streamed exact inference must match jevrons.net.forward with StepNeuron on real test digits. Compared on
    output activations (predictions can differ on random tie breaks). The demo's true z uses the two-decimal
    values actually sent, so a sum within rounding of zero may flip: allow 2% of digits to differ."""
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
        assert same >= 0.98, m["name"]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--live", action="store_true", help="allow the live source (real Jev calls, billed)")
    ap.add_argument("--max-usd", type=float, default=0.05, help="spend cap for live calls in this process")
    ap.add_argument("--check", action="store_true")
    ARGS = ap.parse_args()
    if ARGS.check:
        check()
    else:
        print(f"http://127.0.0.1:{ARGS.port}  live={'on' if ARGS.live else 'off'}", flush=True)
        ThreadingHTTPServer(("127.0.0.1", ARGS.port), Handler).serve_forever()
