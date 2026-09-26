"""Stage 3: learn a circular boundary with a 2-6-1 net of Jev neurons.

Points are uniform in [-1, 1]^2, labeled inside if radius < 0.8 (about half the square).
Arms: straight-through through live Jev (3 seeds) and a swap control (trained with an exact step neuron, evaluated through Jev).
Usage: uv run python -m jevrons.stage3 [local]
"""

import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from jevrons.net import Adam, JevNeuron, StepNeuron, bce, forward, init, ste_grads

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "runs" / "stage3"
SIZES = [2, 6, 1]
STEPS, BATCH, LR, TAU = 150, 16, 0.1, 0.3  # tuned locally on soft and step proxies
SEEDS = (0, 1, 2)
GRID = np.round(np.linspace(-1, 1, 21), 2)


def data(n, seed):
    rng = np.random.default_rng(seed)
    X = np.round(rng.uniform(-1, 1, (n, 2)), 2)
    return X, (np.hypot(X[:, 0], X[:, 1]) < 0.8).astype(float)[:, None]


XTR, YTR = data(250, 100)
XTE, YTE = data(250, 200)


def start(seed):
    """Random hidden biases, so the initial boundary lines don't all pass through the origin."""
    (W1, _), out = init(SIZES, seed)
    return [(W1, np.random.default_rng(seed + 1000).uniform(-1, 1, SIZES[1])), out]


def train(neuron, seed, log):
    params = start(seed)
    opt, rng = Adam(params, LR), np.random.default_rng(seed)
    for s in range(STEPS):
        i = rng.choice(len(XTR), BATCH, replace=False)
        acts, zs = forward(params, XTR[i], neuron, {"step": s, "seed": seed, "split": "train"})
        log.append({"step": s, "loss": bce(acts[-1], YTR[i]),
                    "acc": float(np.mean((acts[-1] >= 0.5) == (YTR[i] >= 0.5)))})
        params = opt.step(params, ste_grads(params, acts, zs, YTR[i], TAU))
    return params


def train_spsa(neuron, seed, log, c=0.3, lr=0.3):
    params, rng = start(seed), np.random.default_rng(seed)
    for s in range(STEPS // 2):
        i = rng.choice(len(XTR), BATCH, replace=False)
        d = [(rng.choice([-1.0, 1.0], W.shape), rng.choice([-1.0, 1.0], b.shape)) for W, b in params]
        lp = bce(forward([(W + c * dW, b + c * db) for (W, b), (dW, db) in zip(params, d)], XTR[i], neuron)[0][-1], YTR[i])
        lm = bce(forward([(W - c * dW, b - c * db) for (W, b), (dW, db) in zip(params, d)], XTR[i], neuron)[0][-1], YTR[i])
        g = (lp - lm) / (2 * c)
        params = [(np.clip(W - lr * g / dW, -9.99, 9.99), np.clip(b - lr * g / db, -9.99, 9.99))
                  for (W, b), (dW, db) in zip(params, d)]
        log.append({"step": s, "loss": (lp + lm) / 2})
    return params


def accuracy(params, neuron, X, Y, repeats, tag=None):
    ps = np.array([forward(params, X, neuron, {**(tag or {}), "eval": r})[0][-1].ravel() for r in range(repeats)])
    return float(np.mean((ps >= 0.5) == (Y.ravel() >= 0.5))), ps.mean(0)


def linear_baseline():
    """Best logistic regression on raw (x, y): what 'beats a linear classifier' means here."""
    W, b = np.zeros(2), 0.0
    for _ in range(2000):
        p = 1 / (1 + np.exp(-(XTR @ W + b)))
        W -= 0.5 * XTR.T @ (p - YTR.ravel()) / len(XTR)
        b -= 0.5 * np.mean(p - YTR.ravel())
    return float(np.mean(((XTE @ W + b) > 0) == (YTE.ravel() > 0.5)))


def run(name, fit, neuron_train, neuron_eval, seed, grid):
    done = OUT / f"result-{name}-{seed}.json"
    if done.exists():
        return json.loads(done.read_text())
    log = []
    params = fit(neuron_train, seed, log)
    tag = {"arm": name, "seed": seed}
    test_acc, _ = accuracy(params, neuron_eval, XTE, YTE, 2, {**tag, "split": "test"})
    res = {"arm": name, "seed": seed, "test_accuracy": test_acc,
           "test_accuracy_exact_step": accuracy(params, StepNeuron(), XTE, YTE, 1)[0],
           "params": [[W.round(2).tolist(), b.round(2).tolist()] for W, b in params]}
    if grid:
        G = np.array([[x, y] for y in GRID for x in GRID])
        res["grid_p"] = accuracy(params, neuron_eval, G, np.zeros((len(G), 1)), 1, {**tag, "split": "grid"})[1].round(3).tolist()
    (OUT / f"curve-{name}-{seed}.json").write_text(json.dumps(log))
    done.write_text(json.dumps(res))
    print(f"{name} seed={seed} test acc {test_acc:.3f} (exact-step {res['test_accuracy_exact_step']:.3f})", flush=True)
    return res


def main(local):
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"linear classifier test acc {linear_baseline():.3f}; inside fraction {YTE.mean():.2f}")
    step = StepNeuron()
    if local:
        for seed in range(10):
            p = train(step, seed, [])
            print(f"local step seed={seed} test acc {accuracy(p, step, XTE, YTE, 1)[0]:.3f}")
        return
    from jevrons.jev import Jev
    jev = Jev(OUT / "journal.jsonl")
    jn = JevNeuron(jev, "folded", threads=8)
    jobs = [("ste_jev", train, jn, jn, s, True) for s in SEEDS]
    jobs += [("swap", train, step, jn, s, s == 0) for s in SEEDS]
    with ThreadPoolExecutor(len(jobs)) as ex:  # seeds share the client's rate limiter
        results = list(ex.map(lambda j: run(*j), jobs))
    summary = {"linear_test_accuracy": linear_baseline(), "results": [
        {k: v for k, v in r.items() if k not in ("grid_p", "params")} for r in results],
        "spend": {"calls": jev.calls, "tokens": jev.tokens, "usd": round(jev.usd, 4)}}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1))
    print(summary["spend"])


if __name__ == "__main__":
    main(len(sys.argv) > 1 and sys.argv[1] == "local")
