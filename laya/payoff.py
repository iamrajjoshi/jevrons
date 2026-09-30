"""Stage 10 payoff: stage 5's 3 vs 8 recipe with every neuron a fine-tuned laya call.

784-32-1, 500 train / 500 val (stage 5's split), straight-through with tau = 3, Adam 0.02, batch 32,
5 epochs, init N(0, 0.3) (stage5.init(0)). To fit laya's context, each hidden neuron keeps only its
top-N weights by magnitude, chosen at init among pixels that are nonzero in at least 5% of the training
images, and the rest are held at zero after every step. The same mask and start are used for both arms:
  laya  trained through the laya neuron, evaluated on it
  swap  trained with an exact step neuron, evaluated on the laya neuron
laya is deterministic, so one validation pass replaces stage 5's two.
Usage: uv run python payoff.py <run name> [--n 64]   -> runs/stage10/payoff-<run>/
"""

import argparse
import json
import time

import numpy as np
import torch

import data
import laya_model as L
from jevrons.digits import mnist, split
from jevrons.net import Adam, StepNeuron, bce, forward, save_params, ste_grads
from jevrons.stage5 import BATCH, EPOCHS, LR, SEED, TAU, init
from jevrons.states import neuron_state

PAIR = (3, 8)


class LayaNeuron:
    """Every (example, neuron) pair is one laya call on the folded state of its nonzero terms."""

    def __init__(self, weights):
        self.tok, self.model = L.tokenizer(), L.load(weights)
        self.calls, self.seconds, self.max_terms = 0, 0.0, 0

    def __call__(self, X, W, b, tag=None):
        states = []
        live_w = np.round(W, 2) != 0  # masked-out weights are exactly zero, so they never reach the state
        for i in range(X.shape[0]):
            keep = np.round(X[i], 2) != 0
            for j in range(W.shape[1]):
                kj = keep & live_w[:, j]
                states.append(neuron_state("folded", X[i][kj], W[kj, j], b[j])[0])
                self.max_terms = max(self.max_terms, int(kj.sum()))
        t0 = time.perf_counter()
        with torch.no_grad():
            p = L.p_yes(self.model, self.tok, states, data.QUESTION, batch=64)
        self.seconds += time.perf_counter() - t0
        self.calls += len(states)
        assert not np.isnan(p).any(), "a state didn't fit the context"
        return p.reshape(X.shape[0], W.shape[1])


def top_n_mask(W, active, n):
    score = np.abs(W) * active[:, None]
    mask = np.zeros_like(W)
    idx = np.argsort(-score, 0)[:n]
    np.put_along_axis(mask, idx, 1.0, 0)
    return mask


def train(params, mask, X, Y, neuron, log, epochs=EPOCHS):
    """stage 5's fit (net.fit) with the top-N mask re-applied after every Adam step."""
    params = [(params[0][0] * mask, params[0][1]), params[1]]
    opt, rng, step = Adam(params, LR), np.random.default_rng(SEED), 0
    for epoch in range(epochs):
        order = rng.permutation(len(X))
        for i in np.array_split(order, max(1, len(X) // BATCH)):
            acts, zs = forward(params, X[i], neuron)
            log.append({"epoch": epoch, "step": step, "loss": bce(acts[-1], Y[i]),
                        "acc": float(np.mean((acts[-1][:, 0] >= 0.5) == (Y[i, 0] >= 0.5)))})
            params = opt.step(params, ste_grads(params, acts, zs, Y[i], TAU, "logit"))
            params = [(params[0][0] * mask, params[0][1]), params[1]]
            step += 1
        print(f"  epoch {epoch}: mean train acc {np.mean([r['acc'] for r in log if r['epoch'] == epoch]):.3f}", flush=True)
    return params


def evaluate(params, X, y, neuron):
    acts, zs = forward(params, X, neuron)
    fires = [float(np.mean((a >= 0.5) == (z > 0))) for a, z in zip(acts[1:], zs)]
    return {"accuracy": float(np.mean((acts[-1][:, 0] >= 0.5) == (y >= 0.5))),
            "fires_as_intended": {"hidden": fires[0], "output": fires[1]}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--n", type=int, default=64)
    args = ap.parse_args()
    out = data.ROOT / f"runs/stage10/payoff-{args.run}-n{args.n}"
    out.mkdir(parents=True, exist_ok=True)
    xtr, ytr, _, _ = mnist()
    tr, va = split(ytr, PAIR, 500, 500)
    X, Xv = xtr[tr], xtr[va]
    Y, yv = (ytr[tr] == PAIR[1]).astype(float)[:, None], (ytr[va] == PAIR[1]).astype(float)
    active = (np.mean(X > 0, 0) >= 0.05).astype(float)
    start = init(SEED)
    mask = top_n_mask(start[0][0], active, args.n)
    neuron = LayaNeuron(data.ROOT / f"runs/stage10/{args.run}/model.safetensors")
    results = {"n": args.n, "active_pixels": int(active.sum())}
    for arm in ("swap", "laya"):
        t0, log = time.time(), []
        print(arm, flush=True)
        params = train(start, mask, X, Y, neuron if arm == "laya" else StepNeuron(), log)
        res = {"val_laya": evaluate(params, Xv, yv, neuron),
               "val_exact_step": float(np.mean((forward(params, Xv, StepNeuron())[0][-1][:, 0] >= 0.5) == (yv >= 0.5))),
               "minutes": round((time.time() - t0) / 60, 1)}
        results[arm] = res
        save_params(out / f"weights-{arm}.npz", params)
        (out / f"curve-{arm}.json").write_text(json.dumps(log))
        print(arm, res, flush=True)
    results["laya_calls"], results["laya_seconds"] = neuron.calls, round(neuron.seconds)
    results["max_terms_sent"] = neuron.max_terms
    (out / "result.json").write_text(json.dumps(results, indent=1))
    print(results)


if __name__ == "__main__":
    main()
