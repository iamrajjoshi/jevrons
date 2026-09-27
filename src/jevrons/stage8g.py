"""Stage 8g: make a Jevron behave like a real neuron, then backprop through what it actually does.

Same task, split, seed and initialization as stage 5's 3 vs 8, so stage 5 is the baseline arm.
Uses Jev's measured response curve (runs/margin): P(fire) = Phi(k(n) * (z / spread - m0(n))),
where n is the number of terms the neuron sends.

  B  measured-curve backward: dp/dz = k/spread * phi(k (z/spread - m0)); neuron unchanged
  C  B's backward on a symmetric curve, plus pre-compensation: the bias sent to Jev is shifted
     by m0 * spread so Jev's drifted threshold lands on the true zero
  D  C plus phrase averaging: three wordings of the question on the same state in one call
  E  no pre-compensation; average the noul with a neutral choice question on the same state (their
     leans point opposite ways and largely cancel); backward on the symmetric curve

The swap control (stage 5's exact-trained 3v8 weights) is re-run on the C and D neurons: if the
neuron now behaves as intended, the swap gap should shrink.
Usage: uv run python -m jevrons.stage8g [local]
"""

import json
import sys
from concurrent.futures import ThreadPoolExecutor
from math import erf
from pathlib import Path

import numpy as np

from jevrons.digits import mnist, split
from jevrons.jev import noul
from jevrons.net import StepNeuron, fit, forward
from jevrons.stage5 import BATCH, EPOCHS, LR, SEED, init
from jevrons.states import FALSE, INSTRUCTIONS, TRUE, r2

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "runs" / "stage8g"
CURVE = json.loads((ROOT / "runs/margin/summary.json").read_text())["offset_fit"]
_N = np.log([int(n) for n in CURVE])
_K = np.array([v["k"] for v in CURVE.values()])
_M0 = np.array([v["threshold_m0"] for v in CURVE.values()])
Phi = np.vectorize(lambda t: 0.5 * (1 + erf(t / np.sqrt(2))))
phi = lambda t: np.exp(-0.5 * t * t) / np.sqrt(2 * np.pi)  # noqa: E731

QUESTIONS = {
    "v1": noul(INSTRUCTIONS["folded"], TRUE, FALSE),
    "v2": noul("Sum every number in the products list.", "The sum is positive.", "The sum is zero or negative."),
    "v3": noul("Compute the total of all values in products.", "The total is above zero.", "The total is zero or below."),
    # A neutral choice leans the other way from the noul (threshold above zero instead of below),
    # so averaging the two cancels much of the lean: runs/choice-jev.
    "choice": {"type": "choice", "instructions": "Add all the numbers in products together, then compare the total with zero.",
               "criteria": {"above": "The total is above zero.", "below": "The total is below zero or equal to zero."}},
}


def curve(n):
    """Measured sharpness k and threshold m0 for a neuron sending n numbers (log-interpolated)."""
    ln = np.log(np.clip(n, 10, 250))
    return np.interp(ln, _N, _K), np.interp(ln, _N, _M0)


def terms_and_spread(a_in, W, b):
    """Per (example, neuron): how many numbers the folded state holds, and the spread of those numbers."""
    active = (np.round(a_in, 2) != 0).astype(float)
    n = active.sum(1, keepdims=True) + 1  # nonzero inputs plus the folded bias
    spread = np.sqrt((a_in**2) @ (W**2) + b**2) + 1e-6
    return np.broadcast_to(n, spread.shape), spread


def slope_for(compensated):
    def slope(layer, z, a_in, W, b):
        n, spread = terms_and_spread(a_in, W, b)
        k, m0 = curve(n)
        shift = 0.0 if compensated else m0
        return k / spread * phi(k * (z / spread - shift))
    return slope


class CalibratedJevNeuron:
    """Folded-bias Jev neuron with optional pre-compensation and phrase averaging."""

    def __init__(self, jev, compensate, phrasings, threads=128):
        self.jev, self.compensate, self.threads = jev, compensate, threads
        self.questions = {k: QUESTIONS[k] for k in phrasings}

    def __call__(self, X, W, b, tag=None):
        jobs = [(i, j) for i in range(X.shape[0]) for j in range(W.shape[1])]

        def one(ij):
            i, j = ij
            keep = np.round(X[i], 2) != 0
            products = [r2(v) for v in X[i][keep] * W[keep, j]]
            bias = r2(b[j])
            z = sum(products) + bias
            shift = 0.0
            if self.compensate:
                spread = np.sqrt(sum(v * v for v in products) + bias * bias) + 1e-6
                shift = float(curve(len(products) + 1)[1]) * spread  # send z + m0 * spread
            state = {"products": products + [r2(bias + shift)]}
            ans = self.jev.ask(state, self.questions, {**(tag or {}), "i": i, "j": j, "z": z, "shift": shift})
            return float(np.mean([a["above"] if isinstance(a, dict) else a for a in ans.values()]))

        with ThreadPoolExecutor(self.threads) as ex:
            return np.array(list(ex.map(one, jobs))).reshape(X.shape[0], W.shape[1])


class ShiftedMock:
    """Local stand-in with the measured quirks, for checking the code before spending."""

    def __init__(self, compensate, seed=0):
        self.compensate, self.rng = compensate, np.random.default_rng(seed)

    def __call__(self, X, W, b, tag=None):
        z = X @ W + b
        n, spread = terms_and_spread(X, W, b)
        k, m0 = curve(n)
        z_sent = z + (m0 * spread if self.compensate else 0.0)
        p = Phi(k * (z_sent / spread - m0))
        return (self.rng.random(p.shape) < p) * 0.94 + 0.03


def evaluate(params, X, y, neuron, tag, passes=2):
    outs, agree = [], []
    for r in range(passes):
        acts, zs = forward(params, X, neuron, {**tag, "eval": r})
        outs.append(acts[-1][:, 0])
        agree.append([float(np.mean((a >= 0.5) == (z > 0))) for a, z in zip(acts[1:], zs)])
    outs = np.array(outs)
    return {"accuracy": float(np.mean((outs >= 0.5) == (y >= 0.5))),
            "fires_as_intended": {"hidden": float(np.mean([a[0] for a in agree])),
                                  "output": float(np.mean([a[1] for a in agree]))}}


def data():
    xtr, ytr, _, _ = mnist()
    tr, va = split(ytr, (3, 8), 500, 500)
    return xtr[tr], (ytr[tr] == 8).astype(float)[:, None], xtr[va], (ytr[va] == 8).astype(float)


def local_check():
    X, Y, Xv, yv = data()
    swap = [tuple(np.load(ROOT / "runs/stage5/weights-3v8-swap.npz")[f"arr_{i}"] for i in pair) for pair in ((0, 1), (2, 3))]
    for comp in (False, True):
        acc = np.mean([np.mean((forward(swap, Xv, ShiftedMock(comp, s))[0][-1][:, 0] >= 0.5) == (yv >= 0.5)) for s in range(3)])
        print(f"swap weights on the quirky mock, compensate={comp}: {acc:.3f}")
    for name, comp in (("B", False), ("C", True)):
        p = fit(init(SEED), X, Y, ShiftedMock(comp, 1), EPOCHS, BATCH, LR, 3.0, SEED, slope=slope_for(comp))
        acc = np.mean((forward(p, Xv, ShiftedMock(comp, 9))[0][-1][:, 0] >= 0.5) == (yv >= 0.5))
        print(f"arm {name} trained on the quirky mock: {acc:.3f}")


def main():
    from jevrons.jev import Jev
    OUT.mkdir(parents=True, exist_ok=True)
    X, Y, Xv, yv = data()
    jev = Jev(OUT / "journal.jsonl")
    swap = [tuple(np.load(ROOT / "runs/stage5/weights-3v8-swap.npz")[f"arr_{i}"] for i in pair) for pair in ((0, 1), (2, 3))]
    arms = {"B": (False, ["v1"]), "C": (True, ["v1"]), "D": (True, ["v1", "v2", "v3"]), "E": ("none", ["v1", "choice"])}
    for name, (comp, phrasings) in arms.items():
        done = OUT / f"result-{name}.json"
        if done.exists():
            continue
        neuron = CalibratedJevNeuron(jev, comp is True, phrasings)
        tag = {"arm": name}
        log = []
        params = fit(init(SEED), X, Y, neuron, EPOCHS, BATCH, LR, 3.0, SEED, log, {**tag, "split": "train"},
                     slope=slope_for(comp is not False), checkpoint=OUT / f"checkpoint-{name}.pkl")
        res = {"arm": name, "compensate": comp, "phrasings": phrasings,
               "trained": evaluate(params, Xv, yv, neuron, {**tag, "split": "val"}),
               "trained_exact_step": float(np.mean((forward(params, Xv, StepNeuron())[0][-1][:, 0] >= 0.5) == (yv >= 0.5)))}
        if comp is not False:  # the neuron itself changed, so the swap control changes too
            res["swap"] = evaluate(swap, Xv, yv, neuron, {**tag, "split": "val", "swap": True})
        np.savez(OUT / f"weights-{name}.npz", *[a for layer in params for a in layer])
        (OUT / f"curve-{name}.json").write_text(json.dumps(log))
        done.write_text(json.dumps(res))
        print(name, json.dumps(res), f"(${jev.usd:.2f})", flush=True)
    print({"calls": jev.calls, "usd": round(jev.usd, 4), "by_backend": jev.by_backend})


if __name__ == "__main__":
    local_check() if len(sys.argv) > 1 and sys.argv[1] == "local" else main()
