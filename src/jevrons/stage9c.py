"""Stage 9c: laya as a scalar neuron. Code computes z; laya answers "is z greater than zero?".

Laya fails that question as a classifier (52% sign accuracy) and its response is jagged and
V-shaped, but it is deterministic, so its measured curve (runs/probe-laya/scalar_curve.json,
2,181 values of z) is its activation function. Backprop goes through a smoothed copy of that
curve. Same task, split, init and schedule as stage 5's 3 vs 8; the swap control is stage 5's
exact-trained network run on laya. Free and local (Ollaya).
Usage: uv run python -m jevrons.stage9c [local]
"""

import json
import sys
from pathlib import Path

import numpy as np

from jevrons.digits import mnist, split
from jevrons.net import ScalarJevNeuron, StepNeuron, fit, forward, sig
from jevrons.stage5 import BATCH, EPOCHS, LR, SEED, init
from jevrons.states import SCALAR_Q

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "runs" / "stage9c"
_C = json.loads((ROOT / "runs/probe-laya/scalar_curve.json").read_text())
_ZT, _PT = np.array(_C["z"]), np.array(_C["p_mean"])
_GRID = np.round(np.arange(-100, 100.001, 0.01), 2)
_RAW = np.interp(_GRID, _ZT, _PT)
_SIGMA = 0.5  # smoothing width in units of z, for the backward pass only
_k = np.exp(-0.5 * (np.arange(-300, 301) * 0.01 / _SIGMA) ** 2)
_SMOOTH = np.convolve(np.pad(_RAW, 300, mode="edge"), _k / _k.sum(), mode="valid")
_DSMOOTH = np.gradient(_SMOOTH, 0.01)


def laya_curve(z):
    """Tabulated laya response at two-decimal z (what the neuron actually returns)."""
    return np.interp(np.clip(np.round(z, 2), -100, 100), _GRID, _RAW)


def slope(layer, z, a_in, W, b):
    return np.interp(np.clip(z, -100, 100), _GRID, _DSMOOTH)


class TabulatedLaya:
    """Local replay of laya's measured curve; laya is deterministic, so this is laya off-line."""

    def __call__(self, X, W, b, tag=None):
        return laya_curve(X @ W + b)


def data():
    xtr, ytr, _, _ = mnist()
    tr, va = split(ytr, (3, 8), 500, 500)
    return xtr[tr], (ytr[tr] == 8).astype(float)[:, None], xtr[va], (ytr[va] == 8).astype(float)


def accuracy(params, X, y, neuron, tag=None):
    out = forward(params, X, neuron, tag)[0][-1][:, 0]
    return float(np.mean((out >= 0.5) == (y >= 0.5))), out


def swap_weights():
    w = np.load(ROOT / "runs/stage5/weights-3v8-swap.npz")
    return [(w["arr_0"], w["arr_1"]), (w["arr_2"], w["arr_3"])]


def local():
    X, Y, Xv, yv = data()
    tab = TabulatedLaya()
    for name, s in (("smoothed-curve backprop", slope), ("sigmoid backprop, tau 3", None)):
        p = fit(init(SEED), X, Y, tab, EPOCHS, BATCH, LR, 3.0, SEED, output="bce", slope=s)
        print(f"trained on tabulated laya, {name}: {accuracy(p, Xv, yv, tab)[0]:.3f}"
              f"  (same weights, exact step: {accuracy(p, Xv, yv, StepNeuron())[0]:.3f})")
    print(f"swap (stage 5 exact-trained) on tabulated laya: {accuracy(swap_weights(), Xv, yv, tab)[0]:.3f}")


NEG, POS = 0.718, 0.919  # laya's mean p for z < -2 and z > 2: where the jagged curve settles


class LayaNet:
    """Layer 0: laya on the raw sum. Layer 1: laya calibrated to [0, 1] by its measured levels
    ("all-laya"), or an ordinary sigmoid in code ("laya-hidden")."""

    def __init__(self, hidden, output_mode):
        self.hidden, self.output_mode = hidden, output_mode

    def __call__(self, X, W, b, tag=None):
        if tag and tag.get("layer") == 0:
            return self.hidden(X, W, b, tag)
        if self.output_mode == "code":
            return sig(X @ W + b)
        return np.clip((self.hidden(X, W, b, tag) - NEG) / (POS - NEG), 0.02, 0.98)


def net_slope(output_mode):
    def f(layer, z, a_in, W, b):
        if layer == 0:
            return slope(layer, z, a_in, W, b)
        if output_mode == "code":
            return sig(z) * (1 - sig(z))
        return slope(layer, z, a_in, W, b) / (POS - NEG)
    return f


def main():
    from jevrons.jev import Jev
    OUT.mkdir(parents=True, exist_ok=True)
    X, Y, Xv, yv = data()
    jev = Jev(OUT / "journal.jsonl", backends=["ollaya"])
    laya = ScalarJevNeuron(jev, threads=32, question=SCALAR_Q)
    res = {}
    for arm, output_mode in (("all-laya", "laya"), ("laya-hidden", "code")):
        live, tabulated = LayaNet(laya, output_mode), LayaNet(TabulatedLaya(), output_mode)
        wpath = OUT / f"weights-{arm}.npz"
        if not wpath.exists():
            log = []
            params = fit(init(SEED), X, Y, live, EPOCHS, BATCH, LR, 3.0, SEED, log, {"arm": arm, "split": "train"},
                         output="bce", slope=net_slope(output_mode), checkpoint=OUT / f"checkpoint-{arm}.pkl")
            np.savez(wpath, *[a for layer in params for a in layer])
            (OUT / f"curve-{arm}.json").write_text(json.dumps(log))
        w = np.load(wpath)
        params = [(w["arr_0"], w["arr_1"]), (w["arr_2"], w["arr_3"])]
        for which, p in (("trained", params), ("swap", swap_weights())):
            acc, out = accuracy(p, Xv, yv, live, {"arm": arm, "which": which, "split": "val"})
            res[f"{arm}/{which}"] = {"val_on_laya": acc, "val_on_tabulated_laya": accuracy(p, Xv, yv, tabulated)[0],
                                     "val_exact_step": accuracy(p, Xv, yv, StepNeuron())[0],
                                     "output_mean_by_class": {"3": float(out[yv < 0.5].mean()), "8": float(out[yv > 0.5].mean())}}
            print(arm, which, res[f"{arm}/{which}"], flush=True)
    res["calls"] = jev.calls
    (OUT / "summary.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    local() if len(sys.argv) > 1 and sys.argv[1] == "local" else main()
