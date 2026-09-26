"""Stage 5: Variant B on two digits. 784-32-1 net where every neuron is a full-state Jev call
(folded-bias products of the nonzero pixels). Pairs 0 vs 1, then 3 vs 8; 500 train / 500 val.

Arms per pair: trained through live Jev, and a swap control (same init, trained with an exact
step neuron, evaluated through live Jev). Settings tuned locally on MockJevNeuron.
Usage: uv run python -m jevrons.stage5
"""

import json
from pathlib import Path

import numpy as np

from jevrons.digits import mnist, split
from jevrons.jev import Jev
from jevrons.net import JevNeuron, StepNeuron, fit, forward

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "runs" / "stage5"
PAIRS = ((0, 1), (3, 8))
EPOCHS, BATCH, LR, TAU, SCALE = 5, 32, 0.02, 3.0, 0.3
EVAL_PASSES = 2
SEED = 0


def init(seed):
    rng = np.random.default_rng(seed)
    return [(rng.normal(0, SCALE, (784, 32)), np.zeros(32)), (rng.normal(0, 1 / np.sqrt(32), (32, 1)), np.zeros(1))]


def evaluate(params, X, y, neuron, tag):
    outs, agree = [], []
    for r in range(EVAL_PASSES):
        acts, zs = forward(params, X, neuron, {**tag, "eval": r})
        outs.append(acts[-1][:, 0])
        agree.append([float(np.mean((a >= 0.5) == (z > 0))) for a, z in zip(acts[1:], zs)])
    outs = np.array(outs)
    return {"accuracy": float(np.mean((outs >= 0.5) == (y >= 0.5))),
            "accuracy_per_pass": np.mean((outs >= 0.5) == (y >= 0.5), axis=1).tolist(),
            "fires_as_intended": {"hidden": float(np.mean([a[0] for a in agree])), "output": float(np.mean([a[1] for a in agree]))}}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    xtr, ytr, _, _ = mnist()
    jev = Jev(OUT / "journal.jsonl")
    neuron = JevNeuron(jev, "folded", threads=128)
    summary = []
    for pair in PAIRS:
        tr, va = split(ytr, pair, 500, 500)
        X, Xv = xtr[tr], xtr[va]
        Y, yv = (ytr[tr] == pair[1]).astype(float)[:, None], (ytr[va] == pair[1]).astype(float)
        name = f"{pair[0]}v{pair[1]}"
        for arm in ("jev", "swap"):
            done = OUT / f"result-{name}-{arm}.json"
            if done.exists():
                summary.append(json.loads(done.read_text()))
                continue
            log, tag = [], {"pair": name, "arm": arm}
            train_neuron = neuron if arm == "jev" else StepNeuron()
            params = fit(init(SEED), X, Y, train_neuron, EPOCHS, BATCH, LR, TAU, SEED, log, {**tag, "split": "train"})
            res = {"pair": name, "arm": arm, "val_jev": evaluate(params, Xv, yv, neuron, {**tag, "split": "val"}),
                   "val_exact_step": float(np.mean((forward(params, Xv, StepNeuron())[0][-1][:, 0] >= 0.5) == (yv >= 0.5)))}
            np.savez(OUT / f"weights-{name}-{arm}.npz", *[a for layer in params for a in layer])
            (OUT / f"curve-{name}-{arm}.json").write_text(json.dumps(log))
            done.write_text(json.dumps(res))
            summary.append(res)
            print(f"{name} {arm}: val on Jev {res['val_jev']['accuracy']:.3f}, exact-step {res['val_exact_step']:.3f}, "
                  f"fires as intended {res['val_jev']['fires_as_intended']}  (${jev.usd:.2f} so far)", flush=True)
    spend = {"calls": jev.calls, "tokens": jev.tokens, "usd": round(jev.usd, 4), "by_backend": jev.by_backend}
    (OUT / "summary.json").write_text(json.dumps({"results": summary, "spend": spend}, indent=1))
    print(spend)


if __name__ == "__main__":
    main()
