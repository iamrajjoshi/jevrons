"""Stage 6: ten digits with full-state Jev neurons (784-32-10, folded-bias products).

3 seeds. Arms per seed: trained through live Jev (10 epochs on 1,000 images), and a swap control
(same init, trained with an exact step neuron, run through live Jev). Validation on 500 images at
epochs 5 and 10; one final test on 1,000 stratified test images. Checkpoints every epoch, so a
crash resumes where it stopped. Usage: uv run python -m jevrons.stage6
"""

import json
from pathlib import Path

import numpy as np

from jevrons.digits import mnist, split, test_subset
from jevrons.jev import Jev
from jevrons.net import JevNeuron, StepNeuron, decide, fit, forward, save_params

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "runs" / "stage6"
SEEDS = (0, 1, 2)
EPOCHS, BATCH, LR, TAU, SCALE = 10, 32, 0.02, 3.0, 0.3
VAL_EPOCHS = (4, 9)  # zero-based: after epochs 5 and 10


def init(seed):
    rng = np.random.default_rng(seed)
    return [(rng.normal(0, SCALE, (784, 32)), np.zeros(32)), (rng.normal(0, 1 / np.sqrt(32), (32, 10)), np.zeros(10))]


def evaluate(params, X, y, neuron, tag):
    acts, zs = forward(params, X, neuron, tag)
    agree = [float(np.mean((a >= 0.5) == (z > 0))) for a, z in zip(acts[1:], zs)]
    return {"accuracy": float(np.mean(decide(acts[-1]) == y)),
            "no_output_fires": float(np.mean((acts[-1] >= 0.5).sum(1) == 0)),
            "fires_as_intended": {"hidden": agree[0], "output": agree[1]},
            "per_digit": [float(np.mean(decide(acts[-1])[y == d] == d)) for d in range(10)],
            "outputs": acts[-1].round(2).tolist()}


def exact_accuracy(params, X, y):
    return float(np.mean(decide(forward(params, X, StepNeuron())[0][-1]) == y))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    xtr, ytr, xte, yte = mnist()
    tr, va = split(ytr, range(10), 1000, 500)
    te = test_subset(yte, 1000)
    X, Y, Xv, yv, Xt, yt = xtr[tr], np.eye(10)[ytr[tr]], xtr[va], ytr[va], xte[te], yte[te]
    jev = Jev(OUT / "journal.jsonl")
    neuron = JevNeuron(jev, "folded", threads=128)
    for seed in SEEDS:
        for arm in ("jev", "swap"):
            done = OUT / f"result-seed{seed}-{arm}.json"
            if done.exists():
                continue
            tag = {"seed": seed, "arm": arm}
            res = {"seed": seed, "arm": arm, "val": {}}
            partial = OUT / f"partial-seed{seed}-{arm}.json"
            if partial.exists():
                res = json.loads(partial.read_text())

            def on_epoch(epoch, params):
                if arm == "jev" and epoch in VAL_EPOCHS and str(epoch + 1) not in res["val"]:
                    r = evaluate(params, Xv, yv, neuron, {**tag, "split": "val", "epoch": epoch})
                    r["exact_step"] = exact_accuracy(params, Xv, yv)
                    res["val"][str(epoch + 1)] = r
                    partial.write_text(json.dumps(res))
                    print(f"seed={seed} jev epoch {epoch + 1}: val on Jev {r['accuracy']:.3f}, exact {r['exact_step']:.3f}, "
                          f"fires {r['fires_as_intended']}  (${jev.usd:.2f})", flush=True)

            log = []
            params = fit(init(seed), X, Y, neuron if arm == "jev" else StepNeuron(), EPOCHS, BATCH, LR, TAU, seed,
                         log, {**tag, "split": "train"}, on_epoch=on_epoch,
                         checkpoint=OUT / f"checkpoint-seed{seed}-{arm}.pkl")
            if arm == "swap":
                r = evaluate(params, Xv, yv, neuron, {**tag, "split": "val"})
                r["exact_step"] = exact_accuracy(params, Xv, yv)
                res["val"][str(EPOCHS)] = r
            res["test"] = evaluate(params, Xt, yt, neuron, {**tag, "split": "test"})
            res["test"]["exact_step"] = exact_accuracy(params, Xt, yt)
            save_params(OUT / f"weights-seed{seed}-{arm}.npz", params)
            (OUT / f"curve-seed{seed}-{arm}.json").write_text(json.dumps(log))
            done.write_text(json.dumps(res))
            print(f"seed={seed} {arm}: val {res['val'][str(EPOCHS)]['accuracy']:.3f}, test on Jev {res['test']['accuracy']:.3f}, "
                  f"test exact {res['test']['exact_step']:.3f}  (${jev.usd:.2f})", flush=True)
    print({"calls": jev.calls, "tokens": jev.tokens, "usd": round(jev.usd, 4), "by_backend": jev.by_backend})


if __name__ == "__main__":
    main()
