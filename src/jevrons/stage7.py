"""Stage 7: more data. Ten digits, 784-32-10 full-state Jev neurons, the stage 6 recipe on 5,000
training images (5x stage 6), 8 epochs, 1 seed, sized to a ~$80 budget with a hard $85 cap.
Validation on 500 images after epochs 4 and 8; one test on 2,000 stratified test images; swap
control (same init, trained with an exact step neuron on the same 5,000 images, run on Jev).
Usage: uv run python -m jevrons.stage7
"""

import json
from pathlib import Path

import numpy as np

from jevrons.digits import mnist, split, test_subset
from jevrons.jev import Jev
from jevrons.net import JevNeuron, StepNeuron, fit
from jevrons.stage6 import BATCH, LR, TAU, evaluate, exact_accuracy, init

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "runs" / "stage7"
SEED, EPOCHS, N_TRAIN, N_VAL, N_TEST = 0, 8, 5000, 500, 2000
VAL_EPOCHS = (3, 7)  # zero-based
MAX_USD = 85.0


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    xtr, ytr, xte, yte = mnist()
    tr, va = split(ytr, range(10), N_TRAIN, N_VAL, seed=7)
    te = test_subset(yte, N_TEST, seed=7)
    X, Y, Xv, yv, Xt, yt = xtr[tr], np.eye(10)[ytr[tr]], xtr[va], ytr[va], xte[te], yte[te]
    jev = Jev(OUT / "journal.jsonl", max_usd=MAX_USD)
    neuron = JevNeuron(jev, "folded", threads=128)
    for arm in ("jev", "swap"):
        done = OUT / f"result-{arm}.json"
        if done.exists():
            continue
        tag = {"seed": SEED, "arm": arm}
        partial = OUT / f"partial-{arm}.json"
        res = json.loads(partial.read_text()) if partial.exists() else {"arm": arm, "val": {}}

        def on_epoch(epoch, params):
            if arm == "jev" and epoch in VAL_EPOCHS and str(epoch + 1) not in res["val"]:
                r = evaluate(params, Xv, yv, neuron, {**tag, "split": "val", "epoch": epoch})
                r["exact_step"] = exact_accuracy(params, Xv, yv)
                res["val"][str(epoch + 1)] = r
                partial.write_text(json.dumps(res))
                print(f"jev epoch {epoch + 1}: val on Jev {r['accuracy']:.3f}, exact {r['exact_step']:.3f}  (${jev.usd:.2f})", flush=True)

        log = []
        params = fit(init(SEED), X, Y, neuron if arm == "jev" else StepNeuron(), EPOCHS, BATCH, LR, TAU, SEED, log,
                     {**tag, "split": "train"}, on_epoch=on_epoch, checkpoint=OUT / f"checkpoint-{arm}.pkl")
        if arm == "swap":
            r = evaluate(params, Xv, yv, neuron, {**tag, "split": "val"})
            r["exact_step"] = exact_accuracy(params, Xv, yv)
            res["val"][str(EPOCHS)] = r
        res["test"] = evaluate(params, Xt, yt, neuron, {**tag, "split": "test"})
        res["test"]["exact_step"] = exact_accuracy(params, Xt, yt)
        np.savez(OUT / f"weights-{arm}.npz", *[a for layer in params for a in layer])
        (OUT / f"curve-{arm}.json").write_text(json.dumps(log))
        done.write_text(json.dumps(res))
        print(f"{arm}: val {res['val'][str(EPOCHS)]['accuracy']:.3f}, test on Jev {res['test']['accuracy']:.3f}, "
              f"test exact {res['test']['exact_step']:.3f}  (${jev.usd:.2f})", flush=True)
    print({"calls": jev.calls, "tokens": jev.tokens, "usd": round(jev.usd, 4), "by_backend": jev.by_backend})


if __name__ == "__main__":
    main()
