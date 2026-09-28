"""Stage 8c: pretrain offline on a simulated arm E neuron, then fine-tune briefly on live Jev.

Same architecture, data, mask and init as stage 8a (sparse 784-64-10, 96 random connections per hidden neuron), so
stage 8a's Jev-only run is the comparison at equal architecture and data. Pretraining runs stage 8a's 6 epochs
through SimE, the stage 6c curve neuron fitted to stage 7's arm E answers (free); fine-tuning runs FT_EPOCHS epochs
through Jev with a fresh Adam at FT_LR. Validation on Jev before fine-tuning and after each fine-tuning epoch; one test.

  local   offline check: pretrain on SimE, fine-tune on a differently-tuned SimE standing in for Jev, at several rates
  run     the paid run
  pretest the pretrained network (no Jev training) on the test set through Jev
Usage: uv run python -m jevrons.stage8c [local | run | pretest]
"""

import json
import sys
from pathlib import Path

import numpy as np

from jevrons.jev import Jev
from jevrons.net import fit, forward
from jevrons.stage8a import (BATCH, EPOCHS, KIND, K, H, LR, SEED, SIM_E, TAU, SimE, SparseENeuron, active_pixels, budget,
                             connections, data, init, slope, summarize)

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "runs" / "stage8c"
FT_EPOCHS, FT_LR = 2, 0.01


def pretrain(X, Y, mask):
    return fit(init(SEED, mask), X, Y, SimE(*SIM_E), EPOCHS, BATCH, LR, TAU, SEED, slope=slope, mask=mask)


def local():
    X, Y, Xv, yv, _, _ = data()
    mask = connections(KIND, K, H, active_pixels(X))
    acc = lambda p, n: float(np.mean(np.argmax(forward(p, Xv, n)[0][-1], 1) == yv))  # noqa: E731
    pre = pretrain(X, Y, mask)
    for name, alt in (("less lean, sharper", ([3.0, 0.04, 0.2], [2.0, 0.04, 0.5])), ("more lean, softer", ([6.0, 0.08, 0.8], [3.0, 0.06, 1.0]))):
        target = SimE(*alt)
        direct = fit(init(SEED, mask), X, Y, target, EPOCHS, BATCH, LR, TAU, SEED, slope=slope, mask=mask)
        print(f"{name}: trained on it for {EPOCHS} epochs {acc(direct, target):.3f}; pretrained on SimE, no fine-tune {acc(pre, target):.3f}", flush=True)
        for lr in (0.02, 0.01, 0.005):
            curve = []
            fit(pre, X, Y, target, FT_EPOCHS, BATCH, lr, TAU, SEED + 1, slope=slope, mask=mask,
                on_epoch=lambda e, p: curve.append(round(acc(p, target), 3)))
            print(f"  fine-tune lr {lr}: after each epoch {curve}", flush=True)


def run():
    OUT.mkdir(parents=True, exist_ok=True)
    X, Y, Xv, yv, Xt, yt = data()
    mask = connections(KIND, K, H, active_pixels(X))
    pre_path = OUT / "weights-pretrained.npz"
    if pre_path.exists():
        w = np.load(pre_path)
        pre = [(w["arr_0"], w["arr_1"]), (w["arr_2"], w["arr_3"])]
    else:
        pre = pretrain(X, Y, mask)
        np.savez(pre_path, *[a for layer in pre for a in layer])
    jev = Jev(OUT / "journal.jsonl", max_usd=budget())
    neuron = SparseENeuron(jev)
    tag = {"seed": SEED, "arm": "pretrain-ft"}
    partial = OUT / "partial.json"
    res = json.loads(partial.read_text()) if partial.exists() else {"val": {}}
    if "0" not in res["val"]:
        r = summarize(pre, Xv, yv, neuron, {**tag, "split": "val", "epoch": -1})
        r["sim_accuracy"] = float(np.mean(np.argmax(forward(pre, Xv, SimE(*SIM_E))[0][-1], 1) == yv))
        r["usd_so_far"] = jev.usd
        res["val"]["0"] = r
        partial.write_text(json.dumps(res))
        print(f"pretrained, before fine-tuning: val on Jev {r['accuracy']:.3f}, exact {r['exact_step']:.3f}, "
              f"on SimE {r['sim_accuracy']:.3f}  (${jev.usd:.2f})", flush=True)

    def on_epoch(epoch, params):
        if str(epoch + 1) not in res["val"]:
            r = summarize(params, Xv, yv, neuron, {**tag, "split": "val", "epoch": epoch})
            r["usd_so_far"] = jev.usd
            res["val"][str(epoch + 1)] = r
            partial.write_text(json.dumps(res))
            print(f"fine-tune epoch {epoch + 1}: val on Jev {r['accuracy']:.3f}, exact {r['exact_step']:.3f}  (${jev.usd:.2f})", flush=True)

    log = []
    params = fit(pre, X, Y, neuron, FT_EPOCHS, BATCH, FT_LR, TAU, SEED + 1, log, {**tag, "split": "train"}, on_epoch=on_epoch,
                 checkpoint=OUT / "checkpoint.pkl", slope=slope, mask=mask)
    res["test"] = summarize(params, Xt, yt, neuron, {**tag, "split": "test"})
    res["spend_process"] = {"calls": jev.calls, "tokens": jev.tokens, "usd": round(jev.usd, 4)}
    np.savez(OUT / "weights-finetuned.npz", *[a for layer in params for a in layer])
    (OUT / "curve.json").write_text(json.dumps(log))
    (OUT / "result.json").write_text(json.dumps(res))
    print(f"test on Jev {res['test']['accuracy']:.3f}, exact {res['test']['exact_step']:.3f}  (${jev.usd:.2f})", flush=True)
    print({"calls": jev.calls, "tokens": jev.tokens, "usd": round(jev.usd, 4), "by_backend": jev.by_backend})


def pretest():
    """The pretrained network, before any Jev training, on the test set through Jev (after `run`)."""
    X, Y, Xv, yv, Xt, yt = data()
    w = np.load(OUT / "weights-pretrained.npz")
    pre = [(w["arr_0"], w["arr_1"]), (w["arr_2"], w["arr_3"])]
    jev = Jev(OUT / "journal-pretest.jsonl", max_usd=min(2.5, budget()))
    r = summarize(pre, Xt, yt, SparseENeuron(jev), {"seed": SEED, "arm": "pretrained", "split": "test"})
    r["sim_accuracy"] = float(np.mean(np.argmax(forward(pre, Xt, SimE(*SIM_E))[0][-1], 1) == yt))
    r["spend_process"] = {"calls": jev.calls, "tokens": jev.tokens, "usd": round(jev.usd, 4)}
    (OUT / "result-pretrained-test.json").write_text(json.dumps(r))
    print(f"pretrained only, test on Jev {r['accuracy']:.3f}, exact {r['exact_step']:.3f}, on SimE {r['sim_accuracy']:.3f}  (${jev.usd:.2f})")


if __name__ == "__main__":
    {"local": local, "run": run, "pretest": pretest}[sys.argv[1] if len(sys.argv) > 1 else "run"]()
