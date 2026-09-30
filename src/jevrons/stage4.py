"""Stage 4: Variant A on ten digits. Train 784-32-10 locally with an exact step neuron, then run
the same weights with Jev as the activation (scalar z, "Is the number z positive?").
Usage: uv run python -m jevrons.stage4
"""

import json
from pathlib import Path

import numpy as np

from jevrons.digits import mnist, split, test_subset
from jevrons.jev import Jev
from jevrons.net import ScalarJevNeuron, StepNeuron, decide, fit, forward, predict, save_params

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "runs" / "stage4"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    xtr, ytr, xte, yte = mnist()
    tr, va = split(ytr, range(10), 1000, 1000)
    te = test_subset(yte, 1000)
    rng = np.random.default_rng(0)
    params = [(rng.normal(0, 1 / 28, (784, 32)), np.zeros(32)), (rng.normal(0, 1 / np.sqrt(32), (32, 10)), np.zeros(10))]
    params = fit(params, xtr[tr], np.eye(10)[ytr[tr]], StepNeuron(), epochs=10)
    step = StepNeuron()
    jev = Jev(OUT / "journal.jsonl")
    neuron = ScalarJevNeuron(jev)
    res = {"weights_trained_with": "exact step, 1,000 images, 10 epochs"}
    for name, X, y in (("val", xtr[va], ytr[va]), ("test", xte[te], yte[te])):
        acts_j, zs = forward(params, X, neuron, {"split": name})
        acts_s, _ = forward(params, X, step)
        fire_agree = [float(np.mean((a >= 0.5) == (z > 0))) for a, z in zip(acts_j[1:], zs)]
        res[name] = {"n": len(y), "acc_exact_step": float(np.mean(predict(params, X, step) == y)),
                     "acc_jev": float(np.mean(decide(acts_j[-1]) == y)),
                     "no_output_fires_jev": float(np.mean((acts_j[-1] >= 0.5).sum(1) == 0)),
                     "jev_fires_as_intended": {"hidden": fire_agree[0], "output": fire_agree[1]},
                     "prediction_agreement": float(np.mean(decide(acts_j[-1]) == decide(acts_s[-1])))}
        print(name, res[name], flush=True)
    res["spend"] = {"calls": jev.calls, "tokens": jev.tokens, "usd": round(jev.usd, 4), "by_backend": jev.by_backend}
    (OUT / "summary.json").write_text(json.dumps(res, indent=1))
    save_params(OUT / "weights.npz", params)
    print(res["spend"])


if __name__ == "__main__":
    main()
