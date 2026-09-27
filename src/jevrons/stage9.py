"""Stage 9b: XOR with an open-weight local neuron (Ollaya), same recipe as stage 2's folded arm,
plus a swap control (trained with an exact step neuron, run on the local model).
Usage: uv run python -m jevrons.stage9 ollaya-decider
"""

import json
import sys
from pathlib import Path

import numpy as np

from jevrons.jev import Jev
from jevrons.net import JevNeuron, StepNeuron, forward, init, train_ste
from jevrons.stage2 import SEEDS, TASKS, X, evaluate

ROOT = Path(__file__).resolve().parents[2]


def main(backend):
    out = ROOT / "runs" / f"stage9-{backend}"
    out.mkdir(parents=True, exist_ok=True)
    jev = Jev(out / "journal.jsonl", backends=[backend])
    neuron = JevNeuron(jev, "folded", threads=4)
    Y, sizes, steps = TASKS["XOR"]
    results = []
    for seed in SEEDS:
        for arm in ("trained", "swap"):
            done = out / f"result-XOR-{arm}-{seed}.json"
            if done.exists():
                results.append(json.loads(done.read_text()))
                continue
            log = []
            params = train_ste(init(sizes, seed), X, Y, neuron if arm == "trained" else StepNeuron(), steps, log=log)
            res = {"task": "XOR", "backend": backend, "arm": arm, "seed": seed, **evaluate(params, neuron, Y),
                   "local_step_correct": bool(np.all(forward(params, X, StepNeuron())[0][-1] == Y)),
                   "params": [[W.round(2).tolist(), b.round(2).tolist()] for W, b in params]}
            done.write_text(json.dumps(res))
            (out / f"curve-XOR-{arm}-{seed}.json").write_text(json.dumps(log))
            results.append(res)
            print(f"XOR {backend} seed={seed} {arm}: pass acc {np.mean(res['pass_accuracy']):.2f}, "
                  f"all-correct {res['all_correct_frac']:.1f}, exact-step correct {res['local_step_correct']}", flush=True)
    (out / "summary.json").write_text(json.dumps({"results": results, "calls": jev.calls}, indent=1))


if __name__ == "__main__":
    main(sys.argv[1])
