"""Stage 2: AND, OR (one Jev neuron) and XOR (2-4-1 Jev net).

Arms at equal Jev-call budgets: straight-through with the products format, straight-through
with bias folded into the products, and SPSA (half the steps, two passes each).
Usage: uv run python -m jevrons.stage2
"""

import json
from pathlib import Path

import numpy as np

from jevrons.jev import Jev
from jevrons.net import JevNeuron, StepNeuron, forward, init, train_spsa, train_ste

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "runs" / "stage2"
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], float)
TASKS = {
    "AND": (np.array([[0], [0], [0], [1]], float), [2, 1], 60),
    "OR": (np.array([[0], [1], [1], [1]], float), [2, 1], 60),
    "XOR": (np.array([[0], [1], [1], [0]], float), [2, 4, 1], 150),
}
SEEDS = (0, 1, 2)
EVAL_REPEATS = 5


def evaluate(params, neuron, Y):
    """Truth table under the real (stochastic) neuron, EVAL_REPEATS independent passes."""
    outs = np.array([forward(params, X, neuron, {"eval": r})[0][-1].ravel() for r in range(EVAL_REPEATS)])
    per_pass = np.mean((outs >= 0.5) == (Y.ravel() >= 0.5), axis=1)
    return {"pass_accuracy": per_pass.tolist(), "all_correct_frac": float(np.mean(per_pass == 1)),
            "mean_p": outs.mean(0).round(3).tolist()}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    jev = Jev(OUT / "journal.jsonl")
    results = []
    for task, (Y, sizes, steps) in TASKS.items():
        for seed in SEEDS:
            arms = {
                "ste_products": ("products", lambda n, p, log: train_ste(p, X, Y, n, steps, log=log)),
                "ste_folded": ("folded", lambda n, p, log: train_ste(p, X, Y, n, steps, log=log)),
                "spsa_products": ("products", lambda n, p, log: train_spsa(p, X, Y, n, steps // 2, lr=0.3, c=0.3, seed=seed, log=log)),
            }
            for arm, (fmt, fit) in arms.items():
                done = OUT / f"result-{task}-{arm}-{seed}.json"
                if done.exists():  # resume: never pay for a finished run twice
                    results.append(json.loads(done.read_text()))
                    continue
                neuron = JevNeuron(jev, fmt)
                log = []
                calls0 = jev.calls
                params = fit(neuron, init(sizes, seed), log)
                res = {"task": task, "seed": seed, "arm": arm, "train_calls": jev.calls - calls0,
                       **evaluate(params, neuron, Y),
                       "local_step_correct": bool(np.all(forward(params, X, StepNeuron())[0][-1] == Y)),
                       "loss_first": log[0]["loss"], "loss_last": log[-1]["loss"],
                       "params": [[W.round(2).tolist(), b.round(2).tolist()] for W, b in params]}
                (OUT / f"curve-{task}-{arm}-{seed}.json").write_text(json.dumps(log))
                done.write_text(json.dumps(res))
                results.append(res)
                print(f"{task} seed={seed} {arm:<14} calls={res['train_calls']:<5} "
                      f"all-correct {res['all_correct_frac']:.1f}  pass acc {np.mean(res['pass_accuracy']):.2f}  "
                      f"loss {res['loss_first']:.3f}->{res['loss_last']:.3f}  step-net correct {res['local_step_correct']}",
                      flush=True)
    summary = {"results": results, "spend": {"calls": jev.calls, "tokens": jev.tokens, "usd": round(jev.usd, 4)}}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1))
    print(summary["spend"])


if __name__ == "__main__":
    main()
