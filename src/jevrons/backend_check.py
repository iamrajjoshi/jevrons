"""Is a backend the same neuron, and how fast is it? Usage: uv run python -m jevrons.backend_check vercel

1. Equivalence: 100 folded-bias neurons x 3 repeats on `typesafe` and on the named backend.
   Same neuron means sign accuracy and mean p agree within the repeat-to-repeat noise.
2. Throughput: 1,000 calls to the named backend alone, paced at 3,000/min, counting retries.
"""

import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from jevrons.digits import pixel_pool
from jevrons.jev import BACKENDS, Jev
from jevrons.states import neuron_question, neuron_state

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "runs" / "backend_check"


def neurons(n=100):
    pool, rng = pixel_pool(), np.random.default_rng(7)
    out = []
    for i in range(n):
        k = (10, 30, 75, 150)[i % 4]
        x, w = rng.choice(pool, k), np.clip(rng.normal(0, 1, k), -9.99, 9.99)
        out.append(neuron_state("folded", x, w, rng.normal(0, np.sqrt(np.sum((x * w) ** 2)))))
    return out


def main(name):
    OUT.mkdir(parents=True, exist_ok=True)
    states, q = neurons(), neuron_question("folded")
    p = {}
    for b in ("typesafe", name):
        jev = Jev(OUT / "journal.jsonl", backends=[b])
        with ThreadPoolExecutor(16) as ex:
            p[b] = np.array(list(ex.map(lambda j: jev.ask(states[j[0]][0], q, {"part": "equiv", "i": j[0], "rep": j[1]})["fires"],
                                        [(i, r) for r in range(3) for i in range(len(states))]))).reshape(3, -1)
    z = np.array([s[1] for s in states])
    repeat_noise = float(np.mean([np.abs(p[b][0] - p[b][1]).mean() for b in p]))
    equiv = {b: {"sign_accuracy": float(np.mean((p[b] >= 0.5) == (z > 0))), "mean_p": float(p[b].mean())} for b in p}
    equiv["mean_abs_diff_between_backends"] = float(np.abs(p["typesafe"].mean(0) - p[name].mean(0)).mean())
    equiv["mean_abs_diff_between_repeats"] = repeat_noise
    print(json.dumps(equiv, indent=1), flush=True)

    fast = BACKENDS[name]
    fast.per_min = 3000
    jev = Jev(OUT / "journal.jsonl", backends=[name])
    t0 = time.monotonic()
    with ThreadPoolExecutor(48) as ex:
        list(ex.map(lambda i: jev.ask(states[i % 100][0], q, {"part": "throughput", "i": i}), range(1000)))
    elapsed = time.monotonic() - t0
    retries = sum(json.loads(line)["attempts"] - 1 for line in open(OUT / "journal.jsonl")
                  if json.loads(line)["tag"].get("part") == "throughput")
    result = {"equivalence": equiv, "throughput_calls_per_min": round(1000 / elapsed * 60), "retries": retries}
    (OUT / f"summary-{name}.json").write_text(json.dumps(result, indent=1))
    print(result)


if __name__ == "__main__":
    main(sys.argv[1])
