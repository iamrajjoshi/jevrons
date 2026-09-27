"""Stage 9a: a smaller neuron probe for slow local models (Ollaya backends).

Scalar: z in [-10, 10] at 0.1, both wordings, one call each. Full state: folded-bias products,
40 neurons per term count with bias drawn independently of the terms (the stage 1 wording probe),
same seed as stage 1, so the numbers line up with Jev's. Usage:
  uv run python -m jevrons.local_probe ollaya-von [terms...]
"""

import gzip
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from jevrons.jev import Jev
from jevrons.states import SCALAR_Q, SCALAR_Q_ALT, neuron_question, neuron_state

ROOT = Path(__file__).resolve().parents[2]


def main(backend, terms):
    out = ROOT / "runs" / f"probe-{backend}"
    jev = Jev(out / "journal.jsonl", backends=[backend])
    pmap = lambda f, xs: list(ThreadPoolExecutor(4).map(f, xs))  # noqa: E731

    zs = [round(v, 1) for v in np.arange(-10, 10.0001, 0.1)]
    scalar = {}
    for name, q in (("v1", SCALAR_Q), ("v2", SCALAR_Q_ALT)):
        ps = pmap(lambda z: jev.ask({"z": z}, q, {"part": "scalar", "wording": name, "z": z})["pos"], zs)
        scalar[name] = {"sign_accuracy": float(np.mean([(p >= 0.5) == (z > 0) for z, p in zip(zs, ps)])),
                        "mean_p_neg": float(np.mean([p for z, p in zip(zs, ps) if z < 0])),
                        "mean_p_pos": float(np.mean([p for z, p in zip(zs, ps) if z > 0]))}
        print(backend, "scalar", name, scalar[name], flush=True)

    imgs = np.frombuffer(gzip.open(ROOT / "data/train-images-idx3-ubyte.gz").read(), np.uint8, offset=16)
    pool, rng = imgs[imgs > 0] / 255.0, np.random.default_rng(3)
    full = []
    for k in terms:
        cells = []
        for i in range(40):
            x, w = rng.choice(pool, k), np.clip(rng.normal(0, 1, k), -9.99, 9.99)
            b = rng.normal(0, np.sqrt(np.sum((x * w) ** 2)))
            cells.append(neuron_state("folded", x, w, b))
        ps = pmap(lambda sz: jev.ask(sz[0], neuron_question("folded"), {"part": "full", "k": k, "z": sz[1]})["fires"], cells)
        acc = float(np.mean([(p >= 0.5) == (z > 0) for (_, z), p in zip(cells, ps)]))
        full.append({"k": k, "sign_accuracy": acc, "frac_yes": float(np.mean([p >= 0.5 for p in ps]))})
        print(backend, "full", full[-1], flush=True)
    lat = [json.loads(line)["latency_s"] for line in open(out / "journal.jsonl")]
    summary = {"backend": backend, "scalar": scalar, "full_folded": full, "calls": jev.calls,
               "median_latency_s": float(np.median(lat))}
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(summary)


if __name__ == "__main__":
    main(sys.argv[1], [int(t) for t in sys.argv[2:]] or [10, 30, 75, 150])
