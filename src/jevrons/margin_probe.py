"""Jev's response curve in the format the digit networks use (folded-bias products, bias 0).

For each term count, 600 neurons whose normalized margin m = z / spread is spread evenly over
[-3, 3] (rejection sampling on random weights, so the bias never carries the answer), each asked
twice. Fits P(fire) = Phi(k m) per term count. Paced slowly so a running stage isn't disturbed.
Usage: uv run python -m jevrons.margin_probe
"""

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from jevrons.digits import pixel_pool
from jevrons.jev import BACKENDS, Jev
from jevrons.net import Phi
from jevrons.states import neuron_question, neuron_state

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "runs" / "margin"
TERMS, PER_TERM, REPEATS = (10, 30, 75, 150, 250), 600, 2


def neurons(rng, pool, k, n):
    """n bias-0 neurons with margins roughly uniform on [-3, 3]."""
    targets, out = rng.uniform(-3, 3, n), []
    for t in targets:
        while True:  # scale the weights' mean so the sum lands near the target margin
            x, w = rng.choice(pool, k), rng.normal(0, 1, k)
            spread = np.sqrt(np.sum((x * w) ** 2))
            w = np.clip(w + (t * spread - np.dot(x, w)) / (np.sum(x) + 1e-9), -9.99, 9.99)
            state, z = neuron_state("folded", x, w, 0.0)
            m = z / np.sqrt(sum(v * v for v in state["products"]))
            if abs(m - t) < 0.25:
                out.append((state, z, m))
                break
    return out


def fit_k(m, fired):
    ks = np.linspace(0.2, 6, 300)
    nll = [-np.mean(fired * np.log(Phi(k * m) + 1e-9) + (1 - fired) * np.log(1 - Phi(k * m) + 1e-9)) for k in ks]
    return float(ks[int(np.argmin(nll))])


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    BACKENDS["typesafe"].per_min, BACKENDS["vercel"].per_min = 150, 450  # leave the rest for stage 6
    jev = Jev(OUT / "journal.jsonl")
    pool, rng = pixel_pool(), np.random.default_rng(11)
    q = neuron_question("folded")
    rows, summary = [], {}
    for k in TERMS:
        cells = neurons(rng, pool, k, PER_TERM)
        jobs = [(i, r) for r in range(REPEATS) for i in range(len(cells))]
        with ThreadPoolExecutor(16) as ex:
            ps = list(ex.map(lambda j: jev.ask(cells[j[0]][0], q, {"part": "margin", "k": k, "i": j[0], "rep": j[1],
                                                                    "z": cells[j[0]][1], "m": cells[j[0]][2]})["fires"], jobs))
        p = np.array(ps).reshape(REPEATS, -1)
        m = np.array([c[2] for c in cells])
        fired = (p >= 0.5).astype(float)
        summary[k] = {"k_fit": fit_k(np.tile(m, REPEATS), fired.ravel()),
                      "sign_accuracy": float(np.mean(fired == (m > 0))),
                      "repeat_flip_rate": float(np.mean(fired[0] != fired[1]))}
        rows += [[k, float(mi), *map(float, p[:, i])] for i, mi in enumerate(m)]
        print(k, summary[k], f"${jev.usd:.3f}", flush=True)
    (OUT / "rows.json").write_text(json.dumps(rows))
    (OUT / "summary.json").write_text(json.dumps({"by_terms": summary, "calls": jev.calls, "usd": round(jev.usd, 4)}, indent=1))


if __name__ == "__main__":
    main()
