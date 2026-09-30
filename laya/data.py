"""Training and test states for stage 10. Every state is exactly what states.neuron_state("folded", x, w, b)
produces, asked with states.neuron_question("folded"), so training matches what Jevrons sends.

Synthetic states ("probe" family): k MNIST pixel values times N(0, s) weights at two decimals plus a bias,
with the weights shifted so the margin m = z / spread lands on a target drawn uniformly from [-3, 3]
(capped by the largest margin k + 1 numbers allow). "generic" family: heavy-tailed numbers with no MNIST
structure, set to the target margin by a shift. Both shift families tie the sign vote to the total, and the
pilot learned to count signs (stage 6b's vote trap fooled it 96% of the time), so "mixed" (added after the
pilot) draws how many numbers are positive independently of the total and scales the two groups to the target.
Family mix 50/20/30.
Label = true sign of the total (z > 0), computed from the rounded numbers sent.
"""

import gzip
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from jevrons import digits  # noqa: E402
from jevrons.stage6b_traps import two_groups  # noqa: E402
from jevrons.states import neuron_question, neuron_state  # noqa: E402

QUESTION = neuron_question("folded")["fires"]
FAMILIES, FAMILY_P = ("probe", "generic", "mixed"), (0.5, 0.2, 0.3)
TRAIN_K = [k for k in range(2, 61) if k not in (10, 30)]  # 10 and 30 held out; 61+ tests length generalization
REAL = ROOT / "runs/stage10/real.jsonl.gz"
_POOL = None


def pixel_pool():
    global _POOL
    if _POOL is None:
        _POOL = digits.pixel_pool()
    return _POOL


def margin(products) -> float:
    v = np.asarray(products, float)
    s = np.sqrt(np.sum(v * v))
    return float(v.sum() / s) if s else 0.0


def synth(rng, k: int, target: float | None = None, family: str | None = None):
    """One synthetic folded state near margin `target`. Returns (state, z)."""
    cap = 0.95 * np.sqrt(k + 1)
    t = float(np.clip(rng.uniform(-3, 3) if target is None else target, -cap, cap))
    family = family or str(rng.choice(FAMILIES, p=FAMILY_P))
    for _ in range(200):
        if family == "mixed":  # sign counts drawn independently of the total, so counting signs can't work
            n_pos = int(rng.integers(1, k + 1))
            ratio = np.exp(rng.uniform(-np.log(10), np.log(10)))
            pos, neg = two_groups(np.abs(rng.standard_t(3, n_pos)) * ratio + 0.01,
                                  np.abs(rng.standard_t(3, k + 1 - n_pos)) + 0.01, t, np.exp(rng.uniform(0, np.log(40))))
            v = rng.permutation(np.r_[pos, neg])
            state, z = neuron_state("folded", np.ones(k), v[:k], v[k])
            if abs(margin(state["products"]) - t) < 0.25:
                return state, z
            continue
        if family == "probe":
            x, w = rng.choice(pixel_pool(), k), rng.normal(0, np.exp(rng.uniform(np.log(0.2), np.log(3))), k)
            b = 0.0 if rng.random() < 0.2 else rng.normal(0, rng.uniform(0, 1) * np.sqrt(np.sum((x * w) ** 2)) / 2)
            for _ in range(2):  # shift the weights so the total lands on the target margin
                spread = np.sqrt(np.sum((x * w) ** 2) + b * b)
                w = np.clip(w + (t * spread - np.dot(x, w) - b) / (np.sum(x) + 1e-9), -9.99, 9.99)
        else:
            x = np.ones(k)
            w = rng.standard_t(3, k) * np.exp(rng.uniform(np.log(0.1), np.log(5)))
            b = rng.standard_t(3) * np.std(w)
            for _ in range(2):
                spread = np.sqrt(np.sum(w * w) + b * b)
                w = np.clip(w + (t * spread - w.sum() - b) / k, -99.99, 99.99)
        state, z = neuron_state("folded", x, w, b)
        if abs(margin(state["products"]) - t) < 0.25:
            return state, z
    return state, z  # rare: accept the closest attempt rather than loop forever


def state_hash(products) -> int:
    return int.from_bytes(hashlib.blake2b(json.dumps(products).encode(), digest_size=8).digest(), "little")


def real_states():
    """Real Jev-answered states, split by state hash: 10% test. The margin probe is test-only."""
    train, test = [], []
    with gzip.open(REAL, "rt") as f:
        for line in f:
            r = json.loads(line)
            held = r["src"] == "margin" or state_hash(r["products"]) % 10 == 0
            (test if held else train).append(r)
    return train, test


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    ms = []
    for k in (2, 5, 30, 60):
        for fam in FAMILIES:
            for t in (-2.5, -0.1, 0.0, 1.0, 2.9):
                s, z = synth(rng, k, t, fam)
                assert s == neuron_state("folded", s["products"][:-1], [1.0] * k, s["products"][-1])[0]
                assert abs(sum(s["products"]) - z) < 1e-9
                ms.append((k, fam, t, round(margin(s["products"]), 2)))
    print(ms)
    assert all(abs(m - np.clip(t, -0.95 * np.sqrt(k + 1), 0.95 * np.sqrt(k + 1))) < 0.26 for k, _, t, m in ms)
    print("data ok")
