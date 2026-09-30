"""Stage 6c, section 6: does Jev repeat its own mistakes? Stage 5 3v8 validation ran every call twice.
Compares how often a misfire repeats with what independent noise at the measured curve would give.
Usage: uv run python -m jevrons.stage6c_repeat
"""
from pathlib import Path

import numpy as np

from jevrons.digits import mnist, split
from jevrons.net import load_params
from jevrons.stage8g import terms_and_spread
from jevrons.stage6c import curve_p, trained_logit_p

RUNS = Path(__file__).resolve().parents[2] / "runs"
d = dict(np.load(RUNS / "stage6c/calls.npz"))
xtr, ytr, _, _ = mnist(); _, va = split(ytr, (3, 8), 500, 500); Xv = xtr[va]
rng = np.random.default_rng(0)
for arm, code in (("jev", 0), ("swap", 1)):
    (W, b), _ = load_params(RUNS / f"stage5/weights-3v8-{arm}.npz")
    Z = Xv @ W + b; n, sp = terms_and_spread(Xv, W, b)
    m = (d["src"] == 0) & (d["arm"] == code) & (d["pair"] == 1) & (d["layer"] == 0) & (d["split"] == 1)
    i, j, ev = d["i"][m].astype(int), d["j"][m].astype(int), d["eval"][m]
    e = (d["p"][m] >= .5) != (d["z"][m] > 0)
    E = np.full((2,) + Z.shape, np.nan); E[ev.astype(int), i, j] = e
    obs = np.nansum(E[0] * E[1]) / np.nansum(E[0])
    cnt = np.nansum(E[0], 1)
    for name, f in (("margin curve", curve_p), ("6b logistic", trained_logit_p)):
        pf = f(Z, sp, n); q = np.where(Z > 0, 1 - pf, pf)  # misfire prob per call
        rep = (q**2).sum() / q.sum()
        sims = [(rng.random(q.shape) < q).sum(1) for _ in range(300)]
        v = np.mean([s.var() for s in sims]); top = np.mean([np.sort(s)[-25:].sum() / s.sum() for s in sims])
        print(f"{arm}: {name}: predicted misfire {q.mean():.3f}, repeat if independent {rep:.2f} (measured {obs:.2f}); "
              f"per-image var {v:.2f} (measured {cnt.var():.2f}); worst 5% share {top:.0%} (measured {np.sort(cnt)[-25:].sum() / cnt.sum():.0%})")
