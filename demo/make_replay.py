"""Cut the demo's replay and mock data out of the stage 7 and 8a training journals (gitignored, GBs).

  uv run python demo/make_replay.py

For each cut it writes, into demo/runs/:
  <stage>-<arm>-replay.jsonl  every call of the test pass for the first PER_DIGIT test images of each digit, keyed
                              like the replay index (sha1 of state + questions), so replay on those digits shows
                              Jev's recorded answers call for call
  <stage>-mock.json           recorded (noul, choice, latency) samples binned by layer and z / spread, which the
                              mock resamples from
"""

import json
import math
import random
from collections import defaultdict

import numpy as np

from jevrons.digits import _load, test_subset
from server import DEMO, ROOT, _key

PER_DIGIT, BINS, PER_BIN = 12, np.linspace(-4, 4, 33), 80
# stage, its test images (as in its training script), the arms to cut replays for, and the file names they get
CUTS = [("stage7", dict(n=2000, seed=7), {"jev": "stage7-replay.jsonl"}),
        ("stage8a", dict(n=1000, seed=0), {"jev": "stage8a-jev-replay.jsonl", "swap": "stage8a-swap-replay.jsonl"})]


def cut(stage, test, arms):
    te = test_subset(_load("t10k-labels-idx1-ubyte.gz", 8).astype(int), test["n"], seed=test["seed"])
    per = test["n"] // 10
    keep = {int(i) for d in range(10) for i in range(d * per, d * per + PER_DIGIT)}
    samples, replay = defaultdict(list), defaultdict(list)
    with (ROOT / f"runs/{stage}/journal.jsonl").open() as f:
        for line in f:
            if '"split": "test"' not in line:
                continue
            r = json.loads(line)
            tag, p = r["tag"], r["state"]["products"]
            spread = math.sqrt(sum(v * v for v in p)) + 1e-6
            a = r["answers"]
            if stage == "stage8a" or tag["arm"] == "jev":  # stage 7's mock was cut from the jev arm only
                samples[f"{tag['layer']}:{int(np.digitize(sum(p) / spread, BINS))}"].append([a["v1"], a["choice"]["above"], r["latency_s"]])
            if tag["arm"] in arms and tag["i"] in keep:
                replay[tag["arm"]].append({"key": _key(r["state"], r["questions"]), "mnist": int(te[tag["i"]]), "answers": a,
                                           "usage": r["usage"], "latency_s": r["latency_s"], "served": r["served"]})
    rng = random.Random(0)
    table = {k: rng.sample(v, min(PER_BIN, len(v))) for k, v in samples.items()}
    (DEMO / f"runs/{stage}-mock.json").write_text(json.dumps({"bins": BINS.tolist(), "samples": table}))
    for arm, name in arms.items():
        with (DEMO / "runs" / name).open("w") as f:
            f.writelines(json.dumps(r) + "\n" for r in replay[arm])
        print(f"{stage} {arm}: {len(replay[arm])} replay calls for {len(keep)} images -> demo/runs/{name}")


if __name__ == "__main__":
    for c in CUTS:
        cut(*c)
