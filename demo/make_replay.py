"""Cut the demo's stage 7 data out of the 3 GB training journal (runs/stage7/journal.jsonl, gitignored).

  uv run python demo/make_replay.py

Writes two small files the server reads:
  demo/runs/stage7-replay.jsonl  every call of the test pass for the first PER_DIGIT test images of each digit,
                                 keyed like the replay index (sha1 of state + questions), so "replay" on
                                 those digits shows Jev's recorded answers
  demo/runs/stage7-mock.json     recorded (noul, choice, latency) samples binned by layer and
                                 z / spread, which the stage 7 mock resamples from
"""

import json
import math
import random
from collections import defaultdict

import numpy as np

from jevrons.digits import _load, test_subset
from server import DEMO, ROOT, _key

JOURNAL = ROOT / "runs/stage7/journal.jsonl"
PER_DIGIT, BINS, PER_BIN = 12, np.linspace(-4, 4, 33), 80


def main():
    te = test_subset(_load("t10k-labels-idx1-ubyte.gz", 8).astype(int), 2000, seed=7)  # stage 7's test images
    keep = {int(i) for d in range(10) for i in range(d * 200, d * 200 + PER_DIGIT)}
    samples = defaultdict(list)
    replay = []
    with JOURNAL.open() as f:
        for line in f:
            if '"split": "test"' not in line or '"arm": "jev"' not in line:
                continue
            r = json.loads(line)
            tag, p = r["tag"], r["state"]["products"]
            spread = math.sqrt(sum(v * v for v in p)) + 1e-6
            b = int(np.digitize(sum(p) / spread, BINS))
            a = r["answers"]
            samples[f"{tag['layer']}:{b}"].append([a["v1"], a["choice"]["above"], r["latency_s"]])
            if tag["i"] in keep:
                replay.append({"key": _key(r["state"], r["questions"]), "mnist": int(te[tag["i"]]), "answers": a,
                               "usage": r["usage"], "latency_s": r["latency_s"], "served": r["served"]})
    rng = random.Random(0)
    table = {k: rng.sample(v, min(PER_BIN, len(v))) for k, v in samples.items()}
    (DEMO / "runs/stage7-mock.json").write_text(json.dumps({"bins": BINS.tolist(), "samples": table}))
    with (DEMO / "runs/stage7-replay.jsonl").open("w") as f:
        f.writelines(json.dumps(r) + "\n" for r in replay)
    print(f"{len(replay)} replay calls for {len(keep)} images")


if __name__ == "__main__":
    main()
