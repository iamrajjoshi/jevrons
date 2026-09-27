"""Step 5 check: the served model through the normal Jevrons client path, Jev(..., backends=[name]).

Sends the margin probe's 1,800 states (10/30/75 terms) and the scalar sweep through the Jev client to
the Ollaya-served model, fits the same curves as evaluate.py, and compares every answer with the
offline PyTorch answer for the same weights. Journal: runs/stage10/client/journal.jsonl.
Usage: uv run python client_check.py ollaya-laya-neuron truth
"""

import json
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch

import data
import laya_model as L
from evaluate import fit_curve
from jevrons.jev import Jev
from jevrons.states import SCALAR_Q

OUT = data.ROOT / "runs/stage10/client"


def main(backend, run):
    jev = Jev(OUT / "journal.jsonl", backends=[backend])
    _, test = data.real_states()
    probe = [r for r in test if r["src"] == "margin"]
    states = [{"products": r["products"]} for r in probe]
    zs = [round(v, 1) for v in np.arange(-10, 10.0001, 0.1)]
    with ThreadPoolExecutor(4) as ex:
        served = list(ex.map(lambda s: jev.ask(s, {"fires": data.QUESTION}, {"part": "margin-states"})["fires"], states))
        served_scalar = list(ex.map(lambda z: jev.ask({"z": z}, SCALAR_Q, {"part": "scalar", "z": z})["pos"], zs))
    tok, model = L.tokenizer(), L.load(data.ROOT / f"runs/stage10/{run}/model.safetensors")
    with torch.no_grad():
        offline = L.p_yes(model, tok, states, data.QUESTION)
        offline_scalar = L.p_yes(model, tok, [{"z": z} for z in zs], SCALAR_Q["pos"])
    served, served_scalar = np.array(served), np.array(served_scalar)
    res = {"backend": backend, "run": run, "calls": jev.calls,
           "max_abs_diff_vs_offline": float(max(np.abs(served - offline).max(), np.abs(served_scalar - offline_scalar).max())),
           "yes_no_agreement_vs_offline": float(np.mean(np.r_[(served >= 0.5) == (offline >= 0.5),
                                                               (served_scalar >= 0.5) == (offline_scalar >= 0.5)])),
           "scalar_sign_accuracy": float(np.mean((served_scalar >= 0.5) == (np.array(zs) > 0))), "curves": {}}
    lat = [json.loads(line)["latency_s"] for line in (OUT / "journal.jsonl").open()]
    res["median_latency_s"] = float(np.median(lat))
    for k in (10, 30, 75):
        sel = np.array([len(r["products"]) - 1 == k for r in probe])
        m = np.array([data.margin(r["products"]) for r in probe])[sel]
        res["curves"][k] = {kk: v for kk, v in fit_curve(m, (served[sel] >= 0.5).astype(float)).items()}
    (OUT / "summary.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main(*sys.argv[1:3])
