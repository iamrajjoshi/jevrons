"""Fine-tune laya end to end on the folded neuron question.

Arms:
  truth  synthetic states (70%) + real journal states (30%), label = true sign of the total
  jev    real journal states only, soft label = Jev's recorded mean p ("copy Jev")
Loss is cross-entropy on softmax(logits / T) with laya's own noul temperature T, so the probability
Ollaya reports after calibration is the one trained. bf16 autocast on MPS, gradient checkpointing.
Usage: uv run python train.py truth --steps 3000 --name truth
"""

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file

import data
import laya_model as L

OUT = data.ROOT / "runs/stage10"


def make_items(tok, recs):
    """recs: [(state, target_p)] -> encoded items with soft targets; drops states that don't fit."""
    out = []
    for state, y in recs:
        it = L.encode(tok, state, data.QUESTION)
        if it:
            it["target"] = [1.0 - y, y]
            out.append(it)
    return out


def sampler(arm, rng, real_train):
    def one():
        if arm == "truth" and rng.random() < 0.7:
            s, z = data.synth(rng, int(rng.choice(data.TRAIN_K)))
            return s, float(z > 0)
        r = real_train[rng.integers(len(real_train))]
        return {"products": r["products"]}, (float(r["z"] > 0) if arm == "truth" else r["p_jev"])
    return one


def val_set(tok, real_test):
    """Fixed validation: synthetic at held-out counts 10, 30, 75 plus a slice of held-out real states."""
    rng = np.random.default_rng(1234)
    recs = [(s, float(z > 0)) for k in (10, 30, 75) for s, z in (data.synth(rng, k, family="probe") for _ in range(100))]
    sel = rng.choice(len(real_test), 300, replace=False)
    recs += [({"products": real_test[i]["products"]}, float(real_test[i]["z"] > 0)) for i in sel]
    tags = ["s10"] * 100 + ["s30"] * 100 + ["s75"] * 100 + ["real"] * 300
    items = make_items(tok, recs)
    assert len(items) == len(recs)
    return items, np.array(tags), np.array([r[1] for r in recs]), np.array([real_test[i]["p_jev"] for i in sel])


@torch.no_grad()
def predict(model, tok, items, bs=48):
    order = sorted(range(len(items)), key=lambda i: len(items[i]["ids"]))
    p = np.zeros(len(items))
    for s in range(0, len(order), bs):
        idx = order[s:s + bs]
        with torch.autocast("mps", dtype=torch.bfloat16):
            lg = L.logits(model, [items[i] for i in idx], tok.pad_token_id).float() / L.NOUL_T
        p[idx] = torch.softmax(lg, -1)[:, 1].cpu().numpy()
    return p


def save(model, path: Path):
    """Same keys and dtypes as laya's original file (F16, temperature F32), so Ollaya can load it."""
    ref = load_file(str(L.HF / "model.safetensors"))
    sd = {k: v.detach().to("cpu", ref[k].dtype).contiguous() for k, v in model.state_dict().items()}
    save_file(sd, str(path))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("arm", choices=["truth", "jev"])
    ap.add_argument("--name", required=True)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    out = OUT / args.name
    out.mkdir(parents=True, exist_ok=True)
    (out / "args.json").write_text(json.dumps(vars(args), indent=1))
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    tok = L.tokenizer()
    real_train, real_test = data.real_states()
    print(f"real states: {len(real_train)} train, {len(real_test)} test", flush=True)
    val_items, val_tags, val_y, val_pjev = val_set(tok, real_test)
    model = L.load()
    model.encoder.gradient_checkpointing_enable()
    model.train()
    params = [p for n, p in model.named_parameters() if not n.startswith("act_head")]
    for n, p in model.named_parameters():
        p.requires_grad_(not n.startswith("act_head"))
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / args.warmup) * (
        0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1, s / args.steps)))))
    draw = sampler(args.arm, rng, real_train)
    queue, log, t0, peak = [], (out / "log.jsonl").open("w"), time.time(), 0
    dev = L.device()

    def evaluate(step):
        model.eval()
        p = predict(model, tok, val_items)
        model.train()
        rec = {"step": step, "elapsed_s": round(time.time() - t0)}
        for tag in ("s10", "s30", "s75", "real"):
            sel = val_tags == tag
            rec[f"acc_{tag}"] = float(np.mean((p[sel] >= 0.5) == (val_y[sel] > 0.5)))
        rec["real_mae_vs_jev"] = float(np.mean(np.abs(p[val_tags == "real"] - val_pjev)))
        print("eval", rec, flush=True)
        log.write(json.dumps({"eval": rec}) + "\n")
        log.flush()

    evaluate(0)
    for step in range(1, args.steps + 1):
        if not queue:  # length-bucketed batches: draw 8 batches' worth, sort by length, split
            items = make_items(tok, [draw() for _ in range(args.bs * 8)])
            items.sort(key=lambda it: len(it["ids"]))
            queue = [items[i:i + args.bs] for i in range(0, len(items), args.bs)]
            rng.shuffle(queue)
        batch = queue.pop()
        target = torch.tensor([it["target"] for it in batch], device=dev)
        with torch.autocast("mps", dtype=torch.bfloat16):
            lg = L.logits(model, batch, tok.pad_token_id)
        logp = F.log_softmax(lg.float() / L.NOUL_T, -1)
        loss = -(target * logp).sum(-1).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        sched.step()
        opt.zero_grad(set_to_none=True)
        peak = max(peak, torch.mps.driver_allocated_memory())
        acc = float(((logp[:, 1].exp() >= 0.5) == (target[:, 1] >= 0.5)).float().mean())
        log.write(json.dumps({"step": step, "loss": round(float(loss), 5), "acc": acc, "lr": sched.get_last_lr()[0],
                              "len": len(batch[-1]["ids"])}) + "\n")
        if step % 50 == 0:
            print(f"step {step} loss {float(loss):.4f} {time.time() - t0:.0f}s peak {peak / 1e9:.1f} GB", flush=True)
        if step % args.eval_every == 0 or step == args.steps:
            evaluate(step)
    save(model, out / "model.safetensors")
    summary = {"steps": args.steps, "train_s": round(time.time() - t0), "peak_mps_gb": round(peak / 1e9, 2)}
    (out / "train_summary.json").write_text(json.dumps(summary, indent=1))
    print(summary, flush=True)


if __name__ == "__main__":
    main()
