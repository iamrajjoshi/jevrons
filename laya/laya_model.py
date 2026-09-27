"""Load laya (ModernBERT-large + laya decision head) and score noul questions exactly as laya's own
rl_agent_api.py does. The official code lives in hf/ (downloaded from the HF repo at the pinned revision);
the weights and tokenizer are symlinks to the identical blobs Ollaya already stores."""

import json
import sys
from pathlib import Path

import numpy as np
import torch

HF = Path(__file__).resolve().parent / "hf"
sys.path.insert(0, str(HF))
from rl_common import QTYPES, build_model, build_sequence, collate_items  # noqa: E402

CFG = json.loads((HF / "rl_agent_config.json").read_text())
NOUL_T = CFG["temperature_by_options"]["noul:2"]  # 1.983; Ollaya applies the same calibration
MAX_LEN, HEAD_MAX = CFG["max_len"], CFG["head_max_len"]


def device():
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def tokenizer():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(HF / "tokenizer")


def load(weights: Path | None = None, dev=None, dtype=torch.float32):
    from safetensors.torch import load_file
    model = build_model(CFG, encoder_dir=str(HF / "encoder"))
    sd = load_file(str(weights or HF / "model.safetensors"))
    model.load_state_dict({k: v.float() for k, v in sd.items()}, strict=True)
    return model.to(dev or device(), dtype).eval()


def encode(tok, state, question: dict):
    """question: one Jev noul question dict. Returns an item for collate_items, or None if the state doesn't fit."""
    q = {"t": "noul", "ins": question["instructions"], "crit": question.get("criteria")}
    ids, markers = build_sequence(tok, state, q, 10**6, HEAD_MAX)  # uncapped, so truncation is detectable
    if len(markers) != 2 or len(ids) > MAX_LEN:  # Ollaya would silently cut the state; we skip it instead
        return None
    return {"ids": ids, "markers": markers, "qtype": QTYPES["noul"], "target": [0.0, 0.0], "label": -1,
            "episode": 0, "ep_step": 0, "ep_len": 1, "src": ""}


def n_state_tokens(tok, state) -> int:
    return len(tok(json.dumps(state, ensure_ascii=False), add_special_tokens=False)["input_ids"])


def logits(model, batch_items, pad_id):
    b = collate_items([batch_items], pad_id)
    dev = next(model.parameters()).device
    lg, _ = model(*(b[k].to(dev) for k in ("input_ids", "attention_mask", "marker_pos", "marker_mask", "qtype")))
    return lg[:, :2]  # [false, true]


@torch.no_grad()
def p_yes(model, tok, states, question, batch=32, temperature=NOUL_T) -> np.ndarray:
    """Calibrated P(true) per state, as Ollaya's noul answer."""
    items = [encode(tok, s, question) for s in states]
    out = np.full(len(items), np.nan)
    order = sorted((i for i, it in enumerate(items) if it), key=lambda i: len(items[i]["ids"]))
    for s in range(0, len(order), batch):
        idx = order[s:s + batch]
        lg = logits(model, [items[i] for i in idx], tok.pad_token_id).float() / temperature
        out[idx] = torch.softmax(lg, -1)[:, 1].cpu().numpy()
    return out
