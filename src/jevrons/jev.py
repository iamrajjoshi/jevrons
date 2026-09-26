"""Minimal Jev client: one POST per call, every call journaled as JSONL, spend tracked."""

import json
import os
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

ENDPOINT = "https://api.typesafe.ai/v1/systemone"
MODEL = "jev-1.13.0"
USD_PER_INPUT_TOKEN = 0.042 / 1_000_000
CREDENTIALS = Path.home() / ".config/jev-research/credentials.env"
MIN_INTERVAL_S = 60 / 1100  # stay just under 1,200 requests/min


def _key() -> str:
    if key := os.environ.get("TYPESAFE_API_KEY"):
        return key
    if CREDENTIALS.exists():
        for line in CREDENTIALS.read_text().splitlines():
            if line.startswith("TYPESAFE_API_KEY="):
                return line.split("=", 1)[1].strip().strip("'\"")
    raise RuntimeError(f"set TYPESAFE_API_KEY or add it to {CREDENTIALS}")


def noul(instructions: str, true: str, false: str) -> dict:
    return {"type": "noul", "instructions": instructions, "criteria": {"true": true, "false": false}}


class Jev:
    def __init__(self, journal: Path, max_usd: float | None = None):
        self.key = _key()
        self.journal = journal
        self.max_usd = max_usd
        self.usd = 0.0
        self.tokens = 0
        self.calls = 0
        self._lock = threading.Lock()
        self._next = 0.0
        journal.parent.mkdir(parents=True, exist_ok=True)

    def ask(self, state, questions: dict, tag: dict | None = None) -> dict[str, float]:
        """Send one request; return {question name: P(yes)}."""
        if self.max_usd is not None and self.usd >= self.max_usd:
            raise RuntimeError(f"spend cap ${self.max_usd} reached")
        body = json.dumps({"model": MODEL, "state": state, "questions": questions}).encode()
        headers = {"Authorization": f"Bearer {self.key}", "Content-Type": "application/json"}
        with self._lock:  # pace requests under the 1,200/min limit
            wait = self._next - time.monotonic()
            self._next = max(self._next, time.monotonic()) + MIN_INTERVAL_S
        if wait > 0:
            time.sleep(wait)
        t0 = time.monotonic()
        for attempt in range(7):
            try:
                req = urllib.request.Request(ENDPOINT, body, headers)
                with urllib.request.urlopen(req, timeout=180) as r:
                    resp = json.load(r)
                break
            except urllib.error.HTTPError as e:
                if e.code != 429 and e.code < 500:  # 429 and any 5xx (incl. Cloudflare 52x) retry
                    detail = e.read()[:500].decode(errors="replace")
                    raise RuntimeError(f"Jev HTTP {e.code}: {detail}") from None
            except (urllib.error.URLError, TimeoutError):
                pass
            time.sleep(min(60, 2**attempt))
        else:
            raise RuntimeError("Jev: retries exhausted")
        latency = time.monotonic() - t0
        if resp.get("model") != MODEL:
            raise RuntimeError(f"Jev returned model {resp.get('model')!r}, expected {MODEL}")
        answers = {name: a["noul"] for name, a in resp["answers"].items()}
        usage = resp.get("usage") or {}
        tokens = usage.get("input_tokens", 0)
        record = {"t": time.time(), "tag": tag, "state": state, "questions": questions,
                  "answers": answers, "usage": usage, "latency_s": round(latency, 3),
                  "attempts": attempt + 1}
        with self._lock:
            self.calls += 1
            self.tokens += tokens
            self.usd += tokens * USD_PER_INPUT_TOKEN
            with self.journal.open("a") as f:
                f.write(json.dumps(record) + "\n")
        return answers
