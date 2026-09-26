"""Jev client: calls are spread across backends, paced per backend, and journaled as JSONL.

A backend is one place that serves Jev (TypeSafe directly, Vercel AI Gateway, ...). To add or
replace one, add a `Backend` to BACKENDS; nothing else changes. Pick backends with the
JEVRONS_BACKENDS env var (comma-separated names); default is every backend with a key.
"""

import json
import os
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

MODEL = "jev-1.13.0"
USD_PER_INPUT_TOKEN = 0.042 / 1_000_000
CREDENTIALS = Path.home() / ".config/jev-research/credentials.env"


def _secret(var: str) -> str | None:
    if value := os.environ.get(var):
        return value
    if CREDENTIALS.exists():
        for line in CREDENTIALS.read_text().splitlines():
            if line.startswith(f"{var}="):
                return line.split("=", 1)[1].strip().strip("'\"") or None
    return None


@dataclass
class Backend:
    name: str
    endpoint: str
    key_var: str
    per_min: float  # our pacing target, just under the backend's request limit
    model: str = MODEL
    _next: float = field(default=0.0, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    @property
    def key(self) -> str | None:
        return _secret(self.key_var)

    def next_slot(self) -> float:
        return max(self._next, time.monotonic())

    def reserve(self) -> float:
        """Claim the next request slot; return seconds to wait for it."""
        with self._lock:
            now = time.monotonic()
            slot = max(self._next, now)
            self._next = slot + 60 / self.per_min
            return slot - now

    def post(self, state, questions: dict) -> tuple[dict, int]:
        """One request with retries on 429, 5xx and network errors. Returns (body, attempts)."""
        body = json.dumps({"model": self.model, "state": state, "questions": questions}).encode()
        headers = {"Authorization": f"Bearer {self.key}", "Content-Type": "application/json"}
        for attempt in range(7):
            try:
                with urllib.request.urlopen(urllib.request.Request(self.endpoint, body, headers),
                                            timeout=180) as r:
                    return json.load(r), attempt + 1
            except urllib.error.HTTPError as e:
                if e.code != 429 and e.code < 500:
                    detail = e.read()[:500].decode(errors="replace")
                    raise RuntimeError(f"{self.name} HTTP {e.code}: {detail}") from None
            except (urllib.error.URLError, TimeoutError):
                pass
            time.sleep(min(60, 2**attempt))
        raise RuntimeError(f"{self.name}: retries exhausted")


BACKENDS = {
    "typesafe": Backend("typesafe", "https://api.typesafe.ai/v1/systemone", "TYPESAFE_API_KEY", 1100),
    # TypeSafe-compatible gateway path: same request and response shape as TypeSafe's API.
    "vercel": Backend("vercel", "https://ai-gateway.vercel.sh/typesafe/v1/systemone", "AI_GATEWAY_API_KEY", 1100),
}


def noul(instructions: str, true: str, false: str) -> dict:
    return {"type": "noul", "instructions": instructions, "criteria": {"true": true, "false": false}}


class Jev:
    def __init__(self, journal: Path, max_usd: float | None = None, backends: list[str] | None = None):
        names = backends or [n for n in os.environ.get("JEVRONS_BACKENDS", "").split(",") if n]
        self.backends = [BACKENDS[n] for n in names] if names else [b for b in BACKENDS.values() if b.key]
        missing = [b.name for b in self.backends if not b.key]
        if missing or not self.backends:
            raise RuntimeError(f"no key for {missing or 'any backend'}; set it in the env or {CREDENTIALS}")
        self.journal = journal
        self.max_usd = max_usd
        self.usd = 0.0
        self.tokens = 0
        self.calls = 0
        self.by_backend: dict[str, int] = {}
        self._lock = threading.Lock()
        journal.parent.mkdir(parents=True, exist_ok=True)

    def ask(self, state, questions: dict, tag: dict | None = None) -> dict[str, float]:
        """Send one request to whichever backend frees up first; return {question: P(yes)}."""
        if self.max_usd is not None and self.usd >= self.max_usd:
            raise RuntimeError(f"spend cap ${self.max_usd} reached")
        backend = min(self.backends, key=Backend.next_slot)
        if (wait := backend.reserve()) > 0:
            time.sleep(wait)
        t0 = time.monotonic()
        resp, attempts = backend.post(state, questions)
        latency = time.monotonic() - t0
        if resp.get("model") != MODEL:
            raise RuntimeError(f"{backend.name} served model {resp.get('model')!r}, expected {MODEL}")
        answers = {name: a["noul"] for name, a in resp["answers"].items()}
        usage = resp.get("usage") or {}
        tokens = usage.get("input_tokens", 0)
        record = {"t": time.time(), "backend": backend.name, "tag": tag, "state": state,
                  "questions": questions, "answers": answers, "usage": usage,
                  "latency_s": round(latency, 3), "attempts": attempts}
        with self._lock:
            self.calls += 1
            self.tokens += tokens
            self.usd += tokens * USD_PER_INPUT_TOKEN
            self.by_backend[backend.name] = self.by_backend.get(backend.name, 0) + 1
            with self.journal.open("a") as f:
                f.write(json.dumps(record) + "\n")
        return answers


if __name__ == "__main__":  # smoke test: one call per backend that has a key
    q = {"pos": noul("Is the number z positive?", "Yes, z is positive.", "No, z is zero or negative.")}
    for name, b in BACKENDS.items():
        if not b.key:
            print(f"{name}: no key ({b.key_var})")
            continue
        jev = Jev(Path("runs/smoke/journal.jsonl"), backends=[name])
        print(name, jev.ask({"z": -3.02}, q, {"part": "smoke"}), f"{jev.tokens} tokens")
