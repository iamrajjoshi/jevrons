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


class BackendDown(RuntimeError):
    """A backend kept failing after its retries; the router moves on to another one."""


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
    key_var: str | None  # None: no key needed (local server)
    per_min: float  # our pacing target, just under the backend's request limit
    model: str = MODEL  # model id we send
    served_model: str = MODEL  # model id the response must report
    provider: str | None = None  # upstream a gateway must route to (None: not a gateway)
    extra: dict = field(default_factory=dict)  # backend-specific request fields
    default: bool = True  # included when no backends are named; False for a different model
    usd_per_token: float = USD_PER_INPUT_TOKEN
    _next: float = field(default=0.0, repr=False)
    _cool_until: float = field(default=0.0, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    @property
    def key(self) -> str | None:
        return _secret(self.key_var) if self.key_var else "local"

    def next_slot(self) -> float:
        return max(self._next, self._cool_until, time.monotonic())

    def cool_off(self, seconds: float = 60):
        self._cool_until = time.monotonic() + seconds

    def reserve(self) -> float:
        """Claim the next request slot; return seconds to wait for it."""
        with self._lock:
            now = time.monotonic()
            slot = max(self._next, now)
            self._next = slot + 60 / self.per_min
            return slot - now

    def served(self, resp: dict) -> str:
        """Check who answered; raise if it isn't the pinned model and upstream."""
        routing = ((resp.get("provider_metadata") or {}).get("gateway") or {}).get("routing") or {}
        upstream = routing.get("finalProvider")
        if resp.get("model") != self.served_model or upstream != self.provider:
            raise RuntimeError(f"{self.name} served {resp.get('model')!r} via {upstream!r}, "
                               f"expected {self.served_model!r} via {self.provider!r}")
        return f"{resp['model']}@{upstream}" if upstream else resp["model"]

    def post(self, state, questions: dict) -> tuple[dict, int]:
        """One request with retries on 429, 5xx and network errors. Returns (body, attempts)."""
        body = json.dumps({"model": self.model, "state": state, "questions": questions, **self.extra}).encode()
        headers = {"Authorization": f"Bearer {self.key}", "Content-Type": "application/json"}
        errors = []
        for attempt in range(6):
            try:
                with urllib.request.urlopen(urllib.request.Request(self.endpoint, body, headers),
                                            timeout=120) as r:
                    return json.load(r), attempt + 1
            except urllib.error.HTTPError as e:
                detail = e.read()[:300].decode(errors="replace")
                if e.code in (401, 402, 403):  # key or credits problem: let another backend carry on
                    raise BackendDown(f"{self.name}: HTTP {e.code} {detail}") from None
                if e.code != 429 and e.code < 500:
                    raise RuntimeError(f"{self.name} HTTP {e.code}: {detail}") from None
                errors.append(str(e.code))
            except (urllib.error.URLError, TimeoutError) as e:
                errors.append(type(e).__name__)
            time.sleep(min(30, 2**attempt))
        raise BackendDown(f"{self.name}: retries exhausted ({', '.join(errors)})")


BACKENDS = {
    "typesafe": Backend("typesafe", "https://api.typesafe.ai/v1/systemone", "TYPESAFE_API_KEY", 1100),
    # TypeSafe-compatible gateway path: same request and response shape as TypeSafe's API.
    # Only the unversioned alias exists here; pin the upstream and journal what served. Measured
    # 2026-09-26: same neuron as direct (backend_check), clean at ~5,900/min in a 1,500-call burst, but ~20% retries sustained at 5,000/min; run at 3,000.
    "vercel": Backend("vercel", "https://ai-gateway.vercel.sh/typesafe/v1/systemone", "VERCEL_API_KEY", 3000,
                      model="typesafe-ai/jev", served_model="typesafe-ai/jev", provider="typesafe-ai",
                      extra={"providerOptions": {"gateway": {"only": ["typesafe-ai"]}}}),
    # Open-weight System-1 model served locally by Ollaya (https://ollaya.dev), same API. A different
    # neuron from Jev, so it is never picked by default: name it with backends=["ollaya"].
    "ollaya": Backend("ollaya", "http://127.0.0.1:11435/v1/systemone", None, 60000,
                      model="laya:en", served_model="laya:en", default=False, usd_per_token=0.0),
}


def noul(instructions: str, true: str, false: str) -> dict:
    return {"type": "noul", "instructions": instructions, "criteria": {"true": true, "false": false}}


class Jev:
    def __init__(self, journal: Path, max_usd: float | None = None, backends: list[str] | None = None):
        names = backends or [n for n in os.environ.get("JEVRONS_BACKENDS", "").split(",") if n]
        self.backends = [BACKENDS[n] for n in names] if names else [b for b in BACKENDS.values() if b.default and b.key]
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

    def _log_failure(self, backend: str, error: str, tag: dict | None):
        with self._lock, self.journal.with_name(self.journal.stem + "-failures.jsonl").open("a") as f:
            f.write(json.dumps({"t": time.time(), "backend": backend, "error": error, "tag": tag}) + "\n")

    def ask(self, state, questions: dict, tag: dict | None = None) -> dict[str, float]:
        """Send one request to whichever backend frees up first; return {question: P(yes)}."""
        if self.max_usd is not None and self.usd >= self.max_usd:
            raise RuntimeError(f"spend cap ${self.max_usd} reached")
        failures = []
        for backend in sorted(self.backends, key=Backend.next_slot):
            if (wait := backend.reserve()) > 0:
                time.sleep(wait)
            t0 = time.monotonic()
            try:
                resp, attempts = backend.post(state, questions)
                break
            except BackendDown as e:  # cool it off and let the next backend take this call
                backend.cool_off()
                failures.append(str(e))
                self._log_failure(backend.name, str(e), tag)
        else:
            raise RuntimeError(f"every backend failed: {failures}")
        latency = time.monotonic() - t0
        served = backend.served(resp)
        answers = {name: a["noul"] for name, a in resp["answers"].items()}
        usage = resp.get("usage") or {}
        tokens = usage.get("input_tokens", 0)
        record = {"t": time.time(), "backend": backend.name, "served": served, "tag": tag,
                  "failed_over_from": [f.split(":")[0] for f in failures] or None, "state": state,
                  "questions": questions, "answers": answers, "usage": usage,
                  "latency_s": round(latency, 3), "attempts": attempts}
        with self._lock:
            self.calls += 1
            self.tokens += tokens
            self.usd += tokens * backend.usd_per_token
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
