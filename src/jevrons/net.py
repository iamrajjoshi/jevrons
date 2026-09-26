"""Small fully connected nets whose neurons are pluggable: exact step, or a Jev call.

Forward uses the neuron's real output. Backward is straight-through on the intended
sum: dL/dz = dL/da * sigmoid'(z / tau) / tau. SPSA is the derivative-free alternative.
"""

import pickle
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from jevrons.states import neuron_question, neuron_state

sig = lambda z: 1 / (1 + np.exp(-np.clip(z, -30, 30)))  # noqa: E731


class StepNeuron:
    """Exact local neuron: fires iff the intended sum is positive."""

    def __call__(self, X, W, b, tag=None):
        return (X @ W + b > 0).astype(float)


class JevNeuron:
    """Each (example, neuron) pair is one Jev call. Zero inputs are dropped from the state."""

    def __init__(self, jev, fmt="paired", threads=64):
        self.jev, self.fmt, self.threads = jev, fmt, threads

    def __call__(self, X, W, b, tag=None):
        jobs = [(i, j) for i in range(X.shape[0]) for j in range(W.shape[1])]

        def one(ij):
            i, j = ij
            keep = np.round(X[i], 2) != 0
            state, z = neuron_state(self.fmt, X[i][keep], W[keep, j], b[j])
            return self.jev.ask(state, neuron_question(self.fmt),
                                {**(tag or {}), "i": i, "j": j, "z": z})["fires"]

        with ThreadPoolExecutor(self.threads) as ex:
            out = list(ex.map(one, jobs))
        return np.array(out).reshape(X.shape[0], W.shape[1])


class ScalarJevNeuron:
    """Variant A: the sum is computed in code; Jev only answers whether z is positive."""

    def __init__(self, jev, threads=64):
        self.jev, self.threads = jev, threads

    def __call__(self, X, W, b, tag=None):
        from jevrons.states import SCALAR_Q_ALT, r2
        Z = X @ W + b
        jobs = [(i, j) for i in range(Z.shape[0]) for j in range(Z.shape[1])]

        def one(ij):
            i, j = ij
            return self.jev.ask({"z": r2(Z[i, j])}, SCALAR_Q_ALT, {**(tag or {}), "i": i, "j": j})["pos"]

        with ThreadPoolExecutor(self.threads) as ex:
            return np.array(list(ex.map(one, jobs))).reshape(Z.shape)


class MockJevNeuron:
    """Local stand-in for the full-state neuron, fitted to stage 1: fires on z + noise > 0 with
    noise sd 0.77 x the spread of the terms (the per-term products and bias), p in {0.03, 0.97}."""

    def __init__(self, seed=0, noise=0.77):
        self.rng, self.noise = np.random.default_rng(seed), noise

    def __call__(self, X, W, b, tag=None):
        z = X @ W + b
        spread = np.sqrt((X**2) @ (W**2) + b**2)
        return np.where(z + self.noise * spread * self.rng.standard_normal(z.shape) > 0, 0.97, 0.03)


def init(sizes, seed):
    rng = np.random.default_rng(seed)
    return [(rng.normal(0, 1, (a, c)), np.zeros(c)) for a, c in zip(sizes, sizes[1:])]


def forward(params, X, neuron, tag=None):
    acts, zs = [X], []
    for layer, (W, b) in enumerate(params):
        zs.append(acts[-1] @ W + b)
        acts.append(neuron(acts[-1], W, b, {**(tag or {}), "layer": layer}))
    return acts, zs


def bce(p, Y):
    p = np.clip(p, 1e-3, 1 - 1e-3)
    return float(np.mean(-Y * np.log(p) - (1 - Y) * np.log(1 - p)))


def ste_grads(params, acts, zs, Y, tau=1.0, output="bce"):
    """output="bce": exact dBCE/dp times the surrogate slope (stages 2-3).
    output="logit": dL/dz = (p - Y), as if p were sigmoid(z); stable when p is near 0 or 1."""
    p = np.clip(acts[-1], 1e-3, 1 - 1e-3)
    da = (p - Y) / (p * (1 - p)) / Y.size  # dBCE/dp
    grads = []
    for layer in reversed(range(len(params))):
        s = sig(zs[layer] / tau)
        dz = da * s * (1 - s) / tau
        if output == "logit" and layer == len(params) - 1:
            dz = (p - Y) / Y.size
        grads.append((acts[layer].T @ dz, dz.sum(0)))
        da = dz @ params[layer][0].T
    return grads[::-1]


class Adam:
    def __init__(self, params, lr):
        self.lr, self.t = lr, 0
        self.m = [np.zeros_like(q) for layer in params for q in layer]
        self.v = [np.zeros_like(q) for q in self.m]

    def step(self, params, grads):
        self.t += 1
        new = []
        flat = [q for layer in params for q in layer]
        for k, (q, g) in enumerate(zip(flat, (g for layer in grads for g in layer))):
            self.m[k] = 0.9 * self.m[k] + 0.1 * g
            self.v[k] = 0.999 * self.v[k] + 0.001 * g * g
            step = (self.m[k] / (1 - 0.9**self.t)) / (np.sqrt(self.v[k] / (1 - 0.999**self.t)) + 1e-8)
            new.append(np.clip(q - self.lr * step, -9.99, 9.99))  # weights stay two-digit on the wire
        return list(zip(new[::2], new[1::2]))


def train_ste(params, X, Y, neuron, steps, lr=0.1, tau=1.0, log=None):
    opt = Adam(params, lr)
    for s in range(steps):
        acts, zs = forward(params, X, neuron, {"step": s})
        if log is not None:
            log.append({"step": s, "loss": bce(acts[-1], Y), "p": acts[-1].ravel().tolist(),
                        "disagree": float(np.mean([np.mean((a > 0.5) != (z > 0)) for a, z in zip(acts[1:], zs)]))})
        params = opt.step(params, ste_grads(params, acts, zs, Y, tau))
    return params


def train_spsa(params, X, Y, neuron, steps, lr=0.1, c=0.3, seed=0, log=None):
    rng = np.random.default_rng(seed)
    for s in range(steps):
        deltas = [(rng.choice([-1.0, 1.0], W.shape), rng.choice([-1.0, 1.0], b.shape)) for W, b in params]
        plus = [(W + c * dW, b + c * db) for (W, b), (dW, db) in zip(params, deltas)]
        minus = [(W - c * dW, b - c * db) for (W, b), (dW, db) in zip(params, deltas)]
        lp = bce(forward(plus, X, neuron, {"step": s, "spsa": "+"})[0][-1], Y)
        lm = bce(forward(minus, X, neuron, {"step": s, "spsa": "-"})[0][-1], Y)
        g = (lp - lm) / (2 * c)
        params = [(np.clip(W - lr * g / dW, -9.99, 9.99), np.clip(b - lr * g / db, -9.99, 9.99))
                  for (W, b), (dW, db) in zip(params, deltas)]
        if log is not None:
            log.append({"step": s, "loss": (lp + lm) / 2})
    return params


if __name__ == "__main__":
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], float)
    Y = np.array([[0], [1], [1], [0]], float)
    solved = 0
    for seed in range(10):
        p = train_ste(init([2, 4, 1], seed), X, Y, StepNeuron(), 300)
        out = forward(p, X, StepNeuron())[0][-1]
        solved += bool(np.all(out == Y))
    print(f"local step XOR solved {solved}/10 seeds")
    assert solved >= 7


def fit(params, X, Y, neuron, epochs, batch=32, lr=0.01, tau=1.0, seed=0, log=None, tag=None, output="logit",
        on_epoch=None, checkpoint=None):
    """Minibatch straight-through training; one Jev forward pass per batch.

    on_epoch(epoch, params) runs after each epoch. With a checkpoint path, params, optimizer state
    and log are saved after every epoch, and a rerun resumes from the last finished epoch.
    """
    opt, rng = Adam(params, lr), np.random.default_rng(seed)
    step, start = 0, 0
    if checkpoint is not None and Path(checkpoint).exists():
        state = pickle.loads(Path(checkpoint).read_bytes())
        params, opt.m, opt.v, opt.t, step, start = (state[k] for k in ("params", "m", "v", "t", "step", "epoch"))
        if log is not None:
            log[:] = state["log"]
    for epoch in range(epochs):
        order = rng.permutation(len(X))  # drawn every epoch so a resumed run sees the same batches
        if epoch < start:
            continue
        for i in np.array_split(order, max(1, len(X) // batch)):
            acts, zs = forward(params, X[i], neuron, {**(tag or {}), "epoch": epoch, "step": step})
            if log is not None:
                log.append({"epoch": epoch, "step": step, "loss": bce(acts[-1], Y[i]),
                            "acc": float(np.mean(decide(acts[-1]) == (np.argmax(Y[i], 1) if Y.shape[1] > 1 else Y[i, 0])))})
            params = opt.step(params, ste_grads(params, acts, zs, Y[i], tau, output))
            step += 1
        if checkpoint is not None:
            Path(checkpoint).write_bytes(pickle.dumps({"params": params, "m": opt.m, "v": opt.v, "t": opt.t,
                                                       "step": step, "epoch": epoch + 1, "log": log or []}))
        if on_epoch is not None:
            on_epoch(epoch, params)
    return params


def decide(out, seed=0):
    """Class from output activations. Ties (common with near-binary outputs) break at random,
    never by the intended sum, which would leak exact arithmetic into the prediction."""
    if out.shape[1] == 1:
        return (out[:, 0] >= 0.5).astype(int)
    return np.argmax(out + 1e-6 * np.random.default_rng(seed).random(out.shape), 1)


def predict(params, X, neuron, tag=None):
    return decide(forward(params, X, neuron, tag)[0][-1])
