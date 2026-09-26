"""Small fully connected nets whose neurons are pluggable: exact step, or a Jev call.

Forward uses the neuron's real output. Backward is straight-through on the intended
sum: dL/dz = dL/da * sigmoid'(z / tau) / tau. SPSA is the derivative-free alternative.
"""

from concurrent.futures import ThreadPoolExecutor

import numpy as np

from jevrons.states import neuron_question, neuron_state

sig = lambda z: 1 / (1 + np.exp(-np.clip(z, -30, 30)))  # noqa: E731


class StepNeuron:
    """Exact local neuron: fires iff the intended sum is positive."""

    def __call__(self, X, W, b, tag=None):
        return (X @ W + b > 0).astype(float)


class JevNeuron:
    """Each (example, neuron) pair is one Jev call. Zero inputs are dropped from the state."""

    def __init__(self, jev, fmt="paired", threads=16):
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


def ste_grads(params, acts, zs, Y, tau=1.0):
    p = np.clip(acts[-1], 1e-3, 1 - 1e-3)
    da = (p - Y) / (p * (1 - p)) / Y.size  # dBCE/dp
    grads = []
    for layer in reversed(range(len(params))):
        s = sig(zs[layer] / tau)
        dz = da * s * (1 - s) / tau
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
