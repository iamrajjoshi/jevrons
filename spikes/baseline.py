"""Local capacity check for the Jev-neuron proposal: 784->H->10, ten sigmoid outputs, mean BCE."""
import gzip
from pathlib import Path

import numpy as np

DATA = Path(__file__).resolve().parents[1] / "data"

def load(name, off):
    return np.frombuffer(gzip.open(DATA / name).read(), np.uint8, offset=off)

Xtr = load("train-images-idx3-ubyte.gz", 16).reshape(-1, 784) / 255.0
ytr = load("train-labels-idx1-ubyte.gz", 8)
Xte = load("t10k-images-idx3-ubyte.gz", 16).reshape(-1, 784) / 255.0
yte = load("t10k-labels-idx1-ubyte.gz", 8)

def stratified(rng, n_per, exclude=None):
    idx = []
    for d in range(10):
        pool = np.where(ytr == d)[0]
        if exclude is not None:
            pool = np.setdiff1d(pool, exclude)
        idx.append(rng.choice(pool, n_per, replace=False))
    return np.concatenate(idx)

sig = lambda z: 1 / (1 + np.exp(-np.clip(z, -30, 30)))

def fwd(W1, b1, W2, b2, X, hidden, out):
    z1 = X @ W1 + b1
    h = sig(z1) if hidden == "soft" else (z1 > 0).astype(float)
    z2 = h @ W2 + b2
    p = sig(z2) if out == "soft" else (z2 > 0).astype(float)
    return z1, h, z2, p

def train(X, y, H, hidden, out, epochs, seed, Xv, yv, bs=32, lr=1e-2):
    rng = np.random.default_rng(seed)
    P = [rng.normal(0, 1 / np.sqrt(784), (784, H)), np.zeros(H),
         rng.normal(0, 1 / np.sqrt(H), (H, 10)), np.zeros(10)]
    m = [np.zeros_like(p) for p in P]; v = [np.zeros_like(p) for p in P]; t = 0
    Y = np.eye(10)[y]
    for _ in range(epochs):
        for i in np.array_split(rng.permutation(len(X)), max(1, len(X) // bs)):
            z1, h, z2, p = fwd(*P, X[i], hidden, out)
            ps = sig(z2)  # surrogate: sigmoid slope for backward (straight-through when hard)
            dz2 = (ps - Y[i]) / (10 * len(i))  # mean BCE over 10 outputs
            dh = dz2 @ P[2].T
            dz1 = dh * sig(z1) * (1 - sig(z1))
            g = [X[i].T @ dz1, dz1.sum(0), h.T @ dz2, dz2.sum(0)]
            t += 1
            for k in range(4):
                m[k] = 0.9 * m[k] + 0.1 * g[k]; v[k] = 0.999 * v[k] + 0.001 * g[k] ** 2
                P[k] -= lr * (m[k] / (1 - 0.9 ** t)) / (np.sqrt(v[k] / (1 - 0.999 ** t)) + 1e-8)
    return P

def acc(P, X, y, hidden, out):
    _, _, z2, p = fwd(*P, X, hidden, out)
    # hard outputs tie often; break ties with z2 so argmax is defined
    return (np.argmax(p + 1e-6 * sig(z2), 1) == y).mean()

rng = np.random.default_rng(0)
tr = stratified(rng, 100)
va = stratified(rng, 100, exclude=tr)
nnz = (Xtr > 0).sum(1)
print(f"nonzero pixels/image: mean {nnz.mean():.0f}, p95 {np.percentile(nnz, 95):.0f}, max {nnz.max()}")

configs = [("soft", "soft"), ("hard", "soft"), ("hard", "hard")]
for n_ep in (10, 50):
    for H in (32, 64, 128):
        for hid, out in configs:
            a, swap = [], []
            for s in range(3):
                P = train(Xtr[tr], ytr[tr], H, hid, out, n_ep, s, None, None)
                a.append(acc(P, Xtr[va], ytr[va], hid, out))
                if hid == "soft":
                    swap.append(acc(P, Xtr[va], ytr[va], "hard", "hard"))
            line = f"1k train ep={n_ep:<3} H={H:<4} hidden={hid:<5} out={out:<5} val acc {np.mean(a):.3f} ±{np.std(a):.3f}"
            if swap:
                line += f" | swap->step {np.mean(swap):.3f}"
            print(line, flush=True)

# reference: 50k train, 10 epochs
full = np.setdiff1d(np.arange(60000), va)[:50000]
for hid, out in configs:
    P = train(Xtr[full], ytr[full], 32, hid, out, 10, 0, None, None)
    z1 = Xtr[va] @ P[0] + P[1]
    print(f"50k train H=32 hidden={hid:<5} out={out:<5} val {acc(P, Xtr[va], ytr[va], hid, out):.3f} test {acc(P, Xte, yte, hid, out):.3f} "
          f"| z1 range p1..p99 {np.percentile(z1,1):.1f}..{np.percentile(z1,99):.1f}, |z1|<0.05 frac {(np.abs(z1)<0.05).mean():.3f}", flush=True)
