"""MNIST loading and the frozen splits used from stage 4 on."""

import gzip
from pathlib import Path

import numpy as np

DATA = Path(__file__).resolve().parents[2] / "data"


def _load(name, offset):
    return np.frombuffer(gzip.open(DATA / name).read(), np.uint8, offset=offset)


def mnist():
    xtr = _load("train-images-idx3-ubyte.gz", 16).reshape(-1, 784) / 255.0
    ytr = _load("train-labels-idx1-ubyte.gz", 8).astype(int)
    xte = _load("t10k-images-idx3-ubyte.gz", 16).reshape(-1, 784) / 255.0
    yte = _load("t10k-labels-idx1-ubyte.gz", 8).astype(int)
    return xtr, ytr, xte, yte


def pixel_pool():
    """Every nonzero pixel value in the training images, scaled to (0, 1]. The probes draw neuron inputs from it."""
    imgs = _load("train-images-idx3-ubyte.gz", 16)
    return imgs[imgs > 0] / 255.0


def split(labels, digits, n_train, n_val, seed=0):
    """Stratified train/val indices from the training partition, disjoint, frozen by seed."""
    rng = np.random.default_rng(seed)
    tr, va = [], []
    for d in digits:
        pool = rng.permutation(np.where(labels == d)[0])
        tr.append(pool[: n_train // len(digits)])
        va.append(pool[n_train // len(digits): n_train // len(digits) + n_val // len(digits)])
    return np.concatenate(tr), np.concatenate(va)


def test_subset(labels, n, seed=0):
    """Stratified subset of the official test set."""
    rng = np.random.default_rng(seed)
    return np.concatenate([rng.permutation(np.where(labels == d)[0])[: n // 10] for d in range(10)])
