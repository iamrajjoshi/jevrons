"""Neuron state formats and their fixed questions. Numbers go on the wire at two decimals."""

import numpy as np

from jevrons.jev import noul

TRUE = "The final total is greater than zero."
FALSE = "The final total is zero or less."

INSTRUCTIONS = {
    "paired": "Each term has an input x and a weight w. Multiply x by w for every term, "
              "add all the products together, then add bias.",
    "parallel": "x and w are lists of equal length. Multiply each x value by the w value at "
                "the same position, add all the products together, then add bias.",
    "products": "Add all the numbers in products together, then add bias.",
    "folded": "Add all the numbers in products together.",
}


def r2(v) -> float:
    return round(float(v), 2)


def neuron_state(fmt: str, x, w, b) -> tuple[dict, float]:
    """Return (state, true z computed from exactly the values sent)."""
    x = [r2(v) for v in x]
    w = [r2(v) for v in w]
    b = r2(b)
    if fmt == "paired":
        state = {"bias": b, "terms": [{"x": xi, "w": wi} for xi, wi in zip(x, w)]}
        z = sum(xi * wi for xi, wi in zip(x, w)) + b
    elif fmt == "parallel":
        state = {"bias": b, "x": x, "w": w}
        z = sum(xi * wi for xi, wi in zip(x, w)) + b
    elif fmt == "products":
        p = [r2(xi * wi) for xi, wi in zip(x, w)]
        state = {"bias": b, "products": p}
        z = sum(p) + b
    elif fmt == "folded":  # bias travels as one more product, so it isn't a standalone number
        p = [r2(xi * wi) for xi, wi in zip(x, w)] + [b]
        state = {"products": p}
        z = sum(p)
    else:
        raise ValueError(fmt)
    return state, z


ALT_TRUE = "Yes, the total is positive."
ALT_FALSE = "No, the total is zero or negative."


def neuron_question(fmt: str, wording: str = "v1") -> dict:
    if wording == "v2":  # question form; clean 0.02/0.99 step in the scalar sweep
        return {"fires": noul(f"{INSTRUCTIONS[fmt]} Is the total positive?", ALT_TRUE, ALT_FALSE)}
    return {"fires": noul(INSTRUCTIONS[fmt], TRUE, FALSE)}


def packed_questions(fmt: str, keys) -> dict:
    return {k: noul(f"Look only at neuron {k}. {INSTRUCTIONS[fmt]}", TRUE, FALSE) for k in keys}


SCALAR_Q = {"pos": noul("z is a number.", "z is greater than zero.", "z is zero or less.")}
SCALAR_Q_ALT = {"pos": noul("Is the number z positive?", "Yes, z is positive.",
                            "No, z is zero or negative.")}


def sample_neuron(rng: np.random.Generator, pixel_pool: np.ndarray, k: int, margin: float):
    """k MNIST pixel values, N(0,1) weights, bias placing z at margin * ||x*w||."""
    x = rng.choice(pixel_pool, k)
    w = np.clip(rng.normal(0, 1, k), -9.99, 9.99)
    spread = np.sqrt(np.sum((x * w) ** 2))
    b = margin * spread - np.dot(x, w)
    return x, w, b, spread


if __name__ == "__main__":
    s, z = neuron_state("paired", [0.5, 1.0], [2.0, -0.25], 0.1)
    assert s == {"bias": 0.1, "terms": [{"x": 0.5, "w": 2.0}, {"x": 1.0, "w": -0.25}]}
    assert abs(z - 0.85) < 1e-9
    _, zp = neuron_state("products", [0.333], [3.0], 0.0)
    assert zp == 0.99  # x rounds to 0.33 first; z uses exactly the values sent
    rng = np.random.default_rng(0)
    x, w, b, spread = sample_neuron(rng, np.array([0.5, 1.0]), 10, 1.5)
    assert abs((np.dot(x, w) + b) - 1.5 * spread) < 1e-9
    sf, zf = neuron_state("folded", [0.5], [2.0], -0.25)
    assert sf == {"products": [1.0, -0.25]} and zf == 0.75
    print("states ok")
