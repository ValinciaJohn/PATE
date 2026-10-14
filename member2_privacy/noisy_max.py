"""Bucket 1 - paper Sec 2.1, Equation 1:  f(x) = argmax_j { n_j(x) + Lap(1/gamma) }.
Pure function, no file I/O. `scale` = 1/gamma (the plan's noise_scale)."""
import numpy as np


def noisy_argmax(counts, scale, rng):
    """counts: (Q, C) or (C,) vote counts. Adds independent Laplace(0, scale) noise to every
    class count of every query and returns the argmax."""
    if scale <= 0:
        raise ValueError("scale must be > 0")
    counts = np.asarray(counts, dtype=float)
    return (counts + rng.laplace(0.0, scale, size=counts.shape)).argmax(axis=-1)
