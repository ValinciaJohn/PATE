"""Buckets 3 + 4 - moments accountant for the Laplace noisy-max, written from the paper.

Per query, integer lambda, gamma = 1/scale:
  Thm 2 (data-independent): alpha(l) <= 2 g^2 l (l+1)       [mechanism is (2g,0)-DP]
  Thm 3 (data-dependent):   alpha(l) <= log((1-q)((1-q)/(1-e^{2g}q))^l + q e^{2g l}),
                            valid if q < (e^{2g}-1)/(e^{4g}-1)
  Lemma 4:                  q = sum_{j!=j*} (2 + g(n*-n_j)) / (4 exp(g(n*-n_j)))
  Pure-DP fallback:         alpha(l) <= 2 g l
Thm 1: moments add over queries; eps = min_l (alpha(l) + ln(1/delta)) / l."""
import numpy as np

LAMBDAS = np.arange(1, 9)   # paper: integers up to 8


def alpha_independent(n_queries, gamma, lambdas=LAMBDAS):
    lam = np.asarray(lambdas, float)
    return np.tile(np.minimum(2 * gamma**2 * lam * (lam + 1), 2 * gamma * lam), (n_queries, 1))


def q_upper_bound(counts, gamma):
    counts = np.asarray(counts, float)
    gap = counts.max(axis=1, keepdims=True) - counts
    terms = (2 + gamma * gap) / (4 * np.exp(gamma * gap))
    terms[np.arange(len(counts)), counts.argmax(axis=1)] = 0.0
    return np.minimum(terms.sum(axis=1), 1.0)


def alpha_dependent(counts, gamma, lambdas=LAMBDAS):
    counts = np.asarray(counts)
    lam = np.asarray(lambdas, float)
    base = alpha_independent(len(counts), gamma, lam)
    q = q_upper_bound(counts, gamma)[:, None]
    e2g = np.exp(2 * gamma)
    valid = q < (e2g - 1) / (np.exp(4 * gamma) - 1)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        thm3 = np.log((1 - q) * ((1 - q) / (1 - e2g * q)) ** lam + q * np.exp(2 * gamma * lam))
    return np.minimum(base, np.where(valid & np.isfinite(thm3), thm3, np.inf))


def eps_from_alpha(alpha_total, delta, lambdas=LAMBDAS):
    lam = np.asarray(lambdas, float)
    eps = (np.asarray(alpha_total) + np.log(1 / delta)) / lam
    i = int(np.argmin(eps))
    return float(eps[i]), int(lam[i])


def epsilon(counts, scale, delta, data_dependent=True):
    """Total (eps, delta)-DP cost of answering every query in `counts` with Laplace scale `scale`."""
    g = 1.0 / scale
    a = alpha_dependent(counts, g) if data_dependent else alpha_independent(len(counts), g)
    return eps_from_alpha(a.sum(axis=0), delta)[0]


def epsilon_curve(counts, scale, delta, checkpoints, data_dependent=True):
    g = 1.0 / scale
    a = alpha_dependent(counts, g) if data_dependent else alpha_independent(len(counts), g)
    cum = np.cumsum(a, axis=0)
    return [eps_from_alpha(cum[k - 1], delta)[0] for k in checkpoints]


def eps_strong_composition(scale, n_queries, delta):
    """Paper Sec 3.2 simple analysis: 4T g^2 + 2g sqrt(2T ln(1/delta))."""
    g = 1.0 / scale
    return 4 * n_queries * g**2 + 2 * g * np.sqrt(2 * n_queries * np.log(1 / delta))
