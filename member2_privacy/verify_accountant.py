"""Extension 2 - verify the accountant (our own check; not in the paper).

Check A  Theorems 2 and 3 hold. For two classes the output distribution of noisy-max is known exactly
         (Lemma 4 is exact there), so we compute the TRUE moment of the privacy loss in high-precision
         arithmetic and confirm it never exceeds min(Theorem 2, Theorem 3). This tests the paper's claims
         (proved in Appendix A) numerically.
Check B  Our float code computes the Theorem 2/3 formulas correctly (same values as the high-precision formulas).
Check C  Tightness: how far above the exact moment the bound sits (the paper says its analysis is
         "rather conservative", Sec. 3.3). Reported for the reference setting.
Check D  Our accountant against the authors' released script
Check E  Our smooth sensitivity against the authors' `smoothed_sens`
         (tensorflow/privacy, research/pate_2017/analysis.py). D and E need internet; skipped if offline.

Run:  python -m member2_privacy.verify_accountant"""
import math
import sys
import urllib.request
from decimal import Decimal as D, getcontext
import numpy as np
from . import accountant as acc

getcontext().prec = 400
AUTHORS_URL = "https://raw.githubusercontent.com/tensorflow/privacy/master/research/pate_2017/analysis.py"


# ---------- exact two-class moments (high precision) ----------
def _tail(a):                                     # Pr[noisy winner flips] when gap*gamma = a  (Lemma 4, exact for 2 classes)
    return (2 + a) / (4 * a.exp())


def _probs(g, gam):                               # (Pr[output 0], Pr[output 1]) when counts differ by g = n0 - n1
    t = _tail(abs(D(g)) * D(str(gam)))
    return (1 - t, t) if g >= 0 else (t, 1 - t)


def exact_alpha(g, gam, lam):
    """True lambda-th log moment of the privacy loss; neighbours change the gap by 2 (one vote moves)."""
    best = None
    for d in (2, -2):
        p0, p1 = _probs(g, gam)
        q0, q1 = _probs(g + d, gam)
        a = (p0 * (p0 / q0) ** lam + p1 * (p1 / q1) ** lam).ln()
        best = a if best is None else max(best, a)
    return best


def thm2(gam, lam):
    return 2 * D(str(gam)) ** 2 * lam * (lam + 1)


def thm3(g, gam, lam):
    q = _tail(D(g) * D(str(gam)))
    e2 = (2 * D(str(gam))).exp()
    if q >= (e2 - 1) / ((4 * D(str(gam))).exp() - 1):
        return None                                # condition of Theorem 3 not met
    return ((1 - q) * ((1 - q) / (1 - e2 * q)) ** lam + q * (2 * D(str(gam)) * lam).exp()).ln()


def bound(g, gam, lam):
    b2, b3 = thm2(gam, lam), thm3(g, gam, lam)
    return b2 if b3 is None else min(b2, b3)


def check_validity(gammas=(0.01, 0.05, 0.2, 0.5, 1.0), gaps=range(0, 300, 3), lambdas=range(1, 9)):
    viol = tot = 0
    for gam in gammas:
        for g in gaps:
            for lam in lambdas:
                tot += 1
                viol += exact_alpha(g, gam, lam) > bound(g, gam, lam)
    return viol, tot


def check_float_code(gammas=(0.05, 0.2), gaps=range(0, 120, 6)):
    """Our float accountant vs the same formulas in high precision, wherever the value is large enough to resolve."""
    worst = 0.0
    for gam in gammas:
        for g in gaps:
            counts = np.array([[(g + 200) / 2, (200 - g) / 2]])
            ours = acc.alpha_dependent(counts, gam)[0]
            for i, lam in enumerate(acc.LAMBDAS):
                ref = float(bound(g, gam, int(lam)))
                if ref > 1e-8:
                    worst = max(worst, abs(ours[i] - ref) / ref)
    return worst


def tightness_table(gam=0.05, gaps=(40, 100, 150), lambdas=(2, 8)):
    rows = []
    for g in gaps:
        for lam in lambdas:
            ex, b = exact_alpha(g, gam, lam), bound(g, gam, lam)
            rows.append((g, lam, float(ex), float(b), float(b / ex)))
    return rows


# ---------- comparison with the authors' script ----------
def load_authors():
    try:
        text = urllib.request.urlopen(AUTHORS_URL, timeout=20).read().decode()
    except Exception as e:                         # offline
        return None, str(e)
    ns = {"math": math, "np": np, "xrange": range}
    exec(text[text.index("def compute_q_noisy_max("):text.index("def main(")], ns)
    return ns, None


def authors_epsilon(ns, counts, scale, delta, L=8):
    g, lam = 1.0 / scale, np.arange(1, L + 1)
    tot = np.zeros(L)
    for c in counts:
        tot += np.array([ns["logmgf_from_counts"](c, g, l) for l in lam])
    return float(((tot - math.log(delta)) / lam).min())


def compare_with_authors(ns, settings=((250, 0.9), (250, 0.7), (100, 0.8), (10, 0.8)), scales=(5, 20, 100), Q=300):
    from .aggregate import vote_counts
    rng = np.random.default_rng(0)
    rows = []
    for n, p in settings:
        true = rng.integers(0, 2, Q)
        agree = rng.random((n, Q)) < p
        other = (true + rng.integers(1, 2, (n, Q))) % 2
        counts = vote_counts(np.where(agree, true, other), 2)
        for s in scales:
            rows.append((n, p, s, acc.epsilon(counts, s, 1e-5), authors_epsilon(ns, counts, s, 1e-5)))
    return rows


def compare_smooth_sensitivity(ns, rows=([200, 50], [150, 100], [240, 10], [180, 70]), gammas=(0.05, 0.1), beta=0.09):
    """Our smooth sensitivity vs the authors' `smoothed_sens` for single queries; returns the worst relative difference."""
    from .smooth_sensitivity import smooth_sensitivity
    worst = 0.0
    for counts in rows:
        for gam in gammas:
            theirs = np.array([ns["smoothed_sens"](np.array(counts), gam, l, beta) for l in range(1, 9)])
            ours = smooth_sensitivity(np.array([counts]), gam, beta)
            worst = max(worst, float(np.max(np.abs(ours - theirs) / np.maximum(theirs, 1e-300))))
    return worst


def main():
    ok = True
    viol, tot = check_validity()
    print(f"A. Theorems 2 and 3 vs exact two-class moments: {viol} violations in {tot} checks "
          f"(gamma 0.01..1, gap 0..300, lambda 1..8)  -> {'PASS' if viol == 0 else 'FAIL'}")
    ok &= viol == 0
    w = check_float_code()
    print(f"B. Float accountant vs high-precision formulas: worst relative difference {w:.1e}  -> {'PASS' if w < 1e-6 else 'FAIL'}")
    ok &= w < 1e-6
    print("C. Tightness at gamma = 0.05 (bound / exact moment):")
    for g, lam, ex, b, r in tightness_table():
        print(f"     gap {g:3d}  lambda {lam}  exact {ex:.2e}  bound {b:.2e}  ratio {r:5.1f}")
    ns, err = load_authors()
    if ns is None:
        print(f"D. Authors' script: skipped (could not download: {err})")
    else:
        print("D. Our epsilon vs the authors' script (delta 1e-5):")
        for n, p, s, ours, theirs in compare_with_authors(ns):
            flag = "" if abs(ours - theirs) < 1e-6 else "   <- differs"
            print(f"     teachers {n:3d} agree {p}  scale {s:3d}  ours {ours:8.3f}  authors {theirs:8.3f}{flag}")
        w = compare_smooth_sensitivity(ns)
        print(f"E. Our smooth sensitivity vs the authors' smoothed_sens: worst relative difference {w:.1e}  -> {'PASS' if w < 1e-6 else 'FAIL'}")
        ok &= w < 1e-6
        print("   Differences in D, if any, come from the extra pure-DP cap (2*gamma*lambda) in the authors' code,"
              " which the paper's text does not include (plan change C5).")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
