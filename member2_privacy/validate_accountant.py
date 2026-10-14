"""Bucket 3 check: compare our accountant with the numbers in the paper (Sec. 3.2).
Run:  python -m member2_privacy.validate_accountant"""
from . import accountant as acc

CASES = [("MNIST setting", 0.05, 100, 1e-5, 5.80),
         ("SVHN setting", 0.05, 1000, 1e-6, 26.0)]

print(f"{'case':14s} {'gamma':>6s} {'T':>5s} {'delta':>7s} | {'paper simple':>12s} {'our simple':>10s} | {'our moments':>11s} {'best lambda':>11s} | ok")
for name, g, T, d, paper in CASES:
    scale = 1 / g
    simple = acc.eps_strong_composition(scale, T, d)
    alpha = acc.alpha_independent(T, g).sum(axis=0)   # Theorem 2, summed over T queries (Theorem 1)
    eps, lam = acc.eps_from_alpha(alpha, d)
    ok = abs(simple - paper) < 1.0 and eps <= simple
    print(f"{name:14s} {g:6.2f} {T:5d} {d:7.0e} | {paper:12.2f} {simple:10.2f} | {eps:11.2f} {lam:11d} | {'PASS' if ok else 'CHECK'}")