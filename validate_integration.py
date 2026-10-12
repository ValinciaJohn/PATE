"""Validate the Member 1 + Member 2 integration on real artifacts.  Run from the repo root.

  python validate_integration.py                         # checks the reference run t250_b20_q500_s42
  python validate_integration.py --run-id t100_b20_q500_s42
  python validate_integration.py --grid                  # also: is the plan grid complete, real, and clean?

Needs artifacts/<run_id>/ to exist, i.e. run first:  python run_step_a.py run --grid ref
Writes nothing into artifacts/ (the determinism check re-runs Member 2 in a temporary copy).
Exit code 0 = no FAIL. WARN = worth reading, not an error."""
import argparse
import math
import os
import re
import shutil
import sys
import tempfile
from itertools import product
from pathlib import Path
import numpy as np
from common import paths
from common.contracts import ExperimentConfig, DataManifest, PrivacyReport, check_votes, check_labels
from member2_privacy import accountant as acc
from member2_privacy._io import art_root
from member2_privacy.aggregate import aggregate_and_account, vote_counts

RUN_RE = re.compile(r"^t(\d+)_b(\d+(?:\.\d+)?)_q(\d+)_s(\d+)$")
RESULTS = []


def rec(level, name, detail=""):
    RESULTS.append(level)
    print(f"[{level:4s}] {name}" + (f"  -  {detail}" if detail else ""))


def ok(cond, name, detail="", fail="FAIL"):
    rec("PASS" if cond else fail, name, detail)


def section(title):
    print(f"\n=== {title}")


# ---------------------------------------------------------------- Member 1 outputs
def check_member1(d, cfg, root):
    section("Member 1 outputs (data splits and teacher votes)   paper: Sec. 2.1, Appendix C")
    need = [paths.MANIFEST, paths.X_PRIVATE, paths.Y_PRIVATE, paths.X_PUBLIC, paths.Y_PUBLIC_TRUE, paths.X_TEST,
            paths.Y_TEST, paths.X_HOLDOUT, paths.Y_HOLDOUT, paths.TEACHER_VOTES]
    missing = [f for f in need if not (d / f).exists()]
    ok(not missing, "all contract files present", "missing: " + ", ".join(missing) if missing else f"{len(need)} files")
    if missing:
        return None
    m = DataManifest.model_validate_json((d / paths.MANIFEST).read_text())
    L = lambda f: np.load(d / f)
    Xpr, ypr, Xpu, ypu, Xte, yte, Xho, yho = (L(paths.X_PRIVATE), L(paths.Y_PRIVATE), L(paths.X_PUBLIC), L(paths.Y_PUBLIC_TRUE),
                                              L(paths.X_TEST), L(paths.Y_TEST), L(paths.X_HOLDOUT), L(paths.Y_HOLDOUT))
    votes = L(paths.TEACHER_VOTES)
    ok((len(Xpr), len(Xpu), len(Xte), len(Xho)) == (m.n_private, m.n_public, m.n_test, m.n_holdout) and
       m.n_features == Xpr.shape[1], "manifest matches the arrays", f"private {len(Xpr)}, public {len(Xpu)}, test {len(Xte)}, holdout {len(Xho)}")
    ok(m.n_teachers == cfg.n_teachers and m.n_classes == 2, "manifest: teachers match the run id, 2 classes")
    try:
        check_votes(votes, m)
        ok(True, "check_votes passes", f"votes shape {votes.shape}, dtype {votes.dtype}")
    except AssertionError as e:
        ok(False, "check_votes passes", str(e))
    ok(np.issubdtype(votes.dtype, np.integer), "votes are integers")

    real = m.n_features == 13 and m.n_private > 20000
    if not real:
        rec("WARN", "THIS RUN HOLDS DUMMY DATA", f"{m.n_features} features, {m.n_private} private rows. Real Adult has 13 features and 27,133 private rows. "
            "Do not quote these numbers in the report; run `python run_step_a.py run --grid ref` with internet.")
    total = m.n_private + m.n_public + m.n_test + m.n_holdout
    shares = np.array([m.n_private, m.n_public, m.n_test, m.n_holdout]) / total
    ok(np.allclose(shares, [.60, .15, .15, .10], atol=.005), "split is 60/15/15/10 (plan section 2)",
       "shares " + "/".join(f"{s:.3f}" for s in shares), fail="FAIL" if real else "WARN")
    pos = [float(y.mean()) for y in (ypr, ypu, yte, yho)]
    ok(max(pos) - min(pos) < .01, "split is stratified (same >50K share in every part)", "shares " + ", ".join(f"{p:.3f}" for p in pos),
       fail="FAIL" if real else "WARN")
    ok(not (np.isnan(Xpr).any() or np.isnan(Xpu).any() or np.isnan(Xte).any() or np.isnan(Xho).any()), "no missing values in X")
    unseen = int((Xpu == -1).sum() + (Xte == -1).sum() + (Xho == -1).sum())
    rec("INFO", "categories unseen in the private pool (encoded -1)", f"{unseen} values")

    acc_t = (votes == ypu).mean(axis=1)
    maj = max(ypu.mean(), 1 - ypu.mean())
    ones = votes.sum(axis=0)
    n = votes.shape[0]
    plur = (ones * 2 > n).astype(int)
    rec("INFO", f"teachers ({n}) on the public pool",
        f"mean accuracy {acc_t.mean():.3f} (min {acc_t.min():.3f}, max {acc_t.max():.3f}); always-predict-majority = {maj:.3f}; rows per teacher ~{m.n_private // n}")
    ok(acc_t.mean() > maj, "teachers beat the majority-class baseline", fail="FAIL" if real else "WARN")
    pa = float((plur == ypu).mean())
    ok(pa >= 0.80 or not real, "non-private plurality vote is accurate (paper Appendix C: non-private model 85% on Adult)", f"{pa:.3f}")
    gap = float((np.abs(ones - (n - ones)) / n).mean())
    rec("INFO", "normalized vote gap (Member 1's 'agreement'; paper Fig. 3)", f"{gap:.3f}; unanimous rows {((ones == 0) | (ones == n)).mean():.3f}")
    cache = root / "_cache" / f"votes_t{cfg.n_teachers}_s{cfg.seed}.npy"
    if cache.exists():
        ok(np.array_equal(np.load(cache), votes), "run votes equal Member 1's cache for this (teachers, seed)")
    else:
        rec("INFO", "no vote cache file for this (teachers, seed)", cache.name)
    return m, votes, ypu, real


# ---------------------------------------------------------------- Member 2 outputs
def rerun_is_identical(d, cfg, rep):
    tmp = Path(tempfile.mkdtemp())
    old_env, old_root = os.environ.get("PATE_ARTIFACT_ROOT"), paths.ARTIFACT_ROOT
    try:
        dst = tmp / cfg.run_id
        dst.mkdir()
        for f in (paths.MANIFEST, paths.TEACHER_VOTES, paths.Y_PUBLIC_TRUE, "query_request.npy"):
            if (d / f).exists():
                shutil.copy(d / f, dst / f)
        os.environ["PATE_ARTIFACT_ROOT"], paths.ARTIFACT_ROOT = str(tmp), Path(tmp)
        rep2 = aggregate_and_account(cfg)
        return (np.array_equal(np.load(dst / paths.QUERY_IDX), np.load(d / paths.QUERY_IDX)) and
                np.array_equal(np.load(dst / paths.NOISY_LABELS), np.load(d / paths.NOISY_LABELS)) and rep2.epsilon == rep.epsilon)
    finally:
        paths.ARTIFACT_ROOT = old_root
        if old_env is None:
            os.environ.pop("PATE_ARTIFACT_ROOT", None)
        else:
            os.environ["PATE_ARTIFACT_ROOT"] = old_env
        shutil.rmtree(tmp, ignore_errors=True)


def check_member2(d, cfg, m, votes, ypu, real):
    section("Member 2 outputs (noisy labels and epsilon)   paper: Sec. 2.1 Eq. 1, Sec. 3.2-3.3, Theorems 1-3, Lemma 4")
    need = [paths.QUERY_IDX, paths.NOISY_LABELS, paths.PRIVACY_REPORT, "accountant_details.json"]
    missing = [f for f in need if not (d / f).exists()]
    ok(not missing, "Member 2 files present", "missing: " + ", ".join(missing) + "  (run `python run_step_a.py run --grid ref` first)" if missing else "")
    if missing:
        return
    import json
    rep = PrivacyReport.model_validate_json((d / paths.PRIVACY_REPORT).read_text())
    det = json.loads((d / "accountant_details.json").read_text())
    idx, lab = np.load(d / paths.QUERY_IDX), np.load(d / paths.NOISY_LABELS)
    ok((rep.n_teachers, rep.n_queries, rep.noise_scale, rep.seed, rep.delta) == (cfg.n_teachers, cfg.n_queries, cfg.noise_scale, cfg.seed, cfg.delta),
       "privacy_report matches the run id (teachers, queries, scale, seed, delta)", f"delta {rep.delta:g}")
    try:
        check_labels(idx, lab, m)
        ok(len(idx) == cfg.n_queries, "check_labels passes; one noisy label per query", f"{len(idx)} queries")
    except AssertionError as e:
        ok(False, "check_labels passes", str(e))
    ok(cfg.n_queries < m.n_public, "queries are a subset of the public pool (plan: never the whole pool)", f"{cfg.n_queries} of {m.n_public}")

    counts = vote_counts(votes[:, idx], m.n_classes)
    g = 1.0 / cfg.noise_scale
    eps_dd = acc.epsilon(counts, cfg.noise_scale, cfg.delta)
    eps_di = acc.epsilon(counts, cfg.noise_scale, cfg.delta, data_dependent=False)
    eps_sc = acc.eps_strong_composition(cfg.noise_scale, cfg.n_queries, cfg.delta)
    ok(abs(eps_dd - rep.epsilon) < 1e-9 and abs(eps_di - det["eps_data_independent"]) < 1e-9 and abs(eps_sc - det["eps_strong_composition"]) < 1e-9,
       "epsilon recomputed from the votes equals the report", f"data-dependent {rep.epsilon:.3f}, data-independent {eps_di:.3f}, simple composition {eps_sc:.3f}")
    ok(eps_dd <= eps_di + 1e-9, "data-dependent epsilon <= data-independent (Sec. 3.3: take the smaller bound)")
    ok(eps_dd >= math.log(1 / cfg.delta) / 8 - 1e-9, "epsilon >= ln(1/delta)/8 (lambda <= 8 floor, Sec. 3.3)", f"floor {math.log(1 / cfg.delta) / 8:.3f}")
    ok(eps_di <= eps_sc + 1e-9, "moments (Thm 2) <= simple composition (Sec. 3.2)",
       "" if eps_di <= eps_sc + 1e-9 else "above simple composition: expected only for very large noise with few queries, because of the lambda <= 8 floor", fail="WARN")

    acc_now = float((lab == ypu[idx]).mean())
    ok(rep.label_accuracy is not None and abs(rep.label_accuracy - acc_now) < 1e-9, "reported label accuracy equals recomputed", f"{acc_now:.3f}")
    ok(acc_now >= 0.55, "noisy labels are useful (accuracy well above chance)", f"{acc_now:.3f}" +
       ("" if acc_now >= 0.55 else "  <- the noise swamps the votes at this setting; more teachers or a smaller noise scale are needed (paper Fig. 2)"), fail="WARN")

    if m.n_classes == 2:   # Eq. 1 end to end: how often does noise flip the plurality label? Lemma 4 gives the exact probability.
        a = g * np.abs(counts[:, 0] - counts[:, 1])
        p = (2 + a) / (4 * np.exp(a))
        flips = int((lab != counts.argmax(axis=1)).sum())
        mu, sd = float(p.sum()), math.sqrt(float((p * (1 - p)).sum()))
        z = (flips - mu) / sd if sd > 0 else 0.0
        ok(abs(z) < 4, "noise follows Eq. 1: observed label flips match Lemma 4's probability", f"observed {flips}, expected {mu:.1f} +/- {sd:.1f} (z = {z:+.1f})")

    ok(rerun_is_identical(d, cfg, rep), "same seed gives identical query rows, labels and epsilon (re-run in a temp copy)")
    try:
        from member2_privacy.smooth_sensitivity import release_epsilon
        r = release_epsilon(counts, cfg.noise_scale, cfg.delta, rng=0)
        ok(math.isfinite(r["noise_scale"]) and abs(r["epsilon"] - rep.epsilon) < 1e-6, "smooth-sensitivity release works (Sec. 3.3)",
           f"noise scale {r['noise_scale']:.4f}; the release itself is ({r['eps_ss']:.2f}, {r['delta']:g})-DP")
    except ValueError as e:
        rec("WARN", "smooth-sensitivity release not available at this noise scale", str(e))


# ---------------------------------------------------------------- accountant against the paper (needs no data)
def check_accountant():
    section("Accountant against the paper (no data needed)   paper: Sec. 3.2 numbers, Theorems 2-3, Appendix A")
    ok(abs(acc.eps_strong_composition(20, 100, 1e-5) - 5.80) < 0.01, "simple composition = 5.80 (gamma 0.05, 100 queries, delta 1e-5)")
    ok(abs(acc.eps_strong_composition(20, 1000, 1e-6) - 26) < 1, "simple composition = about 26 (1000 queries, delta 1e-6)")
    e1 = acc.eps_from_alpha(acc.alpha_independent(100, 0.05).sum(axis=0), 1e-5)[0]
    e2 = acc.eps_from_alpha(acc.alpha_independent(1000, 0.05).sum(axis=0), 1e-6)[0]
    ok(e1 < 5.80 and e2 < 26.62, "moments accountant is below simple composition at the paper's settings", f"{e1:.2f} < 5.80 and {e2:.2f} < 26.62")
    from member2_privacy import verify_accountant as va
    viol, tot = va.check_validity(gammas=(0.05, 0.5), gaps=range(0, 90, 10), lambdas=(1, 4, 8))
    ok(viol == 0, "Theorems 2 and 3 never below the exact two-class moment", f"{viol} violations in {tot} checks")
    w = va.check_float_code(gammas=(0.05,), gaps=range(0, 60, 12))
    ok(w < 1e-6, "float code equals the paper's formulas", f"worst relative difference {w:.1e}")


# ---------------------------------------------------------------- the plan grid
def check_grid(root):
    section("Plan grid: teachers {10,100,250} x scale {1,5,20,100} x queries {100,500,1000} x seeds {42,43,44} = 108 runs")
    import json
    have, dummy, real, junk = set(), [], [], []
    for d in sorted(root.iterdir()) if root.exists() else []:
        if not d.is_dir() or d.name == "_cache":
            continue
        if not RUN_RE.match(d.name):
            junk.append(d.name)
            continue
        if (d / paths.PRIVACY_REPORT).exists() and (d / paths.MANIFEST).exists():
            have.add(d.name)
            mf = json.loads((d / paths.MANIFEST).read_text())
            (real if mf["n_features"] == 13 and mf["n_private"] > 20000 else dummy).append(d.name)
    want = {f"t{n}_b{b}_q{q}_s{s}" for n, b, q, s in product([10, 100, 250], [1, 5, 20, 100], [100, 500, 1000], [42, 43, 44])}
    ok(not (want - have), "all 108 grid runs have a privacy report", f"{len(have & want)} of 108 present")
    ok(not dummy, "no run in the grid holds dummy data", f"{len(real)} real, {len(dummy)} dummy" if dummy else f"{len(real)} real", fail="WARN")
    ok(not junk, "no leftover test folders in artifacts/", ", ".join(junk) if junk else "", fail="WARN")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", default="t250_b20_q500_s42")
    ap.add_argument("--grid", action="store_true")
    a = ap.parse_args()
    root = art_root()
    m = RUN_RE.match(a.run_id)
    if not m:
        sys.exit("run id must look like t250_b20_q500_s42")
    n, b, q, s = m.groups()
    cfg = ExperimentConfig(run_id=a.run_id, seed=int(s), n_teachers=int(n), noise_scale=float(b), n_queries=int(q))
    d = root / a.run_id
    print(f"Validating {a.run_id} in {root.resolve()}")
    if not d.exists():
        sys.exit(f"{d} does not exist. Run first:  python run_step_a.py run --grid ref")
    out = check_member1(d, cfg, root)
    if out:
        check_member2(d, cfg, *out)
    check_accountant()
    if a.grid:
        check_grid(root)
    print(f"\nSUMMARY: {RESULTS.count('PASS')} PASS, {RESULTS.count('WARN')} WARN, {RESULTS.count('FAIL')} FAIL")
    sys.exit(1 if "FAIL" in RESULTS else 0)


if __name__ == "__main__":
    main()
