import numpy as np
import pytest
from common.contracts import ExperimentConfig
from member2_privacy.dummy_artifacts import make_dummy_artifacts
from member2_privacy._io import art_dir
from common import paths
from member2_privacy import accountant as acc
from member2_privacy import smooth_sensitivity as ss
from member2_privacy import verify_accountant as va
from member2_privacy.aggregate import aggregate_and_account


# ---------- extension 1: smooth sensitivity ----------
def _ref_ss(row, gamma, lam, beta):
    """Scalar, loop-based reference for one query and one lambda."""
    top, sec = sorted(row, reverse=True)[:2]
    a = lambda t, s: acc.alpha_dependent(np.array([[t, s]], float), gamma, [lam])[0, 0]
    best, k = 0.0, 0
    while top >= sec + k:
        best = max(best, np.exp(-beta * k) * max(a(top - k - 1, sec + k + 1) - a(top - k, sec + k), 0.0))
        k += 1
    return best


def test_smooth_sensitivity_matches_scalar_reference():
    for row in ([200, 50], [150, 100], [180, 70]):
        got = ss.smooth_sensitivity(np.array([row]), 0.05)
        want = [_ref_ss(row, 0.05, l, ss.BETA) for l in range(1, 9)]
        assert np.allclose(got, want, rtol=1e-9, atol=0)


def test_close_votes_have_larger_sensitivity_than_clear_votes():
    close = ss.smooth_sensitivity(np.array([[150, 100]]), 0.05)[7]
    clear = ss.smooth_sensitivity(np.array([[240, 10]]), 0.05)[7]
    assert close > clear > 0


def test_data_independent_regime_has_zero_sensitivity():
    # near-tied votes: the Theorem 2 bound is the smaller one, which does not depend on the data
    assert (ss.smooth_sensitivity(np.array([[130, 120]]), 0.05) == 0).all()


def test_unusable_lambdas_are_marked_inf():
    out = ss.smooth_sensitivity(np.array([[100, 10]]), 1.0)     # gamma = 1: only lambda <= 2 satisfies 0.5*gamma*lambda <= 1
    assert np.isfinite(out[:2]).all() and np.isinf(out[2:]).all()


def test_release_epsilon_is_reproducible_and_consistent():
    rng = np.random.default_rng(0)
    counts = np.stack([rng.integers(180, 250, 300), rng.integers(0, 40, 300)], axis=1)
    counts[:, 0] = 250 - counts[:, 1]
    a = ss.release_epsilon(counts, 20, 1e-5, rng=1)
    b = ss.release_epsilon(counts, 20, 1e-5, rng=1)
    assert a == b
    assert np.isclose(a["epsilon"], acc.epsilon(counts, 20, 1e-5))      # gamma = 0.05: every lambda <= 8 is usable
    assert np.isclose(a["eps_ss"], 2 * ss.BETA * np.log(1e5))
    assert a["noise_scale"] >= 0


def test_release_for_run_writes_file(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "ARTIFACT_ROOT", str(tmp_path))
    cfg = ExperimentConfig(run_id="rel", n_teachers=250, noise_scale=20, n_queries=200)
    make_dummy_artifacts(cfg)
    aggregate_and_account(cfg)
    out = ss.release_for_run("rel")
    assert (art_dir("rel") / "epsilon_release.json").exists() and out["noise_scale"] > 0


# ---------- extension 2: verification ----------
def test_theorems_hold_against_exact_moments():
    viol, tot = va.check_validity(gammas=(0.05, 0.5), gaps=range(0, 60, 6), lambdas=(1, 4, 8))
    assert tot == 2 * 10 * 3 and viol == 0


def test_float_accountant_matches_formulas():
    assert va.check_float_code(gammas=(0.05,), gaps=range(0, 60, 12)) < 1e-6


def test_bounds_are_conservative():
    assert all(r > 1 for *_, r in va.tightness_table(gaps=(40, 100), lambdas=(2, 8)))


def test_matches_authors_script_when_online():
    ns, err = va.load_authors()
    if ns is None:
        pytest.skip(f"offline: {err}")
    assert va.compare_smooth_sensitivity(ns) < 1e-6
    for n, p, s, ours, theirs in va.compare_with_authors(ns, settings=((250, 0.9), (100, 0.8)), scales=(20,), Q=100):
        assert abs(ours - theirs) < 1e-6
