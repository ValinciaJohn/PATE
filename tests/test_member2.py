import numpy as np
import pytest
from common.contracts import ExperimentConfig, PrivacyReport, DataManifest, check_votes, check_labels
from member2_privacy.dummy_artifacts import make_dummy_artifacts
from member2_privacy._io import art_dir
from common import paths
from member2_privacy.noisy_max import noisy_argmax
from member2_privacy.aggregate import aggregate_and_account, vote_counts
from member2_privacy import accountant as acc


def _counts(n_teachers, p_agree, Q=500, C=2, seed=0):
    rng = np.random.default_rng(seed)
    true = rng.integers(0, C, Q)
    agree = rng.random((n_teachers, Q)) < p_agree
    other = (true + rng.integers(1, C, (n_teachers, Q))) % C
    return vote_counts(np.where(agree, true, other), C), true


# ---- noisy_argmax (Bucket 1) ----
def test_tiny_noise_gives_plurality():
    c, _ = _counts(250, 0.8)
    keep = c[:, 0] != c[:, 1]
    assert (noisy_argmax(c, 1e-6, np.random.default_rng(0))[keep] == c.argmax(1)[keep]).all()


def test_huge_noise_is_about_chance():
    c, true = _counts(250, 0.95, Q=5000)
    assert abs((noisy_argmax(c, 1e6, np.random.default_rng(0)) == true).mean() - 0.5) < 0.03


def test_flip_rate_matches_lemma4():
    gam, g, n = 0.1, 10, 200_000
    flip = (noisy_argmax(np.tile([130, 120], (n, 1)), 1 / gam, np.random.default_rng(1)) != 0).mean()
    assert abs(flip - (2 + gam * g) / (4 * np.exp(gam * g))) < 0.005


def test_invalid_scale():
    with pytest.raises(ValueError):
        noisy_argmax(np.ones((2, 2)), 0, np.random.default_rng(0))


# ---- accountant (Buckets 3 + 4) ----
def test_paper_strong_composition_numbers():
    assert abs(acc.eps_strong_composition(20, 100, 1e-5) - 5.80) < 0.01     # MNIST, gamma=.05
    assert abs(acc.eps_strong_composition(20, 1000, 1e-6) - 26) < 1         # SVHN


def test_moments_tighter_than_strong_composition():
    c, _ = _counts(250, 0.9, Q=100)
    di = acc.epsilon(c, 20, 1e-5, data_dependent=False)
    assert di < acc.eps_strong_composition(20, 100, 1e-5)
    assert acc.epsilon(c, 20, 1e-5) <= di + 1e-9


def test_epsilon_rises_with_queries():
    c, _ = _counts(250, 0.8)
    e = acc.epsilon_curve(c, 20, 1e-5, [50, 100, 300, 500])
    assert all(a < b for a, b in zip(e, e[1:]))


def test_epsilon_falls_when_teachers_agree_more():
    lo = acc.epsilon(_counts(250, 0.6)[0], 20, 1e-5)
    hi = acc.epsilon(_counts(250, 0.95)[0], 20, 1e-5)
    assert hi < lo


def test_more_noise_means_less_epsilon_data_independent():
    c, _ = _counts(25, 0.8)
    e = [acc.epsilon(c, s, 1e-5, data_dependent=False) for s in (1, 2, 5, 10, 20)]
    assert all(a > b for a, b in zip(e, e[1:]))


def test_data_dependent_epsilon_never_exceeds_data_independent():
    for n in (25, 250):
        c, _ = _counts(n, 0.8)
        for s in (1, 5, 20):
            assert acc.epsilon(c, s, 1e-5) <= acc.epsilon(c, s, 1e-5, data_dependent=False) + 1e-9


def test_epsilon_floor_from_lambda_cap():
    # with lambda <= 8 the tail bound can never go below ln(1/delta)/8
    c, _ = _counts(250, 0.95)
    assert acc.epsilon(c, 5, 1e-5) >= np.log(1e5) / 8 - 1e-9


# ---- entry point on dummy artifacts ----
def test_aggregate_and_account_on_dummy(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "ARTIFACT_ROOT", str(tmp_path))
    cfg = ExperimentConfig(run_id="t25_b2_q300_s42", n_teachers=25, noise_scale=2, n_queries=300)
    make_dummy_artifacts(cfg)
    rep = aggregate_and_account(cfg)
    d = art_dir(cfg.run_id)
    m = DataManifest.model_validate_json((d / paths.MANIFEST).read_text())
    idx, lab = np.load(d / paths.QUERY_IDX), np.load(d / paths.NOISY_LABELS)
    check_labels(idx, lab, m)
    assert len(idx) == 300 and np.isfinite(rep.epsilon) and rep.epsilon > 0
    PrivacyReport.model_validate_json((d / paths.PRIVACY_REPORT).read_text())
    assert aggregate_and_account(cfg).epsilon == rep.epsilon   # same seed -> same output


# ---- optional query_request.npy from Member 3 (Appendix B.2) ----
def test_query_request_is_used_when_present(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "ARTIFACT_ROOT", str(tmp_path))
    cfg = ExperimentConfig(run_id="req", n_teachers=25, noise_scale=5, n_queries=50)
    make_dummy_artifacts(cfg)
    d = art_dir(cfg.run_id)
    want = np.random.default_rng(1).permutation(1000)[:50]
    np.save(d / "query_request.npy", want)
    aggregate_and_account(cfg)
    assert (np.load(d / paths.QUERY_IDX) == want).all()            # Member 3's order is kept
    np.save(d / "query_request.npy", np.zeros(50, dtype=int))      # duplicates must be rejected
    with pytest.raises(AssertionError):
        aggregate_and_account(cfg)


def test_query_request_wrong_length_is_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "ARTIFACT_ROOT", str(tmp_path))
    cfg = ExperimentConfig(run_id="req2", n_teachers=25, noise_scale=5, n_queries=50)
    make_dummy_artifacts(cfg)
    np.save(art_dir(cfg.run_id) / "query_request.npy", np.arange(40))
    with pytest.raises(ValueError):
        aggregate_and_account(cfg)
