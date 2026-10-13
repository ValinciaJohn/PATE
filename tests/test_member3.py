# tests/test_member3.py
# Tests for Member 3. Run:  python -m pytest tests/test_member3.py -v
# Every test uses a temporary folder (tmp_path), so the real artifacts/ folder is never touched.
import numpy as np
import pytest
from common.contracts import ExperimentConfig, PrivacyReport, ResultRow
from common.dummy import make_dummy_artifacts
from common import paths
from member3_student.student import train_student
from member3_student.baseline import train_baseline
from member3_student.mia import run_mia, per_example_loss
from member3_student.evaluate import train_and_audit
from member3_student.active import pick_unconfident, build_request, confidence


@pytest.fixture
def cfg(tmp_path, monkeypatch):
    """Fake run in a temporary folder: Member 1's files from common/dummy.py,
    plus small fake versions of Member 2's files (query_idx, noisy_labels, privacy_report)."""
    monkeypatch.setenv("PATE_ARTIFACT_ROOT", str(tmp_path))
    cfg = ExperimentConfig(run_id="t10_b20_q100_s42", n_teachers=10, noise_scale=20, n_queries=100, seed=42)
    m = make_dummy_artifacts(cfg)
    rng = np.random.default_rng(0)
    np.save(paths.artifact_path(cfg.run_id, paths.QUERY_IDX), rng.permutation(m.n_public)[:cfg.n_queries])
    np.save(paths.artifact_path(cfg.run_id, paths.NOISY_LABELS), rng.integers(0, m.n_classes, cfg.n_queries))
    report = PrivacyReport(run_id=cfg.run_id, seed=cfg.seed, epsilon=2.5, delta=cfg.delta,
                           noise_scale=cfg.noise_scale, n_teachers=cfg.n_teachers, n_queries=cfg.n_queries)
    paths.artifact_path(cfg.run_id, paths.PRIVACY_REPORT).write_text(report.model_dump_json())
    return cfg


# ---- the entry point on dummy data ----
def test_train_and_audit_on_dummy_writes_a_valid_result_row(cfg):
    row = train_and_audit(cfg)
    saved = ResultRow.model_validate_json(paths.artifact_path(cfg.run_id, paths.RESULT_ROW).read_text())
    assert saved == row                                            # the file holds what was returned
    assert (row.run_id, row.seed, row.n_teachers, row.n_queries) == (cfg.run_id, cfg.seed, cfg.n_teachers, cfg.n_queries)
    assert row.epsilon == 2.5                                      # copied from Member 2's report
    for value in (row.student_acc, row.baseline_acc, row.mia_student, row.mia_baseline):
        assert 0.0 <= value <= 1.0


def test_same_seed_gives_same_result(cfg):
    assert train_and_audit(cfg) == train_and_audit(cfg)


# ---- the privacy rule: the student never reads the real public answers ----
def test_never_loads_y_public_true(cfg, monkeypatch):
    opened = []
    real_load = np.load
    def spy(file, *args, **kwargs):                                # records every file np.load opens
        opened.append(str(file))
        return real_load(file, *args, **kwargs)
    monkeypatch.setattr(np, "load", spy)
    train_and_audit(cfg)
    assert opened, "spy saw no files"
    assert not any(f.endswith(paths.Y_PUBLIC_TRUE) for f in opened)


def test_works_without_y_public_true(cfg):
    paths.artifact_path(cfg.run_id, paths.Y_PUBLIC_TRUE).unlink()   # delete the file: nothing should miss it
    train_and_audit(cfg)


# ---- shapes ----
def test_student_learns_only_the_queried_rows():
    rng = np.random.default_rng(0)
    X_public = rng.normal(size=(1000, 5))
    query_idx = np.arange(50)
    student = train_student(X_public, query_idx, rng.integers(0, 2, 50))
    assert student.n_features_in_ == 5
    assert sum(len(t) for t in [student.estimators_[0].tree_.value]) > 0   # it trained
    with pytest.raises(ValueError):                                # 50 rows but 60 labels -> must fail
        train_student(X_public, query_idx, rng.integers(0, 2, 60))


# ---- the attack ----
def test_mia_score_is_between_0_and_1_and_finds_a_leak():
    rng = np.random.default_rng(0)
    X_priv, y_priv = rng.normal(size=(1000, 5)), rng.integers(0, 2, 1000)
    X_hold, y_hold = rng.normal(size=(300, 5)), rng.integers(0, 2, 300)
    X_other, y_other = rng.normal(size=(1000, 5)), rng.integers(0, 2, 1000)
    leaky = run_mia(train_baseline(X_priv, y_priv), X_priv, y_priv, X_hold, y_hold)
    safe = run_mia(train_baseline(X_other, y_other), X_priv, y_priv, X_hold, y_hold)
    assert 0.0 <= leaky <= 1.0 and 0.0 <= safe <= 1.0
    assert leaky > 0.9                                             # studied the members -> leaks
    assert abs(safe - 0.5) < 0.1                                   # never saw them -> coin flip


def test_loss_works_when_student_saw_only_one_class():
    # with huge noise the student could get only one label; the loss must still work (no crash, no NaN)
    rng = np.random.default_rng(0)
    X = rng.normal(size=(100, 5))
    model = train_student(X, np.arange(20), np.zeros(20, dtype=int))
    loss = per_example_loss(model, X, rng.integers(0, 2, 100))
    assert np.isfinite(loss).all() and (loss >= 0).all()

# ---- active learning (Appendix B.2) ----



def test_active_picks_new_unique_rows_least_confident_first():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(1000, 5))
    first = np.arange(50)
    model = train_student(X, first, (X[first, 0] > 0).astype(int))
    new = pick_unconfident(model, X, first, 100)
    assert len(set(new.tolist())) == 100 and not set(new.tolist()) & set(first.tolist())
    c = confidence(model, X[new])
    assert (np.diff(c) >= 0).all()                                 # least sure first
    rest = np.setdiff1d(np.arange(1000), np.concatenate([first, new]))
    assert c.max() <= confidence(model, X[rest]).min()             # nobody left out is less sure


def test_active_request_keeps_round1_rows_first():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(500, 5))
    first = rng.permutation(500)[:40]
    model = train_student(X, first, rng.integers(0, 2, 40))
    req = build_request(first, 200, model, X)
    assert len(req) == 200 and len(set(req.tolist())) == 200
    assert (req[:40] == first).all()
    with pytest.raises(ValueError):
        build_request(first, 600, model, X)                        # more rows than exist