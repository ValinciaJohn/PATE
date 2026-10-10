# Tests for Member 1. Run:  python -m pytest tests/test_member1.py
import numpy as np
import pandas as pd
from common.contracts import ExperimentConfig, check_votes
from common import paths
from member1_teachers.preprocess import load_adult, make_transformer
from member1_teachers.teachers import make_partitions
from member1_teachers.pipeline import build_data_and_teachers


def make_cfg(run_id="test_run", n_teachers=10, seed=42):
    return ExperimentConfig(run_id=run_id, n_teachers=n_teachers, noise_scale=20, n_queries=100, seed=seed)


def test_partitions_are_separate_and_cover_everything():
    groups = make_partitions(1003, 10)
    all_rows = np.concatenate(groups)
    assert len(all_rows) == len(set(all_rows)) == 1003


def test_shapes_and_vote_range_match_manifest():
    cfg = make_cfg()
    m = build_data_and_teachers(cfg)
    votes = np.load(paths.artifact_path(cfg.run_id, paths.TEACHER_VOTES))
    check_votes(votes, m)
    X_public = np.load(paths.artifact_path(cfg.run_id, paths.X_PUBLIC))
    assert X_public.shape == (m.n_public, m.n_features)


def test_same_seed_gives_same_votes():
    a, b = make_cfg("run_a"), make_cfg("run_b")
    build_data_and_teachers(a)
    build_data_and_teachers(b)
    va = np.load(paths.artifact_path("run_a", paths.TEACHER_VOTES))
    vb = np.load(paths.artifact_path("run_b", paths.TEACHER_VOTES))
    assert (va == vb).all()


def test_text_becomes_integers_and_unseen_becomes_minus_one():
    private = pd.DataFrame({"job": ["a", "b", "a", "c"], "age": [20, 30, 40, 50]})
    other = pd.DataFrame({"job": ["b", "zzz"], "age": [25, 35]})      # "zzz" never seen in private
    out = make_transformer(private).transform(other)
    assert out.shape == (2, 2)                       # one column per feature: no one-hot
    assert out[1, 0] == -1                           # unseen category -> -1
    assert out[0, 1] == 25                           # numbers are not scaled


def test_no_one_hot_no_missing_values():
    X, _ = load_adult()
    cfg = make_cfg("run_cols")
    m = build_data_and_teachers(cfg)
    X_private = np.load(paths.artifact_path(cfg.run_id, paths.X_PRIVATE))
    assert m.n_features == X.shape[1]                # same number of columns as the clean table
    assert not np.isnan(X_private).any()