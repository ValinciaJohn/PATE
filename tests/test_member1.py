# tests/test_member1.py
# Tests for Member 1. Run:  python -m pytest tests/test_member1.py
import numpy as np
from common.contracts import ExperimentConfig, check_votes
from common import paths
from member1_teachers.teachers import make_partitions
from member1_teachers.pipeline import build_data_and_teachers


def make_cfg(run_id="test_run", n_teachers=10, seed=42):
    return ExperimentConfig(run_id=run_id, n_teachers=n_teachers, noise_scale=2, n_queries=100, seed=seed)


def test_partitions_are_separate_and_cover_everything():
    groups = make_partitions(1003, 10)
    all_rows = np.concatenate(groups)
    assert len(all_rows) == len(set(all_rows)) == 1003


def test_shapes_and_vote_range_match_manifest():
    cfg = make_cfg()
    m = build_data_and_teachers(cfg)
    votes = np.load(paths.artifact_path(cfg.run_id, paths.TEACHER_VOTES))
    check_votes(votes, m)
    assert np.load(paths.artifact_path(cfg.run_id, paths.X_PUBLIC)).shape == (m.n_public, m.n_features)


def test_same_seed_gives_same_votes():
    a, b = make_cfg("run_a"), make_cfg("run_b")
    build_data_and_teachers(a); build_data_and_teachers(b)
    va = np.load(paths.artifact_path("run_a", paths.TEACHER_VOTES))
    vb = np.load(paths.artifact_path("run_b", paths.TEACHER_VOTES))
    assert (va == vb).all()