"""Member 2's own dummy artifacts, for unit tests and `dev_sweep` (common/dummy.py is still empty).
Writes only what Member 2 reads: manifest.json, teacher_votes.npy, y_public_true.npy.
Votes are biased: each teacher agrees with a hidden true label ~80% of the time, with the per-query rate
varying between 60% and 100%, so both strong and weak quorums appear."""
import numpy as np
from common import paths
from common.contracts import ExperimentConfig, DataManifest
from ._io import art_dir


def make_dummy_artifacts(cfg: ExperimentConfig, n_public=1000, n_classes=2) -> DataManifest:
    rng = np.random.default_rng(cfg.seed)
    d = art_dir(cfg.run_id)
    m = DataManifest(run_id=cfg.run_id, n_features=13, n_classes=n_classes, n_private=2000, n_public=n_public,
                     n_test=500, n_holdout=300, n_teachers=cfg.n_teachers)
    (d / paths.MANIFEST).write_text(m.model_dump_json())
    y_true = rng.integers(0, n_classes, n_public)
    np.save(d / paths.Y_PUBLIC_TRUE, y_true)
    p = rng.uniform(0.6, 1.0, n_public)
    agree = rng.random((cfg.n_teachers, n_public)) < p
    other = (y_true + rng.integers(1, n_classes, (cfg.n_teachers, n_public))) % n_classes
    np.save(d / paths.TEACHER_VOTES, np.where(agree, y_true, other).astype(int))
    return m
