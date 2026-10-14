"""Random artifacts in exactly the contract shapes/names, so every member can build before real data exists."""
import numpy as np
from .contracts import ExperimentConfig, DataManifest
from . import paths
from pathlib import Path


def make_dummy_artifacts(cfg: ExperimentConfig, n_public=1000, n_private=2000, n_test=500,
                         n_holdout=300, n_features=20, n_classes=2) -> DataManifest:
    rng = np.random.default_rng(cfg.seed)
    d = Path(paths.artifact_dir(cfg.run_id))
    m = DataManifest(run_id=cfg.run_id, n_features=n_features, n_classes=n_classes,
                     n_private=n_private, n_public=n_public, n_test=n_test,
                     n_holdout=n_holdout, n_teachers=cfg.n_teachers)
    (d / paths.MANIFEST).write_text(m.model_dump_json())
    for name, n in [("private", n_private), ("public", n_public), ("test", n_test), ("holdout", n_holdout)]:
        np.save(d / f"X_{name}.npy", rng.normal(size=(n, n_features)))
        if name != "public":
            np.save(d / f"y_{name}.npy", rng.integers(0, n_classes, n))
    y_true = rng.integers(0, n_classes, n_public)
    np.save(d / paths.Y_PUBLIC_TRUE, y_true)
    # biased votes: each teacher agrees with the hidden label ~80% on average (per-query rate in U(0.6, 1.0))
    p = rng.uniform(0.6, 1.0, n_public)
    agree = rng.random((cfg.n_teachers, n_public)) < p
    other = (y_true + rng.integers(1, n_classes, (cfg.n_teachers, n_public))) % n_classes
    np.save(d / paths.TEACHER_VOTES, np.where(agree, y_true, other).astype(int))
    return m
