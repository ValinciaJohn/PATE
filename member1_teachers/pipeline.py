# MEMBER 1: the one function the rest of the team calls.
# Reads nothing from teammates. Writes the data files + teacher votes to artifacts/<run_id>/.
# The votes only depend on (number of teachers, seed), so they are trained once and reused (saved in artifacts/_cache).
# If you ever change how the teachers are trained, delete the folder artifacts/_cache.
import os
import numpy as np
from common.contracts import ExperimentConfig, DataManifest, check_votes
from common import paths
from member1_teachers.split import prepare_data
from member1_teachers.teachers import train_teachers, get_votes


def votes_for(cfg, data):
    cache_dir = os.path.join(paths.ARTIFACT_ROOT, "_cache")
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, f"votes_t{cfg.n_teachers}_s{cfg.seed}.npy")
    if os.path.exists(cache_file):
        votes = np.load(cache_file)
        if votes.shape == (cfg.n_teachers, len(data["X_public"])):
            return votes                                  # already trained for this (teachers, seed)
    teachers = train_teachers(data["X_private"], data["y_private"], cfg.n_teachers, cfg.seed)
    votes = get_votes(teachers, data["X_public"])         # teachers are NOT saved, only their votes
    np.save(cache_file, votes)
    return votes


def build_data_and_teachers(cfg: ExperimentConfig) -> DataManifest:
    data, _ = prepare_data(cfg.seed)
    votes = votes_for(cfg, data)

    save = lambda name, array: np.save(paths.artifact_path(cfg.run_id, name), array)
    save(paths.X_PRIVATE, data["X_private"]);   save(paths.Y_PRIVATE, data["y_private"])
    save(paths.X_PUBLIC, data["X_public"]);     save(paths.Y_PUBLIC_TRUE, data["y_public"])
    save(paths.X_TEST, data["X_test"]);         save(paths.Y_TEST, data["y_test"])
    save(paths.X_HOLDOUT, data["X_holdout"]);   save(paths.Y_HOLDOUT, data["y_holdout"])
    save(paths.TEACHER_VOTES, votes)

    manifest = DataManifest(
        run_id=cfg.run_id,
        n_features=data["X_private"].shape[1],
        n_classes=2,
        n_private=len(data["X_private"]),
        n_public=len(data["X_public"]),
        n_test=len(data["X_test"]),
        n_holdout=len(data["X_holdout"]),
        n_teachers=cfg.n_teachers,
    )
    check_votes(votes, manifest)                          # stop here if something is wrong
    with open(paths.artifact_path(cfg.run_id, paths.MANIFEST), "w") as f:
        f.write(manifest.model_dump_json(indent=2))
    return manifest


if __name__ == "__main__":
    # the team's reference run: 250 teachers, noise scale 20, 500 queries
    cfg = ExperimentConfig(run_id="t250_b20_q500_s42", n_teachers=250, noise_scale=20, n_queries=500, seed=42)
    manifest = build_data_and_teachers(cfg)
    print(manifest)
    print("files saved in:", paths.artifact_dir(cfg.run_id))