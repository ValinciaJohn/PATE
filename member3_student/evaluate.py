# member3_student/evaluate.py
# MEMBER 3: the one function the rest of the team calls: train_and_audit(cfg) -> ResultRow.
# Reads Member 1's and Member 2's files from artifacts/<run_id>/, trains the student and the baseline,
# grades both on the test set, runs the attack on both, and writes result_row.json.
# Never loads y_public_true.npy (the real answers of the public rows).
import numpy as np
from common.contracts import ExperimentConfig, DataManifest, PrivacyReport, ResultRow, check_labels
from common import paths
from member3_student.student import train_student
from member3_student.baseline import train_baseline
from member3_student.mia import run_mia


def load(run_id, file_name):
    return np.load(paths.artifact_path(run_id, file_name))


def train_and_audit(cfg: ExperimentConfig) -> ResultRow:
    run = cfg.run_id

    # 1. read the small summary files
    manifest = DataManifest.model_validate_json(paths.artifact_path(run, paths.MANIFEST).read_text())
    privacy = PrivacyReport.model_validate_json(paths.artifact_path(run, paths.PRIVACY_REPORT).read_text())

    # 2. read the data
    X_public = load(run, paths.X_PUBLIC)                                   # from Member 1
    query_idx, noisy_labels = load(run, paths.QUERY_IDX), load(run, paths.NOISY_LABELS)   # from Member 2
    check_labels(query_idx, noisy_labels, manifest)                       # stop if Member 2's files look wrong
    X_private, y_private = load(run, paths.X_PRIVATE), load(run, paths.Y_PRIVATE)
    X_test, y_test = load(run, paths.X_TEST), load(run, paths.Y_TEST)
    X_holdout, y_holdout = load(run, paths.X_HOLDOUT), load(run, paths.Y_HOLDOUT)

    # 3. train both models
    student = train_student(X_public, query_idx, noisy_labels, cfg.seed)
    baseline = train_baseline(X_private, y_private, cfg.seed)

    # 4. grade on the test set (people neither model has seen) + 5. run the attack on both
    row = ResultRow(
        run_id=run, seed=cfg.seed, n_teachers=cfg.n_teachers,
        noise_scale=cfg.noise_scale, n_queries=cfg.n_queries,
        epsilon=privacy.epsilon,                                            # copied from Member 2
        student_acc=float(student.score(X_test, y_test)),
        baseline_acc=float(baseline.score(X_test, y_test)),
        mia_student=run_mia(student, X_private, y_private, X_holdout, y_holdout, cfg.seed),
        mia_baseline=run_mia(baseline, X_private, y_private, X_holdout, y_holdout, cfg.seed),
    )

    # 6. save
    paths.artifact_path(run, paths.RESULT_ROW).write_text(row.model_dump_json(indent=2))
    return row


if __name__ == "__main__":
    # the team's reference run: 250 teachers, noise scale 20, 500 queries, seed 42
    cfg = ExperimentConfig(run_id="t250_b20_q500_s42", n_teachers=250, noise_scale=20, n_queries=500, seed=42)
    print(train_and_audit(cfg).model_dump_json(indent=2))