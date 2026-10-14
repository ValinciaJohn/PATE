"""Optional experiment C10 (paper Appendix B.2): active learning vs random queries.
Lives at the repo root because it calls Member 1's, Member 2's and Member 3's entry points
(like run_step_a.py). It only uses their public functions and changes none of their code.

  python run_active.py        # 250 teachers, noise scale 20, 500 and 1000 queries, seeds 42, 43, 44
                              # needs the main grid done first (python -m member3_student.plots)

For each setting:
  round 1  Member 2 answers N_INIT random public rows -> a first student is trained on them
  pick     the first student picks the rows it is least sure about (member3_student/active.py)
  round 2  Member 2 answers [round-1 rows, same order] + [least-sure rows], via query_request.npy,
           and the final student is trained on all of them (train_and_audit)
Compared with the main-grid run at the same setting, where every row is random.

Privacy: round 2 re-derives the round-1 answers with the same seed and order, so they come out
identical (checked below) and are not new answers; round 2's epsilon covers every answered row once.
Which rows round 2 asks depends on round-1 answers; the moments accountant allows such adaptive
steps (paper Theorem 1, composability)."""
from itertools import product
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import paths
from common.contracts import ExperimentConfig, PrivacyReport, ResultRow
from member1_teachers.pipeline import build_data_and_teachers
from member2_privacy.aggregate import aggregate_and_account
from member3_student.student import train_student
from member3_student.active import build_request
from member3_student.evaluate import train_and_audit

TEACHERS, SCALE, N_INIT = 250, 20, 100
TOTALS, SEEDS = [500, 1000], [42, 43, 44]
REQUEST = "query_request.npy"          # the file name Member 2's aggregate.py looks for
RESULTS = Path("results")


def load(run_id, name):
    return np.load(paths.artifact_path(run_id, name))


def read_json(model, run_id, name):
    return model.model_validate_json((Path(paths.ARTIFACT_ROOT) / run_id / name).read_text())


def one_setting(total, seed):
    tag = f"t{TEACHERS}_b{SCALE}_q{total}_s{seed}"            # same name as the main-grid run

    # round 1: N_INIT random rows
    r1 = ExperimentConfig(run_id=f"al_{tag}_round1", seed=seed, n_teachers=TEACHERS,
                          noise_scale=SCALE, n_queries=N_INIT)
    build_data_and_teachers(r1)
    paths.artifact_path(r1.run_id, REQUEST).unlink(missing_ok=True)   # make sure round 1 is random
    aggregate_and_account(r1)
    idx1, labels1 = load(r1.run_id, paths.QUERY_IDX), load(r1.run_id, paths.NOISY_LABELS)

    # pick: the first student chooses the rows it is least sure about
    X_public = load(r1.run_id, paths.X_PUBLIC)
    first_student = train_student(X_public, idx1, labels1, seed)
    request = build_request(idx1, total, first_student, X_public, seed)

    # round 2: round-1 rows (same order) + least-sure rows
    r2 = ExperimentConfig(run_id=f"al_{tag}", seed=seed, n_teachers=TEACHERS,
                          noise_scale=SCALE, n_queries=total)
    build_data_and_teachers(r2)
    np.save(paths.artifact_path(r2.run_id, REQUEST), request)
    active_report = aggregate_and_account(r2)
    if not (load(r2.run_id, paths.NOISY_LABELS)[:N_INIT] == labels1).all():
        raise RuntimeError(f"{tag}: the round-1 rows got different noisy answers in round 2, so they "
                           "would be paid for twice and the epsilon would be too small. Stopping.")
    active = train_and_audit(r2)

    # the random run from the main grid (Step 8)
    if not (Path(paths.ARTIFACT_ROOT) / tag / paths.RESULT_ROW).exists():
        raise FileNotFoundError(f"{tag} has no result_row.json: run  python -m member3_student.plots  first")
    random = read_json(ResultRow, tag, paths.RESULT_ROW)
    random_report = read_json(PrivacyReport, tag, paths.PRIVACY_REPORT)

    return dict(n_teachers=TEACHERS, noise_scale=SCALE, n_queries=total, seed=seed,
                eps_random=random.epsilon, eps_active=active.epsilon,
                acc_random=random.student_acc, acc_active=active.student_acc,
                label_acc_random=random_report.label_accuracy, label_acc_active=active_report.label_accuracy,
                mia_random=random.mia_student, mia_active=active.mia_student)


def plot(summary):
    x, w = np.arange(len(summary)), 0.35
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for k, (name, label) in enumerate([("random", "random queries"), ("active", "least-confident first")]):
        bars = ax.bar(x + (k - 0.5) * w, 100 * summary[f"acc_{name}_mean"], w,
                      yerr=100 * summary[f"acc_{name}_std"].fillna(0), capsize=3, label=label)
        for bar, eps in zip(bars, summary[f"eps_{name}_mean"]):
            ax.text(bar.get_x() + bar.get_width() / 2, 70.5, f"ε {eps:.2f}", ha="center", fontsize=8)
    ax.set_xticks(x, [f"{q} queries" for q in summary.n_queries])
    ax.set_ylim(70, 90); ax.set_ylabel("student test accuracy (%)")
    ax.set_title(f"Active learning vs random ({TEACHERS} teachers, noise scale {SCALE}, mean ± std over seeds)",
                 fontsize=9)
    ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3)
    out = RESULTS / "fig_m3_active_learning.png"
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)
    return out


if __name__ == "__main__":
    rows = []
    for total, seed in product(TOTALS, SEEDS):
        r = one_setting(total, seed)
        rows.append(r)
        print(f"q={total:4d} seed={seed}  random: acc={r['acc_random']:.3f} eps={r['eps_random']:.2f}   "
              f"active: acc={r['acc_active']:.3f} eps={r['eps_active']:.2f}")
    df = pd.DataFrame(rows)
    RESULTS.mkdir(exist_ok=True)
    df.to_csv(RESULTS / "active_learning.csv", index=False)
    cols = [c for c in df.columns if c.startswith(("eps_", "acc_", "label_acc_", "mia_"))]
    summary = df.groupby("n_queries")[cols].agg(["mean", "std"])
    summary.columns = [f"{c}_{s}" for c, s in summary.columns]
    summary = summary.reset_index()
    summary.to_csv(RESULTS / "active_learning_summary.csv", index=False)
    print(summary[["n_queries", "acc_random_mean", "acc_active_mean", "eps_random_mean", "eps_active_mean",
                   "label_acc_random_mean", "label_acc_active_mean"]].round(3).to_string(index=False))
    print("plot:", plot(summary))