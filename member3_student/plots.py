# member3_student/plots.py
# MEMBER 3: run train_and_audit on every finished run of the plan's grid, save results/results.csv,
# and draw Member 3's two plots (plan section 6, C14):
#   1. epsilon vs student accuracy (baseline accuracy as a reference line)
#   2. epsilon vs MIA success (student and baseline; 0.5 = coin flip)
# Run from the repo root:  python -m member3_student.plots          (reuses saved result_row.json)
#                          python -m member3_student.plots --redo   (recomputes every result row)
import argparse
from itertools import product
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")                              # draw to files, no window
import matplotlib.pyplot as plt
from common.contracts import ExperimentConfig, PrivacyReport, ResultRow
from common import paths
from member3_student.evaluate import train_and_audit

# the plan's sweep grid (section 2, C3): 3 x 4 x 3 settings x 3 seeds = 108 runs
GRID = dict(teachers=[10, 100, 250], scales=[1, 5, 20, 100], queries=[100, 500, 1000], seeds=[42, 43, 44])
RESULTS = Path("results")
METRICS = ["epsilon", "student_acc", "baseline_acc", "mia_student", "mia_baseline"]


def grid_configs():
    for n, b, q, s in product(GRID["teachers"], GRID["scales"], GRID["queries"], GRID["seeds"]):
        yield ExperimentConfig(run_id=f"t{n}_b{b}_q{q}_s{s}", seed=s, n_teachers=n, noise_scale=b, n_queries=q)


def collect(redo=False):
    """One ResultRow per finished run. Reuses result_row.json unless redo=True or Member 2's
    epsilon changed since it was written (then the row is stale and is recomputed)."""
    rows, missing = [], []
    for cfg in grid_configs():
        folder = Path(paths.ARTIFACT_ROOT) / cfg.run_id
        if not (folder / paths.PRIVACY_REPORT).exists():          # Member 2 has not run this one yet
            missing.append(cfg.run_id)
            continue
        report = PrivacyReport.model_validate_json((folder / paths.PRIVACY_REPORT).read_text())
        saved = folder / paths.RESULT_ROW
        row = ResultRow.model_validate_json(saved.read_text()) if saved.exists() else None
        if redo or row is None or row.epsilon != report.epsilon:
            row = train_and_audit(cfg)
            print(f"{cfg.run_id:22s} student_acc={row.student_acc:.3f}  mia_student={row.mia_student:.3f}")
        rows.append(row.model_dump())
    return pd.DataFrame(rows), missing


def summarize(df):
    """Mean and std over the 3 seeds for each of the 36 settings."""
    s = df.groupby(["n_teachers", "noise_scale", "n_queries"])[METRICS].agg(["mean", "std"])
    s.columns = [f"{metric}_{stat}" for metric, stat in s.columns]
    s["n_seeds"] = df.groupby(["n_teachers", "noise_scale", "n_queries"]).seed.nunique()
    return s.reset_index()


def plot_accuracy(summary, df):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for n, g in summary.groupby("n_teachers"):
        ax.errorbar(g.epsilon_mean, 100 * g.student_acc_mean, yerr=100 * g.student_acc_std.fillna(0),
                    fmt="o", capsize=3, label=f"student, {n} teachers")
    ax.axhline(100 * df.baseline_acc.mean(), color="black", ls="--", label="baseline (no privacy)")
    ax.set_xscale("log")
    ax.set_xlabel("epsilon (log scale, smaller = more private)")
    ax.set_ylabel("test accuracy (%)")
    ax.set_title("Privacy vs accuracy (mean ± std over seeds)")
    ax.grid(alpha=0.3); ax.legend(fontsize=8)
    out = RESULTS / "fig_m3_epsilon_vs_accuracy.png"
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)
    return out


def plot_mia(summary, df):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for n, g in summary.groupby("n_teachers"):
        ax.errorbar(g.epsilon_mean, g.mia_student_mean, yerr=g.mia_student_std.fillna(0),
                    fmt="o", capsize=3, label=f"student, {n} teachers")
    ax.axhline(df.mia_baseline.mean(), color="black", ls="--", label="baseline (no privacy)")
    ax.axhline(0.5, color="grey", ls=":", label="0.5 = coin flip (no leak)")
    ax.set_xscale("log")
    ax.set_xlabel("epsilon (log scale, smaller = more private)")
    ax.set_ylabel("MIA success (AUC)")
    ax.set_title("Privacy vs membership-inference attack (mean ± std over seeds)")
    ax.grid(alpha=0.3); ax.legend(fontsize=8)
    out = RESULTS / "fig_m3_epsilon_vs_mia.png"
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--redo", action="store_true", help="recompute every result row")
    df, missing = collect(ap.parse_args().redo)
    if missing:
        print(f"{len(missing)} of 108 runs not ready yet (no privacy_report.json), e.g. {missing[0]}")
    if df.empty:
        raise SystemExit("no finished runs: first run  python run_step_a.py run --grid main")
    RESULTS.mkdir(exist_ok=True)
    df.to_csv(RESULTS / "results.csv", index=False)
    summary = summarize(df)
    summary.to_csv(RESULTS / "m3_summary.csv", index=False)
    print(f"{len(df)} runs -> results/results.csv, results/m3_summary.csv")
    print("plots:", plot_accuracy(summary, df), plot_mia(summary, df))