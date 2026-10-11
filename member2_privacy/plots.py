"""Member 2 plots and results table (moved from Member 3, change C14).

Plot 1  paper Sec. 4.1, Fig. 2: noisy-label accuracy vs gamma, one line per teacher count
Plot 2  paper Sec. 4.1, Fig. 3: vote gap (top minus second, divided by n teachers) vs number of teachers
Plot 3  paper Sec. 4.2, Fig. 4: epsilon vs number of queries (data-dependent, data-independent, simple composition)
Table   epsilon, delta, queries (mean and std over seeds)

Reads artifacts/<run_id>/privacy_report.json + accountant_details.json (+ teacher_votes.npy for Plot 2).
Writes to results/."""
import json
import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import paths


def _root():
    return Path(os.environ.get("PATE_ARTIFACT_ROOT", paths.ROOT / "artifacts"))


def _results():
    d = Path(os.environ.get("PATE_RESULTS_DIR", paths.ROOT / "results"))
    d.mkdir(parents=True, exist_ok=True)
    return d


def load_runs():
    rows = []
    for d in sorted(_root().iterdir()):
        rp = d / "privacy_report.json"
        if not rp.exists():
            continue
        r = json.loads(rp.read_text())
        det = d / "accountant_details.json"
        if det.exists():
            r.update(json.loads(det.read_text()))
        r["dir"] = str(d)
        rows.append(r)
    return pd.DataFrame(rows)


def plot_accuracy_vs_noise(df, n_queries=500):
    """Fig. 2: accuracy of the noisy aggregation vs gamma = 1/noise_scale."""
    d = df[df.n_queries == n_queries]
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for n, g in d.groupby("n_teachers"):
        s = g.groupby("noise_scale").label_accuracy.mean().sort_index()
        ax.plot(1.0 / s.index.values, 100 * s.values, marker="o", label=f"{n} teachers")
    ax.set_xscale("log")
    ax.set_xlabel("gamma = 1 / noise scale   (larger gamma = less noise)")
    ax.set_ylabel("accuracy of noisy labels (%)")
    ax.set_title(f"Noisy-label accuracy vs noise ({n_queries} queries)")
    ax.grid(alpha=.3); ax.legend(); fig.tight_layout()
    out = _results() / "fig2_accuracy_vs_noise.png"; fig.savefig(out, dpi=150); plt.close(fig)
    return out


def vote_gap(votes):
    """Mean over queries of (top count - second count) / n_teachers."""
    n_classes = int(votes.max()) + 1
    counts = np.stack([(votes == j).sum(axis=0) for j in range(n_classes)], axis=0)
    top2 = np.sort(counts, axis=0)[-2:]
    return float(((top2[1] - top2[0]) / votes.shape[0]).mean())


def plot_vote_gap(df):
    """Fig. 3: how decisive the teachers are, vs number of teachers."""
    d = df.drop_duplicates(["n_teachers", "seed"])
    d = d.assign(gap=[vote_gap(np.load(Path(p) / paths.TEACHER_VOTES)) for p in d["dir"]])
    s = d.groupby("n_teachers").gap.mean().sort_index()
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.bar([str(n) for n in s.index], 100 * s.values)
    ax.set_xlabel("number of teachers"); ax.set_ylabel("vote gap / number of teachers (%)")
    ax.set_title("How certain is the aggregation? (normalized vote gap)")
    ax.grid(alpha=.3, axis="y"); fig.tight_layout()
    out = _results() / "fig3_vote_gap.png"; fig.savefig(out, dpi=150); plt.close(fig)
    return out


def plot_epsilon_vs_queries(df, n_teachers=250, noise_scale=20):
    """Fig. 4 columns: epsilon for each number of queries at the reference setting."""
    d = df[(df.n_teachers == n_teachers) & (df.noise_scale == noise_scale)]
    g = d.groupby("n_queries")[["epsilon", "eps_data_independent", "eps_strong_composition"]].mean().sort_index()
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.plot(g.index, g.epsilon, marker="o", label="data-dependent (Thm 3)")
    ax.plot(g.index, g.eps_data_independent, marker="s", ls="--", label="data-independent (Thm 2)")
    ax.plot(g.index, g.eps_strong_composition, marker="^", ls=":", label="simple composition (Sec. 3.2)")
    ax.set_xlabel("number of queries"); ax.set_ylabel(f"epsilon (delta = {d.delta.iloc[0]:g})")
    ax.set_title(f"Privacy cost vs queries ({n_teachers} teachers, noise scale {noise_scale})")
    ax.grid(alpha=.3); ax.legend(); fig.tight_layout()
    out = _results() / "fig4_epsilon_vs_queries.png"; fig.savefig(out, dpi=150); plt.close(fig)
    return out


def epsilon_table(df):
    """Table: epsilon, delta, queries for every setting, mean and std over seeds."""
    keys = ["n_teachers", "noise_scale", "n_queries", "delta"]
    g = df.groupby(keys)
    t = g.agg(epsilon_mean=("epsilon", "mean"), epsilon_std=("epsilon", "std"),
              eps_data_independent=("eps_data_independent", "mean"),
              eps_simple_composition=("eps_strong_composition", "mean"),
              label_accuracy=("label_accuracy", "mean"), n_seeds=("seed", "nunique")).reset_index()
    t.to_csv(_results() / "epsilon_table.csv", index=False)
    return t


def main(n_queries=500, ref_teachers=250, ref_scale=20):
    df = load_runs()
    out = [plot_accuracy_vs_noise(df, n_queries), plot_vote_gap(df),
           plot_epsilon_vs_queries(df, ref_teachers, ref_scale)]
    epsilon_table(df)
    return out


if __name__ == "__main__":
    for p in main():
        print("wrote", p)
