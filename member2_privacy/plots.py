"""Member 2 plots and results table (moved from Member 3, change C14).

Plot 1  paper Sec. 4.1, Fig. 2: noisy-label accuracy vs gamma, one line per teacher count
Plot 2  paper Sec. 4.1, Fig. 3: vote gap (top minus second, divided by n teachers) vs number of teachers
Plot 3  paper Sec. 4.2, Fig. 4: epsilon vs number of queries (data-dependent, data-independent, simple composition)
Plot 4  extension (illustrates Lemma 4, Sec. 3.3): per-query privacy cost vs teacher vote gap
Plot 5  extension (Sec. 3.3, "How does the number of teachers affect the privacy cost?"): epsilon, label accuracy
        and average teacher accuracy vs number of teachers
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
from common.contracts import DataManifest, PrivacyReport
from . import accountant as acc
from ._io import art_root, results_dir
from .aggregate import vote_counts


def _root():
    return art_root()


def _results():
    return results_dir()


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


def plot_cost_vs_gap(run_id="t250_b20_q500_s42"):
    """Extension 3: privacy cost of each answered query vs how far ahead the winning class is.
    Cost of query i at the best lambda* is alpha_i(lambda*) / lambda* (its share of epsilon beyond the
    ln(1/delta)/lambda* term). Lemma 4 / Theorem 3: a wide gap makes the outcome near-certain, so the
    data-dependent cost falls far below the data-independent Theorem 2 cost (dashed line)."""
    d = _root() / run_id
    m = DataManifest.model_validate_json((d / paths.MANIFEST).read_text())
    rep = PrivacyReport.model_validate_json((d / paths.PRIVACY_REPORT).read_text())
    idx = np.load(d / paths.QUERY_IDX)
    counts = vote_counts(np.load(d / paths.TEACHER_VOTES)[:, idx], m.n_classes)
    gamma = 1.0 / rep.noise_scale
    alpha = acc.alpha_dependent(counts, gamma)                       # (queries, lambda)
    eps, lam = acc.eps_from_alpha(alpha.sum(axis=0), rep.delta)
    i = int(lam) - 1
    cost = alpha[:, i] / lam
    srt = np.sort(counts, axis=1)
    gap = srt[:, -1] - srt[:, -2]
    indep = acc.alpha_independent(1, gamma)[0, i] / lam
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.scatter(gap, np.maximum(cost, 1e-12), s=10, alpha=.5, label="data-dependent cost per query (Thm 3)")
    ax.axhline(indep, ls="--", color="C1", label="data-independent cost per query (Thm 2)")
    ax.set_yscale("log")
    ax.set_xlabel("teacher vote gap (top count minus second count)")
    ax.set_ylabel(f"cost per query at lambda = {lam}")
    ax.set_title(f"Close votes cost the most ({m.n_teachers} teachers, noise scale {rep.noise_scale:g}, {rep.n_queries} queries)",
                 fontsize=9)
    ax.grid(alpha=.3); ax.legend(fontsize=8); fig.tight_layout()
    out = _results() / "fig5_cost_vs_gap.png"; fig.savefig(out, dpi=150); plt.close(fig)
    return out


def plot_teacher_tradeoff(df, n_queries=500, noise_scale=20):
    """Extension 4: more teachers widen the vote gap (lower epsilon, better noisy labels) but each teacher
    sees less data (lower teacher accuracy). Sec. 3.3, last paragraph: "true up to a point".
    Teacher accuracy is measured on the public pool against y_public_true (a diagnostic; Member 2 may read it)."""
    d = df[(df.n_queries == n_queries) & (df.noise_scale == noise_scale)].copy()
    d["teacher_acc"] = [float((np.load(Path(p) / paths.TEACHER_VOTES) == np.load(Path(p) / paths.Y_PUBLIC_TRUE)).mean())
                        for p in d["dir"]]
    g = d.groupby("n_teachers")[["epsilon", "label_accuracy", "teacher_acc"]].mean().sort_index()
    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    ax.plot(g.index, g.epsilon, "o-", color="C0", label="epsilon (data-dependent)")
    ax.set_xscale("log"); ax.set_xticks(list(g.index)); ax.set_xticklabels([str(n) for n in g.index])
    ax.set_xlabel("number of teachers"); ax.set_ylabel("epsilon", color="C0")
    ax2 = ax.twinx()
    ax2.plot(g.index, 100 * g.label_accuracy, "s--", color="C1", label="noisy-label accuracy (%)")
    ax2.plot(g.index, 100 * g.teacher_acc, "^:", color="C2", label="average teacher accuracy (%)")
    ax2.set_ylabel("accuracy (%)")
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    fig.legend(h1 + h2, l1 + l2, fontsize=8, loc="lower center", ncol=3, frameon=False)
    ax.set_title(f"Teacher-count trade-off (noise scale {noise_scale:g}, {n_queries} queries)", fontsize=10)
    ax.grid(alpha=.3); fig.tight_layout(rect=(0, 0.07, 1, 1))
    out = _results() / "fig6_teacher_tradeoff.png"; fig.savefig(out, dpi=150); plt.close(fig)
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


def main(n_queries=500, ref_teachers=250, ref_scale=20, ref_seed=42):
    df = load_runs()
    out = [plot_accuracy_vs_noise(df, n_queries), plot_vote_gap(df),
           plot_epsilon_vs_queries(df, ref_teachers, ref_scale)]
    run_id = f"t{ref_teachers}_b{int(ref_scale)}_q{n_queries}_s{ref_seed}"
    if (_root() / run_id / paths.PRIVACY_REPORT).exists():
        out.append(plot_cost_vs_gap(run_id))
    ref = df[(df.n_queries == n_queries) & (df.noise_scale == ref_scale)]
    if ref.n_teachers.nunique() >= 2:
        out.append(plot_teacher_tradeoff(df, n_queries, ref_scale))
    epsilon_table(df)
    return out


if __name__ == "__main__":
    for p in main():
        print("wrote", p)
