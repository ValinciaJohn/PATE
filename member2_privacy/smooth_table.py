"""Smooth-sensitivity table for the report (paper Sec. 3.3).

For every run where all lambda <= 8 are usable (noise scale >= 4, i.e. 0.5 * gamma * 8 <= 1) this computes the
Laplace noise that the smooth-sensitivity release would add to the data-dependent epsilon, and writes
results/smooth_sensitivity_table.csv (mean and max over seeds). The paper reports this noise as below 0.5 (MNIST)
and below 1.0 (SVHN) and publishes epsilon un-noised; so do we, and this table shows how large the noise would be.

Run:  python -m member2_privacy.smooth_table"""
import json
from pathlib import Path
import numpy as np
from common import paths
from .aggregate import vote_counts
from .plots import load_runs, _results
from .smooth_sensitivity import release_epsilon


def smooth_table():
    df = load_runs()
    df = df[df.noise_scale >= 4]
    rows = []
    for _, r in df.iterrows():
        d = Path(r["dir"])
        n_classes = json.loads((d / paths.MANIFEST).read_text())["n_classes"]
        counts = vote_counts(np.load(d / paths.TEACHER_VOTES)[:, np.load(d / paths.QUERY_IDX)], n_classes)
        out = release_epsilon(counts, r["noise_scale"], r["delta"], rng=0)
        rows.append(dict(n_teachers=r["n_teachers"], noise_scale=r["noise_scale"], n_queries=r["n_queries"], seed=r["seed"],
                         epsilon=out["epsilon"], release_noise_scale=out["noise_scale"]))
    import pandas as pd
    t = (pd.DataFrame(rows).groupby(["n_teachers", "noise_scale", "n_queries"])
         .agg(epsilon_mean=("epsilon", "mean"), release_noise_mean=("release_noise_scale", "mean"),
              release_noise_max=("release_noise_scale", "max"), n_seeds=("seed", "nunique")).reset_index())
    t.to_csv(_results() / "smooth_sensitivity_table.csv", index=False)
    return t


if __name__ == "__main__":
    t = smooth_table()
    print(t.round(3).to_string(index=False))
    print("\nwrote", _results() / "smooth_sensitivity_table.csv")
