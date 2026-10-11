"""Member 2 dev helper: run the privacy side of the sweep on DUMMY artifacts.
Same grid as the plan (teachers x noise scale x queries x seeds). Replace with real
votes later: `run_sweep.py` (integration) does the same loop with Member 1's real pipeline."""
from itertools import product
from common.contracts import ExperimentConfig
from common.dummy import make_dummy_artifacts
from .aggregate import aggregate_and_account

TEACHERS = [10, 100, 250]
SCALES = [1, 5, 20, 100]          # gamma = 1, 0.2, 0.05, 0.01  (paper Sec. 4.1)
QUERIES = [100, 500, 1000]        # paper Sec. 4.2
SEEDS = [42, 43, 44]


def run(teachers=TEACHERS, scales=SCALES, queries=QUERIES, seeds=SEEDS, n_public=2000):
    for n, b, q, s in product(teachers, scales, queries, seeds):
        cfg = ExperimentConfig(run_id=f"t{n}_b{b}_q{q}_s{s}", seed=s,
                               n_teachers=n, noise_scale=b, n_queries=q)
        make_dummy_artifacts(cfg, n_public=n_public)
        aggregate_and_account(cfg)


if __name__ == "__main__":
    run()
    print("done")
