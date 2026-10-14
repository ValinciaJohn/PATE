"""Integration Step A (Member 1 + Member 2): run Member 1's REAL pipeline, then Member 2's aggregation.
Lives at the repo root because it imports both members' entry points (the plan lets only integration code do that).

  python run_step_a.py check                 # does common/ + Member 1's code match what Member 2 expects?
  python run_step_a.py run --grid ref        # one reference run: 250 teachers, scale 20, 500 queries, seed 42
  python run_step_a.py run --grid main       # the plan's grid: 36 settings x 3 seeds = 108 runs
  python run_step_a.py run --grid tradeoff   # extension 4: 25 and 50 teachers at the reference setting
  python run_step_a.py validate              # sanity summary of the real artifacts (votes, agreement, accuracy)
"""
import argparse
import sys
from itertools import product
import numpy as np
from common import paths
from common.contracts import ExperimentConfig, DataManifest, check_votes
from member2_privacy._io import art_dir, art_root

GRIDS = {
    "ref":      dict(teachers=[250], scales=[20], queries=[500], seeds=[42]),
    "main":     dict(teachers=[10, 100, 250], scales=[1, 5, 20, 100], queries=[100, 500, 1000], seeds=[42, 43, 44]),
    "tradeoff": dict(teachers=[25, 50], scales=[20], queries=[500], seeds=[42, 43, 44]),
}


def configs(name):
    g = GRIDS[name]
    for n, b, q, s in product(g["teachers"], g["scales"], g["queries"], g["seeds"]):
        yield ExperimentConfig(run_id=f"t{n}_b{b}_q{q}_s{s}", seed=s, n_teachers=n, noise_scale=b, n_queries=q)


def check():
    import common.contracts as C
    ok = True
    for name in ["ARTIFACT_ROOT", "MANIFEST", "TEACHER_VOTES", "Y_PUBLIC_TRUE", "QUERY_IDX", "NOISY_LABELS",
                 "PRIVACY_REPORT", "RESULT_ROW", "artifact_dir", "artifact_path"]:
        has = hasattr(paths, name)
        ok &= has
        print(f"common.paths.{name:15s} {'ok' if has else 'MISSING -> Member 2 code uses it'}")
    for model, fields in {"ExperimentConfig": ["run_id", "seed", "n_teachers", "noise_scale", "n_queries", "delta"],
                          "DataManifest": ["n_classes", "n_public", "n_teachers"],
                          "PrivacyReport": ["run_id", "seed", "epsilon", "delta", "noise_scale", "n_teachers", "n_queries", "label_accuracy"],
                          "ResultRow": ["run_id", "seed"]}.items():
        have = set(getattr(C, model).model_fields)
        miss = [f for f in fields if f not in have]
        ok &= not miss
        print(f"contracts.{model:17s} {'ok' if not miss else 'MISSING FIELDS: ' + ', '.join(miss)}")
    try:
        from member1_teachers.pipeline import build_data_and_teachers  # noqa: F401
        print("member1_teachers.pipeline.build_data_and_teachers  ok")
    except Exception as e:
        ok = False
        print(f"member1_teachers.pipeline.build_data_and_teachers  PROBLEM: {e}")
    try:
        import ucimlrepo  # noqa: F401
        print("ucimlrepo (Member 1's pipeline downloads Adult with it)  ok")
    except Exception:
        ok = False
        print("ucimlrepo  MISSING -> run: pip install ucimlrepo   (and ask Member 1 to add it to requirements.txt)")
    print(f"artifacts folder: {art_root().resolve()}   (run every command from the repo root)")
    return 0 if ok else 1


def run(grid, rebuild=False):
    from member1_teachers.pipeline import build_data_and_teachers
    from member2_privacy.aggregate import aggregate_and_account
    for cfg in configs(grid):
        d = art_dir(cfg.run_id)
        if rebuild or not (d / paths.TEACHER_VOTES).exists():
            build_data_and_teachers(cfg)
        r = aggregate_and_account(cfg)
        acc = "n/a" if r.label_accuracy is None else f"{r.label_accuracy:.3f}"
        print(f"{cfg.run_id:22s} epsilon={r.epsilon:7.3f}  noisy-label accuracy={acc}")


def validate():
    root = art_root()
    rows = []
    for d in sorted(p for p in root.iterdir() if (p / paths.TEACHER_VOTES).exists()):
        m = DataManifest.model_validate_json((d / paths.MANIFEST).read_text())
        votes = np.load(d / paths.TEACHER_VOTES)
        check_votes(votes, m)
        counts = np.stack([(votes == j).sum(axis=0) for j in range(m.n_classes)])
        srt = np.sort(counts, axis=0)
        gap = float(((srt[-1] - srt[-2]) / m.n_teachers).mean())
        unanimous = float((srt[-1] == m.n_teachers).mean())
        yt = d / paths.Y_PUBLIC_TRUE
        tacc = float((votes == np.load(yt)).mean()) if yt.exists() else float("nan")
        rows.append((d.name, m.n_teachers, m.n_public, tacc, gap, unanimous))
    seen = set()
    print(f"{'run':22s} {'teachers':>8s} {'n_public':>8s} {'teacher acc':>11s} {'norm. gap':>9s} {'unanimous':>9s}")
    for r in rows:
        key = (r[1], r[2], round(r[3], 6))
        if key in seen:        # same ensemble reused across runs: print once
            continue
        seen.add(key)
        print(f"{r[0]:22s} {r[1]:8d} {r[2]:8d} {r[3]:11.3f} {r[4]:9.3f} {r[5]:9.3f}")
    print("\nWhat to look for: teacher accuracy near or above the majority-class rate (about 0.76 on Adult;"
          " the paper's Adult teachers are around 0.8); normalized gap clearly above 0 and rising with"
          " the number of teachers; unanimous fraction not exactly 0 or 1.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("check")
    r = sub.add_parser("run"); r.add_argument("--grid", choices=GRIDS, default="ref"); r.add_argument("--rebuild", action="store_true")
    sub.add_parser("validate")
    a = ap.parse_args()
    if a.cmd == "check": sys.exit(check())
    elif a.cmd == "run": run(a.grid, a.rebuild)
    else: validate()
