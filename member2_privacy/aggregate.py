"""Bucket 6 / entry point: aggregate_and_account(cfg) -> PrivacyReport.
Reads manifest + teacher_votes, answers n_queries public queries (random, or the rows in the optional
query_request.npy from Member 3) with noisy-max,
writes query_idx.npy, noisy_labels.npy, privacy_report.json (+ accountant_details.json)."""
import json
import numpy as np
from common.contracts import ExperimentConfig, DataManifest, PrivacyReport, check_votes, check_labels
from common import paths
from .noisy_max import noisy_argmax
from . import accountant
from ._io import art_dir


def vote_counts(votes, n_classes):
    """(n_teachers, Q) labels -> (Q, n_classes) counts n_j(x)."""
    return np.stack([(votes == j).sum(axis=0) for j in range(n_classes)], axis=1)


def aggregate_and_account(cfg: ExperimentConfig) -> PrivacyReport:
    d = art_dir(cfg.run_id)
    m = DataManifest.model_validate_json((d / paths.MANIFEST).read_text())
    votes = np.load(d / paths.TEACHER_VOTES)
    check_votes(votes, m)
    if cfg.n_queries > m.n_public:
        raise ValueError("n_queries exceeds public pool size")

    # Optional input from Member 3 (Appendix B.2 experiment). If the file is absent, we pick
    # random rows (seeded), so this module runs and tests fine without Member 3.
    req = d / "query_request.npy"
    if req.exists():
        idx = np.load(req)          # keep Member 3's order (least confident first)
        if len(idx) != cfg.n_queries or idx.min() < 0:
            raise ValueError("query_request.npy must hold n_queries valid, unique public indices")
    else:
        # first n_queries of one seeded permutation: sets for 100 < 500 < 1000 queries are nested (as the paper's
        # "first N inputs"), so epsilon can only grow with the number of queries for a fixed seed and noise scale
        idx = np.sort(np.random.default_rng(cfg.seed).permutation(m.n_public)[:cfg.n_queries])
    counts = vote_counts(votes[:, idx], m.n_classes)
    labels = noisy_argmax(counts, cfg.noise_scale, np.random.default_rng(cfg.seed + 10_000))
    check_labels(idx, labels, m)
    np.save(d / paths.QUERY_IDX, idx)
    np.save(d / paths.NOISY_LABELS, labels)

    eps_dd = accountant.epsilon(counts, cfg.noise_scale, cfg.delta, data_dependent=True)
    eps_di = accountant.epsilon(counts, cfg.noise_scale, cfg.delta, data_dependent=False)
    eps_sc = accountant.eps_strong_composition(cfg.noise_scale, cfg.n_queries, cfg.delta)

    acc = None  # diagnostic only; Member 2 is the only reader of y_public_true
    yt = d / paths.Y_PUBLIC_TRUE
    if yt.exists():
        acc = float((labels == np.load(yt)[idx]).mean())

    report = PrivacyReport(run_id=cfg.run_id, seed=cfg.seed,
                       epsilon=eps_dd, delta=cfg.delta,
                       noise_scale=cfg.noise_scale, n_teachers=cfg.n_teachers,
                       n_queries=cfg.n_queries, label_accuracy=acc)
    (d / paths.PRIVACY_REPORT).write_text(report.model_dump_json())
    (d / "accountant_details.json").write_text(json.dumps(
        dict(eps_data_dependent=eps_dd, eps_data_independent=eps_di, eps_strong_composition=eps_sc)))
    return report
