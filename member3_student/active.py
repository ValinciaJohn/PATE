# member3_student/active.py
# MEMBER 3: active learning (paper Appendix B.2, "identifying unconfident points using the student").
# A first student, trained on a small random batch, looks at the public rows that were NOT asked yet
# and picks the ones it is least sure about. Those are asked next, instead of random rows.
# No file I/O and no teammates' code here: run_active.py (repo root) does the two rounds.
import numpy as np


def confidence(model, X):
    """Highest class probability for each row: 0.5 = completely unsure (2 classes), 1.0 = certain."""
    return model.predict_proba(X).max(axis=1)


def pick_unconfident(model, X_public, already_asked, n_new, seed=42):
    """Row numbers of the n_new public rows the model is least sure about, least sure first.
    Rows in already_asked are skipped. Ties (random forests give many equal probabilities)
    are broken randomly with the seed, so we do not favour rows with small row numbers."""
    candidates = np.setdiff1d(np.arange(len(X_public)), np.asarray(already_asked))
    if n_new > len(candidates):
        raise ValueError(f"asked for {n_new} new rows but only {len(candidates)} are left")
    candidates = np.random.default_rng(seed).permutation(candidates)      # random order for ties
    order = np.argsort(confidence(model, X_public[candidates]), kind="stable")
    return candidates[order[:n_new]]


def build_request(first_idx, n_total, model, X_public, seed=42):
    """The query_request for round 2: the round-1 rows first (same order), then the least
    confident new rows, n_total rows in all. Every one of them counts toward epsilon."""
    new = pick_unconfident(model, X_public, first_idx, n_total - len(first_idx), seed)
    return np.concatenate([np.asarray(first_idx), new]).astype(int)


if __name__ == "__main__":
    # quick self-check with fake data that HAS a pattern: answer = 1 when column 0 is above 0
    from member3_student.student import train_student
    rng = np.random.default_rng(0)
    X_public = rng.normal(size=(5000, 5))
    y_hidden = (X_public[:, 0] > 0).astype(int)                 # the "truth" (only for this check)

    first = rng.permutation(5000)[:100]                           # round 1: 100 random rows
    student = train_student(X_public, first, y_hidden[first])
    request = build_request(first, 500, student, X_public)      # round 2: 100 old + 400 unsure

    new = request[100:]
    random_rows = np.setdiff1d(np.arange(5000), first)[:400]
    print("request length:", len(request), " unique:", len(set(request)) == len(request))
    print("round-1 rows kept first:", (request[:100] == first).all())
    print("avg confidence, unsure picks:", round(confidence(student, X_public[new]).mean(), 3))
    print("avg confidence, random rows: ", round(confidence(student, X_public[random_rows]).mean(), 3))
    print("avg |column 0|, unsure picks:", round(np.abs(X_public[new, 0]).mean(), 3), " (near the 0 border)")
    print("avg |column 0|, random rows: ", round(np.abs(X_public[random_rows, 0]).mean(), 3))