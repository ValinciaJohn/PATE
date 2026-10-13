# member3_student/mia.py
# MEMBER 3: membership inference attack (MIA), the "hacker test".
# Members = rows from the PRIVATE pool. Non-members = rows from the HOLDOUT (never trained on).
# Idea: models are more confident (smaller loss) on rows they studied. If the attacker can use
# the loss to tell members from non-members, the model leaks. Score 0.5 = coin flip = no leak.
import numpy as np
from sklearn.metrics import roc_auc_score


def per_example_loss(model, X, y):
    """How wrong the model is about each row: -log(probability it gives to the real answer).
    Small = confident and right. Big = unsure or wrong."""
    proba = model.predict_proba(X)
    p_true = np.zeros(len(y))
    for col, label in enumerate(model.classes_):  # find the column that belongs to each row's real label
        p_true[y == label] = proba[y == label, col]
    return -np.log(np.clip(p_true, 1e-12, 1.0))   # clip so we never take log(0)


def mia_score(model, X_members, y_members, X_nonmembers, y_nonmembers):
    """AUC of the rule 'smaller loss -> member'. 0.5 = no leak, 1.0 = total leak."""
    loss_in = per_example_loss(model, X_members, y_members)
    loss_out = per_example_loss(model, X_nonmembers, y_nonmembers)
    is_member = np.concatenate([np.ones(len(loss_in)), np.zeros(len(loss_out))])
    member_score = -np.concatenate([loss_in, loss_out])   # minus: small loss -> high "member" score
    return float(roc_auc_score(is_member, member_score))


def run_mia(model, X_private, y_private, X_holdout, y_holdout, seed=42):
    """Balanced attack: the same number of members (sampled from private) and non-members (holdout)."""
    rng = np.random.default_rng(seed)
    n = min(len(X_private), len(X_holdout))
    pick_in = rng.choice(len(X_private), n, replace=False)
    pick_out = rng.choice(len(X_holdout), n, replace=False)
    return mia_score(model, X_private[pick_in], y_private[pick_in],
                     X_holdout[pick_out], y_holdout[pick_out])


if __name__ == "__main__":
    # quick self-check with fake numbers (no real data needed)
    from member3_student.baseline import train_baseline
    rng = np.random.default_rng(0)
    X_private, y_private = rng.normal(size=(2000, 13)), rng.integers(0, 2, 2000)
    X_holdout, y_holdout = rng.normal(size=(500, 13)), rng.integers(0, 2, 500)
    X_other, y_other = rng.normal(size=(2000, 13)), rng.integers(0, 2, 2000)

    # model A studied the private rows -> should leak (score well above 0.5)
    leaky = train_baseline(X_private, y_private)
    print("model that studied private rows:      ", round(run_mia(leaky, X_private, y_private, X_holdout, y_holdout), 3))

    # model B never saw the private rows -> should not leak (score near 0.5)
    safe = train_baseline(X_other, y_other)
    print("model that never saw private rows:    ", round(run_mia(safe, X_private, y_private, X_holdout, y_holdout), 3))