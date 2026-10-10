# MEMBER 1: cut the private pool into N separate groups, train one random forest per group, collect votes.
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from member1_teachers.split import prepare_data


def make_partitions(n_rows, n_teachers, seed=42):
    """Shuffle the row numbers, then cut them into N groups. No row can be in two groups."""
    order = np.random.default_rng(seed).permutation(n_rows)
    return np.array_split(order, n_teachers)


def train_teachers(X_private, y_private, n_teachers, seed=42):
    """One random forest (100 trees, paper Appendix C) per group. A teacher only sees its own group."""
    teachers = []
    for i, rows in enumerate(make_partitions(len(X_private), n_teachers, seed)):
        model = RandomForestClassifier(n_estimators=100, random_state=seed + i)
        model.fit(X_private[rows], y_private[rows])
        teachers.append(model)
    return teachers


def get_votes(teachers, X_public):
    """Every teacher predicts a label for every public row. Shape = (N teachers, n public rows)."""
    return np.stack([t.predict(X_public) for t in teachers]).astype(int)


if __name__ == "__main__":
    data, _ = prepare_data()
    n_private = len(data["X_private"])

    for n_teachers in [10, 100, 250]:
        # CHECK 1: the groups are separate and together cover the private pool exactly once
        groups = make_partitions(n_private, n_teachers)
        all_rows = np.concatenate(groups)
        assert len(all_rows) == len(set(all_rows)) == n_private, "groups overlap or miss rows!"

        teachers = train_teachers(data["X_private"], data["y_private"], n_teachers)
        votes = get_votes(teachers, data["X_public"])

        # CHECK 2: the votes table has the right shape and only 0/1 values
        assert votes.shape == (n_teachers, len(data["X_public"]))
        assert votes.min() >= 0 and votes.max() <= 1

        # Numbers for your report (paper Sec. 4.1)
        teacher_acc = np.mean([t.score(data["X_test"], data["y_test"]) for t in teachers])
        ones = votes.sum(axis=0)                                  # how many teachers said "1" per row
        agreement = np.abs(ones - (n_teachers - ones)).mean() / n_teachers   # 1.0 = all teachers always agree
        print(f"teachers={n_teachers:3d}  rows per teacher={len(groups[0]):6d}  "
              f"avg teacher accuracy on test={teacher_acc:.3f}  agreement={agreement:.3f}  votes shape={votes.shape}")
    print("OK: all checks passed")