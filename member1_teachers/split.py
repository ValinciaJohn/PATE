# MEMBER 1: split the data into 4 parts and turn the tables into numbers.
#   private 60%  -> teachers train here (and the baseline, and the MIA "members")
#   public  15%  -> inputs for the student
#   test    15%  -> measures accuracy
#   holdout 10%  -> never trained on, used as the MIA "non-members"
from functools import lru_cache
import numpy as np
from sklearn.model_selection import train_test_split
from member1_teachers.preprocess import load_adult, make_transformer


def make_splits(X, y, seed=42):
    """Cut the clean tables into 4 parts. Each row goes to exactly one part."""
    X_private, X_rest, y_private, y_rest = train_test_split(
        X, y, test_size=0.40, random_state=seed, stratify=y)
    X_public, X_rest, y_public, y_rest = train_test_split(
        X_rest, y_rest, train_size=15 / 40, random_state=seed, stratify=y_rest)
    X_test, X_holdout, y_test, y_holdout = train_test_split(
        X_rest, y_rest, train_size=15 / 25, random_state=seed, stratify=y_rest)
    return {
        "private": (X_private, y_private),
        "public": (X_public, y_public),
        "test": (X_test, y_test),
        "holdout": (X_holdout, y_holdout),
    }


@lru_cache(maxsize=3)          # same seed asked again -> reuse the result instead of recomputing
def prepare_data(seed=42):
    """Load -> split -> text to integers (the map is learned from the private part only)."""
    X, y = load_adult()
    splits = make_splits(X, y, seed)
    transformer = make_transformer(splits["private"][0])
    data = {}
    for name, (X_part, y_part) in splits.items():
        data["X_" + name] = transformer.transform(X_part)   # numbers table
        data["y_" + name] = y_part.to_numpy()                # 0/1 labels
    return data, splits


if __name__ == "__main__":
    data, splits = prepare_data()

    total = sum(len(v[0]) for v in splits.values())
    print("total rows:", total)
    for name, (X_part, y_part) in splits.items():
        print(f"{name:8s} rows={len(X_part):6d}  share={len(X_part) / total:.2f}  earn>50K={y_part.mean():.3f}")

    # CHECK 1: no row is in two parts
    seen = set()
    for X_part, _ in splits.values():
        ids = set(X_part.index)
        assert not (seen & ids), "a row appears in two parts!"
        seen |= ids
    assert len(seen) == total
    print("OK: every row is in exactly one part")

    # CHECK 2: no NaN, and the number of columns equals the number of features (no one-hot)
    assert not np.isnan(data["X_private"]).any()
    assert data["X_private"].shape[1] == splits["private"][0].shape[1]
    print("OK: columns =", data["X_private"].shape[1], "(one integer column per feature)")
    unseen = int((data["X_test"] == -1).sum() + (data["X_public"] == -1).sum() + (data["X_holdout"] == -1).sum())
    print("values never seen in the private part (mapped to -1):", unseen)