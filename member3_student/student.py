# member3_student/student.py
# MEMBER 3: the student. It learns ONLY from public rows + the noisy labels from Member 2.
# It never sees the private data and never loads y_public_true.npy.
import numpy as np
from sklearn.ensemble import RandomForestClassifier


def train_student(X_public, query_idx, noisy_labels, seed=42):
    """Train a random forest (100 trees, same as the teachers, paper Appendix C)
    on the public rows that Member 2 answered."""
    X_train = X_public[query_idx]                 # keep only the rows the teachers answered
    model = RandomForestClassifier(n_estimators=100, random_state=seed)
    model.fit(X_train, noisy_labels)              # learn: these rows -> these noisy answers
    return model


if __name__ == "__main__":
    # quick self-check with fake numbers (no real data needed)
    rng = np.random.default_rng(0)
    X_public = rng.normal(size=(1000, 13))        # 1000 fake public people, 13 columns each
    query_idx = rng.permutation(1000)[:100]       # pretend Member 2 picked 100 of them
    noisy_labels = rng.integers(0, 2, 100)        # pretend noisy answers (0 or 1)

    student = train_student(X_public, query_idx, noisy_labels)
    print("student trained on", len(query_idx), "rows")
    print("first 10 guesses:", student.predict(X_public[:10]))