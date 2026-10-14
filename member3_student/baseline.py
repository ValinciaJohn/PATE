# member3_student/baseline.py
# MEMBER 3: the baseline. A normal model trained directly on the PRIVATE pool with the real labels.
# No privacy at all. It shows the best accuracy we could get, so we can see what privacy costs.
import numpy as np
from sklearn.ensemble import RandomForestClassifier


def train_baseline(X_private, y_private, seed=42):
    """Same model as the student and teachers (random forest, 100 trees, paper Appendix C),
    so the comparison is fair. The only difference is the data it learns from."""
    model = RandomForestClassifier(n_estimators=100, random_state=seed)
    model.fit(X_private, y_private)               # learn from the secret list + real answers
    return model


if __name__ == "__main__":
    # quick self-check with fake numbers (no real data needed)
    rng = np.random.default_rng(0)
    X_private = rng.normal(size=(2000, 13))       # 2000 fake private people, 13 columns each
    y_private = rng.integers(0, 2, 2000)          # their fake real answers (0 or 1)

    baseline = train_baseline(X_private, y_private)
    print("baseline trained on", len(X_private), "rows")
    print("first 10 guesses:", baseline.predict(X_private[:10]))
    print("first 10 real answers:", y_private[:10])