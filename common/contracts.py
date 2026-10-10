# common/contracts.py
# SHARED by all 3 members. Changes need all 3 approvals.
# These classes describe the settings and the small result files that move between members.
from pydantic import BaseModel, Field


class ExperimentConfig(BaseModel):
    run_id: str                       # e.g. "t25_b2_q300_s42"
    dataset: str = "adult"
    seed: int = 42
    n_teachers: int = Field(ge=2)
    noise_scale: float = Field(gt=0)  # Laplace scale b = 1/gamma
    n_queries: int = Field(gt=0)
    delta: float = 1e-5


class DataManifest(BaseModel):        # written by Member 1
    run_id: str
    n_features: int
    n_classes: int
    n_private: int
    n_public: int
    n_test: int
    n_holdout: int
    n_teachers: int


class PrivacyReport(BaseModel):       # written by Member 2
    run_id: str
    epsilon: float
    delta: float
    noise_scale: float
    n_teachers: int
    n_queries: int
    label_accuracy: float | None = None   # diagnostic only


class ResultRow(BaseModel):           # written by Member 3
    run_id: str
    n_teachers: int
    noise_scale: float
    n_queries: int
    epsilon: float
    student_acc: float
    baseline_acc: float
    mia_student: float                # 0.5 = no leakage
    mia_baseline: float


def check_votes(votes, m: DataManifest):
    assert votes.shape == (m.n_teachers, m.n_public), votes.shape
    assert votes.min() >= 0 and votes.max() < m.n_classes


def check_labels(idx, labels, m: DataManifest):
    assert idx.shape == labels.shape
    assert idx.max() < m.n_public and len(set(idx.tolist())) == len(idx)
    assert labels.min() >= 0 and labels.max() < m.n_classes