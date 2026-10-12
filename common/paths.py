# common/paths.py
# SHARED. Where every hand-off file lives: artifacts/<run_id>/<file name>
import os
from pathlib import Path

# Allow tests and users to override the artifact root.
ARTIFACT_ROOT = Path(os.environ.get("PATE_ARTIFACT_ROOT", "artifacts"))
ROOT = Path(__file__).resolve().parent.parent

# file names (everyone uses these names, nobody types them by hand)
MANIFEST = "manifest.json"
X_PRIVATE, Y_PRIVATE = "X_private.npy", "y_private.npy"
X_PUBLIC = "X_public.npy"
Y_PUBLIC_TRUE = "y_public_true.npy"   # diagnostics only. Member 3 must never load it.
X_TEST, Y_TEST = "X_test.npy", "y_test.npy"
X_HOLDOUT, Y_HOLDOUT = "X_holdout.npy", "y_holdout.npy"
TEACHER_VOTES = "teacher_votes.npy"
QUERY_IDX = "query_idx.npy"
NOISY_LABELS = "noisy_labels.npy"
PRIVACY_REPORT = "privacy_report.json"
RESULT_ROW = "result_row.json"



def artifact_dir(run_id):
    """Folder for one run. Created if it does not exist."""
    root = Path(os.environ.get("PATE_ARTIFACT_ROOT", str(ARTIFACT_ROOT)))
    folder = root / run_id
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def artifact_path(run_id, file_name):
    return artifact_dir(run_id) / file_name
