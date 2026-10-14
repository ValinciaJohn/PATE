"""Small adapter between Member 2's code and the shared `common/paths.py`.
Every Member 2 module goes through here, so the code works with the shared paths API.
Run everything from the repo root: the artifacts folder is `paths.ARTIFACT_ROOT` ("artifacts")."""
import os
from pathlib import Path
from common import paths


def art_dir(run_id) -> Path:
    """Artifact folder of one run, as a Path (created if missing, as common/paths.py does)."""
    return Path(paths.artifact_dir(run_id))


def art_root() -> Path:
    return Path(paths.ARTIFACT_ROOT)


def results_dir() -> Path:
    d = Path(os.environ.get("PATE_RESULTS_DIR", "results"))
    d.mkdir(parents=True, exist_ok=True)
    return d
