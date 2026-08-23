import os
from pathlib import Path

# Base Paths
ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent
DATA_DIR = ROOT_DIR / "data"
ARTIFACTS_DIR = ROOT_DIR / "artifacts"
MODELS_DIR = ARTIFACTS_DIR / "models"
REPORTS_DIR = ARTIFACTS_DIR / "reports"
RUNS_DIR = ARTIFACTS_DIR / "runs"

# Ensure runtime directories exist
for directory in [DATA_DIR, ARTIFACTS_DIR, MODELS_DIR, REPORTS_DIR, RUNS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Orchestration retry configuration
MAX_RETRIES = int(os.environ.get("MAX_RETRIES", "2"))

