import shutil
from pathlib import Path
root_dir = Path(__file__).resolve().parent.parent
artifacts_dir = root_dir / "artifacts" / "runs"
if artifacts_dir.exists():
    print(f"Cleaning temporary artifacts in {artifacts_dir}")
