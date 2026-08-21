import os, json, joblib
from pathlib import Path

def save_json(data: dict, filepath: Path):
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)

def load_json(filepath: Path) -> dict:
    with open(filepath, 'r') as f:
        return json.load(f)
