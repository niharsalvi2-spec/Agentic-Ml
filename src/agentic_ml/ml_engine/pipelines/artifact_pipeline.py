import joblib
import json
from pathlib import Path
from typing import Any, Dict
from sklearn.pipeline import Pipeline
from src.agentic_ml.core.constants import MODELS_DIR

class ArtifactSerializer:
    """Builds and serializes complete sklearn pipelines with metadata."""
    
    @staticmethod
    def save_artifact(pipeline: Any, metadata: Dict[str, Any], filename: str = "model.pkl") -> str:
        filepath = MODELS_DIR / filename
        meta_path = MODELS_DIR / f"{Path(filename).stem}_metadata.json"
        
        # Save model pipeline
        joblib.dump(pipeline, filepath)
        
        # Save metadata
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2)
            
        return str(filepath)
