import joblib, json
from src.agentic_ml.core.constants import MODELS_DIR

class ArtifactService:
    @staticmethod
    def save_model(model, metadata, name="model.pkl"):
        path = MODELS_DIR / name
        joblib.dump(model, path)
        with open(MODELS_DIR / f"{name}_metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
        return str(path)
