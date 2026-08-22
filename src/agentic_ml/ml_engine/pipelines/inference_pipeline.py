"""
Inference Pipeline Runner.
Loads serialized PKL bundles and serves predictions.
"""

from typing import Any, Dict, List, Optional
import pandas as pd
from src.agentic_ml.ml_engine.pipelines.artifact_pipeline import PKLGeneratorAgent, PKLBundleLoader


class InferenceEngine:
    """Production serving layer for serialized ML model bundles."""

    def __init__(self, model_path: str, verify_hash: bool = True):
        self.loader = PKLGeneratorAgent.load(model_path, verify_hash=verify_hash)

    def predict(self, X: pd.DataFrame) -> Any:
        return self.loader.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> Any:
        return self.loader.predict_proba(X)

    def get_metadata(self) -> Dict[str, Any]:
        return self.loader.summary()
