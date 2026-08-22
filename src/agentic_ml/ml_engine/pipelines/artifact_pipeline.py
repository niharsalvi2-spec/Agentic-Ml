"""
Artifact Packaging and Self-Contained Deployment Bundler.
Packages trained models, schema contracts, metrics, and preprocessors into secure, hash-verified .pkl bundles.
"""

import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
import pandas as pd

from src.agentic_ml.core.constants import MODELS_DIR
import src.agentic_ml.ml_engine.pipelines.pkl_utils as pu


class PKLBundleLoader:
    """Consumes and validates self-contained model bundles at inference time."""

    def __init__(self, bundle: Dict[str, Any], filepath: Optional[str] = None):
        self.bundle = bundle
        self.filepath = filepath
        self.pipeline = bundle.get("pipeline")
        if self.pipeline is None:
            raise ValueError("Loaded bundle has no 'pipeline' attribute.")

    def _validate_columns(self, X: Any) -> Any:
        expected = self.bundle.get("feature_columns")
        if expected is None:
            return X
        if hasattr(X, "columns"):
            missing = set(expected) - set(X.columns)
            if missing:
                raise ValueError(f"Input is missing expected feature columns: {missing}")
            return X[expected]
        return X

    def predict(self, X: Any) -> Any:
        X_val = self._validate_columns(X)
        return self.pipeline.predict(X_val)

    def predict_proba(self, X: Any) -> Any:
        if not hasattr(self.pipeline, "predict_proba"):
            raise AttributeError(f"Model '{self.bundle.get('model_name')}' does not support predict_proba.")
        X_val = self._validate_columns(X)
        return self.pipeline.predict_proba(X_val)

    def summary(self) -> Dict[str, Any]:
        return {
            "model_name": self.bundle.get("model_name"),
            "task": self.bundle.get("task"),
            "created_at": self.bundle.get("created_at"),
            "feature_columns": self.bundle.get("feature_columns"),
            "metrics": self.bundle.get("metrics"),
        }


class PKLGeneratorAgent:
    """Assembles upstream pipeline outputs into one production .pkl bundle."""

    def __init__(self, save_dir: Optional[str] = None):
        self.save_dir = Path(save_dir) if save_dir else MODELS_DIR
        self.save_dir.mkdir(exist_ok=True, parents=True)
        self.version_manager = pu.PKLVersionManager(str(self.save_dir / "registry"))

    def build_bundle(
        self,
        pipeline_or_model: Any,
        task: str,
        model_name: str,
        feature_columns: Optional[List[str]] = None,
        numeric_cols: Optional[List[str]] = None,
        categorical_cols: Optional[List[str]] = None,
        target_column: Optional[str] = None,
        classes: Optional[List[Any]] = None,
        metrics: Optional[Dict[str, Any]] = None,
        dataset_info: Optional[Dict[str, Any]] = None,
        preprocessing_objects: Optional[Dict[str, Any]] = None,
        description: str = "",
        extra_metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        return {
            "pipeline": pipeline_or_model,
            "model_name": model_name,
            "task": task,
            "description": description,
            "feature_columns": list(feature_columns) if feature_columns is not None else None,
            "numeric_cols": list(numeric_cols) if numeric_cols is not None else None,
            "categorical_cols": list(categorical_cols) if categorical_cols is not None else None,
            "target_column": target_column,
            "classes": list(classes) if classes is not None else None,
            "metrics": metrics or {},
            "dataset_info": dataset_info or {},
            "preprocessing_objects": preprocessing_objects or {},
            "created_at": datetime.now().isoformat(),
            "extra_metadata": extra_metadata or {},
        }

    def generate(
        self,
        pipeline_or_model: Any,
        task: str,
        model_name: str,
        feature_columns: Optional[List[str]] = None,
        numeric_cols: Optional[List[str]] = None,
        categorical_cols: Optional[List[str]] = None,
        target_column: Optional[str] = None,
        classes: Optional[List[Any]] = None,
        metrics: Optional[Dict[str, Any]] = None,
        dataset_info: Optional[Dict[str, Any]] = None,
        preprocessing_objects: Optional[Dict[str, Any]] = None,
        description: str = "",
        compress: int = 3,
        register_version: bool = False,
    ) -> Dict[str, Any]:
        bundle = self.build_bundle(
            pipeline_or_model=pipeline_or_model,
            task=task,
            model_name=model_name,
            feature_columns=feature_columns,
            numeric_cols=numeric_cols,
            categorical_cols=categorical_cols,
            target_column=target_column,
            classes=classes,
            metrics=metrics,
            dataset_info=dataset_info,
            preprocessing_objects=preprocessing_objects,
            description=description,
        )

        safe_name = model_name.lower().replace(" ", "_")
        filepath = str(self.save_dir / f"{safe_name}.pkl")
        filepath, file_hash = pu.save_pkl_with_hash(bundle, filepath, compress=compress)

        result = {
            "filepath": filepath,
            "sha256": file_hash,
            "size_bytes": os.path.getsize(filepath),
            "model_name": model_name,
            "task": task,
        }

        if register_version:
            v_str = self.version_manager.register(bundle, model_name, metrics=metrics, description=description)
            result["registry_version"] = v_str

        return result

    @staticmethod
    def load(filepath: str, verify_hash: bool = True) -> PKLBundleLoader:
        bundle = pu.safe_load_pkl(filepath, verify_hash=verify_hash)
        return PKLBundleLoader(bundle, filepath)


class ArtifactSerializer:
    """Backward-compatible serializer wrapper."""

    @staticmethod
    def save_artifact(pipeline: Any, metadata: Dict[str, Any], filename: str = "model.pkl") -> str:
        generator = PKLGeneratorAgent()
        model_name = metadata.get("model_name", "model")
        task = metadata.get("task_type", "classification")
        feature_cols = metadata.get("selected_features")
        metrics = metadata.get("metrics")
        
        target_path = str(MODELS_DIR / filename)
        bundle = generator.build_bundle(
            pipeline_or_model=pipeline,
            task=task,
            model_name=model_name,
            feature_columns=feature_cols,
            metrics=metrics,
            extra_metadata=metadata,
        )
        filepath, _ = pu.save_pkl_with_hash(bundle, target_path)
        return filepath
