"""
Data Loader module.
Handles dataset ingestion from diverse sources (CSV, Parquet, JSON)
and synthetic generation benchmarks with deterministic reproducible seeds.
"""

from pathlib import Path
from typing import Tuple, Optional, Dict, Any
import pandas as pd
import numpy as np
from sklearn.datasets import make_classification, make_regression, make_blobs

from src.agentic_ml.ml_engine.data.collector_utils import quick_quality_check


class DataLoader:
    """Loads datasets from file formats or synthesizes high-quality ML benchmarks."""

    @staticmethod
    def load_or_synthesize(
        task_type: str = "classification",
        dataset_path: str = "",
        n_samples: int = 200,
        n_features: int = 6,
        random_state: int = 42
    ) -> Tuple[pd.DataFrame, str]:
        """Loads data from path or generates synthetic data for the given task."""
        if dataset_path and Path(dataset_path).exists():
            path_obj = Path(dataset_path)
            if path_obj.suffix.lower() == ".parquet":
                df = pd.read_parquet(dataset_path)
            elif path_obj.suffix.lower() == ".json":
                df = pd.read_json(dataset_path)
            else:
                df = pd.read_csv(dataset_path)

            target = df.columns[-1]
            return df, target

        # Synthetic generation
        if task_type == "classification":
            X, y = make_classification(
                n_samples=n_samples,
                n_features=n_features,
                n_informative=max(2, n_features - 2),
                n_redundant=min(2, max(0, n_features - 4)),
                n_classes=2,
                random_state=random_state
            )
            feature_names = [f"feature_{i+1}" for i in range(n_features)]
            df = pd.DataFrame(X, columns=feature_names)
            df["target"] = y
            return df, "target"

        elif task_type == "clustering":
            X, y = make_blobs(
                n_samples=n_samples,
                n_features=n_features,
                centers=3,
                random_state=random_state
            )
            feature_names = [f"feature_{i+1}" for i in range(n_features)]
            df = pd.DataFrame(X, columns=feature_names)
            # In clustering, target might be optional ground truth
            df["cluster_ground_truth"] = y
            return df, "cluster_ground_truth"

        else:  # regression default
            X, y = make_regression(
                n_samples=n_samples,
                n_features=n_features,
                n_informative=max(2, n_features - 2),
                noise=0.1,
                random_state=random_state
            )
            feature_names = [f"feature_{i+1}" for i in range(n_features)]
            df = pd.DataFrame(X, columns=feature_names)
            df["target"] = y
            return df, "target"

    @staticmethod
    def audit_quality(df: pd.DataFrame) -> Dict[str, Any]:
        """Runs quick quality audit on DataFrame."""
        return quick_quality_check(df)
