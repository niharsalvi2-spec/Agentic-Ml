import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any
from sklearn.datasets import make_classification, make_regression
from pathlib import Path

class DataLoader:
    """Loads CSV/Parquet or synthesizes benchmark data if missing."""
    
    @staticmethod
    def load_or_synthesize(task_type: str = "classification", dataset_path: str = "") -> Tuple[pd.DataFrame, str]:
        if dataset_path and Path(dataset_path).exists():
            df = pd.read_csv(dataset_path)
            target = df.columns[-1]
            return df, target
            
        # Synthetic generation if no data provided
        if task_type == "classification":
            X, y = make_classification(
                n_samples=200, n_features=6, n_informative=4, n_classes=2, random_state=42
            )
            feature_names = [f"feature_{i+1}" for i in range(6)]
            df = pd.DataFrame(X, columns=feature_names)
            df["target"] = y
            return df, "target"
        else:
            X, y = make_regression(
                n_samples=200, n_features=6, n_informative=4, noise=0.1, random_state=42
            )
            feature_names = [f"feature_{i+1}" for i in range(6)]
            df = pd.DataFrame(X, columns=feature_names)
            df["target"] = y
            return df, "target"
