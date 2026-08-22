"""
Model Training and Benchmarking Engine.
Trains candidate model suites, measures execution latencies, and computes comparative benchmarks.
"""

import time
import logging
from typing import Dict, Any, Tuple, List, Optional
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, r2_score, mean_squared_error

from src.agentic_ml.ml_engine.models.registry import ModelRegistry

logger = logging.getLogger("agentic_ml.models.training")


class ModelTrainer:
    """Trains and compares candidate models with latency and performance metrics."""

    @staticmethod
    def train_candidates(
        X: pd.DataFrame,
        y: pd.Series,
        task_type: str = "classification",
        model_names: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Trains standard candidate models on the provided training set."""
        all_models = ModelRegistry.get_models_for_task(task_type)
        
        if model_names:
            models = {k: v for k, v in all_models.items() if k in model_names}
        else:
            # Default baseline subset
            if task_type == "classification":
                models = {k: all_models[k] for k in ["RandomForest", "GradientBoosting", "LogisticRegression"] if k in all_models}
            elif task_type == "clustering":
                models = all_models
            else:
                models = {k: all_models[k] for k in ["RandomForest", "GradientBoosting", "Ridge"] if k in all_models}

        trained = {}
        for name, model in models.items():
            try:
                if task_type == "clustering":
                    model.fit(X)
                else:
                    model.fit(X, y)
                trained[name] = model
            except Exception as e:
                logger.error(f"Failed training candidate {name}: {e}")

        return trained

    @staticmethod
    def compare_all(
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        task_type: str = "classification",
        models: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Trains and benchmarks multiple candidate models with accuracy, F1, and timing."""
        if models is None:
            models = ModelRegistry.get_models_for_task(task_type)

        results = []
        for name, model in models.items():
            t0 = time.time()
            try:
                model.fit(X_train, y_train)
                train_time = time.time() - t0

                t1 = time.time()
                preds = model.predict(X_test)
                infer_time = time.time() - t1

                if task_type == "classification":
                    acc = float(accuracy_score(y_test, preds))
                    prec = float(precision_score(y_test, preds, average="weighted", zero_division=0))
                    rec = float(recall_score(y_test, preds, average="weighted", zero_division=0))
                    f1 = float(f1_score(y_test, preds, average="weighted", zero_division=0))
                    results.append({
                        "model": name,
                        "accuracy": round(acc, 4),
                        "precision": round(prec, 4),
                        "recall": round(rec, 4),
                        "f1": round(f1, 4),
                        "train_time_sec": round(train_time, 4),
                        "infer_time_sec": round(infer_time, 4),
                        "error": None
                    })
                else:
                    r2 = float(r2_score(y_test, preds))
                    mse = float(mean_squared_error(y_test, preds))
                    rmse = float(np.sqrt(mse))
                    results.append({
                        "model": name,
                        "r2": round(r2, 4),
                        "rmse": round(rmse, 4),
                        "mse": round(mse, 4),
                        "train_time_sec": round(train_time, 4),
                        "infer_time_sec": round(infer_time, 4),
                        "error": None
                    })
            except Exception as e:
                results.append({
                    "model": name,
                    "error": str(e),
                    "train_time_sec": None,
                    "infer_time_sec": None
                })

        # Sort by primary metric
        if task_type == "classification":
            results.sort(key=lambda r: (r.get("accuracy") is not None, r.get("accuracy") or 0.0), reverse=True)
        else:
            results.sort(key=lambda r: (r.get("r2") is not None, r.get("r2") or -float("inf")), reverse=True)

        return results
