"""
AGENT-READABLE MODULE
======================
name: supervised_classifier_agent
purpose: Top-level orchestrator. Registers every classifier (sklearn version AND
         from-scratch version) behind one interface, recommends a classifier
         family given dataset characteristics, and benchmarks all registered
         models on a given dataset.

USAGE (agent-facing)
---------------------
    from supervised_classifier_agent import SupervisedClassifierAgent

    agent = SupervisedClassifierAgent()

    # 1) Ask for a recommendation before training anything
    rec = agent.recommend(
        n_samples=5000, n_features=40, is_high_dim=False,
        need_interpretability=True, need_proba=True,
        suspect_nonlinear=True, has_outliers=False,
        is_imbalanced=False, need_fast_inference=True,
    )

    # 2) Train + evaluate every registered model (or a subset) and rank them
    results = agent.compare_all(X_train, y_train, X_test, y_test,
                                 models=["logistic_regression_sklearn", "random_forest_sklearn"])

    # 3) Get a single trained model by name
    model = agent.get_model("random_forest_sklearn")
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

REGISTRY KEYS
-------------
    logistic_regression_sklearn / logistic_regression_scratch
    naive_bayes_sklearn        / naive_bayes_scratch
    knn_sklearn                / knn_scratch
    decision_tree_sklearn      / decision_tree_scratch
    random_forest_sklearn      / random_forest_scratch
    boosting_sklearn            / boosting_scratch
    svm_sklearn                 / svm_scratch

DECISION GUIDE (mirrors docs/README.md; used by recommend())
--------------------------------------------------------------
    need interpretability + linear data           -> logistic_regression
    need interpretability + nonlinear/rules        -> decision_tree
    high-dim, few samples (text/bio)               -> svm or logistic_regression (linear kernel)
    small data, fast baseline, calibrated priors   -> naive_bayes
    local/neighborhood structure, low-dim, small n -> knn
    general-purpose tabular, robust, no tuning     -> random_forest
    max accuracy, willing to tune, tabular         -> boosting
    imbalanced classes                              -> logistic_regression/random_forest with
                                                        class_weight='balanced', or naive_bayes
"""

import time
import numpy as np

from logistic_regression import SklearnLogisticRegression, ScratchLogisticRegression
from naive_bayes import SklearnNaiveBayes, ScratchNaiveBayes
from knn import SklearnKNN, ScratchKNN
from decision_tree import SklearnDecisionTree, ScratchDecisionTree
from random_forest import SklearnRandomForest, ScratchRandomForest
from boosting import SklearnBoosting, ScratchBoosting
from svm import SklearnSVM, ScratchSVM


class SupervisedClassifierAgent:
    """Registers, recommends, trains, and benchmarks classifiers."""

    def __init__(self):
        self.registry = {
            "logistic_regression_sklearn": SklearnLogisticRegression,
            "logistic_regression_scratch": ScratchLogisticRegression,
            "naive_bayes_sklearn": SklearnNaiveBayes,
            "naive_bayes_scratch": ScratchNaiveBayes,
            "knn_sklearn": SklearnKNN,
            "knn_scratch": ScratchKNN,
            "decision_tree_sklearn": SklearnDecisionTree,
            "decision_tree_scratch": ScratchDecisionTree,
            "random_forest_sklearn": SklearnRandomForest,
            "random_forest_scratch": ScratchRandomForest,
            "boosting_sklearn": SklearnBoosting,
            "boosting_scratch": ScratchBoosting,
            "svm_sklearn": SklearnSVM,
            "svm_scratch": ScratchSVM,
        }

    # ------------------------------------------------------------------ #
    # Model access
    # ------------------------------------------------------------------ #
    def list_models(self):
        return list(self.registry.keys())

    def get_model(self, name, **kwargs):
        """Instantiate a fresh, untrained model by registry key."""
        if name not in self.registry:
            raise KeyError(f"Unknown model '{name}'. Available: {self.list_models()}")
        return self.registry[name](**kwargs)

    def describe_model(self, name):
        """Return the METADATA dict for a registered model (no training needed)."""
        return self.registry[name].METADATA

    def describe_all(self):
        return {name: cls.METADATA for name, cls in self.registry.items()}

    # ------------------------------------------------------------------ #
    # Recommendation logic (rule-based, mirrors docs/README.md decision guide)
    # ------------------------------------------------------------------ #
    def recommend(self, n_samples=None, n_features=None, is_high_dim=None,
                   need_interpretability=False, need_proba=False,
                   suspect_nonlinear=False, has_outliers=False,
                   is_imbalanced=False, need_fast_inference=False,
                   prefer_scratch=False, top_k=3):
        """
        Score every registered model against the stated dataset/requirement
        characteristics and return the top_k best-fitting model names with reasons.
        """
        if is_high_dim is None and n_samples and n_features:
            is_high_dim = n_features > n_samples

        scores = {}
        reasons = {}

        for name, cls in self.registry.items():
            meta = cls.METADATA
            score = 0
            why = []

            if need_interpretability and meta["interpretable"]:
                score += 2
                why.append("interpretable")
            if need_proba and meta["supports_proba"]:
                score += 1
                why.append("supports probability output")
            if suspect_nonlinear and meta["handles_nonlinear"]:
                score += 2
                why.append("handles nonlinear boundaries")
            if not suspect_nonlinear and not meta["handles_nonlinear"]:
                score += 1
                why.append("well-suited to linear structure")
            if has_outliers and not meta["sensitive_to_outliers"]:
                score += 2
                why.append("robust to outliers")
            if is_imbalanced and meta["handles_imbalance_well"]:
                score += 1
                why.append("handles class imbalance reasonably")
            if need_fast_inference and meta["inference_speed"] == "fast":
                score += 2
                why.append("fast inference")
            if is_high_dim and meta["good_for_high_dim"]:
                score += 2
                why.append("suited to high-dimensional data")
            if n_samples is not None:
                if n_samples < 2000 and meta["good_for_small_data"]:
                    score += 1
                    why.append("suited to small datasets")
                if n_samples >= 50000 and meta["good_for_large_data"]:
                    score += 2
                    why.append("scales to large datasets")

            is_scratch = name.endswith("_scratch")
            if prefer_scratch == is_scratch:
                score += 0.5

            scores[name] = score
            reasons[name] = why

        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        return [
            {"model": name, "score": score, "reasons": reasons[name]}
            for name, score in ranked[:top_k]
        ]

    # ------------------------------------------------------------------ #
    # Benchmarking
    # ------------------------------------------------------------------ #
    def compare_all(self, X_train, y_train, X_test, y_test, models=None, verbose=True):
        """
        Train + evaluate a set of registered models (default: all) and return
        a results table sorted by test accuracy, including timing.
        """
        from sklearn.metrics import precision_score, recall_score, f1_score

        names = models if models is not None else self.list_models()
        results = []

        for name in names:
            model = self.get_model(name)
            t0 = time.time()
            try:
                model.fit(X_train, y_train)
                train_time = time.time() - t0

                t1 = time.time()
                preds = model.predict(X_test)
                infer_time = time.time() - t1

                acc = float(np.mean(preds == np.asarray(y_test)))
                precision = precision_score(y_test, preds, average="weighted", zero_division=0)
                recall = recall_score(y_test, preds, average="weighted", zero_division=0)
                f1 = f1_score(y_test, preds, average="weighted", zero_division=0)

                results.append({
                    "model": name,
                    "accuracy": round(acc, 4),
                    "precision": round(precision, 4),
                    "recall": round(recall, 4),
                    "f1": round(f1, 4),
                    "train_time_sec": round(train_time, 4),
                    "infer_time_sec": round(infer_time, 4),
                    "error": None,
                })
                if verbose:
                    print(f"[ok]   {name:32s} acc={acc:.4f} f1={f1:.4f} "
                          f"train={train_time:.3f}s infer={infer_time:.3f}s")
            except Exception as e:
                results.append({
                    "model": name, "accuracy": None, "precision": None, "recall": None,
                    "f1": None, "train_time_sec": None, "infer_time_sec": None, "error": str(e),
                })
                if verbose:
                    print(f"[fail] {name:32s} error={e}")

        results.sort(key=lambda r: (r["accuracy"] is not None, r["accuracy"]), reverse=True)
        return results


if __name__ == "__main__":
    # Minimal smoke test using a synthetic dataset
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    X, y = make_classification(n_samples=600, n_features=10, n_informative=6,
                                n_redundant=2, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    scaler = StandardScaler().fit(X_train)
    X_train, X_test = scaler.transform(X_train), scaler.transform(X_test)

    agent = SupervisedClassifierAgent()

    print("=== Recommendation ===")
    for r in agent.recommend(n_samples=600, n_features=10, need_interpretability=True,
                              need_proba=True, suspect_nonlinear=True):
        print(r)

    print("\n=== Benchmark (sklearn versions only, for speed) ===")
    sklearn_models = [m for m in agent.list_models() if m.endswith("_sklearn")]
    agent.compare_all(X_train, y_train, X_test, y_test, models=sklearn_models)
