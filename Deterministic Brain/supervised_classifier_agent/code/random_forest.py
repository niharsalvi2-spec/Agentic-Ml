"""
AGENT-READABLE MODULE
======================
name: random_forest
classifier: Random Forest
versions_in_file: [SklearnRandomForest, ScratchRandomForest]

WHEN TO USE:
- Strong general-purpose tabular classifier, robust to overfitting vs single trees
- Handles nonlinear relationships and feature interactions well
- Gives feature importances "for free"
- Robust to outliers and unscaled features

WHEN NOT TO USE:
- Very high-dimensional sparse data (text) - linear models often better
- When model size / inference latency is tightly constrained (many trees)
- When maximal interpretability is required (forest is less interpretable than single tree)

KEY HYPERPARAMETERS:
- n_estimators: number of trees (more = more stable, diminishing returns, slower)
- max_depth / min_samples_leaf: per-tree complexity control
- max_features: features considered per split ('sqrt' typical for classification)
- bootstrap: whether to sample with replacement (bagging)
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier as _SkRandomForest

from base_classifier import BaseClassifier
from decision_tree import ScratchDecisionTree


class SklearnRandomForest(BaseClassifier):
    """Wrapper around sklearn.ensemble.RandomForestClassifier."""

    METADATA = {
        "name": "Random Forest (sklearn)",
        "family": "ensemble-bagging",
        "supports_proba": True,
        "handles_nonlinear": True,
        "sensitive_to_scaling": False,
        "sensitive_to_outliers": False,
        "good_for_high_dim": False,
        "good_for_small_data": True,
        "good_for_large_data": True,
        "interpretable": False,  # black-box, but offers feature_importances_
        "training_speed": "medium",
        "inference_speed": "medium",
        "handles_multiclass_natively": True,
        "handles_imbalance_well": False,  # unless class_weight='balanced'
        "typical_use_cases": [
            "general-purpose tabular classification",
            "feature importance ranking",
            "strong baseline before trying gradient boosting",
        ],
    }

    def __init__(self, n_estimators=200, max_depth=None, max_features="sqrt",
                 min_samples_leaf=1, class_weight=None, random_state=42, n_jobs=-1):
        self.model = _SkRandomForest(
            n_estimators=n_estimators, max_depth=max_depth, max_features=max_features,
            min_samples_leaf=min_samples_leaf, class_weight=class_weight,
            random_state=random_state, n_jobs=n_jobs,
        )

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def feature_importances(self):
        return self.model.feature_importances_


class ScratchRandomForest(BaseClassifier):
    """
    From-scratch Random Forest: bagging of ScratchDecisionTree instances,
    each trained on a bootstrap sample with a random feature subset per split
    (approximated here via random feature subsampling per tree, not per split,
    for simplicity/speed).
    """

    METADATA = {
        "name": "Random Forest (from-scratch)",
        "family": "ensemble-bagging",
        "supports_proba": True,
        "handles_nonlinear": True,
        "sensitive_to_scaling": False,
        "sensitive_to_outliers": False,
        "good_for_high_dim": False,
        "good_for_small_data": True,
        "good_for_large_data": False,  # pure-python tree growth is slow at scale
        "interpretable": False,
        "training_speed": "slow",
        "inference_speed": "medium",
        "handles_multiclass_natively": True,
        "handles_imbalance_well": False,
        "typical_use_cases": [
            "teaching / interview implementation",
            "small-to-medium tabular datasets without sklearn",
        ],
    }

    def __init__(self, n_estimators=20, max_depth=5, max_features="sqrt", random_state=42):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.max_features = max_features
        self.random_state = random_state
        self.trees = []
        self.feature_subsets = []
        self.classes_ = None

    def _n_selected_features(self, n_features):
        if self.max_features == "sqrt":
            return max(1, int(np.sqrt(n_features)))
        if self.max_features == "log2":
            return max(1, int(np.log2(n_features)))
        if isinstance(self.max_features, int):
            return min(self.max_features, n_features)
        return n_features

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        rng = np.random.RandomState(self.random_state)
        n_samples, n_features = X.shape
        k = self._n_selected_features(n_features)

        self.trees = []
        self.feature_subsets = []
        for i in range(self.n_estimators):
            boot_idx = rng.randint(0, n_samples, n_samples)  # bootstrap sample
            feat_idx = rng.choice(n_features, size=k, replace=False)

            tree = ScratchDecisionTree(max_depth=self.max_depth)
            tree.fit(X[boot_idx][:, feat_idx], y[boot_idx])

            self.trees.append(tree)
            self.feature_subsets.append(feat_idx)
        return self

    def predict_proba(self, X):
        X = np.asarray(X, dtype=float)
        n_samples = X.shape[0]
        n_classes = len(self.classes_)
        agg = np.zeros((n_samples, n_classes))
        class_to_idx = {c: i for i, c in enumerate(self.classes_)}

        for tree, feat_idx in zip(self.trees, self.feature_subsets):
            probs = tree.predict_proba(X[:, feat_idx])
            for local_i, c in enumerate(tree.classes_):
                agg[:, class_to_idx[c]] += probs[:, local_i]

        agg /= len(self.trees)
        return agg

    def predict(self, X):
        probs = self.predict_proba(X)
        idx = np.argmax(probs, axis=1)
        return self.classes_[idx]
