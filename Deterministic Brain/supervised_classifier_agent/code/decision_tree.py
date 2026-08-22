"""
AGENT-READABLE MODULE
======================
name: decision_tree
classifier: Decision Tree
versions_in_file: [SklearnDecisionTree, ScratchDecisionTree]

WHEN TO USE:
- Need a fully interpretable / visualizable model (if-then rules)
- Mixed feature types, nonlinear relationships, feature interactions
- No need to scale features
- Building block for ensembles (Random Forest, Boosting)

WHEN NOT TO USE:
- Alone, when high accuracy is critical (single trees overfit / high variance)
- Data has smooth linear relationships (trees approximate with step functions)

KEY HYPERPARAMETERS:
- max_depth: controls overfitting (shallower = more bias, less variance)
- min_samples_split / min_samples_leaf: minimum samples to split/be a leaf
- criterion: 'gini' (default, faster) or 'entropy' (information gain)
- max_features: number of features considered per split
"""

import numpy as np
from sklearn.tree import DecisionTreeClassifier as _SkDecisionTree

from base_classifier import BaseClassifier


class SklearnDecisionTree(BaseClassifier):
    """Wrapper around sklearn.tree.DecisionTreeClassifier."""

    METADATA = {
        "name": "Decision Tree (sklearn)",
        "family": "tree",
        "supports_proba": True,
        "handles_nonlinear": True,
        "sensitive_to_scaling": False,
        "sensitive_to_outliers": False,
        "good_for_high_dim": False,
        "good_for_small_data": True,
        "good_for_large_data": True,
        "interpretable": True,
        "training_speed": "fast",
        "inference_speed": "fast",
        "handles_multiclass_natively": True,
        "handles_imbalance_well": False,  # unless class_weight='balanced'
        "typical_use_cases": [
            "rule-based / interpretable decisions",
            "feature importance analysis",
            "base learner for ensembles",
        ],
    }

    def __init__(self, max_depth=None, min_samples_split=2, min_samples_leaf=1,
                 criterion="gini", class_weight=None, random_state=42):
        self.model = _SkDecisionTree(
            max_depth=max_depth, min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf, criterion=criterion,
            class_weight=class_weight, random_state=random_state,
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


class _Node:
    __slots__ = ("feature", "threshold", "left", "right", "value")

    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value  # class-probability dict at leaf


class ScratchDecisionTree(BaseClassifier):
    """
    From-scratch CART-style decision tree classifier using Gini impurity,
    binary splits on numeric thresholds, recursive greedy growth.
    """

    METADATA = {
        "name": "Decision Tree (from-scratch)",
        "family": "tree",
        "supports_proba": True,
        "handles_nonlinear": True,
        "sensitive_to_scaling": False,
        "sensitive_to_outliers": False,
        "good_for_high_dim": False,
        "good_for_small_data": True,
        "good_for_large_data": False,  # naive O(features * samples * log(samples)) per split
        "interpretable": True,
        "training_speed": "medium",
        "inference_speed": "fast",
        "handles_multiclass_natively": True,
        "handles_imbalance_well": False,
        "typical_use_cases": [
            "teaching / interview implementation",
            "base learner reused by ScratchRandomForest / ScratchBoosting",
        ],
    }

    def __init__(self, max_depth=5, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None
        self.classes_ = None

    @staticmethod
    def _gini(y):
        if len(y) == 0:
            return 0.0
        _, counts = np.unique(y, return_counts=True)
        probs = counts / counts.sum()
        return 1.0 - np.sum(probs ** 2)

    def _best_split(self, X, y):
        n_samples, n_features = X.shape
        best_gain, best_feat, best_thresh = -1, None, None
        parent_impurity = self._gini(y)

        for feat in range(n_features):
            thresholds = np.unique(X[:, feat])
            # subsample thresholds for speed on large unique-value columns
            if len(thresholds) > 20:
                thresholds = np.percentile(thresholds, np.linspace(0, 100, 20))
            for t in thresholds:
                left_mask = X[:, feat] <= t
                right_mask = ~left_mask
                if left_mask.sum() == 0 or right_mask.sum() == 0:
                    continue
                left_impurity = self._gini(y[left_mask])
                right_impurity = self._gini(y[right_mask])
                weighted = (left_mask.sum() * left_impurity + right_mask.sum() * right_impurity) / n_samples
                gain = parent_impurity - weighted
                if gain > best_gain:
                    best_gain, best_feat, best_thresh = gain, feat, t
        return best_feat, best_thresh, best_gain

    def _leaf_value(self, y):
        classes, counts = np.unique(y, return_counts=True)
        probs = counts / counts.sum()
        return dict(zip(classes, probs))

    def _grow(self, X, y, depth):
        if (depth >= self.max_depth or len(y) < self.min_samples_split
                or len(np.unique(y)) == 1):
            return _Node(value=self._leaf_value(y))

        feat, thresh, gain = self._best_split(X, y)
        if feat is None or gain <= 0:
            return _Node(value=self._leaf_value(y))

        left_mask = X[:, feat] <= thresh
        left = self._grow(X[left_mask], y[left_mask], depth + 1)
        right = self._grow(X[~left_mask], y[~left_mask], depth + 1)
        return _Node(feature=feat, threshold=thresh, left=left, right=right)

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        self.root = self._grow(X, y, depth=0)
        return self

    def _predict_one_proba(self, x, node):
        if node.value is not None:
            return node.value
        if x[node.feature] <= node.threshold:
            return self._predict_one_proba(x, node.left)
        return self._predict_one_proba(x, node.right)

    def predict_proba(self, X):
        X = np.asarray(X, dtype=float)
        out = np.zeros((X.shape[0], len(self.classes_)))
        class_to_idx = {c: i for i, c in enumerate(self.classes_)}
        for i, x in enumerate(X):
            leaf_dist = self._predict_one_proba(x, self.root)
            for c, p in leaf_dist.items():
                out[i, class_to_idx[c]] = p
        return out

    def predict(self, X):
        probs = self.predict_proba(X)
        idx = np.argmax(probs, axis=1)
        return self.classes_[idx]
