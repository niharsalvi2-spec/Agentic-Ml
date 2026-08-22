"""
AGENT-READABLE MODULE
======================
name: boosting
classifier: Boosting (Gradient Boosting / AdaBoost)
versions_in_file: [SklearnBoosting, ScratchBoosting]

WHEN TO USE:
- Need highest possible accuracy on tabular data (often outperforms Random Forest)
- Willing to tune hyperparameters carefully (more sensitive than RF)
- Have time budget for sequential (harder to parallelize) training

WHEN NOT TO USE:
- Very noisy data / many outliers (boosting can overfit to hard/noisy examples)
- Need fast training on very large data without a histogram-based implementation
  (prefer LightGBM/XGBoost/HistGradientBoosting for that; not included here)
- Need strong interpretability

KEY HYPERPARAMETERS (Gradient Boosting):
- n_estimators: number of boosting stages (trees)
- learning_rate: shrinks contribution of each tree (trade off vs n_estimators)
- max_depth: usually shallow trees (3-5) as weak learners
- subsample: <1.0 adds stochastic boosting (reduces overfitting)

KEY HYPERPARAMETERS (AdaBoost, used in scratch version):
- n_estimators: number of weak learners (decision stumps)
- learning_rate: shrinks each learner's contribution to the ensemble vote
"""

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier as _SkGB

from base_classifier import BaseClassifier


class SklearnBoosting(BaseClassifier):
    """Wrapper around sklearn.ensemble.GradientBoostingClassifier."""

    METADATA = {
        "name": "Gradient Boosting (sklearn)",
        "family": "ensemble-boosting",
        "supports_proba": True,
        "handles_nonlinear": True,
        "sensitive_to_scaling": False,
        "sensitive_to_outliers": True,  # more sensitive than RF
        "good_for_high_dim": False,
        "good_for_small_data": True,
        "good_for_large_data": False,  # use HistGradientBoosting for very large data
        "interpretable": False,
        "training_speed": "slow",
        "inference_speed": "medium",
        "handles_multiclass_natively": True,
        "handles_imbalance_well": False,
        "typical_use_cases": [
            "top-accuracy tabular classification",
            "Kaggle-style structured data competitions",
            "risk models where accuracy > interpretability",
        ],
    }

    def __init__(self, n_estimators=200, learning_rate=0.1, max_depth=3,
                 subsample=1.0, random_state=42):
        self.model = _SkGB(
            n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth,
            subsample=subsample, random_state=random_state,
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


class _DecisionStump:
    """Weak learner: single-feature threshold split, used by ScratchBoosting (AdaBoost)."""

    def __init__(self):
        self.feature = None
        self.threshold = None
        self.polarity = 1  # 1 or -1, flips which side predicts +1
        self.alpha = None  # this stump's vote weight in the ensemble

    def predict(self, X):
        n = X.shape[0]
        preds = np.ones(n)
        if self.polarity == 1:
            preds[X[:, self.feature] < self.threshold] = -1
        else:
            preds[X[:, self.feature] >= self.threshold] = -1
        return preds


class ScratchBoosting(BaseClassifier):
    """
    From-scratch AdaBoost (binary classification, labels must be {0,1} -> internally {-1,+1}).
    Sequentially fits decision stumps, reweighting misclassified samples each round.
    """

    METADATA = {
        "name": "AdaBoost (from-scratch)",
        "family": "ensemble-boosting",
        "supports_proba": True,
        "handles_nonlinear": True,
        "sensitive_to_scaling": False,
        "sensitive_to_outliers": True,
        "good_for_high_dim": False,
        "good_for_small_data": True,
        "good_for_large_data": False,
        "interpretable": False,
        "training_speed": "medium",
        "inference_speed": "fast",
        "handles_multiclass_natively": False,  # binary only as implemented
        "handles_imbalance_well": False,
        "typical_use_cases": [
            "teaching / interview implementation of AdaBoost",
            "small binary classification tasks",
        ],
    }

    def __init__(self, n_estimators=50, learning_rate=1.0):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.stumps = []
        self.classes_ = None

    def _best_stump(self, X, y, weights):
        n_samples, n_features = X.shape
        best_stump = _DecisionStump()
        min_error = np.inf

        for feature in range(n_features):
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                for polarity in (1, -1):
                    preds = np.ones(n_samples)
                    if polarity == 1:
                        preds[X[:, feature] < threshold] = -1
                    else:
                        preds[X[:, feature] >= threshold] = -1

                    error = np.sum(weights[preds != y])
                    if error < min_error:
                        min_error = error
                        best_stump.feature = feature
                        best_stump.threshold = threshold
                        best_stump.polarity = polarity
        return best_stump, min_error

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        if len(self.classes_) != 2:
            raise ValueError("ScratchBoosting (AdaBoost) supports binary classification only")

        # map labels to {-1, +1}
        y_signed = np.where(y == self.classes_[1], 1, -1)
        n_samples = X.shape[0]
        weights = np.full(n_samples, 1.0 / n_samples)
        self.stumps = []

        for _ in range(self.n_estimators):
            stump, error = self._best_stump(X, y_signed, weights)
            error = np.clip(error, 1e-10, 1 - 1e-10)
            stump.alpha = self.learning_rate * 0.5 * np.log((1 - error) / error)

            preds = stump.predict(X)
            weights *= np.exp(-stump.alpha * y_signed * preds)
            weights /= np.sum(weights)

            self.stumps.append(stump)
        return self

    def _decision_score(self, X):
        X = np.asarray(X, dtype=float)
        score = np.zeros(X.shape[0])
        for stump in self.stumps:
            score += stump.alpha * stump.predict(X)
        return score

    def predict_proba(self, X):
        score = self._decision_score(X)
        p_pos = 1 / (1 + np.exp(-2 * score))  # logistic-style calibration of AdaBoost margin
        return np.column_stack([1 - p_pos, p_pos])

    def predict(self, X):
        score = self._decision_score(X)
        return np.where(score >= 0, self.classes_[1], self.classes_[0])
