"""
AGENT-READABLE MODULE
======================
name: svm
classifier: Support Vector Machine
versions_in_file: [SklearnSVM, ScratchSVM]

WHEN TO USE:
- Clear margin of separation between classes exists (or can be created via kernel trick)
- High-dimensional data (text, bio data) with fewer samples than features
- Need robust classifier resistant to overfitting when margin is clear
- Kernel trick handles nonlinear boundaries (RBF, polynomial) without manual feature engineering

WHEN NOT TO USE:
- Very large datasets (training scales poorly, roughly O(n^2) to O(n^3) for kernel SVM)
- Need probability estimates that are cheap to compute (requires extra Platt scaling / CV)
- Need direct multiclass without one-vs-rest/one-vs-one overhead
- Noisy data with overlapping classes (very sensitive to C and outliers near margin)

KEY HYPERPARAMETERS:
- C: regularization strength (small C = wider margin/more tolerant of misclassification,
     large C = fits training data more tightly, risk of overfitting)
- kernel: 'linear', 'rbf' (default, good general-purpose), 'poly', 'sigmoid'
- gamma: kernel coefficient for 'rbf'/'poly' (controls influence radius of a single sample)
- MUST scale features before using SVM (distance/margin-based)
"""

import numpy as np
from sklearn.svm import SVC as _SkSVC

from base_classifier import BaseClassifier


class SklearnSVM(BaseClassifier):
    """Wrapper around sklearn.svm.SVC."""

    METADATA = {
        "name": "Support Vector Machine (sklearn)",
        "family": "margin-based",
        "supports_proba": True,  # only if probability=True (adds cost via internal CV)
        "handles_nonlinear": True,  # via kernel trick
        "sensitive_to_scaling": True,
        "sensitive_to_outliers": True,
        "good_for_high_dim": True,
        "good_for_small_data": True,
        "good_for_large_data": False,
        "interpretable": False,  # linear kernel is somewhat interpretable, rbf is not
        "training_speed": "slow",
        "inference_speed": "medium",
        "handles_multiclass_natively": False,  # sklearn does OvO internally
        "handles_imbalance_well": False,  # unless class_weight='balanced'
        "typical_use_cases": [
            "text/document classification (high-dim, linear kernel)",
            "bioinformatics (few samples, many features)",
            "image classification with engineered features",
        ],
    }

    def __init__(self, C=1.0, kernel="rbf", gamma="scale", class_weight=None,
                 probability=True, random_state=42):
        self.model = _SkSVC(
            C=C, kernel=kernel, gamma=gamma, class_weight=class_weight,
            probability=probability, random_state=random_state,
        )

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def support_vectors(self):
        return self.model.support_vectors_


class ScratchSVM(BaseClassifier):
    """
    From-scratch LINEAR SVM (binary) trained via subgradient descent on hinge loss:
        L = (1/2)||w||^2 + C * mean(max(0, 1 - y*(w.x+b)))
    No kernel trick (linear boundary only). Labels internally mapped to {-1, +1}.
    """

    METADATA = {
        "name": "Linear SVM (from-scratch)",
        "family": "margin-based",
        "supports_proba": False,  # raw hinge-loss SVM has no native probability output
        "handles_nonlinear": False,  # linear only, no kernel trick implemented
        "sensitive_to_scaling": True,
        "sensitive_to_outliers": True,
        "good_for_high_dim": True,
        "good_for_small_data": True,
        "good_for_large_data": False,
        "interpretable": True,  # linear weight vector is interpretable
        "training_speed": "medium",
        "inference_speed": "fast",
        "handles_multiclass_natively": False,
        "handles_imbalance_well": False,
        "typical_use_cases": [
            "teaching / interview implementation of linear SVM",
            "linearly-separable or near-separable binary tasks",
        ],
    }

    def __init__(self, C=1.0, lr=0.001, n_iters=1000, random_state=42):
        self.C = C
        self.lr = lr
        self.n_iters = n_iters
        self.random_state = random_state
        self.w = None
        self.b = 0.0
        self.classes_ = None

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        if len(self.classes_) != 2:
            raise ValueError("ScratchSVM supports binary classification only")

        y_signed = np.where(y == self.classes_[1], 1, -1)
        n_samples, n_features = X.shape
        rng = np.random.RandomState(self.random_state)
        self.w = rng.normal(scale=0.01, size=n_features)
        self.b = 0.0

        for _ in range(self.n_iters):
            margins = y_signed * (X @ self.w + self.b)
            misclassified = margins < 1

            # gradient of (1/2)||w||^2 is w; gradient of hinge term only from violating points
            grad_w = self.w - self.C * (X[misclassified].T @ y_signed[misclassified]) / n_samples
            grad_b = -self.C * np.sum(y_signed[misclassified]) / n_samples

            self.w -= self.lr * grad_w
            self.b -= self.lr * grad_b
        return self

    def decision_function(self, X):
        X = np.asarray(X, dtype=float)
        return X @ self.w + self.b

    def predict(self, X):
        scores = self.decision_function(X)
        return np.where(scores >= 0, self.classes_[1], self.classes_[0])
