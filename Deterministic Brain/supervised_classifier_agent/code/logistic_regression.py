"""
AGENT-READABLE MODULE
======================
name: logistic_regression
classifier: Logistic Regression
versions_in_file: [SklearnLogisticRegression, ScratchLogisticRegression]

WHEN TO USE (quick reference for agent):
- Baseline classifier, binary or multiclass
- Data is roughly linearly separable in feature space (or after feature engineering)
- Need interpretable coefficients / probability outputs
- Fast to train, fast to predict, low variance

WHEN NOT TO USE:
- Complex nonlinear decision boundaries with no feature engineering
- Heavily correlated / high-multicollinearity features without regularization

KEY HYPERPARAMETERS:
- C (inverse regularization strength): smaller C = stronger regularization
- penalty: 'l1' (sparse/feature-selecting), 'l2' (default, ridge-like), 'elasticnet'
- solver: 'lbfgs' (default, small-medium data), 'saga' (large data, supports l1/elasticnet)
- class_weight: 'balanced' for imbalanced classes
"""

import numpy as np
from sklearn.linear_model import LogisticRegression as _SkLogReg

from base_classifier import BaseClassifier


class SklearnLogisticRegression(BaseClassifier):
    """Thin, agent-friendly wrapper around sklearn.linear_model.LogisticRegression."""

    METADATA = {
        "name": "Logistic Regression (sklearn)",
        "family": "linear",
        "supports_proba": True,
        "handles_nonlinear": False,
        "sensitive_to_scaling": True,
        "sensitive_to_outliers": True,
        "good_for_high_dim": True,
        "good_for_small_data": True,
        "good_for_large_data": True,
        "interpretable": True,
        "training_speed": "fast",
        "inference_speed": "fast",
        "handles_multiclass_natively": True,
        "handles_imbalance_well": False,  # unless class_weight='balanced'
        "typical_use_cases": [
            "binary classification baseline",
            "credit scoring / churn prediction",
            "medical risk scoring (interpretable coefficients)",
            "text classification with TF-IDF features",
        ],
    }

    def __init__(self, C=1.0, solver="lbfgs", class_weight=None,
                 max_iter=1000, random_state=42, **kwargs):
        # Note: newer scikit-learn versions deprecate the `penalty` kwarg in favor
        # of `l1_ratio`/`C`; we rely on the solver's default (L2) regularization.
        self.model = _SkLogReg(
            C=C, solver=solver, class_weight=class_weight,
            max_iter=max_iter, random_state=random_state, **kwargs
        )

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def coefficients(self):
        """Return (coef, intercept) for interpretability."""
        return self.model.coef_, self.model.intercept_


class ScratchLogisticRegression(BaseClassifier):
    """
    From-scratch binary logistic regression via batch gradient descent.
    Implements: sigmoid, cross-entropy loss, L2 regularization, gradient descent.
    For multiclass, wrap with one-vs-rest (see fit_multiclass note below).
    """

    METADATA = {
        "name": "Logistic Regression (from-scratch)",
        "family": "linear",
        "supports_proba": True,
        "handles_nonlinear": False,
        "sensitive_to_scaling": True,
        "sensitive_to_outliers": True,
        "good_for_high_dim": True,
        "good_for_small_data": True,
        "good_for_large_data": False,  # naive full-batch gradient descent
        "interpretable": True,
        "training_speed": "medium",
        "inference_speed": "fast",
        "handles_multiclass_natively": False,  # binary only, OvR needed for multiclass
        "handles_imbalance_well": False,
        "typical_use_cases": [
            "teaching / interview implementation",
            "small binary classification tasks",
            "when sklearn dependency is unavailable",
        ],
    }

    def __init__(self, lr=0.1, n_iters=2000, l2=0.0, tol=1e-7, verbose=False):
        self.lr = lr
        self.n_iters = n_iters
        self.l2 = l2
        self.tol = tol
        self.verbose = verbose
        self.weights = None
        self.bias = 0.0
        self.loss_history = []

    @staticmethod
    def _sigmoid(z):
        z = np.clip(z, -500, 500)
        return 1.0 / (1.0 + np.exp(-z))

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0.0
        prev_loss = np.inf

        for i in range(self.n_iters):
            linear = X @ self.weights + self.bias
            preds = self._sigmoid(linear)

            error = preds - y
            grad_w = (X.T @ error) / n_samples + (self.l2 / n_samples) * self.weights
            grad_b = np.mean(error)

            self.weights -= self.lr * grad_w
            self.bias -= self.lr * grad_b

            eps = 1e-12
            loss = -np.mean(y * np.log(preds + eps) + (1 - y) * np.log(1 - preds + eps))
            loss += (self.l2 / (2 * n_samples)) * np.sum(self.weights ** 2)
            self.loss_history.append(loss)

            if self.verbose and i % 200 == 0:
                print(f"iter {i}: loss={loss:.6f}")
            if abs(prev_loss - loss) < self.tol:
                break
            prev_loss = loss

        return self

    def predict_proba(self, X):
        X = np.asarray(X, dtype=float)
        p1 = self._sigmoid(X @ self.weights + self.bias)
        return np.column_stack([1 - p1, p1])

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X)[:, 1] >= threshold).astype(int)
