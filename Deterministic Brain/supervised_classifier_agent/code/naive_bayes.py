"""
AGENT-READABLE MODULE
======================
name: naive_bayes
classifier: Naive Bayes (Gaussian)
versions_in_file: [SklearnNaiveBayes, ScratchNaiveBayes]

WHEN TO USE:
- Text classification (spam detection, sentiment) - typically with Multinomial NB, not included here
- Very fast baseline, works well with high-dimensional sparse data
- Small training sets (strong prior assumptions help generalize)
- Features are (approximately) conditionally independent given the class

WHEN NOT TO USE:
- Features are strongly correlated (violates independence assumption -> biased probability estimates,
  though class predictions can still be decent)
- Need well-calibrated probabilities (NB probabilities are often overconfident)

KEY HYPERPARAMETERS (Gaussian NB):
- var_smoothing: small value added to variances for numerical stability
- priors: optionally set explicit class priors instead of learning from data
"""

import numpy as np
from sklearn.naive_bayes import GaussianNB as _SkGaussianNB

from base_classifier import BaseClassifier


class SklearnNaiveBayes(BaseClassifier):
    """Wrapper around sklearn.naive_bayes.GaussianNB."""

    METADATA = {
        "name": "Gaussian Naive Bayes (sklearn)",
        "family": "probabilistic",
        "supports_proba": True,
        "handles_nonlinear": True,  # decision boundary can be nonlinear for unequal variances
        "sensitive_to_scaling": False,
        "sensitive_to_outliers": True,  # affects mean/variance estimates
        "good_for_high_dim": True,
        "good_for_small_data": True,
        "good_for_large_data": True,
        "interpretable": True,
        "training_speed": "fast",
        "inference_speed": "fast",
        "handles_multiclass_natively": True,
        "handles_imbalance_well": True,  # priors adjust automatically
        "typical_use_cases": [
            "quick baseline classifier",
            "real-time / low-latency scoring",
            "small labeled datasets",
            "sensor / continuous-feature classification",
        ],
    }

    def __init__(self, var_smoothing=1e-9, priors=None):
        self.model = _SkGaussianNB(var_smoothing=var_smoothing, priors=priors)

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)


class ScratchNaiveBayes(BaseClassifier):
    """
    From-scratch Gaussian Naive Bayes.
    For each class, learns per-feature mean/variance; assumes features are
    conditionally independent given the class (the "naive" assumption).
    """

    METADATA = {
        "name": "Gaussian Naive Bayes (from-scratch)",
        "family": "probabilistic",
        "supports_proba": True,
        "handles_nonlinear": True,
        "sensitive_to_scaling": False,
        "sensitive_to_outliers": True,
        "good_for_high_dim": True,
        "good_for_small_data": True,
        "good_for_large_data": True,
        "interpretable": True,
        "training_speed": "fast",
        "inference_speed": "fast",
        "handles_multiclass_natively": True,
        "handles_imbalance_well": True,
        "typical_use_cases": [
            "teaching / interview implementation",
            "quick baseline without sklearn",
        ],
    }

    def __init__(self, var_smoothing=1e-9):
        self.var_smoothing = var_smoothing
        self.classes_ = None
        self.mean_ = {}
        self.var_ = {}
        self.priors_ = {}

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        eps = self.var_smoothing * np.var(X, axis=0).max() if X.shape[0] > 1 else self.var_smoothing

        for c in self.classes_:
            X_c = X[y == c]
            self.mean_[c] = X_c.mean(axis=0)
            self.var_[c] = X_c.var(axis=0) + eps
            self.priors_[c] = X_c.shape[0] / X.shape[0]
        return self

    def _log_gaussian_likelihood(self, X, mean, var):
        # log N(x; mean, var) summed across features (independence assumption)
        coeff = -0.5 * np.log(2 * np.pi * var)
        exponent = -((X - mean) ** 2) / (2 * var)
        return np.sum(coeff + exponent, axis=1)

    def predict_proba(self, X):
        X = np.asarray(X, dtype=float)
        log_joint = []
        for c in self.classes_:
            log_prior = np.log(self.priors_[c])
            log_likelihood = self._log_gaussian_likelihood(X, self.mean_[c], self.var_[c])
            log_joint.append(log_prior + log_likelihood)
        log_joint = np.array(log_joint).T  # shape (n_samples, n_classes)

        # softmax-style normalization for stability
        max_log = log_joint.max(axis=1, keepdims=True)
        probs = np.exp(log_joint - max_log)
        probs /= probs.sum(axis=1, keepdims=True)
        return probs

    def predict(self, X):
        probs = self.predict_proba(X)
        idx = np.argmax(probs, axis=1)
        return self.classes_[idx]
