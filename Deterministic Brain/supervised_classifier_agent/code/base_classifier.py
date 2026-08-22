"""
AGENT-READABLE MODULE
======================
name: base_classifier
type: interface
purpose: Defines the common contract every classifier module (sklearn version
         and from-scratch version) must follow, so `supervised_classifier_agent.py`
         can call any of them interchangeably.

CONTRACT
--------
Every classifier class in this package (Sklearn* and Scratch*) implements:

    fit(X, y)                -> self
    predict(X)                -> np.ndarray, shape (n_samples,)
    predict_proba(X)          -> np.ndarray, shape (n_samples, n_classes)   [if supported]
    score(X, y)                -> float (accuracy)
    METADATA (class attribute) -> dict describing the model for agent reasoning

METADATA schema (used by supervised_classifier_agent.py to pick a model):
{
    "name": str,
    "family": str,                # e.g. "linear", "tree", "instance-based", "ensemble", "margin"
    "supports_proba": bool,
    "handles_nonlinear": bool,
    "sensitive_to_scaling": bool,
    "sensitive_to_outliers": bool,
    "good_for_high_dim": bool,
    "good_for_small_data": bool,
    "good_for_large_data": bool,
    "interpretable": bool,
    "training_speed": "fast" | "medium" | "slow",
    "inference_speed": "fast" | "medium" | "slow",
    "handles_multiclass_natively": bool,
    "handles_imbalance_well": bool,
    "typical_use_cases": [str, ...],
}
"""

from abc import ABC, abstractmethod
import numpy as np


class BaseClassifier(ABC):
    """Abstract base class. All Sklearn*/Scratch* classifiers inherit this."""

    METADATA = {}

    @abstractmethod
    def fit(self, X, y):
        ...

    @abstractmethod
    def predict(self, X):
        ...

    def predict_proba(self, X):
        raise NotImplementedError(f"{self.__class__.__name__} does not support predict_proba")

    def score(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)
        preds = self.predict(X)
        return float(np.mean(preds == y))

    @classmethod
    def describe(cls):
        """Return the agent-readable metadata block."""
        return cls.METADATA
