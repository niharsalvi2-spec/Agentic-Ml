"""
AGENT-READABLE MODULE
======================
name: knn
classifier: K-Nearest Neighbors
versions_in_file: [SklearnKNN, ScratchKNN]

WHEN TO USE:
- Nonlinear decision boundaries, no assumption about data distribution
- Small-to-medium datasets (inference cost grows with training set size)
- Low-dimensional or moderately-dimensional feature space (curse of dimensionality hurts KNN)
- Local structure matters more than global patterns

WHEN NOT TO USE:
- Very large datasets (slow inference: O(n) per query without indexing structures)
- High-dimensional data without dimensionality reduction (distances become meaningless)
- Need fast real-time predictions at scale

KEY HYPERPARAMETERS:
- n_neighbors (k): small k -> low bias/high variance (overfits), large k -> high bias/low variance
- weights: 'uniform' or 'distance' (closer neighbors count more)
- metric: 'minkowski' (p=2 is euclidean, p=1 is manhattan), 'cosine' for text/embeddings
- MUST scale features before using KNN (distance-based)
"""

import numpy as np
from collections import Counter
from sklearn.neighbors import KNeighborsClassifier as _SkKNN

from base_classifier import BaseClassifier


class SklearnKNN(BaseClassifier):
    """Wrapper around sklearn.neighbors.KNeighborsClassifier."""

    METADATA = {
        "name": "K-Nearest Neighbors (sklearn)",
        "family": "instance-based",
        "supports_proba": True,
        "handles_nonlinear": True,
        "sensitive_to_scaling": True,
        "sensitive_to_outliers": True,
        "good_for_high_dim": False,
        "good_for_small_data": True,
        "good_for_large_data": False,
        "interpretable": True,  # locally interpretable (nearest examples), not globally
        "training_speed": "fast",  # lazy learner, "training" = storing data
        "inference_speed": "slow",
        "handles_multiclass_natively": True,
        "handles_imbalance_well": False,
        "typical_use_cases": [
            "recommendation-style nearest-neighbor lookup",
            "small tabular datasets",
            "anomaly detection via distance to neighbors",
        ],
    }

    def __init__(self, n_neighbors=5, weights="uniform", metric="minkowski", p=2):
        self.model = _SkKNN(n_neighbors=n_neighbors, weights=weights, metric=metric, p=p)

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)


class ScratchKNN(BaseClassifier):
    """
    From-scratch KNN using brute-force distance computation (vectorized with numpy).
    Supports euclidean/manhattan distance and uniform/distance weighting.
    """

    METADATA = {
        "name": "K-Nearest Neighbors (from-scratch)",
        "family": "instance-based",
        "supports_proba": True,
        "handles_nonlinear": True,
        "sensitive_to_scaling": True,
        "sensitive_to_outliers": True,
        "good_for_high_dim": False,
        "good_for_small_data": True,
        "good_for_large_data": False,
        "interpretable": True,
        "training_speed": "fast",
        "inference_speed": "slow",
        "handles_multiclass_natively": True,
        "handles_imbalance_well": False,
        "typical_use_cases": [
            "teaching / interview implementation",
            "small datasets without sklearn",
        ],
    }

    def __init__(self, k=5, distance="euclidean", weights="uniform"):
        self.k = k
        self.distance = distance
        self.weights = weights
        self.X_train = None
        self.y_train = None
        self.classes_ = None

    def fit(self, X, y):
        self.X_train = np.asarray(X, dtype=float)
        self.y_train = np.asarray(y)
        self.classes_ = np.unique(self.y_train)
        return self

    def _pairwise_distances(self, X):
        X = np.asarray(X, dtype=float)
        if self.distance == "euclidean":
            # (a-b)^2 = a^2 - 2ab + b^2, vectorized
            sq_train = np.sum(self.X_train ** 2, axis=1)
            sq_test = np.sum(X ** 2, axis=1)
            cross = X @ self.X_train.T
            dist_sq = sq_test[:, None] - 2 * cross + sq_train[None, :]
            return np.sqrt(np.maximum(dist_sq, 0))
        elif self.distance == "manhattan":
            return np.sum(np.abs(X[:, None, :] - self.X_train[None, :, :]), axis=2)
        else:
            raise ValueError(f"Unsupported distance: {self.distance}")

    def predict_proba(self, X):
        dists = self._pairwise_distances(X)
        n_samples = dists.shape[0]
        n_classes = len(self.classes_)
        probs = np.zeros((n_samples, n_classes))
        class_to_idx = {c: i for i, c in enumerate(self.classes_)}

        k = min(self.k, self.X_train.shape[0])
        nn_idx = np.argsort(dists, axis=1)[:, :k]

        for row in range(n_samples):
            neighbor_idx = nn_idx[row]
            neighbor_labels = self.y_train[neighbor_idx]
            if self.weights == "distance":
                d = dists[row, neighbor_idx]
                w = 1.0 / (d + 1e-9)
            else:
                w = np.ones(k)
            for label, weight in zip(neighbor_labels, w):
                probs[row, class_to_idx[label]] += weight
            probs[row] /= probs[row].sum()
        return probs

    def predict(self, X):
        probs = self.predict_proba(X)
        idx = np.argmax(probs, axis=1)
        return self.classes_[idx]
