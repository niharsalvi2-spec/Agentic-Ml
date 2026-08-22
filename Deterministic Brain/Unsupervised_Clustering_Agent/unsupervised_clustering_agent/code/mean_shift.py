"""
mean_shift.py
=============
Every point climbs uphill toward the nearest peak of the data's density
landscape (gradient ascent on a kernel density estimate). Points that
converge to the same peak belong to the same cluster. Number of
clusters is discovered automatically -- it's the number of density
peaks found, not a parameter you set. See docs/04_mean_shift.md.

Two implementations:
    MeanShiftScratch  - pure NumPy, Gaussian-kernel mean shift
    MeanShiftSklearn    - wraps sklearn.cluster.MeanShift
"""
import numpy as np

try:
    from sklearn.cluster import MeanShift as _SkMeanShift, estimate_bandwidth as _sk_estimate_bandwidth
    _SKLEARN_AVAILABLE = True
except ImportError:
    _SKLEARN_AVAILABLE = False


def estimate_bandwidth(X, quantile=0.3, random_state=42):
    """
    Simple heuristic: bandwidth ~ median pairwise distance among a random
    sample of points, scaled by `quantile`. Matches the spirit of
    sklearn's estimate_bandwidth without requiring sklearn.
    """
    X = np.asarray(X, dtype=float)
    rng = np.random.RandomState(random_state)
    n = len(X)
    sample_idx = rng.choice(n, size=min(n, 500), replace=False)
    sample = X[sample_idx]
    dists = np.sqrt(((sample[:, None, :] - sample[None, :, :]) ** 2).sum(-1))
    upper = dists[np.triu_indices(len(sample), k=1)]
    return float(np.quantile(upper, quantile)) if len(upper) else 1.0


class MeanShiftScratch:
    """Gaussian-kernel Mean Shift clustering."""

    def __init__(self, bandwidth=None, max_iterations=300, tol=1e-3, cluster_merge_tol=None):
        self.bandwidth = bandwidth
        self.max_iterations = max_iterations
        self.tol = tol
        # points that converge within this distance of each other -> same cluster
        self.cluster_merge_tol = cluster_merge_tol
        self.labels_ = None
        self.cluster_centers_ = None

    def _gaussian_kernel(self, dist_sq, h):
        return np.exp(-dist_sq / (2 * h ** 2))

    def _shift_point(self, x, X, h):
        dist_sq = np.sum((X - x) ** 2, axis=1)
        weights = self._gaussian_kernel(dist_sq, h)
        return (weights[:, None] * X).sum(axis=0) / weights.sum()

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        h = self.bandwidth or estimate_bandwidth(X)
        merge_tol = self.cluster_merge_tol or h / 2

        shifted = X.copy()
        for _ in range(self.max_iterations):
            new_shifted = np.array([self._shift_point(p, X, h) for p in shifted])
            movement = np.max(np.sqrt(np.sum((new_shifted - shifted) ** 2, axis=1)))
            shifted = new_shifted
            if movement < self.tol:
                break

        # merge points that converged to (nearly) the same peak
        centers = []
        labels = np.full(len(X), -1, dtype=int)
        for i, point in enumerate(shifted):
            assigned = False
            for c_idx, center in enumerate(centers):
                if np.sqrt(np.sum((point - center) ** 2)) < merge_tol:
                    labels[i] = c_idx
                    assigned = True
                    break
            if not assigned:
                centers.append(point)
                labels[i] = len(centers) - 1

        self.cluster_centers_ = np.array(centers)
        self.labels_ = labels
        self.bandwidth_used_ = h
        return self

    def fit_predict(self, X):
        return self.fit(X).labels_


class MeanShiftSklearn:
    """Thin wrapper around sklearn's MeanShift."""

    def __init__(self, bandwidth=None):
        if not _SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn is not installed.")
        self.model = _SkMeanShift(bandwidth=bandwidth)

    def fit(self, X):
        self.model.fit(X)
        self.labels_ = self.model.labels_
        self.cluster_centers_ = self.model.cluster_centers_
        return self

    def fit_predict(self, X):
        return self.model.fit_predict(X)


if __name__ == "__main__":
    rng = np.random.RandomState(0)
    X = np.vstack([rng.randn(40, 2) * 0.6 + [5, 5], rng.randn(40, 2) * 0.6 + [-5, -5]])

    m = MeanShiftScratch().fit(X)
    print("Scratch clusters found:", len(m.cluster_centers_))

    if _SKLEARN_AVAILABLE:
        m2 = MeanShiftSklearn().fit(X)
        print("Sklearn clusters found:", len(m2.cluster_centers_))
