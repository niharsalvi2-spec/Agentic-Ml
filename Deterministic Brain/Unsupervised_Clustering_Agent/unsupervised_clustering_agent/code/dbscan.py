"""
dbscan.py
=========
Density-Based Spatial Clustering: a cluster is a dense region of points
separated by sparse regions. Finds arbitrary-shaped clusters and labels
isolated points as noise (-1) rather than forcing them into a cluster.
See docs/03_dbscan.md.

Two implementations:
    DBSCANScratch  - pure NumPy, core/border/noise point expansion
    DBSCANSklearn   - wraps sklearn.cluster.DBSCAN
"""
import numpy as np

try:
    from utils import pairwise_distances
except ImportError:
    from code.utils import pairwise_distances

try:
    from sklearn.cluster import DBSCAN as _SkDBSCAN
    _SKLEARN_AVAILABLE = True
except ImportError:
    _SKLEARN_AVAILABLE = False


class DBSCANScratch:
    """
    eps: radius of the neighborhood.
    min_pts: minimum neighbors (including self) required for a point to
             be a core point. Rule of thumb: min_pts >= dimensions + 1.
    Noise points end up labeled -1, matching sklearn's convention.
    """

    def __init__(self, eps=0.5, min_pts=5):
        self.eps = eps
        self.min_pts = min_pts
        self.labels_ = None

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        n = len(X)
        dist = pairwise_distances(X)
        neighbors = [np.where(dist[i] <= self.eps)[0] for i in range(n)]

        UNVISITED, NOISE = -2, -1
        labels = np.full(n, UNVISITED, dtype=int)
        cluster_id = 0

        for i in range(n):
            if labels[i] != UNVISITED:
                continue

            if len(neighbors[i]) < self.min_pts:
                labels[i] = NOISE  # may be reclassified as a border point below
                continue

            # i is a core point -> start a new cluster and expand it
            labels[i] = cluster_id
            seed_set = list(neighbors[i])
            idx = 0
            while idx < len(seed_set):
                q = seed_set[idx]
                idx += 1
                if labels[q] == NOISE:
                    labels[q] = cluster_id  # noise point turns out to be a border point
                if labels[q] != UNVISITED:
                    continue
                labels[q] = cluster_id
                if len(neighbors[q]) >= self.min_pts:
                    seed_set.extend(neighbors[q])  # q is also core -> expand further

            cluster_id += 1

        self.labels_ = labels
        self.core_sample_mask_ = np.array([len(neighbors[i]) >= self.min_pts for i in range(n)])
        return self

    def fit_predict(self, X):
        return self.fit(X).labels_


class DBSCANSklearn:
    """Thin wrapper around sklearn's DBSCAN."""

    def __init__(self, eps=0.5, min_pts=5):
        if not _SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn is not installed.")
        self.model = _SkDBSCAN(eps=eps, min_samples=min_pts)

    def fit(self, X):
        self.model.fit(X)
        self.labels_ = self.model.labels_
        self.core_sample_mask_ = np.zeros(len(X), dtype=bool)
        self.core_sample_mask_[self.model.core_sample_indices_] = True
        return self

    def fit_predict(self, X):
        return self.model.fit_predict(X)


if __name__ == "__main__":
    rng = np.random.RandomState(0)
    X = np.vstack([
        rng.randn(30, 2) * 0.5 + [5, 5],
        rng.randn(30, 2) * 0.5 + [-5, -5],
        rng.uniform(-10, 10, size=(5, 2)),  # scattered noise
    ])

    m = DBSCANScratch(eps=1.0, min_pts=5).fit(X)
    print("Scratch labels found:", sorted(set(m.labels_)), " noise count:", np.sum(m.labels_ == -1))

    if _SKLEARN_AVAILABLE:
        m2 = DBSCANSklearn(eps=1.0, min_pts=5).fit(X)
        print("Sklearn labels found:", sorted(set(m2.labels_)), " noise count:", np.sum(m2.labels_ == -1))
