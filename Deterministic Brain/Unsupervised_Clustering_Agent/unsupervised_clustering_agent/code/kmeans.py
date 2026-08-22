"""
kmeans.py
=========
Partition data into K clusters by iteratively assigning points to the
nearest centroid, then moving centroids to the mean of their assigned
points. See docs/01_kmeans.md.

Two implementations:
    KMeansScratch  - pure NumPy, K-Means++ init, assign/update loop
    KMeansSklearn   - wraps sklearn.cluster.KMeans
"""
import numpy as np

try:
    from utils import pairwise_distances, inertia as _inertia
except ImportError:
    from code.utils import pairwise_distances, inertia as _inertia

try:
    from sklearn.cluster import KMeans as _SkKMeans
    _SKLEARN_AVAILABLE = True
except ImportError:
    _SKLEARN_AVAILABLE = False


class KMeansScratch:
    """K-Means with K-Means++ initialization (spreads out starting centroids)."""

    def __init__(self, k=3, max_iterations=300, n_init=10, tol=1e-4, random_state=42):
        self.k = k
        self.max_iterations = max_iterations
        self.n_init = n_init  # run multiple times, keep the lowest-WCSS result
        self.tol = tol
        self.random_state = random_state
        self.centroids_ = None
        self.labels_ = None
        self.inertia_ = None

    def _kmeans_plusplus_init(self, X, rng):
        n_samples = len(X)
        centroids = [X[rng.randint(n_samples)]]
        for _ in range(self.k - 1):
            dist_sq = np.min(
                [np.sum((X - c) ** 2, axis=1) for c in centroids], axis=0
            )
            probs = dist_sq / dist_sq.sum() if dist_sq.sum() > 0 else None
            next_idx = rng.choice(n_samples, p=probs)
            centroids.append(X[next_idx])
        return np.array(centroids)

    def _run_once(self, X, rng):
        centroids = self._kmeans_plusplus_init(X, rng)

        for _ in range(self.max_iterations):
            # assignment step: nearest centroid wins
            dist_to_centroids = np.array([
                np.sum((X - c) ** 2, axis=1) for c in centroids
            ]).T
            labels = np.argmin(dist_to_centroids, axis=1)

            # update step: move centroid to mean of assigned points
            new_centroids = np.array([
                X[labels == j].mean(axis=0) if np.any(labels == j) else centroids[j]
                for j in range(self.k)
            ])

            shift = np.sum((new_centroids - centroids) ** 2)
            centroids = new_centroids
            if shift < self.tol:
                break

        wcss = _inertia(X, labels, centroids)
        return centroids, labels, wcss

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        rng = np.random.RandomState(self.random_state)

        best_wcss = np.inf
        for run in range(self.n_init):
            centroids, labels, wcss = self._run_once(X, np.random.RandomState(self.random_state + run))
            if wcss < best_wcss:
                best_wcss, self.centroids_, self.labels_ = wcss, centroids, labels

        self.inertia_ = best_wcss
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        dist_to_centroids = np.array([
            np.sum((X - c) ** 2, axis=1) for c in self.centroids_
        ]).T
        return np.argmin(dist_to_centroids, axis=1)

    def fit_predict(self, X):
        return self.fit(X).labels_


class KMeansSklearn:
    """Thin wrapper around sklearn's KMeans (K-Means++ init by default)."""

    def __init__(self, k=3, n_init=10, random_state=42):
        if not _SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn is not installed.")
        self.model = _SkKMeans(n_clusters=k, n_init=n_init, random_state=random_state)

    def fit(self, X):
        self.model.fit(X)
        self.centroids_ = self.model.cluster_centers_
        self.labels_ = self.model.labels_
        self.inertia_ = self.model.inertia_
        return self

    def predict(self, X):
        return self.model.predict(X)

    def fit_predict(self, X):
        return self.model.fit_predict(X)


if __name__ == "__main__":
    rng = np.random.RandomState(0)
    X = np.vstack([rng.randn(50, 2) + [5, 5], rng.randn(50, 2) + [-5, -5], rng.randn(50, 2) + [5, -5]])

    m = KMeansScratch(k=3).fit(X)
    print("Scratch inertia:", round(m.inertia_, 2), " unique labels:", len(set(m.labels_)))

    if _SKLEARN_AVAILABLE:
        m2 = KMeansSklearn(k=3).fit(X)
        print("Sklearn inertia:", round(m2.inertia_, 2), " unique labels:", len(set(m2.labels_)))
