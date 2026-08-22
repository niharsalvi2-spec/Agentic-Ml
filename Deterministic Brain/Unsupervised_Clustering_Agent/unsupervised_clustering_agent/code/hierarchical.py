"""
hierarchical.py
================
Agglomerative (bottom-up) hierarchical clustering: start with every
point as its own cluster, repeatedly merge the two closest clusters,
building a full merge history (dendrogram) you can cut at any level.
See docs/02_hierarchical.md.

Two implementations:
    AgglomerativeScratch  - pure NumPy, single/complete/average/ward linkage
    AgglomerativeSklearn   - wraps sklearn.cluster.AgglomerativeClustering
"""
import numpy as np

try:
    from utils import pairwise_distances
except ImportError:
    from code.utils import pairwise_distances

try:
    from sklearn.cluster import AgglomerativeClustering as _SkAgg
    _SKLEARN_AVAILABLE = True
except ImportError:
    _SKLEARN_AVAILABLE = False


class AgglomerativeScratch:
    """
    Agglomerative clustering with a chosen linkage criterion.
        linkage='single'   : distance = min distance between any two points across clusters
        linkage='complete' : distance = max distance between any two points across clusters
        linkage='average'  : distance = mean distance between all cross-cluster pairs
        linkage='ward'      : merge the pair that increases total WCSS the least
    Stores the full merge history in self.merges_ (for drawing a dendrogram)
    and cuts the tree at n_clusters to produce self.labels_.
    """

    def __init__(self, n_clusters=2, linkage="ward"):
        self.n_clusters = n_clusters
        self.linkage = linkage
        self.labels_ = None
        self.merges_ = None  # list of (cluster_a_id, cluster_b_id, distance, new_size)

    def _cluster_distance(self, X, members_a, members_b):
        pts_a, pts_b = X[members_a], X[members_b]
        if self.linkage == "single":
            return np.min(pairwise_distances(np.vstack([pts_a, pts_b]))[: len(pts_a), len(pts_a):])
        if self.linkage == "complete":
            return np.max(pairwise_distances(np.vstack([pts_a, pts_b]))[: len(pts_a), len(pts_a):])
        if self.linkage == "average":
            return np.mean(pairwise_distances(np.vstack([pts_a, pts_b]))[: len(pts_a), len(pts_a):])
        if self.linkage == "ward":
            # increase in WCSS if these two clusters were merged
            combined = np.vstack([pts_a, pts_b])
            merged_sse = np.sum((combined - combined.mean(axis=0)) ** 2)
            sse_a = np.sum((pts_a - pts_a.mean(axis=0)) ** 2)
            sse_b = np.sum((pts_b - pts_b.mean(axis=0)) ** 2)
            return merged_sse - sse_a - sse_b
        raise ValueError("linkage must be 'single', 'complete', 'average', or 'ward'")

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        n = len(X)
        # each cluster = {id: list of point indices}
        clusters = {i: [i] for i in range(n)}
        active_ids = list(clusters.keys())
        self.merges_ = []
        next_id = n

        while len(active_ids) > 1:
            # find closest pair of active clusters
            best = (np.inf, None, None)
            for i in range(len(active_ids)):
                for j in range(i + 1, len(active_ids)):
                    a, b = active_ids[i], active_ids[j]
                    d = self._cluster_distance(X, clusters[a], clusters[b])
                    if d < best[0]:
                        best = (d, a, b)

            dist, a, b = best
            merged_members = clusters[a] + clusters[b]
            self.merges_.append((a, b, dist, len(merged_members)))

            clusters[next_id] = merged_members
            del clusters[a]
            del clusters[b]
            active_ids = list(clusters.keys())
            next_id += 1

        # replay merges, stopping once n_clusters remain, to build labels
        clusters = {i: [i] for i in range(n)}
        next_id = n
        n_merges_to_do = n - self.n_clusters
        for step, (a, b, dist, size) in enumerate(self.merges_):
            if step >= n_merges_to_do:
                break
            clusters[next_id] = clusters[a] + clusters[b]
            del clusters[a]
            del clusters[b]
            next_id += 1

        labels = np.empty(n, dtype=int)
        for label_idx, members in enumerate(clusters.values()):
            for m in members:
                labels[m] = label_idx
        self.labels_ = labels
        return self

    def fit_predict(self, X):
        return self.fit(X).labels_


class AgglomerativeSklearn:
    """Thin wrapper around sklearn's AgglomerativeClustering."""

    def __init__(self, n_clusters=2, linkage="ward"):
        if not _SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn is not installed.")
        self.model = _SkAgg(n_clusters=n_clusters, linkage=linkage)

    def fit(self, X):
        self.model.fit(X)
        self.labels_ = self.model.labels_
        return self

    def fit_predict(self, X):
        return self.model.fit_predict(X)


if __name__ == "__main__":
    rng = np.random.RandomState(0)
    X = np.vstack([rng.randn(20, 2) + [5, 5], rng.randn(20, 2) + [-5, -5]])

    m = AgglomerativeScratch(n_clusters=2, linkage="ward").fit(X)
    print("Scratch unique labels:", len(set(m.labels_)), " sizes:", np.bincount(m.labels_))

    if _SKLEARN_AVAILABLE:
        m2 = AgglomerativeSklearn(n_clusters=2, linkage="ward").fit(X)
        print("Sklearn unique labels:", len(set(m2.labels_)), " sizes:", np.bincount(m2.labels_))
