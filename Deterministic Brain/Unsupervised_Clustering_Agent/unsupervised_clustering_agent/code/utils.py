"""
utils.py
========
Shared helpers for every clustering model in this repo (NumPy only,
no sklearn dependency, so the "Scratch" classes never need it).

Provides:
    StandardScaler
    pairwise_distances       - full n x n Euclidean distance matrix
    euclidean_distance
    inertia / wcss           - Within-Cluster Sum of Squares (K-Means objective)
    elbow_curve               - WCSS for a range of K, for the elbow method
    silhouette_score          - per-point and averaged silhouette
    k_distance_plot_values    - sorted k-th nearest neighbor distances, for
                                 picking DBSCAN's eps
"""
import numpy as np


class StandardScaler:
    """Mandatory preprocessing for any distance-based clustering method."""

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0)
        self.std_[self.std_ == 0] = 1.0
        return self

    def transform(self, X):
        return (np.asarray(X, dtype=float) - self.mean_) / self.std_

    def fit_transform(self, X):
        return self.fit(X).transform(X)


def euclidean_distance(a, b):
    return np.sqrt(np.sum((a - b) ** 2))


def pairwise_distances(X):
    """Full n x n Euclidean distance matrix, vectorized."""
    X = np.asarray(X, dtype=float)
    sq = np.sum(X ** 2, axis=1)
    dist_sq = sq[:, None] + sq[None, :] - 2 * X @ X.T
    dist_sq = np.maximum(dist_sq, 0)  # clip tiny negative values from float error
    return np.sqrt(dist_sq)


def inertia(X, labels, centroids):
    """Within-Cluster Sum of Squares (WCSS) -- the K-Means objective."""
    X = np.asarray(X, dtype=float)
    total = 0.0
    for k, c in enumerate(centroids):
        pts = X[labels == k]
        if len(pts):
            total += np.sum((pts - c) ** 2)
    return total


def elbow_curve(kmeans_scratch_cls, X, k_range=range(1, 11), **kmeans_kwargs):
    """Fit kmeans_scratch_cls for each K in k_range, return list of (K, wcss)."""
    results = []
    for k in k_range:
        model = kmeans_scratch_cls(k=k, **kmeans_kwargs).fit(X)
        results.append((k, model.inertia_))
    return results


def silhouette_score(X, labels):
    """
    Average silhouette score across all points.
        a(i) = mean distance to other points in the SAME cluster
        b(i) = mean distance to points in the NEAREST other cluster
        s(i) = (b(i) - a(i)) / max(a(i), b(i))
    Range -1..+1, higher is better. Points in a singleton cluster get s(i)=0.
    """
    X = np.asarray(X, dtype=float)
    labels = np.asarray(labels)
    unique_labels = [l for l in np.unique(labels) if l != -1]  # exclude DBSCAN noise
    if len(unique_labels) < 2:
        return 0.0

    dist = pairwise_distances(X)
    scores = []
    for i in range(len(X)):
        own_label = labels[i]
        if own_label == -1:
            continue  # noise points are not scored
        same_cluster = np.where((labels == own_label))[0]
        same_cluster = same_cluster[same_cluster != i]
        if len(same_cluster) == 0:
            scores.append(0.0)
            continue
        a_i = dist[i, same_cluster].mean()

        b_i = np.inf
        for other_label in unique_labels:
            if other_label == own_label:
                continue
            other_cluster = np.where(labels == other_label)[0]
            if len(other_cluster) == 0:
                continue
            b_i = min(b_i, dist[i, other_cluster].mean())

        s_i = (b_i - a_i) / max(a_i, b_i) if max(a_i, b_i) > 0 else 0.0
        scores.append(s_i)

    return float(np.mean(scores)) if scores else 0.0


def k_distance_plot_values(X, k):
    """
    Sorted (descending) distance-to-kth-nearest-neighbor for every point.
    Used to eyeball the "knee" -> a good DBSCAN eps value.
    """
    dist = pairwise_distances(X)
    sorted_dist = np.sort(dist, axis=1)  # column 0 is self (distance 0)
    kth = sorted_dist[:, k]  # k-th neighbor (0-indexed self excluded by k>=1)
    return np.sort(kth)[::-1]
