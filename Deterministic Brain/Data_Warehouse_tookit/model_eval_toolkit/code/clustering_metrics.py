"""
clustering_metrics.py
-----------------------
Pure-numpy implementations of standard clustering evaluation metrics.

Internal metrics (no ground truth needed):
    silhouette_score, davies_bouldin_index, inertia

External metrics (need ground truth labels):
    adjusted_rand_index, normalized_mutual_info

No external dependencies besides numpy.
"""

import numpy as np
from itertools import combinations


# --------------------------------------------------------------------------
# Internal metrics
# --------------------------------------------------------------------------

def _pairwise_distances(X):
    """Euclidean distance matrix, shape (n, n)."""
    X = np.asarray(X, dtype=float)
    diff = X[:, None, :] - X[None, :, :]
    return np.sqrt(np.sum(diff ** 2, axis=-1))


def silhouette_score(X, labels):
    """
    For each point i:
        a(i) = mean distance to other points in the same cluster (cohesion)
        b(i) = mean distance to points in the nearest *other* cluster (separation)
        s(i) = (b(i) - a(i)) / max(a(i), b(i))
    Returns the mean s(i) across all points.

    Range: -1 (likely wrong cluster) to +1 (well clustered), 0 = on the
    boundary between two clusters. Use to pick K or compare algorithms.
    """
    X = np.asarray(X, dtype=float)
    labels = np.asarray(labels)
    n = len(X)
    unique_labels = np.unique(labels)

    if len(unique_labels) < 2:
        raise ValueError("Silhouette score requires at least 2 clusters")

    dist = _pairwise_distances(X)
    scores = np.zeros(n)

    for i in range(n):
        own_label = labels[i]
        own_mask = (labels == own_label)
        own_mask[i] = False  # exclude self

        if own_mask.sum() == 0:
            scores[i] = 0.0  # singleton cluster
            continue

        a_i = dist[i, own_mask].mean()

        b_i = np.inf
        for other_label in unique_labels:
            if other_label == own_label:
                continue
            other_mask = (labels == other_label)
            mean_dist = dist[i, other_mask].mean()
            b_i = min(b_i, mean_dist)

        scores[i] = (b_i - a_i) / max(a_i, b_i) if max(a_i, b_i) > 0 else 0.0

    return float(np.mean(scores))


def davies_bouldin_index(X, labels):
    """
    For each cluster pair (i, j):
        R_ij = (sigma_i + sigma_j) / d(mu_i, mu_j)
    where sigma_k = mean intra-cluster distance to centroid k (spread,
    want small) and d(mu_i, mu_j) = distance between centroids (want large).

    DB Index = mean over clusters of max_j!=i(R_ij). Lower is better:
    compact clusters that are far apart from each other.
    """
    X = np.asarray(X, dtype=float)
    labels = np.asarray(labels)
    unique_labels = np.unique(labels)
    k = len(unique_labels)

    if k < 2:
        raise ValueError("Davies-Bouldin index requires at least 2 clusters")

    centroids = np.array([X[labels == lbl].mean(axis=0) for lbl in unique_labels])
    sigmas = np.array([
        np.mean(np.sqrt(np.sum((X[labels == lbl] - centroids[idx]) ** 2, axis=1)))
        for idx, lbl in enumerate(unique_labels)
    ])

    db_values = []
    for i in range(k):
        max_r = -np.inf
        for j in range(k):
            if i == j:
                continue
            centroid_dist = np.sqrt(np.sum((centroids[i] - centroids[j]) ** 2))
            r_ij = (sigmas[i] + sigmas[j]) / centroid_dist if centroid_dist > 0 else np.inf
            max_r = max(max_r, r_ij)
        db_values.append(max_r)

    return float(np.mean(db_values))


def inertia(X, labels):
    """
    Within-Cluster Sum of Squares (WCSS): sum of squared distances from each
    point to its own cluster's centroid. Lower = tighter clusters. Always
    decreases as K increases, so only meaningful compared across K via the
    Elbow Method (and only well-defined for centroid-based clustering like
    K-Means).
    """
    X = np.asarray(X, dtype=float)
    labels = np.asarray(labels)
    unique_labels = np.unique(labels)

    total = 0.0
    for lbl in unique_labels:
        cluster_points = X[labels == lbl]
        centroid = cluster_points.mean(axis=0)
        total += np.sum((cluster_points - centroid) ** 2)

    return float(total)


# --------------------------------------------------------------------------
# External metrics (need ground truth labels)
# --------------------------------------------------------------------------

def _contingency_matrix(labels_true, labels_pred):
    labels_true = np.asarray(labels_true)
    labels_pred = np.asarray(labels_pred)
    true_classes = np.unique(labels_true)
    pred_classes = np.unique(labels_pred)

    matrix = np.zeros((len(true_classes), len(pred_classes)), dtype=int)
    true_idx = {lbl: i for i, lbl in enumerate(true_classes)}
    pred_idx = {lbl: i for i, lbl in enumerate(pred_classes)}

    for t, p in zip(labels_true, labels_pred):
        matrix[true_idx[t], pred_idx[p]] += 1

    return matrix


def adjusted_rand_index(labels_true, labels_pred):
    """
    Adjusted Rand Index (ARI): chance-corrected agreement between a
    clustering and ground-truth labels, over all pairs of samples.
    1 = perfect match, 0 = agreement expected from random labeling,
    <0 = worse than random.
    """
    contingency = _contingency_matrix(labels_true, labels_pred)
    n = contingency.sum()

    def comb2(x):
        return x * (x - 1) / 2

    sum_comb_c = np.sum([comb2(x) for x in contingency.flatten()])
    sum_comb_rows = np.sum([comb2(x) for x in contingency.sum(axis=1)])
    sum_comb_cols = np.sum([comb2(x) for x in contingency.sum(axis=0)])
    total_comb = comb2(n)

    expected_index = (sum_comb_rows * sum_comb_cols) / total_comb if total_comb else 0
    max_index = 0.5 * (sum_comb_rows + sum_comb_cols)

    denom = max_index - expected_index
    if denom == 0:
        return 1.0  # both trivially agree (e.g. all singletons or one cluster)

    return float((sum_comb_c - expected_index) / denom)


def normalized_mutual_info(labels_true, labels_pred):
    """
    Normalized Mutual Information (NMI): information shared between cluster
    assignments and true labels, normalized by the average entropy of both.
    0 = no shared information (random), 1 = perfect correspondence.

    Handles differing numbers of clusters vs. true classes more gracefully
    than ARI.
    """
    labels_true = np.asarray(labels_true)
    labels_pred = np.asarray(labels_pred)
    n = len(labels_true)

    contingency = _contingency_matrix(labels_true, labels_pred)
    p_true = contingency.sum(axis=1) / n
    p_pred = contingency.sum(axis=0) / n
    p_joint = contingency / n

    mi = 0.0
    for i in range(contingency.shape[0]):
        for j in range(contingency.shape[1]):
            if p_joint[i, j] > 0:
                mi += p_joint[i, j] * np.log(p_joint[i, j] / (p_true[i] * p_pred[j]))

    def entropy(p):
        p = p[p > 0]
        return float(-np.sum(p * np.log(p)))

    h_true = entropy(p_true)
    h_pred = entropy(p_pred)

    denom = np.sqrt(h_true * h_pred)
    if denom == 0:
        return 1.0 if mi == 0 else 0.0

    return float(mi / denom)


# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------

def clustering_report(X, labels_pred, labels_true=None):
    """
    Returns a dict of internal metrics (always), plus external metrics
    (ARI, NMI) if ground-truth labels_true is provided.
    """
    report = {
        "silhouette": silhouette_score(X, labels_pred),
        "davies_bouldin": davies_bouldin_index(X, labels_pred),
        "inertia": inertia(X, labels_pred),
    }
    if labels_true is not None:
        report["adjusted_rand_index"] = adjusted_rand_index(labels_true, labels_pred)
        report["nmi"] = normalized_mutual_info(labels_true, labels_pred)
    return report


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    X = np.vstack([
        rng.normal(loc=[0, 0], scale=0.5, size=(20, 2)),
        rng.normal(loc=[5, 5], scale=0.5, size=(20, 2)),
    ])
    labels = np.array([0] * 20 + [1] * 20)
    print(clustering_report(X, labels, labels_true=labels))
