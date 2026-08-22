"""
Example: clustering_metrics.py

Run with:  python3 example_clustering_metrics.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "code"))

import numpy as np
from clustering_metrics import (
    silhouette_score, davies_bouldin_index, inertia,
    adjusted_rand_index, normalized_mutual_info, clustering_report,
)


def make_three_blobs(rng):
    return np.vstack([
        rng.normal([0, 0], 0.5, (30, 2)),
        rng.normal([5, 5], 0.5, (30, 2)),
        rng.normal([0, 5], 0.5, (30, 2)),
    ])


def elbow_and_silhouette_example():
    print("=" * 70)
    print("Example 1: choosing K with Inertia (elbow) + Silhouette")
    print("=" * 70)

    rng = np.random.default_rng(0)
    X = make_three_blobs(rng)  # true structure: 3 clusters

    # naive K-means-style assignment by nearest of K evenly spaced seed centroids,
    # just to have *some* labeling to score without pulling in sklearn
    def assign(X, k, seed=1):
        rng2 = np.random.default_rng(seed)
        centroids = X[rng2.choice(len(X), k, replace=False)]
        for _ in range(10):
            dists = np.linalg.norm(X[:, None, :] - centroids[None, :, :], axis=-1)
            labels = dists.argmin(axis=1)
            for k_i in range(k):
                if np.any(labels == k_i):
                    centroids[k_i] = X[labels == k_i].mean(axis=0)
        return labels

    for k in [2, 3, 4, 5]:
        labels = assign(X, k)
        inert = inertia(X, labels)
        sil = silhouette_score(X, labels) if k > 1 else float("nan")
        print(f"K={k}  inertia={inert:8.2f}  silhouette={sil:.3f}")

    print("-> Inertia keeps dropping as K grows (expected); silhouette peaks "
          "near the true K=3, which is the signal to actually use.\n")


def davies_bouldin_example():
    print("=" * 70)
    print("Example 2: Davies-Bouldin Index -- good vs. messy clustering")
    print("=" * 70)

    rng = np.random.default_rng(2)
    X = make_three_blobs(rng)
    good_labels = np.array([0] * 30 + [1] * 30 + [2] * 30)

    # deliberately messy labeling: shuffle 25% of the labels
    messy_labels = good_labels.copy()
    idx = rng.choice(len(messy_labels), size=len(messy_labels) // 4, replace=False)
    messy_labels[idx] = rng.integers(0, 3, size=len(idx))

    print("Good clustering  DBI:", davies_bouldin_index(X, good_labels))
    print("Messy clustering DBI:", davies_bouldin_index(X, messy_labels))
    print("-> Lower DBI = better; the messy assignment scores worse.\n")


def ari_nmi_example():
    print("=" * 70)
    print("Example 3: ARI / NMI against ground truth")
    print("=" * 70)

    rng = np.random.default_rng(4)
    true_labels = np.array([0] * 30 + [1] * 30 + [2] * 30)

    perfect_pred = true_labels.copy()
    # relabel perfect_pred with different label IDs but same grouping -- ARI/NMI
    # should still show a perfect match since only grouping matters, not the label values
    relabeled = np.where(perfect_pred == 0, 7, np.where(perfect_pred == 1, 3, 9))

    noisy_pred = true_labels.copy()
    idx = rng.choice(len(noisy_pred), size=15, replace=False)
    noisy_pred[idx] = rng.integers(0, 3, size=len(idx))

    for name, pred in [("relabeled (same grouping)", relabeled), ("noisy", noisy_pred)]:
        ari = adjusted_rand_index(true_labels, pred)
        nmi = normalized_mutual_info(true_labels, pred)
        print(f"{name:28s} ARI={ari:.3f}  NMI={nmi:.3f}")

    print("-> ARI/NMI care about grouping structure, not the actual label "
          "values, so a relabeling still scores as a perfect match.\n")


def full_report_example():
    print("=" * 70)
    print("Example 4: clustering_report() end to end")
    print("=" * 70)

    rng = np.random.default_rng(5)
    X = make_three_blobs(rng)
    labels_true = np.array([0] * 30 + [1] * 30 + [2] * 30)
    labels_pred = labels_true.copy()
    idx = rng.choice(len(labels_pred), size=10, replace=False)
    labels_pred[idx] = rng.integers(0, 3, size=len(idx))

    report = clustering_report(X, labels_pred, labels_true=labels_true)
    for k, v in report.items():
        print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")


if __name__ == "__main__":
    elbow_and_silhouette_example()
    davies_bouldin_example()
    ari_nmi_example()
    full_report_example()
