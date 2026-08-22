# Clustering Metrics

code: `code/clustering_metrics.py`

Unsupervised learning has no true labels to check predictions against, so
evaluation splits into two families:

- **No ground truth (most common):** internal metrics based on the
  cluster structure itself — Silhouette, Davies-Bouldin, Inertia.
- **Ground truth available (evaluation/benchmark scenario):** external
  metrics that compare discovered clusters to true labels — ARI, NMI.

## Internal Metric — Silhouette Score

For each point `i`:
```
a(i) = mean distance to other points in the same cluster   (cohesion, want small)
b(i) = mean distance to points in the nearest *other* cluster (separation, want large)
s(i) = (b(i) - a(i)) / max(a(i), b(i))
```
Overall score = mean `s(i)` across all points.

| Value | Meaning |
|---|---|
| +1 | perfectly clustered |
| 0 | point sits on the boundary between two clusters |
| -1 | point is likely in the wrong cluster |

Use for choosing K (pick the K with the highest average silhouette) or
comparing clustering algorithms.

## Internal Metric — Davies-Bouldin Index (DBI)

For each pair of clusters `(i, j)`:
```
R_ij = (sigma_i + sigma_j) / d(mu_i, mu_j)
```
- `sigma_k` = average intra-cluster distance in cluster k (spread — want small)
- `d(mu_i, mu_j)` = distance between cluster centers (separation — want large)

```
DBI = (1/K) * sum_i max_{j != i}(R_ij)
```

**Lower is better.** The best clustering has compact clusters that are far
apart from each other.

## Internal Metric — Inertia (WCSS)

```
WCSS = sum_k sum_{i in cluster k} ||x_i - mu_k||^2
```

Sum of squared distances from each point to its own cluster's centroid.
Lower = tighter clusters, but **inertia always decreases as K increases**,
so it can't be compared across K directly — it's only meaningful via the
**Elbow Method** (plot inertia vs. K, look for the bend). Also only
well-defined for centroid-based algorithms like K-Means.

## External Metric — Adjusted Rand Index (ARI)

Consider every pair of samples. For each pair, check whether the clustering
and the true labels *agree* (both put the pair together, or both put the
pair apart) or *disagree*.

```
Rand Index = agreements / total pairs
```

The **Adjusted** Rand Index corrects for agreement expected purely by chance
(the same way Adjusted R² corrects for chance improvement from extra
features):

| ARI | Meaning |
|---|---|
| 1 | perfect match with true labels |
| 0 | no better than random labeling |
| < 0 | worse than random |

## External Metric — Normalized Mutual Information (NMI)

Measures how much information cluster assignments share with true labels,
using entropy:

```
MI(U, V)  = how much knowing the cluster tells you about the true label
NMI(U, V) = MI(U, V) / sqrt(H(U) * H(V))
```

| NMI | Meaning |
|---|---|
| 0 | clusters carry no information about true labels |
| 1 | perfect correspondence |

**Advantage over ARI:** handles cases where the number of clusters found
doesn't match the number of true classes more gracefully.

## Decision Guide

```
No ground truth:
├── Choosing K              → Elbow (Inertia) + Silhouette Score, pick highest
├── Comparing algorithms     → Silhouette Score (higher better) / Davies-Bouldin (lower better)
└── Visual sanity check      → 2D PCA / UMAP colored by cluster

Ground truth available:
├── ARI  → chance-corrected agreement
└── NMI  → information-theoretic, robust to differing cluster/class counts
    (report both when possible)
```

## Comparison Table

| Metric | Range | Better | Needs Ground Truth |
|---|---|---|---|
| Silhouette | -1 to 1 | higher | No |
| Davies-Bouldin | ≥ 0 | lower | No |
| Inertia (WCSS) | ≥ 0 | lower (only meaningful via Elbow) | No, K-Means only |
| Adjusted Rand Index | -1 to 1 | higher | Yes |
| NMI | 0 to 1 | higher | Yes |

## Function Reference

```python
silhouette_score(X, labels)
davies_bouldin_index(X, labels)
inertia(X, labels)
adjusted_rand_index(labels_true, labels_pred)
normalized_mutual_info(labels_true, labels_pred)
clustering_report(X, labels_pred, labels_true=None)
```
