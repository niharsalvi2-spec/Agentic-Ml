# Clustering Model Selection Guide

Quick decision tree for picking a clustering algorithm. Point your
agent here first before writing new clustering code — it may already
exist in `code/`.

```
Do you know how many clusters (K) you expect?
├── Yes, roughly                     → K-Means
└── No / unknown
    ├── Want the full hierarchy too  → Hierarchical (Agglomerative)
    ├── Clusters are irregular shape
    │   or you need outlier detection → DBSCAN
    └── Clusters have very different
        sizes/densities, dataset small → Mean Shift

Cluster shape?
├── Roughly spherical, similar sizes → K-Means (or Ward-linkage Hierarchical)
├── Arbitrary shape (rings, crescents,
│   spirals), separated by sparse space → DBSCAN
└── Density peaks of varying shape/size → Mean Shift

Need outlier / noise detection built in?
└── DBSCAN (explicitly labels noise as -1); others force every point
    into some cluster.

Dataset size?
├── Large (100k+ points)              → K-Means (fastest) or DBSCAN w/ KD-tree
├── Medium                            → any of the four
└── Small (<10k)                      → Hierarchical or Mean Shift are viable

Need determinism (same result every run)?
├── Yes                               → Hierarchical or DBSCAN
└── K-Means/Mean Shift are usually stable in practice but not
    guaranteed deterministic (random init / point ordering)

Need to pick K objectively?
├── Elbow method (WCSS vs K)          → utils.elbow_curve
└── Silhouette score (more reliable)  → utils.silhouette_score
```

## Comparison table

| Model | Needs K upfront | Cluster shape | Handles noise/outliers | Deterministic | Speed |
|---|---|---|---|---|---|
| K-Means | Yes | Spherical | No (sensitive to outliers) | No (random init) | Fast, O(nKt) |
| Hierarchical | No (cut dendrogram) | Depends on linkage | No | Yes | Slow, O(n²)-O(n³) |
| DBSCAN | No | Arbitrary | Yes (explicit -1 label) | Yes | O(n²), O(n log n) w/ KD-tree |
| Mean Shift | No (found automatically) | Arbitrary (density-based) | Partial | Mostly | Slow, O(n²) |

## Repo map for agents
```
code/utils.py           StandardScaler, pairwise_distances, inertia,
                         elbow_curve, silhouette_score, k_distance_plot_values
code/kmeans.py           KMeansScratch, KMeansSklearn
code/hierarchical.py     AgglomerativeScratch, AgglomerativeSklearn
code/dbscan.py           DBSCANScratch, DBSCANSklearn
code/mean_shift.py       MeanShiftScratch, MeanShiftSklearn
docs/01-05_*.md          theory reference, one file per model + this guide
examples/example_*.py    runnable end-to-end usage per model
```
Every "Scratch" class and every "Sklearn"-wrapper class in this repo
share a `fit(X)` / `fit_predict(X)` API and expose `.labels_`. Swap one
for another without changing calling code.

## Preprocessing checklist
- **Always scale features** (`utils.StandardScaler`) before any of
  these — all four algorithms are distance-based.
- K-Means & Hierarchical (Ward): sensitive to outliers — consider
  removing extreme points first, or use DBSCAN instead which handles
  them natively.
- DBSCAN & Mean Shift: sensitive to high dimensionality — consider
  dimensionality reduction (PCA) first if you have many features.
