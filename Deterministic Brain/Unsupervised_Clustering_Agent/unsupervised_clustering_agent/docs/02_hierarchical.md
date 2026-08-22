# Hierarchical (Agglomerative) Clustering

**Code:** `code/hierarchical.py` — `AgglomerativeScratch`, `AgglomerativeSklearn`

## Core idea
K-Means: you pick K, the algorithm finds clusters. Hierarchical: the
algorithm builds **every possible level** of clustering as a tree, and
you decide afterward where to "cut" it.

- **Agglomerative** (bottom-up, the common approach): start with every
  point as its own cluster, repeatedly merge the two closest clusters.
- **Divisive** (top-down): start with one cluster, repeatedly split —
  rarely used, not implemented here.

## Algorithm
1. Start with n clusters (one per point).
2. Find and merge the two closest clusters → n-1 clusters.
3. Recompute distances from the new merged cluster to all others.
4. Repeat until everything is in one cluster.
5. Every merge is recorded → this history is the **dendrogram**.

## Reading a dendrogram
The height of a merge = the distance at which it happened. Low merges
= very similar points; high merges = very different clusters.
Cutting the tree at a given height yields a specific number of
clusters — cut low for more clusters, high for fewer, with no need to
decide K in advance.

## Linkage methods (how cluster-to-cluster distance is measured)
| Linkage | Distance definition | Effect |
|---|---|---|
| Single | min distance between any two cross-cluster points | chains easily; good for elongated shapes / outlier detection |
| Complete | max distance between any two cross-cluster points | compact, similar-sized clusters; sensitive to outliers |
| Average | mean of all cross-cluster pairwise distances | balanced, general purpose |
| Ward | minimum increase in total WCSS from merging | compact, K-Means-like clusters; **sklearn default** |

## Hierarchical vs K-Means
| | K-Means | Hierarchical |
|---|---|---|
| Speed | Fast, O(nKt) | Slow, O(n²)–O(n³) |
| Need K upfront? | Yes | No — see the whole dendrogram first |
| Determinism | No (random init) | Yes, same result every run |
| Cluster shapes | Spherical only | Depends on linkage |
| Large datasets | Good | Struggles |

## When to use
✓ Small-to-medium datasets, want the full hierarchy / dendrogram, need
determinism, unsure how many clusters exist.
✗ Large datasets (too slow), need to undo a bad early merge (greedy —
cannot backtrack).
