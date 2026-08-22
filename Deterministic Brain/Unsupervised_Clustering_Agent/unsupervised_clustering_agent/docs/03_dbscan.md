# DBSCAN — Density-Based Clustering

**Code:** `code/dbscan.py` — `DBSCANScratch`, `DBSCANSklearn`

## Core idea
K-Means and Hierarchical: a cluster is a group of *close* points.
DBSCAN: a cluster is a **dense region** separated from other dense
regions by sparse space. Like constellations of stars: dense clumps
are clusters, isolated stars are noise, empty space between them is
just... space.

## Three kinds of points
For a point p, its ε-neighborhood is every point within distance ε.
- **Core point**: has ≥ `min_pts` neighbors within ε (dense region) →
  becomes part of a cluster.
- **Border point**: fewer than `min_pts` neighbors, but within ε of a
  core point → joins that core point's cluster.
- **Noise point**: neither → labeled **-1**, not part of any cluster.

## Algorithm
For each unvisited point:
1. Count its ε-neighbors.
2. If < `min_pts` → mark noise (may later be reclaimed as a border point).
3. If ≥ `min_pts` (it's core) → start a new cluster, then expand: add
   every neighbor, and if that neighbor is *also* core, add its
   neighbors too — cluster grows through the dense region until it
   runs out of density.

## Why DBSCAN beats K-Means on irregular shapes
K-Means clusters are always convex/spherical — it cannot separate two
concentric rings, a crescent, or a spiral (centroids end up in the
wrong place). DBSCAN just follows density, so it correctly separates
an inner ring from an outer ring regardless of shape.

## Choosing eps and min_pts
- **min_pts** rule of thumb: ≥ dimensions + 1 (2D data → at least 3;
  common choice for higher-D is 2×dimensions). Larger min_pts → more
  conservative, more noise.
- **eps** is the critical, hard-to-choose parameter:
  - too small → everything is noise
  - too large → everything is one cluster
  - **k-distance plot method**: compute each point's distance to its
    k-th nearest neighbor (k = min_pts), sort descending, plot it, and
    look for the "knee" — that distance is a good eps. See
    `utils.k_distance_plot_values`.

## Advantages / limitations
✓ No need to specify K, finds arbitrary shapes, automatically flags
outliers as noise, robust to outliers, deterministic.
✗ eps/min_pts can be hard to tune, fails when clusters have very
different densities (one eps can't fit both), struggles in
high dimensions (distances become less meaningful), O(n²) without a
spatial index (KD-tree gets this to O(n log n) in low dimensions).

## When to use
✓ Unknown number of clusters, irregular shapes, need outlier detection.
✗ Clusters of very different densities, very high-dimensional data.
