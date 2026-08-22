# K-Means Clustering

**Code:** `code/kmeans.py` — `KMeansScratch`, `KMeansSklearn`

## Core idea
"I believe there are K groups. Find K centers such that each point is
closest to its own group's center." Like placing K school buses so
every student rides the nearest one, then moving buses to the center
of their assigned students, repeating until buses stop moving.

## Algorithm
1. **Initialize**: place K centroids (K-Means++, see below).
2. **Assign**: each point joins its nearest centroid.
3. **Update**: move each centroid to the mean of its assigned points.
4. **Repeat** steps 2–3 until centroids stop moving (or max iterations).

## Objective: WCSS / Inertia
```
WCSS = Σ_k Σ_{i in cluster k} ||x_i - μ_k||²
```
Every iteration can only decrease WCSS, so K-Means always converges —
but only to a **local** minimum, not necessarily the global one.

## The initialization problem → K-Means++
Pure random initialization can put two centroids inside the same true
cluster, splitting a different cluster in half and settling into a bad
local minimum. **K-Means++** fixes this:
1. Pick the first centroid randomly.
2. For each remaining centroid, pick a point with probability
   proportional to its squared distance to the nearest existing
   centroid — farther points are more likely to be chosen.
This spreads centroids out from the start. It's sklearn's default and
what `KMeansScratch` implements.

## Choosing K
### Elbow method
Plot K vs WCSS for K = 1..10. WCSS always decreases as K grows; look
for the point where the curve bends sharply ("the elbow") — beyond
that, adding clusters gives diminishing returns.

### Silhouette score (more reliable)
For point i:
```
a(i) = avg distance to points in the SAME cluster   (want small)
b(i) = avg distance to points in the NEAREST other cluster (want large)
s(i) = (b(i) - a(i)) / max(a(i), b(i))         range: -1 .. +1
```
s(i) near +1: well clustered. Near 0: on a boundary. Negative: probably
in the wrong cluster. Choose the K that **maximizes** the average
silhouette score. See `utils.silhouette_score` and `utils.elbow_curve`.

## Limitations
- Assumes **spherical** clusters (Euclidean distance) — fails on
  elongated, crescent, or ring shapes (see DBSCAN).
- Sensitive to **outliers** — the mean gets pulled toward them.
- Must **specify K** upfront.
- Sensitive to **feature scale** — always `StandardScaler` first.
- Local minimum — run multiple times (`n_init`) and keep the best WCSS.

## When to use
✓ Roughly spherical, similarly-sized clusters, large datasets, need speed.
✗ Unknown/irregular cluster shapes, heavy outliers, unclear K.
