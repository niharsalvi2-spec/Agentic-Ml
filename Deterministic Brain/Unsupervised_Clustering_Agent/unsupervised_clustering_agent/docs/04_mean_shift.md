# Mean Shift Clustering

**Code:** `code/mean_shift.py` — `MeanShiftScratch`, `MeanShiftSklearn`

## Core idea
Treat the data as a landscape where height = point density. Every
point "rolls uphill" toward the nearest peak. Points that converge to
the same peak form a cluster. The **number of clusters is discovered
automatically** — it's just the number of density peaks found.

## Algorithm
For each point:
1. Draw a circle of radius `bandwidth` (h) around it.
2. Compute the mean of all points inside the circle.
3. Move the point to that mean (shift toward higher density).
4. Repeat until movement is below a threshold (converged).
5. Points that converge to (nearly) the same location → same cluster.

## The math — kernel density estimation
Mean Shift is gradient ascent on a kernel density estimate:
```
f(x) = (1/nh^d) * Σ K((x - x_i)/h)
shift vector m(x) = weighted_mean(neighbors) - x
```
Moving by the shift vector each iteration moves the point toward the
nearest density peak.

## Bandwidth — the critical parameter
- **Small bandwidth**: small neighborhoods → many small local peaks →
  risk of over-segmenting into too many clusters.
- **Large bandwidth**: large neighborhoods → few broad peaks → risk of
  under-segmenting into too few clusters.
- `estimate_bandwidth()` (in both `code/mean_shift.py` and sklearn)
  gives a reasonable heuristic starting point based on data scale.

## Mean Shift vs K-Means
| | K-Means | Mean Shift |
|---|---|---|
| Cluster shape | Spherical | Arbitrary density-based |
| Need K? | Yes | No — found automatically |
| Speed | Fast | Slow, O(n²) |

## When to use
✓ Unknown number of clusters, clusters of different sizes/densities,
small-to-medium data (<10k points), need smooth boundaries.
✗ Large datasets (too slow), high-dimensional data (bandwidth becomes
less meaningful), need very fast clustering.
