# K-Nearest Neighbors (KNN)

code: `code/knn.py` — classes `SklearnKNN`, `ScratchKNN`

## The Core Idea — Vote of the Nearest Points

KNN makes no assumptions about the data distribution. To classify a new
point, it finds the `k` closest training points (by some distance metric) and
lets them vote on the label.

```
predict(x) = majority_class( k nearest neighbors of x in training data )
```

There is no real "training" phase beyond storing the data (a "lazy learner") —
all the work happens at prediction time.

## The Math — Step by Step

**Step 1: Choose a distance metric.**
```
Euclidean:  d(a,b) = sqrt(sum((a_i - b_i)^2))
Manhattan:  d(a,b) = sum(|a_i - b_i|)
```

**Step 2: For a query point, compute distance to every training point.**

**Step 3: Select the k smallest distances (the neighbors).**

**Step 4: Vote.**
```
uniform weighting:  each neighbor gets 1 vote
distance weighting: each neighbor gets 1/distance votes (closer = louder vote)
```

**Step 5: Predict the class with the most (weighted) votes.**

## Key Hyperparameters

| Param | Effect |
|---|---|
| `n_neighbors` (k) | Small k → low bias/high variance (overfits to noise). Large k → high bias/smoother boundary. |
| `weights` | `'uniform'` or `'distance'` |
| `metric` | `'minkowski'` (p=2 euclidean, p=1 manhattan), `'cosine'` for text/embeddings |

⚠️ **Must scale features first** — KNN is purely distance-based, so any
unscaled large-range feature will dominate the distance calculation.

## When to Use

✓ Nonlinear decision boundaries, no distributional assumptions
✓ Small-to-medium datasets
✓ Low-to-moderate dimensional feature space
✓ Local structure matters (similar things behave similarly)

## When NOT to Use

✗ Large datasets — inference is O(n) per query without an index structure
✗ High-dimensional data — distances become less meaningful ("curse of dimensionality")
✗ Real-time predictions at scale

## Agent Metadata Summary

```python
SklearnKNN.METADATA
# family: instance-based | good_for_large_data: False | inference_speed: slow
# sensitive_to_scaling: True | good_for_high_dim: False
```
