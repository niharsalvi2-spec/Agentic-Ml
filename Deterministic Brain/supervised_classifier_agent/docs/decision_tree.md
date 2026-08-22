# Decision Tree

code: `code/decision_tree.py` — classes `SklearnDecisionTree`, `ScratchDecisionTree`

## The Core Idea — Recursive If/Then Splits

A decision tree recursively splits the feature space into regions, each split
chosen to make the resulting groups as "pure" (single-class) as possible.

```
if feature_j <= threshold:
    go left
else:
    go right
... repeat recursively until stopping condition ...
leaf = majority class (or class distribution) of samples that land there
```

## The Math — Step by Step

**Step 1: Measure impurity of a node.** Gini impurity (default in sklearn/scratch):
```
Gini = 1 - sum(p_c^2)     for each class c, p_c = fraction of that class in the node
```
Gini = 0 → node is pure (single class). Higher Gini → more mixed.

Alternative: **entropy** (information gain): `-sum(p_c * log2(p_c))`

**Step 2: For every feature and every candidate threshold, compute the
weighted impurity of the two child nodes that split would create.**

**Step 3: Information gain** = parent impurity − weighted child impurity.
Pick the (feature, threshold) pair with the highest gain.

**Step 4: Recurse** on each child node until a stopping condition is hit
(`max_depth`, `min_samples_split`, or the node is already pure).

**Step 5: Leaf prediction** = class distribution of training samples that
ended up in that leaf.

## Key Hyperparameters

| Param | Effect |
|---|---|
| `max_depth` | Controls overfitting — shallow = more bias, less variance |
| `min_samples_split` / `min_samples_leaf` | Minimum samples required to split / to be a leaf |
| `criterion` | `'gini'` (fast) or `'entropy'` (information gain) |
| `max_features` | Number of features considered at each split |

## When to Use

✓ Need a fully interpretable, visualizable model (if-then rules)
✓ Mixed feature types, nonlinear relationships, feature interactions
✓ No need to scale features
✓ As a building block for ensembles (Random Forest, Boosting)

## When NOT to Use

✗ Alone, when accuracy is critical — single trees overfit easily (high variance)
✗ Smooth linear relationships (trees approximate with step functions, inefficient)

## Agent Metadata Summary

```python
SklearnDecisionTree.METADATA
# family: tree | interpretable: True | sensitive_to_scaling: False
# handles_nonlinear: True | training_speed: fast
```
