# Random Forest

code: `code/random_forest.py` — classes `SklearnRandomForest`, `ScratchRandomForest`

## The Core Idea — Bagging of Decorrelated Trees

A single decision tree overfits easily (high variance). Random Forest trains
many trees on randomized versions of the data and averages their votes —
individual tree errors tend to cancel out.

```
for each of n_estimators trees:
    1. Draw a bootstrap sample (sample with replacement) from training data
    2. Grow a decision tree, but at each split only consider a random subset
       of features (this decorrelates the trees from each other)
predict(x) = majority vote (classification) across all trees
```

Randomizing both the **rows** (bootstrap) and the **columns** (feature
subsampling per split) is what makes the forest more powerful than simply
averaging trees trained on the same data.

## The Math — Step by Step

**Step 1: Bootstrap sampling.** For n_samples rows, draw n_samples rows
*with replacement* — roughly 63% of unique rows appear per tree, the rest are
duplicates; ~37% are left out entirely ("out-of-bag" samples).

**Step 2: Feature subsampling per split.** Typically `sqrt(n_features)` for
classification — each split only "sees" a random subset of features, so
strong features don't dominate every tree identically.

**Step 3: Grow each tree** using the same Gini-impurity splitting as a single
Decision Tree (see `decision_tree.md`).

**Step 4: Aggregate.** For classification: majority vote (or average of
predicted class probabilities) across all trees.

**Step 5 (bonus): Feature importance.** Measured by how much each feature
reduces impurity on average, across all trees and all splits using it.

## Key Hyperparameters

| Param | Effect |
|---|---|
| `n_estimators` | Number of trees. More = more stable, diminishing returns, slower. |
| `max_depth` / `min_samples_leaf` | Per-tree complexity control (same as single tree) |
| `max_features` | Features considered per split — `'sqrt'` typical for classification |
| `bootstrap` | Whether to sample with replacement (True = classic Random Forest) |

## When to Use

✓ General-purpose, robust tabular classifier — strong default choice
✓ Nonlinear relationships and feature interactions
✓ Want feature importances "for free"
✓ Robust to outliers and unscaled features, minimal tuning required

## When NOT to Use

✗ Very high-dimensional sparse data (text) — linear models often win
✗ Tight inference-latency or model-size constraints (many trees to store/run)
✗ Maximal interpretability required (forest is a black box vs a single tree)

## Agent Metadata Summary

```python
SklearnRandomForest.METADATA
# family: ensemble-bagging | interpretable: False | sensitive_to_outliers: False
# training_speed: medium | good_for_large_data: True
```
