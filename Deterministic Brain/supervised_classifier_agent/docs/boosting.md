# Boosting (Gradient Boosting / AdaBoost)

code: `code/boosting.py` — classes `SklearnBoosting` (Gradient Boosting), `ScratchBoosting` (AdaBoost)

## The Core Idea — Sequentially Fix Previous Mistakes

Unlike Random Forest (parallel, independent trees averaged together),
Boosting trains weak learners **sequentially**, where each new learner
focuses on the mistakes of the ensemble so far.

```
AdaBoost intuition:
  1. Train a weak learner (decision stump) on the data
  2. Increase the weight of misclassified samples
  3. Train the next weak learner on the reweighted data (it focuses on hard cases)
  4. Repeat; final prediction = weighted vote of all weak learners
     (learners with lower error get a bigger vote weight, "alpha")

Gradient Boosting intuition:
  1. Start with a simple prediction (e.g. log-odds of the base rate)
  2. Compute the residual error of current predictions
  3. Train a new (shallow) tree to predict that residual/gradient
  4. Add the new tree's predictions (scaled by learning_rate) to the ensemble
  5. Repeat for n_estimators rounds
```

## The Math — Step by Step (AdaBoost, binary {-1,+1} labels)

**Step 1: Initialize** equal sample weights `w_i = 1/n`.

**Step 2: Fit a weak learner** (decision stump) that minimizes weighted
classification error.

**Step 3: Compute the learner's vote weight:**
```
alpha = 0.5 * ln( (1 - error) / error )
```
Lower error → higher alpha → this learner's opinion counts more.

**Step 4: Reweight samples** — misclassified samples get their weight
multiplied by `exp(alpha)`, correctly classified get `exp(-alpha)`, then
renormalize so weights sum to 1.

**Step 5: Repeat** for `n_estimators` rounds.

**Step 6: Final prediction** = sign of `sum(alpha_t * stump_t(x))`.

## Key Hyperparameters

| Param | Effect |
|---|---|
| `n_estimators` | Number of boosting stages/weak learners |
| `learning_rate` | Shrinks each stage's contribution — trade off against n_estimators |
| `max_depth` (GBM) | Usually shallow (3-5) trees as weak learners |
| `subsample` (GBM) | <1.0 adds stochastic boosting, reduces overfitting |

## When to Use

✓ Need the highest possible accuracy on tabular data (often beats Random Forest)
✓ Willing to tune hyperparameters carefully
✓ Have time budget for sequential (less parallelizable) training

## When NOT to Use

✗ Very noisy data / many outliers — boosting can overfit to hard/mislabeled examples
✗ Need fast training on very large data (prefer histogram-based implementations
  like LightGBM/XGBoost/HistGradientBoosting — not included in this package)
✗ Need strong interpretability

## Agent Metadata Summary

```python
SklearnBoosting.METADATA
# family: ensemble-boosting | interpretable: False | sensitive_to_outliers: True
# training_speed: slow | good_for_large_data: False
```
