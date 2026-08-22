# Support Vector Machine (SVM)

code: `code/svm.py` — classes `SklearnSVM`, `ScratchSVM` (linear only)

## The Core Idea — Maximum Margin Separator

SVM finds the decision boundary (hyperplane) that maximizes the margin — the
distance between the boundary and the nearest points of each class (the
"support vectors"). A wider margin generally generalizes better.

```
decision boundary: w·x + b = 0
margin boundaries: w·x + b = +1  and  w·x + b = -1
margin width = 2 / ||w||   → minimizing ||w|| maximizes the margin
```

For data that isn't linearly separable, the **kernel trick** implicitly maps
data into a higher-dimensional space where a linear separator DOES exist,
without ever computing the transformation explicitly.

## The Math — Step by Step (linear, soft-margin)

**Step 1: Objective** — maximize margin while allowing some misclassification
(soft margin), controlled by `C`:
```
minimize:  (1/2)||w||^2  +  C * sum( hinge_loss(y_i, w·x_i + b) )
where hinge_loss = max(0, 1 - y_i*(w·x_i + b))     (y_i in {-1, +1})
```
- `(1/2)||w||^2` term → wants a wide margin
- hinge loss term → penalizes points on the wrong side / inside the margin
- `C` controls the trade-off: large C = fit training data tightly (risk
  overfitting), small C = prioritize wide margin (more tolerant of errors)

**Step 2: Optimize** via (sub)gradient descent or quadratic programming —
only points *inside the margin or misclassified* contribute gradient (the
"support vectors"); correctly classified points outside the margin contribute
nothing.

**Step 3 (kernel version, sklearn only):** Replace the dot product `x_i·x_j`
with a kernel function `K(x_i, x_j)` (e.g. RBF: `exp(-gamma*||x_i-x_j||^2)`)
to implicitly work in a higher-dimensional, possibly infinite-dimensional,
feature space — enabling nonlinear boundaries without explicit transformation.

## Key Hyperparameters

| Param | Effect |
|---|---|
| `C` | Regularization strength — small C = wide/tolerant margin, large C = tight fit |
| `kernel` | `'linear'`, `'rbf'` (default, general-purpose), `'poly'`, `'sigmoid'` |
| `gamma` | Kernel coefficient for rbf/poly — controls influence radius of one sample |

⚠️ **Must scale features first** — SVM is margin/distance-based.

## When to Use

✓ Clear (or kernel-inducible) margin of separation between classes
✓ High-dimensional data with fewer samples than features (text, bio data)
✓ Robust classifier resistant to overfitting when margin is clear

## When NOT to Use

✗ Very large datasets — kernel SVM training scales roughly O(n^2) to O(n^3)
✗ Need cheap probability estimates (requires extra Platt scaling / CV in sklearn)
✗ Noisy data with heavily overlapping classes — sensitive to C and near-margin outliers

## Agent Metadata Summary

```python
SklearnSVM.METADATA
# family: margin-based | good_for_high_dim: True | good_for_large_data: False
# sensitive_to_scaling: True | training_speed: slow
```

Note: `ScratchSVM` implements the **linear** case only (no kernel trick) —
use `SklearnSVM(kernel='rbf')` for nonlinear boundaries.
