# Logistic Regression

code: `code/logistic_regression.py` — classes `SklearnLogisticRegression`, `ScratchLogisticRegression`

## The Core Idea — Linear Boundary + Probability

Logistic Regression fits a linear decision boundary, then squashes the raw
linear score through a sigmoid to output a probability between 0 and 1.

```
z = w·x + b                      (linear combination, same as linear regression)
p = sigmoid(z) = 1 / (1 + e^-z)  (squash to [0,1])
predict 1 if p >= 0.5 else 0
```

It is NOT a regression model despite the name — it's a classifier that happens
to use a linear model internally.

## The Math — Step by Step

**Step 1: Linear score.** `z = w1*x1 + w2*x2 + ... + wn*xn + b`

**Step 2: Sigmoid.** Maps any real number to (0, 1):
`sigmoid(z) = 1 / (1 + exp(-z))`

**Step 3: Loss — Binary Cross-Entropy.**
```
Loss = -[y*log(p) + (1-y)*log(1-p)]     (per sample, then averaged)
```
Penalizes confident wrong predictions heavily.

**Step 4: Optimization.** Gradient descent updates weights to minimize loss:
```
w := w - lr * dLoss/dw
b := b - lr * dLoss/db
```
No closed-form solution (unlike linear regression) — always iterative.

**Step 5: Regularization.** L2 penalty (`+ lambda*sum(w^2)`) shrinks weights
toward zero, reducing overfitting; L1 penalty can zero out weights entirely
(automatic feature selection).

## Key Hyperparameters

| Param | Effect |
|---|---|
| `C` (sklearn) | Inverse regularization strength. Small C = strong regularization. |
| `penalty` | `l2` (default), `l1` (sparse), `elasticnet` (mix) |
| `solver` | `lbfgs` (small/medium data), `saga` (large data, supports l1) |
| `class_weight` | `'balanced'` reweights loss for imbalanced classes |
| `lr`, `n_iters` (scratch) | Learning rate and gradient descent iterations |

## When to Use

✓ Baseline for any binary/multiclass task
✓ Need interpretable coefficients (odds ratios via `exp(weight)`)
✓ Data is linearly separable or close to it after feature engineering
✓ Need fast train + fast inference
✓ High-dimensional sparse data (e.g. TF-IDF text features)

## When NOT to Use

✗ Strongly nonlinear decision boundary with no feature engineering
✗ Severe multicollinearity without regularization
✗ Need to capture complex feature interactions automatically

## Agent Metadata Summary

```python
SklearnLogisticRegression.METADATA
# family: linear | interpretable: True | handles_nonlinear: False
# training_speed: fast | inference_speed: fast | good_for_high_dim: True
```
