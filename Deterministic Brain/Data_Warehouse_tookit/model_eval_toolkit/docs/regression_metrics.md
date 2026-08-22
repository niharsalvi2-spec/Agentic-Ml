# Regression Metrics

code: `code/regression_metrics.py`

## MAE — Mean Absolute Error

```
MAE = (1/n) * sum(|y_true - y_pred|)
```

Average of absolute errors, in the same units as the target (e.g. "off by
₹8.75 lakhs on average"). **Robust to outliers**: an error of 100 contributes
exactly 10x an error of 10 (linear relationship), so one bad prediction
doesn't dominate the score. Downside: `|error|` isn't differentiable at 0,
which is less convenient for gradient-based optimization than MSE.

## MSE — Mean Squared Error

```
MSE = (1/n) * sum((y_true - y_pred)^2)
```

Squaring changes everything: an error of 20 contributes 4x an error of 10
(not 2x). MSE punishes large errors much more than small ones, which is a
strength when large errors are genuinely worse (model learns to avoid
catastrophic predictions) and a weakness when outliers are just noisy data
(they unfairly dominate training/evaluation). Units are squared, so it's not
directly interpretable — mostly used internally or converted to RMSE.

## RMSE — Root Mean Squared Error

```
RMSE = sqrt(MSE)
```

Same units as the target, so it's interpretable like MAE, but still
penalizes large errors more heavily. **RMSE is always >= MAE.**

| Pattern | Meaning |
|---|---|
| RMSE ≈ MAE | Errors are consistent across all predictions — no extreme outliers |
| RMSE >> MAE | Some predictions have very large errors — investigate them |

Example: errors `[1,1,1,1,21]` → MAE = 5, RMSE = 9.43. The single 21-sized
error drags RMSE far above MAE even though 4 of 5 predictions were fine.

## R² — Coefficient of Determination

```
R2 = 1 - SS_residual / SS_total
SS_residual = sum((y_true - y_pred)^2)     # error the model has
SS_total    = sum((y_true - mean(y_true))^2)  # error of just predicting the mean
```

"What fraction of variance in the target did the model explain, relative to
a baseline that always predicts the mean?"

| R² | Interpretation |
|---|---|
| 1.0 | perfect predictions |
| 0.8 | explains 80% of variance — good |
| 0.5 | explains 50% — mediocre |
| 0.0 | no better than predicting the mean |
| < 0 | *worse* than predicting the mean |

**Caveats:**
- Sensitive to outliers in `y` — one extreme value inflates `SS_total` and
  can make R² look artificially good.
- Never decreases when you add features, even useless/random ones — a model
  can look better purely from overfitting. Use **Adjusted R²** to compare
  models with different feature counts.

## Adjusted R²

```
Adjusted R2 = 1 - (1 - R2) * (n - 1) / (n - p - 1)
```
where `n` = number of samples, `p` = number of features.

Adding a genuinely useful feature increases R² enough to outweigh the growing
penalty term, so Adjusted R² increases too. Adding a useless feature barely
moves R², so the penalty wins and Adjusted R² can *decrease* — correctly
signalling the extra feature didn't help. **Always prefer Adjusted R² over
plain R² when comparing models with different numbers of features.**

## Comparison Table

| Metric | Units | Outlier Sensitive | Use When |
|---|---|---|---|
| MAE | same as target | No (robust) | Outliers exist, want a directly interpretable typical error |
| MSE | squared units | Yes (heavily) | Large errors must be penalized; smooth for optimization |
| RMSE | same as target | Yes | Standard metric, penalizes large errors, still interpretable |
| R² | unitless (0–1, can be negative) | Yes | Explain overall fit quality as a percentage |
| Adjusted R² | unitless (0–1) | Yes | Comparing models with different numbers of features |

**Recommended default reporting:** MAE + RMSE + R² together — MAE gives the
typical error, the RMSE/MAE ratio flags outlier predictions, and R² gives an
overall fit-quality summary.

## Function Reference

```python
mean_absolute_error(y_true, y_pred)
mean_squared_error(y_true, y_pred)
root_mean_squared_error(y_true, y_pred)
r2_score(y_true, y_pred)
adjusted_r2_score(y_true, y_pred, n_features)
regression_report(y_true, y_pred, n_features=None)
```
