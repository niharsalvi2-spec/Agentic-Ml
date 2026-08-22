# Supervised Regression Agent

A multi-skill agent for building, diagnosing, and tuning ML regression models —
covers the full Phase 5 Part A tree: **Linear Family**, **Tree-Based**, and
**Other Methods**.

## Skill Map

| File | Covers |
|---|---|
| `01_linear_family.md` | Linear Regression (OLS), Ridge, Lasso, Elastic Net + assumption diagnostics |
| `02_tree_based.md` | Decision Tree, Random Forest, Gradient Boosting, XGBoost, LightGBM, CatBoost |
| `03_other_methods.md` | SVR, KNN Regressor, Bayesian Regression |
| `04_evaluation_and_selection.md` | Loss/metrics, CV strategy, model comparison harness |

## How to use this agent

1. Every snippet assumes `X` (features, DataFrame/array), `y` (target) already loaded,
   and train/test already split via `X_train, X_test, y_train, y_test`.
2. Install once:

```bash
pip install scikit-learn xgboost lightgbm catboost statsmodels scipy \
            --break-system-packages
```

## Decision shortcut

```
Need INTERPRETABLE coefficients / small-data uncertainty?
 └── YES
       ├── Multicollinearity present         → Ridge
       ├── Want automatic feature selection   → Lasso
       ├── Correlated groups + selection      → Elastic Net
       ├── Need uncertainty / prediction interval → Bayesian Regression
       └── Plain, few features, linear signal → Linear Regression (OLS)
 └── NO, want best raw accuracy / nonlinear patterns?
       ├── Tabular, need max accuracy          → XGBoost / LightGBM / CatBoost
       ├── Need SOME interpretability + trees  → Random Forest
       ├── Small-medium data, smooth nonlinear  → SVR (RBF kernel)
       ├── Low-dim, local structure matters     → KNN Regressor
       └── Quick baseline, explainable splits   → Decision Tree
```

## Assumption Checklist (run before trusting a linear model)

```python
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy import stats

residuals = y_train - model.predict(X_train)

# 1. Linearity — scatter each feature vs target (visual)
# 2. Independence — residuals vs row index (no pattern)
plt.scatter(range(len(residuals)), residuals); plt.axhline(0, color='r')

# 3. Homoscedasticity — residuals vs fitted values (constant spread)
plt.scatter(model.predict(X_train), residuals); plt.axhline(0, color='r')

# 4. Normality — Q-Q plot + Shapiro-Wilk
sm.qqplot(residuals, line='45'); stat, p = stats.shapiro(residuals)

# 5. Multicollinearity — VIF (see 01_linear_family.md)
```
