# Skill: Regression — Evaluation & Model Selection

## Loss Functions / Metrics

```python
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    mean_absolute_percentage_error, median_absolute_error
)
import numpy as np

def regression_report(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return {"RMSE": rmse, "MAE": mae, "MAPE": mape, "R2": r2}

# Huber loss (robust to outliers, combines MSE + MAE)
from sklearn.linear_model import HuberRegressor
huber = HuberRegressor(epsilon=1.35).fit(X_train, y_train)
```

| Metric | Sensitive to outliers | Units | Use when |
|---|---|---|---|
| MSE / RMSE | Yes (squared) | same as y (RMSE) | Standard choice, differentiable |
| MAE | No | same as y | Outliers present, want robust metric |
| MAPE | No (but blows up near y=0) | % | Business-facing, relative error matters |
| R² | N/A | unitless (0-1) | Explain variance captured |
| Huber | Partial | same as y | Want MSE benefits without outlier sensitivity |

## Cross-Validation Strategy

```python
from sklearn.model_selection import KFold, cross_validate

cv = KFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_validate(model, X, y, cv=cv,
                         scoring=['neg_root_mean_squared_error', 'r2'])
print("RMSE:", -scores['test_neg_root_mean_squared_error'].mean())
print("R2:", scores['test_r2'].mean())

# For time series -> NEVER shuffle, use TimeSeriesSplit
from sklearn.model_selection import TimeSeriesSplit
tscv = TimeSeriesSplit(n_splits=5)
```

## Model Comparison Harness

```python
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, BayesianRidge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import xgboost as xgb, lightgbm as lgb
from catboost import CatBoostRegressor

models = {
    "Linear":     LinearRegression(),
    "Ridge":      Ridge(alpha=1.0),
    "Lasso":      Lasso(alpha=0.01),
    "ElasticNet": ElasticNet(alpha=0.01, l1_ratio=0.5),
    "Bayesian":   BayesianRidge(),
    "DecisionTree": DecisionTreeRegressor(max_depth=6, random_state=42),
    "RandomForest": RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1),
    "GradBoost":  GradientBoostingRegressor(random_state=42),
    "XGBoost":    xgb.XGBRegressor(random_state=42),
    "LightGBM":   lgb.LGBMRegressor(random_state=42),
    "CatBoost":   CatBoostRegressor(verbose=0, random_state=42),
    "SVR":        make_pipeline(StandardScaler(), SVR(kernel='rbf')),
    "KNN":        make_pipeline(StandardScaler(), KNeighborsRegressor(n_neighbors=5)),
}

results = []
for name, m in models.items():
    m.fit(X_train, y_train)
    y_pred = m.predict(X_test)
    metrics = regression_report(y_test, y_pred)
    metrics["Model"] = name
    results.append(metrics)

leaderboard = pd.DataFrame(results).set_index("Model").sort_values("RMSE")
print(leaderboard)
```

## Bias-Variance Diagnostic (learning curve)

```python
from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt

train_sizes, train_scores, val_scores = learning_curve(
    model, X, y, cv=5, scoring='neg_root_mean_squared_error',
    train_sizes=np.linspace(0.1, 1.0, 10))

plt.plot(train_sizes, -train_scores.mean(axis=1), label='Train RMSE')
plt.plot(train_sizes, -val_scores.mean(axis=1), label='Validation RMSE')
plt.legend(); plt.xlabel("Training set size"); plt.ylabel("RMSE")
# Gap between curves = variance (overfitting); both high & converged = bias (underfitting)
```
