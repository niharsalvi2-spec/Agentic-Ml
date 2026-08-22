# Skill: Regression — Linear Family
OLS, Ridge, Lasso, Elastic Net — assumes `X_train, X_test, y_train, y_test` exist.

## Linear Regression (OLS)

```python
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)

print("Coefficients:", dict(zip(X_train.columns, lr.coef_)))
print("Intercept:", lr.intercept_)
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
print("R2:", r2_score(y_test, y_pred))

# Closed-form normal equation (manual, for understanding)
X_design = np.c_[np.ones(len(X_train)), X_train.values]  # add bias column
w = np.linalg.inv(X_design.T @ X_design) @ X_design.T @ y_train
```

### Statsmodels version (for p-values, confidence intervals, full diagnostics)

```python
import statsmodels.api as sm

X_sm = sm.add_constant(X_train)
ols_model = sm.OLS(y_train, X_sm).fit()
print(ols_model.summary())   # coefficients, p-values, R2, F-stat, condition number
```

### Multicollinearity — VIF

```python
from statsmodels.stats.outliers_influence import variance_inflation_factor
import pandas as pd

vif_df = pd.DataFrame()
vif_df["feature"] = X_train.columns
vif_df["VIF"] = [variance_inflation_factor(X_train.values, i) for i in range(X_train.shape[1])]
# VIF > 10 -> severe multicollinearity -> use Ridge or drop features
```

### Gradient Descent (manual, for large n where normal equation O(p^3) is slow)

```python
def gradient_descent_ols(X, y, lr=0.01, n_iter=1000):
    n, p = X.shape
    X_b = np.c_[np.ones(n), X]
    w = np.zeros(p + 1)
    for _ in range(n_iter):
        grad = -(2/n) * X_b.T @ (y - X_b @ w)
        w -= lr * grad
    return w
```

---

## Ridge Regression (L2)

```python
from sklearn.linear_model import Ridge, RidgeCV

ridge = Ridge(alpha=1.0)   # alpha = lambda, higher = more shrinkage
ridge.fit(X_train, y_train)

# Auto-tune alpha via built-in leave-one-out / k-fold CV
ridge_cv = RidgeCV(alphas=np.logspace(-3, 3, 50), cv=5)
ridge_cv.fit(X_train, y_train)
print("Best alpha:", ridge_cv.alpha_)
```

## Lasso Regression (L1)

```python
from sklearn.linear_model import Lasso, LassoCV

lasso = Lasso(alpha=0.01)
lasso.fit(X_train, y_train)
selected_features = X_train.columns[lasso.coef_ != 0]   # sparsity -> feature selection

# Auto-tune alpha
lasso_cv = LassoCV(alphas=np.logspace(-4, 1, 50), cv=5, max_iter=10000)
lasso_cv.fit(X_train, y_train)
print("Best alpha:", lasso_cv.alpha_)

# Regularization path — see which features survive as alpha increases
from sklearn.linear_model import lasso_path
alphas, coefs, _ = lasso_path(X_train, y_train)
import matplotlib.pyplot as plt
plt.plot(alphas, coefs.T); plt.xscale('log')
plt.xlabel("alpha"); plt.ylabel("coefficient value")
```

## Elastic Net

```python
from sklearn.linear_model import ElasticNet, ElasticNetCV

enet = ElasticNet(alpha=0.01, l1_ratio=0.5)   # l1_ratio=1 -> Lasso, 0 -> Ridge
enet.fit(X_train, y_train)

enet_cv = ElasticNetCV(alphas=np.logspace(-4, 1, 50),
                        l1_ratio=[.1, .3, .5, .7, .9, .95, 1], cv=5)
enet_cv.fit(X_train, y_train)
print("Best alpha:", enet_cv.alpha_, "Best l1_ratio:", enet_cv.l1_ratio_)
```

## Lasso vs Ridge vs Elastic Net — quick picker

```python
# Rule of thumb, verify with CV:
# - Multicollinearity, keep all features       -> Ridge
# - Want automatic feature elimination          -> Lasso
# - Correlated feature GROUPS + want selection  -> Elastic Net (keeps group together)
```
