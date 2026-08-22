# Skill: Regression — Other Methods
SVR, KNN Regressor, Bayesian Regression.

## Support Vector Regression (SVR)

```python
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# SVR is scale-sensitive -> always standardize first
svr_pipeline = make_pipeline(
    StandardScaler(),
    SVR(kernel='rbf', C=10, epsilon=0.1, gamma='scale')
)
svr_pipeline.fit(X_train, y_train)
y_pred = svr_pipeline.predict(X_test)

# Hyperparameter search
from sklearn.model_selection import GridSearchCV
param_grid = {
    'svr__C': [0.1, 1, 10, 100],
    'svr__epsilon': [0.01, 0.1, 0.5],
    'svr__gamma': ['scale', 0.001, 0.01, 0.1],
}
grid = GridSearchCV(svr_pipeline, param_grid, cv=5, scoring='neg_mean_squared_error')
grid.fit(X_train, y_train)
print("Best params:", grid.best_params_)

# Kernel choices: 'linear', 'rbf' (default, nonlinear), 'poly', 'sigmoid'
# C: high = fit tightly (low bias, high variance) | low = more slack (high bias, low variance)
# epsilon: tube width - larger = simpler model, fewer support vectors
```

## KNN Regressor

```python
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# KNN is distance-based -> always standardize first
knn_pipeline = make_pipeline(
    StandardScaler(),
    KNeighborsRegressor(n_neighbors=5, weights='distance', metric='minkowski', p=2)
)
knn_pipeline.fit(X_train, y_train)
y_pred = knn_pipeline.predict(X_test)

# weights: 'uniform' (simple average) vs 'distance' (closer neighbors weigh more)
# metric/p: p=1 Manhattan, p=2 Euclidean

# Choosing K via cross-validation
from sklearn.model_selection import cross_val_score
import numpy as np
k_range = range(1, 31)
cv_scores = [cross_val_score(
    make_pipeline(StandardScaler(), KNeighborsRegressor(n_neighbors=k)),
    X_train, y_train, cv=5, scoring='neg_mean_squared_error').mean()
    for k in k_range]
best_k = k_range[np.argmax(cv_scores)]

# Curse of dimensionality check: if X has >20 features, reduce dims first
# from sklearn.decomposition import PCA
# X_reduced = PCA(n_components=10).fit_transform(X_scaled)
```

## Bayesian Regression

```python
from sklearn.linear_model import BayesianRidge, ARDRegression

# BayesianRidge — Gaussian prior on weights, MAP estimate = Ridge with lambda = alpha/beta
br = BayesianRidge(alpha_1=1e-6, alpha_2=1e-6, lambda_1=1e-6, lambda_2=1e-6)
br.fit(X_train, y_train)

y_pred, y_std = br.predict(X_test, return_std=True)   # <- predictive uncertainty!
print("95% interval:", y_pred - 1.96*y_std, "to", y_pred + 1.96*y_std)

# ARD Regression — Automatic Relevance Determination
# (separate prior precision per feature -> aggressive built-in feature selection)
ard = ARDRegression()
ard.fit(X_train, y_train)
relevant_features = X_train.columns[ard.coef_ != 0]

# Full Bayesian workflow with PyMC (for custom priors / hierarchical models)
# import pymc as pm
# with pm.Model() as model:
#     w = pm.Normal('w', mu=0, sigma=1, shape=X_train.shape[1])
#     b = pm.Normal('b', mu=0, sigma=1)
#     sigma = pm.HalfNormal('sigma', sigma=1)
#     mu = pm.math.dot(X_train, w) + b
#     y_obs = pm.Normal('y_obs', mu=mu, sigma=sigma, observed=y_train)
#     trace = pm.sample(1000, tune=1000)
```

## When to reach for these

```
SVR      -> small-medium data, smooth nonlinear signal, no need for feature importance
KNN      -> low-dimensional data (<20 features), local patterns matter, fast prototyping
Bayesian -> small data, need calibrated uncertainty / prediction intervals, active learning
```
