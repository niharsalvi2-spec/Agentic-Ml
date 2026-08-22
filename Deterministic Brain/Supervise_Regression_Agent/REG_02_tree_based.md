# Skill: Regression — Tree Based
Decision Tree, Random Forest, Gradient Boosting, XGBoost, LightGBM, CatBoost.

## Decision Tree Regressor

```python
from sklearn.tree import DecisionTreeRegressor, plot_tree
import matplotlib.pyplot as plt

dt = DecisionTreeRegressor(
    max_depth=5,             # pre-pruning: limit growth
    min_samples_split=10,
    min_samples_leaf=5,
    min_impurity_decrease=0.0,
    random_state=42,
)
dt.fit(X_train, y_train)
y_pred = dt.predict(X_test)

plt.figure(figsize=(20, 10))
plot_tree(dt, feature_names=X_train.columns, filled=True, max_depth=3)

# Cost-complexity pruning (post-pruning) — find best alpha via CV
path = dt.cost_complexity_pruning_path(X_train, y_train)
ccp_alphas = path.ccp_alphas
from sklearn.model_selection import cross_val_score
scores = [cross_val_score(DecisionTreeRegressor(ccp_alpha=a, random_state=42),
                           X_train, y_train, cv=5).mean() for a in ccp_alphas]
best_alpha = ccp_alphas[np.argmax(scores)]
dt_pruned = DecisionTreeRegressor(ccp_alpha=best_alpha, random_state=42).fit(X_train, y_train)
```

## Random Forest Regressor

```python
from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(
    n_estimators=300,        # more trees = better, diminishing returns ~200+
    max_depth=None,          # fully grown trees usually best
    max_features=1/3,        # "sqrt" for clf, 1/3 or "log2" typical for regression
    min_samples_leaf=2,
    bootstrap=True,
    oob_score=True,          # free validation estimate
    n_jobs=-1,
    random_state=42,
)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
print("OOB R2:", rf.oob_score_)

# Feature importance — MDI (biased toward high-cardinality features)
importances_mdi = pd.Series(rf.feature_importances_, index=X_train.columns).sort_values(ascending=False)

# Permutation importance (MDA) — more reliable
from sklearn.inspection import permutation_importance
perm = permutation_importance(rf, X_test, y_test, n_repeats=10, random_state=42)
importances_mda = pd.Series(perm.importances_mean, index=X_train.columns).sort_values(ascending=False)
```

## Gradient Boosting Regressor (sklearn)

```python
from sklearn.ensemble import GradientBoostingRegressor

gbr = GradientBoostingRegressor(
    n_estimators=500,
    learning_rate=0.01,      # lower lr + more trees = better generalization
    max_depth=3,             # shallow trees = weak learners
    subsample=0.8,           # stochastic GB — reduces variance, speeds training
    random_state=42,
)
gbr.fit(X_train, y_train)

# Early stopping via staged predictions (find best n_estimators)
from sklearn.metrics import mean_squared_error
val_errors = [mean_squared_error(y_test, pred) for pred in gbr.staged_predict(X_test)]
best_n = np.argmin(val_errors) + 1
```

## XGBoost

```python
import xgboost as xgb

xgb_model = xgb.XGBRegressor(
    n_estimators=1000,
    learning_rate=0.01,
    max_depth=4,
    min_child_weight=3,      # min sum of hessians per leaf, controls overfitting
    subsample=0.8,
    colsample_bytree=0.8,
    gamma=0.1,                # min gain required to split (prune)
    reg_alpha=0.01,           # L1 on leaf weights
    reg_lambda=1.0,           # L2 on leaf weights
    early_stopping_rounds=50,
    eval_metric='rmse',
    random_state=42,
)
xgb_model.fit(X_train, y_train,
              eval_set=[(X_test, y_test)], verbose=False)

y_pred = xgb_model.predict(X_test)

# Feature importance (3 types)
xgb.plot_importance(xgb_model, importance_type='gain')   # weight / gain / cover
```

## LightGBM

```python
import lightgbm as lgb

lgb_model = lgb.LGBMRegressor(
    n_estimators=1000,
    learning_rate=0.01,
    num_leaves=31,            # leaf-wise growth control (more direct than max_depth)
    max_depth=-1,             # -1 = no limit, but watch overfitting on small data
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.01,
    reg_lambda=1.0,
    random_state=42,
)
lgb_model.fit(X_train, y_train,
              eval_set=[(X_test, y_test)],
              callbacks=[lgb.early_stopping(50)])

y_pred = lgb_model.predict(X_test)
```

## CatBoost

```python
from catboost import CatBoostRegressor

# List categorical column names/indices directly — no manual encoding needed
cat_features = ['category_col1', 'category_col2']

cb_model = CatBoostRegressor(
    iterations=1000,
    learning_rate=0.03,
    depth=6,
    l2_leaf_reg=3.0,
    cat_features=cat_features,     # handled internally via ordered target statistics
    random_strength=1.0,
    bagging_temperature=1.0,
    early_stopping_rounds=50,
    verbose=100,
    random_state=42,
)
cb_model.fit(X_train, y_train, eval_set=(X_test, y_test))
y_pred = cb_model.predict(X_test)
```

## Quick comparison

```python
# XGBoost/LightGBM/CatBoost: level-wise vs leaf-wise vs ordered boosting
# - Large tabular data, need speed          -> LightGBM
# - Heavy categorical features              -> CatBoost
# - Best general accuracy, competitions     -> XGBoost
# - Small dataset, need robustness          -> XGBoost or Random Forest (LightGBM overfits small data)
```
