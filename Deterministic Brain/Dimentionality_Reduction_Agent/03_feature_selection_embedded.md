# Skill: Feature Selection — Embedded Methods
Selection happens *during* model training. Faster than wrapper, more accurate than filter.

## 3.1 Regularization Based

```python
from sklearn.linear_model import Lasso, LogisticRegression, ElasticNet
from sklearn.svm import LinearSVC

# Lasso (L1) Regression — zeros out irrelevant feature weights
lasso = Lasso(alpha=0.01)
lasso.fit(X, y)
selected = X.columns[lasso.coef_ != 0]

# Lasso Logistic Regression (classification)
lasso_clf = LogisticRegression(penalty='l1', solver='liblinear', C=1.0)
lasso_clf.fit(X, y)
selected_clf = X.columns[(lasso_clf.coef_ != 0).any(axis=0)]

# Ridge (L2) — shrinks, does NOT zero out (not a selector, but a baseline)
from sklearn.linear_model import Ridge
ridge = Ridge(alpha=1.0).fit(X, y)

# Elastic Net (L1 + L2) — sparse AND stable with correlated groups
enet = ElasticNet(alpha=0.01, l1_ratio=0.5)
enet.fit(X, y)
selected_enet = X.columns[enet.coef_ != 0]

# Group Lasso — pip install group-lasso
from group_lasso import GroupLasso
gl = GroupLasso(groups=group_ids, group_reg=0.05, l1_reg=0)
gl.fit(X.values, y.values)
```

## 3.2 Tree-Based Importance

```python
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.feature_selection import SelectFromModel
import xgboost as xgb
import lightgbm as lgb

# Random Forest — Mean Decrease Impurity (MDI)
rf = RandomForestClassifier(n_estimators=300, random_state=42).fit(X, y)
importances_mdi = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)

# Mean Decrease Accuracy (MDA) == Permutation Importance on the trained RF (see 3.4)

# Extra Trees
et = ExtraTreesClassifier(n_estimators=300, random_state=42).fit(X, y)

# Gradient Boosting
gb = GradientBoostingClassifier().fit(X, y)

# XGBoost — 3 importance types
xgb_model = xgb.XGBClassifier().fit(X, y)
weight = xgb_model.get_booster().get_score(importance_type='weight')  # split frequency
gain   = xgb_model.get_booster().get_score(importance_type='gain')    # avg gain per split
cover  = xgb_model.get_booster().get_score(importance_type='cover')   # avg coverage

# LightGBM
lgb_model = lgb.LGBMClassifier(importance_type='gain').fit(X, y)

# SelectFromModel — generic threshold wrapper for any of the above
sfm = SelectFromModel(rf, threshold='median')
sfm.fit(X, y)
selected_tree = X.columns[sfm.get_support()]
```

## 3.3 Coefficient Based

```python
# Linear/Logistic Regression coefficients (already shown above)
# SVM coefficients (linear kernel only — nonlinear kernels have no simple coef_)
svm_linear = LinearSVC().fit(X, y)
svm_importance = pd.Series(abs(svm_linear.coef_[0]), index=X.columns)
```

## 3.4 Model-Agnostic Importance

```python
from sklearn.inspection import permutation_importance
import shap

# Permutation Importance — works with ANY fitted model
result = permutation_importance(rf, X, y, n_repeats=10, random_state=42)
perm_importance = pd.Series(result.importances_mean, index=X.columns).sort_values(ascending=False)

# SHAP — gold standard, model-agnostic exact/approximate attribution
explainer = shap.TreeExplainer(rf)          # TreeSHAP: fast + exact for tree models
shap_values = explainer.shap_values(X)
shap.summary_plot(shap_values, X)           # visual ranking
mean_abs_shap = pd.Series(abs(shap_values).mean(axis=0), index=X.columns)

# KernelSHAP — any black-box model (slow, approximate)
kexplainer = shap.KernelExplainer(model.predict, shap.sample(X, 100))
kshap_values = kexplainer.shap_values(X.iloc[:50])

# LinearSHAP — exact & fast for linear models
lexplainer = shap.LinearExplainer(lasso_clf, X)

# DeepSHAP — for neural networks (torch/keras models)
# dexplainer = shap.DeepExplainer(nn_model, background_data)

# LIME — local surrogate explanation per prediction
from lime.lime_tabular import LimeTabularExplainer
lime_exp = LimeTabularExplainer(X.values, feature_names=X.columns, mode='classification')
explanation = lime_exp.explain_instance(X.iloc[0].values, rf.predict_proba)

# Drop-Column Importance — retrain without each feature, measure performance delta
from sklearn.model_selection import cross_val_score
baseline = cross_val_score(rf, X, y, cv=5).mean()
drop_importance = {}
for col in X.columns:
    score = cross_val_score(rf, X.drop(columns=[col]), y, cv=5).mean()
    drop_importance[col] = baseline - score
```

## 3.4 Gradient Based (Neural Networks)

```python
import torch

# Vanilla Gradient
X_t = torch.tensor(X.values, dtype=torch.float32, requires_grad=True)
out = nn_model(X_t)
out.sum().backward()
grad_importance = X_t.grad.abs().mean(dim=0)

# Integrated Gradients / SmoothGrad / GradCAM — use captum (pip install captum)
from captum.attr import IntegratedGradients, NoiseTunnel, LayerGradCam
ig = IntegratedGradients(nn_model)
attributions = ig.attribute(X_t, target=0)
```
