# Skill: Feature Selection — Filter Methods
Model-independent statistical tests. Fast, no overfitting risk.

## 1.1 Univariate — Correlation Based

```python
import pandas as pd
from scipy.stats import pearsonr, spearmanr, kendalltau, pointbiserialr

# Pearson (numeric-numeric, linear)
pearson_scores = X.corrwith(y)  # X: DataFrame, y: Series

# Spearman (monotonic, rank-based)
spearman_scores = X.corrwith(y, method='spearman')

# Kendall Tau (robust, rank-based)
kendall_scores = X.corrwith(y, method='kendall')

# Point-Biserial (continuous feature vs binary target)
pb_scores = {c: pointbiserialr(X[c], y)[0] for c in X.columns}
```

## 1.1 Univariate — Statistical Tests

```python
from sklearn.feature_selection import chi2, f_classif, f_regression
from scipy.stats import kruskal, mannwhitneyu, ks_2samp, ttest_ind

# Chi-Squared (categorical features, non-negative, classification)
chi2_scores, chi2_pvals = chi2(X, y)

# ANOVA F-test (classification)
f_scores, f_pvals = f_classif(X, y)

# ANOVA F-test (regression)
f_scores_reg, f_pvals_reg = f_regression(X, y)

# Kruskal-Wallis (non-parametric ANOVA alt., >2 groups)
stat, p = kruskal(*[X[col][y == cls] for cls in y.unique()])

# Mann-Whitney U (non-parametric, 2 groups)
stat, p = mannwhitneyu(X[col][y == 0], X[col][y == 1])

# Kolmogorov-Smirnov (distribution difference between classes)
stat, p = ks_2samp(X[col][y == 0], X[col][y == 1])

# Welch's T-Test (unequal variance)
stat, p = ttest_ind(X[col][y == 0], X[col][y == 1], equal_var=False)
```

## 1.1 Univariate — Information Theoretic

```python
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression

# Mutual Information (classification) — captures nonlinear dependence
mi_scores = mutual_info_classif(X, y, discrete_features='auto', random_state=42)

# Mutual Information (regression)
mi_scores_reg = mutual_info_regression(X, y, random_state=42)

# Information Gain == Mutual Information for discrete features (same formula)
# Gain Ratio = InfoGain / SplitInfo(feature) — normalizes for high-cardinality bias
import numpy as np
def gain_ratio(feature, target):
    from sklearn.metrics import mutual_info_score
    ig = mutual_info_score(feature, target)
    values, counts = np.unique(feature, return_counts=True)
    probs = counts / counts.sum()
    split_info = -np.sum(probs * np.log2(probs))
    return ig / split_info if split_info != 0 else 0
```

## 1.1 Variance Based

```python
from sklearn.feature_selection import VarianceThreshold

vt = VarianceThreshold(threshold=0.01)
X_reduced = vt.fit_transform(X)
kept_features = X.columns[vt.get_support()]

# Coefficient of Variation (scale-independent spread)
cv = X.std() / X.mean()
low_info_features = cv[cv < 0.1].index
```

## 1.2 Multivariate — Multicollinearity

```python
from statsmodels.stats.outliers_influence import variance_inflation_factor
import pandas as pd

# VIF — Variance Inflation Factor  (VIF > 5 or 10 -> drop)
vif_data = pd.DataFrame()
vif_data["feature"] = X.columns
vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

# Cramer's V (categorical-categorical association)
from scipy.stats.contingency import association
def cramers_v(x, y):
    ct = pd.crosstab(x, y)
    return association(ct, method='cramer')
```

## 1.2 Multivariate — mRMR / Relief

```python
# mRMR — Maximum Relevance Minimum Redundancy
# pip install mrmr_selection
from mrmr import mrmr_classif
selected_features = mrmr_classif(X=X, y=y, K=10)

# Relief / ReliefF — pip install skrebate
from skrebate import ReliefF
rf = ReliefF(n_features_to_select=10, n_neighbors=100)
rf.fit(X.values, y.values)
top_features = X.columns[rf.top_features_[:10]]
```

## 1.3 Regression-Specific: Distance Correlation

```python
import dcor  # pip install dcor
dcor_score = dcor.distance_correlation(X[col], y)  # captures linear + nonlinear
```
