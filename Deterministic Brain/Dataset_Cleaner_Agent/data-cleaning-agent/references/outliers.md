# Outliers

## What are outliers?

Data points significantly far from the rest of the distribution. Not always errors — can be
genuine extreme values. Context determines whether an outlier is valid or erroneous.

| Type | Description | Example |
|---|---|---|
| Point outlier | Single value far from the rest | Salary = 9,999,999 in a dataset of 50k salaries |
| Contextual outlier | Normal globally, abnormal in context | 30°C in December in Pune |
| Collective outlier | A group of points that together is abnormal | Sudden spike in network traffic |

## Why outliers are harmful

**Statistical measures:** mean is pulled toward the outlier; standard deviation inflates
(overestimated variance); correlation can be distorted or even flipped in sign by one extreme
point.

**ML model sensitivity:**

| Model | Sensitivity |
|---|---|
| Linear Regression | Very high — minimizes squared error, outliers dominate |
| Lasso / Ridge | High — still squared loss |
| Logistic Regression | Moderate — decision boundary can shift |
| Decision Tree | Low — splits on value ranges |
| Random Forest | Low — averaging reduces impact |
| KNN | High — distance metrics distorted |
| SVM | Moderate — depends on kernel |
| Neural Networks | High — gradient updates dominated by outlier loss |

## Detection methods

### Z-score

```
z = (x - μ) / σ
Rule: |z| > 3 -> outlier, |z| > 2 -> mild outlier
```

Why 3σ: under normality, 68%/95%/99.7% of data falls within 1σ/2σ/3σ, so `P(|z|>3) ≈ 0.3%` —
a value that rare is flagged. **Limitation:** assumes normal distribution; μ and σ are themselves
distorted by the very outliers you're trying to detect (circular); unsuitable for heavily skewed data.

```python
from scipy import stats
import numpy as np

z_scores = np.abs(stats.zscore(df["col"].dropna()))
outliers = df[z_scores > 3]
```

### IQR method (Tukey's rule) — preferred default

```
Q1 = 25th percentile, Q3 = 75th percentile, IQR = Q3 - Q1
Lower fence = Q1 - 1.5*IQR
Upper fence = Q3 + 1.5*IQR
Outlier: x < lower fence OR x > upper fence
```

Worked example — salaries `[20k,25k,30k,35k,40k,45k,50k,500k]`: Q1=27.5k, Q3=47.5k, IQR=20k,
upper fence = 47.5k + 30k = 77.5k → 500k is an outlier.

Robust because it doesn't use mean/std, so it isn't distorted by the outliers it's detecting;
works on skewed distributions. `1.5×IQR` captures ~99.3% of a normal distribution; use `3.0×IQR`
for "extreme" outliers only.

```python
Q1, Q3 = df["col"].quantile(0.25), df["col"].quantile(0.75)
IQR = Q3 - Q1
lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
outliers = df[(df["col"] < lower) | (df["col"] > upper)]
```

### Modified Z-score (MAD-based) — most robust univariate method

```
MAD = median(|xᵢ - median(x)|)
Modified Z = 0.6745 * (xᵢ - median(x)) / MAD
Rule: |Modified Z| > 3.5 -> outlier
```

0.6745 is a scaling constant so MAD ≈ std under normality, keeping the modified score comparable
to a regular Z-score. Use when data is heavily skewed or outliers have already distorted mean/std.

```python
median = df["col"].median()
mad = (df["col"] - median).abs().median()
modified_z = 0.6745 * (df["col"] - median) / mad
outliers = df[modified_z.abs() > 3.5]
```

### Isolation Forest — multivariate, no distribution assumption

Core idea: outliers are *easier to isolate* than normal points.

```
1. Randomly select a feature
2. Randomly select a split value between its min and max
3. Split data, recurse on each partition
4. Repeat until every point is isolated (alone in its partition)
Anomaly score s(x,n) = 2^(-E[h(x)]/c(n))   [h = path length, c(n) = normalization]
s -> 1: anomaly.  s -> 0.5: normal.
```

Short path to isolate = low-density region = outlier. Works in high dimensions, no distribution
assumption, O(n log n).

```python
from sklearn.ensemble import IsolationForest

iso = IsolationForest(contamination=0.02, random_state=42)
df["is_outlier"] = iso.fit_predict(df.select_dtypes(include="number")) == -1
```

### Local Outlier Factor (LOF) — local density comparison

```
For point p, k nearest neighbors:
reach_dist(p,o) = max(k-dist(o), d(p,o))
lrd(p) = 1 / (avg reach_dist(p, neighbors))
LOF(p) = avg(lrd(neighbors)) / lrd(p)
LOF >> 1 -> outlier (much lower density than neighbors). LOF ≈ 1 -> normal.
```

Catches outliers that are only outliers *relative to their local neighborhood* — useful when
clusters of different densities exist.

```python
from sklearn.neighbors import LocalOutlierFactor

lof = LocalOutlierFactor(n_neighbors=20)
df["is_outlier"] = lof.fit_predict(df.select_dtypes(include="number")) == -1
```

## Handling strategy

| Option | What | When |
|---|---|---|
| Remove | Delete the row | Clear data entry error, MCAR, small % of data |
| Cap / Winsorize | Clamp to the fence value, keep the row | Genuine but extreme — reduce influence without deleting |
| Transform | Log/sqrt/Box-Cox to compress the range | Right-skewed data, model sensitive to scale |
| Keep | Do nothing | Outlier IS the signal (fraud, disease), or model is tree-based/robust |
| Impute as missing | Set to NaN, then KNN/MICE impute | Clear measurement error but rest of the row is useful |

**Winsorization:**

```python
x_new = x.clip(lower=lower_fence, upper=upper_fence)
```

**Log transform** (right-skewed, positive values only): `x_new = log(x)` — e.g. salaries
`[20k, 30k, 500k] -> [9.9, 10.3, 13.1]`; 500k stops dominating.

**Box-Cox:** `x_new = (x^t - 1)/t` if t≠0 else `log(x)`, with `t` optimized to make the result
most normal — `scipy.stats.boxcox`.

## Decision guide

```
Is the outlier a genuine data error?
├── Yes (typo, measurement failure)
│   ├── Row otherwise useful -> impute as missing
│   └── Row useless -> remove
└── No (genuine extreme value)
    ├── The outlier IS what you're detecting (fraud, disease) -> keep
    ├── Model sensitive to outliers (linear, KNN, NN) -> cap or transform
    └── Model robust to outliers (tree-based) -> keep
```
