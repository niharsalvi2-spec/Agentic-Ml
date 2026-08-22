# Anomaly Detection

## Outlier vs anomaly — the distinction

| | Outlier | Anomaly |
|---|---|---|
| Definition | Statistical extreme value | Unexpected pattern violating expected behavior |
| Scope | Single feature, univariate | Can be multivariate, contextual, temporal |
| Detection | Statistical rules | Pattern recognition, model-based |
| Example | Salary = 10 crore | Normal salary but system accessed at 3am from a new country |
| Basis | Distribution-based | Behavior-based |

**Key insight:** an anomaly may not be a statistical outlier at all. A login at 3am with a
completely normal transaction amount — each value is individually normal, but the *combination*
is anomalous. This is why anomaly detection needs to consider context and sequence, not just
per-column statistics.

## Types of anomalies

| Type | Description | Example |
|---|---|---|
| Point | A single data point is anomalous — most common type | A ₹50 lakh transaction on an account averaging ₹5000 |
| Contextual | Anomalous only given context (time, location) | 35°C is normal in May in Pune, anomalous in December |
| Collective | Individual points are normal, the pattern together isn't | 100 failed logins in 10 seconds; stock moving opposite to market for 5 straight days |

## Statistical methods

### 3-sigma rule

Same mechanism as Z-score outlier detection: `anomaly if |x - μ| > 3σ`. Assumes normality, O(n),
fast — good as a quick first pass, not a final answer for skewed or multivariate data.

### Grubbs' test

Formally tests whether a single outlier exists in univariate, normally-distributed data.

```
G = max|xᵢ - x̄| / s
```

Compare `G` to a critical value from the Grubbs table at your chosen significance level `α`; if
`G > G_critical`, a significant outlier exists. **Limitation:** tests for one outlier at a time —
a "masking effect" can hide additional outliers when several are present simultaneously.

```python
from scipy import stats
import numpy as np

def grubbs_test(data, alpha=0.05):
    n = len(data)
    mean, std = np.mean(data), np.std(data, ddof=1)
    G = np.max(np.abs(data - mean)) / std
    t_crit = stats.t.ppf(1 - alpha / (2 * n), n - 2)
    G_crit = ((n - 1) / np.sqrt(n)) * np.sqrt(t_crit**2 / (n - 2 + t_crit**2))
    return G > G_crit, G, G_crit
```

## DBSCAN for anomaly detection

Core idea: cluster normal points into dense regions; points that don't belong to any cluster are
anomalies (noise points).

```
ε (epsilon)  = neighborhood radius
MinPts       = minimum points to form a dense region
Core point   = has >= MinPts neighbors within ε
Border point = within ε of a core point but < MinPts neighbors itself
Noise point  = not reachable from any core point -> ANOMALY

N_ε(p) = {q ∈ D : d(p,q) <= ε}
|N_ε(p)| >= MinPts -> p is a core point
```

Works because normal data forms dense clusters and anomalies are isolated (can't reach MinPts
neighbors → labeled noise). Doesn't assume any particular cluster shape.

```python
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

X = StandardScaler().fit_transform(df.select_dtypes(include="number"))
labels = DBSCAN(eps=0.5, min_samples=5).fit_predict(X)
df["is_anomaly"] = labels == -1   # -1 = noise = anomaly
```

## Time series anomaly detection

### Moving average residual

```
μ_t = mean of window [t-w, t-1]
σ_t = std of same window
Anomaly if |xₜ - μ_t| > k·σ_t   (k=3 typical)
```

```python
window = 7
rolling_mean = df["value"].rolling(window).mean()
rolling_std = df["value"].rolling(window).std()
df["is_anomaly"] = (df["value"] - rolling_mean).abs() > 3 * rolling_std
```

### Seasonal decomposition residual

```
time series = trend + seasonality + residual
```
Detect anomalies on the **residual only**, after removing expected trend/seasonal patterns —
remaining spikes are the anomalies.

```python
from statsmodels.tsa.seasonal import seasonal_decompose

result = seasonal_decompose(df.set_index("date")["value"], model="additive", period=7)
residual = result.resid.dropna()
threshold = 3 * residual.std()
anomalies = residual[residual.abs() > threshold]
```

## Why this matters

- **Fraud detection:** the anomalies *are* the fraud cases.
- **Predictive maintenance:** a sensor anomaly predicts machine failure before it happens.
- **Medical diagnosis:** anomalous readings indicate disease.
- **Network security:** anomalous traffic patterns indicate intrusion.
- **Data quality:** anomalous values in *training* data corrupt the model — this is why anomaly
  checks belong in the cleaning phase, not just in production monitoring.
