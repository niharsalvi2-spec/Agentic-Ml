# Validating Collected / Synthetic Data Quality

Never hand off collected data without checking it. For scraped/API/DB data this means a schema
and null check; for synthetic data it means confirming it actually behaves like real data.

## Quick checklist (every source)

```python
print(df.shape)
print(df.dtypes)
print(df.isnull().sum())
print(df.duplicated().sum())
print(df.describe(include="all"))
```

## Synthetic data — Method 1: visual comparison

```python
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
real_df["salary"].hist(ax=axes[0], bins=30, alpha=0.7)
axes[0].set_title("Real")
synthetic_df["salary"].hist(ax=axes[1], bins=30, alpha=0.7)
axes[1].set_title("Synthetic")
plt.tight_layout()
plt.show()
```

## Method 2: statistical tests (KS test, t-test)

```python
from scipy import stats
import pandas as pd

def validate_column(real_col, synthetic_col):
    ks_stat, ks_p = stats.ks_2samp(real_col.dropna(), synthetic_col.dropna())
    t_stat, t_p = stats.ttest_ind(real_col.dropna(), synthetic_col.dropna())
    return {
        "KS_pvalue": round(ks_p, 4), "KS_pass": "PASS" if ks_p > 0.05 else "FAIL",
        "Ttest_pvalue": round(t_p, 4), "Ttest_pass": "PASS" if t_p > 0.05 else "FAIL",
        "Real_mean": round(real_col.mean(), 4), "Synthetic_mean": round(synthetic_col.mean(), 4),
    }

results = {col: validate_column(real_df[col], synthetic_df[col])
           for col in real_df.select_dtypes(include="number").columns}
print(pd.DataFrame(results).T)
```

## Method 3: correlation preservation

```python
import seaborn as sns
import matplotlib.pyplot as plt

real_corr = real_df.select_dtypes(include="number").corr()
synth_corr = synthetic_df.select_dtypes(include="number").corr()
diff = (real_corr - synth_corr).abs()

corr_score = 1 - diff.mean().mean()
print(f"Correlation similarity score: {corr_score:.4f}")  # closer to 1.0 = better
```

## Method 4: Train-Synthetic-Test-Real (TSTR) — gold standard

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder

target = "purchased"
X_real, y_real = real_df.drop(columns=[target]), real_df[target]
X_synth, y_synth = synthetic_df.drop(columns=[target]), synthetic_df[target]

le = LabelEncoder()
for col in X_real.select_dtypes(include="object").columns:
    X_real[col] = le.fit_transform(X_real[col].astype(str))
    X_synth[col] = le.transform(X_synth[col].astype(str))

model = RandomForestClassifier(n_estimators=100, random_state=42)

model.fit(X_real[:700], y_real[:700])                       # baseline: train real, test real
trtr_auc = roc_auc_score(y_real[700:], model.predict_proba(X_real[700:])[:, 1])

model.fit(X_synth, y_synth)                                  # train synthetic, test real
tstr_auc = roc_auc_score(y_real[700:], model.predict_proba(X_real[700:])[:, 1])

gap = abs(trtr_auc - tstr_auc)
print(f"TRTR {trtr_auc:.4f} | TSTR {tstr_auc:.4f} | gap {gap:.4f}")
print("SYNTHETIC DATA IS HIGH QUALITY" if gap < 0.05 else "SYNTHETIC DATA NEEDS IMPROVEMENT")
```

## Method 5: privacy check (membership inference risk)

Confirms synthetic rows aren't just near-copies of real records.

```python
from sklearn.neighbors import NearestNeighbors

real_num = real_df.select_dtypes(include="number").values
synth_num = synthetic_df.select_dtypes(include="number").values

nn = NearestNeighbors(n_neighbors=1).fit(real_num)
distances, _ = nn.kneighbors(synth_num)

print(f"Avg nearest-real-neighbor distance: {distances.mean():.4f}")
print(f"Min distance (risk of copying): {distances.min():.4f}")
if distances.min() < 0.01:
    print("WARNING: synthetic data may be copying real records — privacy risk")
```

## Method 6: SDV's built-in report (if using SDV)

```python
from sdv.evaluation.single_table import evaluate_quality, run_diagnostic

quality_report = evaluate_quality(real_df, synthetic_df, metadata)
print(quality_report.get_score())            # 0.0-1.0, higher is better
diagnostic = run_diagnostic(real_df, synthetic_df, metadata)
print(diagnostic.get_results())
```

## When there's no "real_df" to compare against

If the data is fully synthetic with no real counterpart (e.g. pure `sklearn.make_classification`
for algorithm testing), skip methods 1-3 and 5 — there's nothing to compare to — and just confirm
the schema/shape checklist above plus that class balance and feature ranges match what you asked
for.
