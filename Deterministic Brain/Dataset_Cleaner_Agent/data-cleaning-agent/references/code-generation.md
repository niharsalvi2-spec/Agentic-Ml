# Code Generation — Leakage-Safe Cleaning Pipeline Skeleton

Every cleaning script this skill produces should be built on these patterns instead of ad hoc
`fillna()`/`drop_duplicates()` calls scattered through a notebook. The two failure modes this
guards against: (1) data leakage from fitting cleaning statistics on the full dataset, and
(2) cleaning that silently destroys signal with no record of what changed.

## 1. Split before you fit anything

```python
from sklearn.model_selection import train_test_split

train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
```

Every statistic used for cleaning (mean, median, IQR fences, KNN neighbors, MICE models) must be
**fit on `train_df` only**, then applied to `test_df` with `.transform()`. Never call `.fit()` or
compute quantiles/means on the concatenation of train+test.

## 2. A single leakage-safe pipeline (missing values + scaling, sklearn-native)

```python
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import StandardScaler

numeric_cols = train_df.select_dtypes(include="number").columns.tolist()
categorical_cols = train_df.select_dtypes(include="object").columns.tolist()

numeric_pipeline = Pipeline([
    ("imputer", KNNImputer(n_neighbors=5)),   # swap for SimpleImputer(strategy="median") if MCAR
    ("scaler", StandardScaler()),
])
categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
])

preprocessor = ColumnTransformer([
    ("num", numeric_pipeline, numeric_cols),
    ("cat", categorical_pipeline, categorical_cols),
])

preprocessor.fit(train_df)                       # fit ONCE, on train only
train_clean = preprocessor.transform(train_df)
test_clean = preprocessor.transform(test_df)      # transform only, never re-fit
```

## 3. Outlier fences — compute on train, apply to both

```python
def compute_iqr_fences(train_series, k=1.5):
    q1, q3 = train_series.quantile(0.25), train_series.quantile(0.75)
    iqr = q3 - q1
    return q1 - k * iqr, q3 + k * iqr

fences = {col: compute_iqr_fences(train_df[col]) for col in numeric_cols}

def apply_fences(df, fences):
    df = df.copy()
    for col, (lower, upper) in fences.items():
        df[col] = df[col].clip(lower, upper)
    return df

train_df = apply_fences(train_df, fences)
test_df = apply_fences(test_df, fences)   # same fences, computed from train only
```

## 4. Before/after report — never hand back silent output

```python
import pandas as pd

def cleaning_report(before: pd.DataFrame, after: pd.DataFrame, label: str = "") -> pd.DataFrame:
    rows = []
    for col in before.columns:
        if col not in after.columns:
            continue
        rows.append({
            "column": col,
            "missing_before": before[col].isnull().sum(),
            "missing_after": after[col].isnull().sum() if col in after else None,
            "n_before": len(before),
            "n_after": len(after),
        })
    report = pd.DataFrame(rows)
    if label:
        print(f"--- Cleaning report: {label} ---")
    print(report.to_string(index=False))
    return report
```

Call this after every major step (missing-value handling, dedup, outlier handling) so row counts
and null counts are visible at each stage, not just at the end.

## 5. Dedup with an audit trail

```python
def dedupe_with_log(df, subset=None, keep="first"):
    before_n = len(df)
    completeness = df.notnull().sum(axis=1)
    df = df.assign(_completeness=completeness)
    if subset:
        df = df.sort_values("_completeness", ascending=False).drop_duplicates(subset=subset, keep="first")
    else:
        df = df.drop_duplicates(keep=keep)
    df = df.drop(columns="_completeness")
    print(f"Dedup: {before_n} -> {len(df)} rows ({before_n - len(df)} removed)")
    return df
```

## 6. Full skeleton — assemble into one script

```python
"""
Generic cleaning template — copy per dataset, adjust the mechanism decisions per column.
"""
import logging
import pandas as pd
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("cleaner")


def clean(df: pd.DataFrame, target_col: str = None):
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    logger.info(f"Split: {len(train_df)} train / {len(test_df)} test")

    # Step 1 — missing values (fit imputer on train only; see missing-values.md for mechanism choice)
    # ... apply the ColumnTransformer pattern from section 2 above

    # Step 2 — duplicates (see duplicates.md)
    train_df = dedupe_with_log(train_df)
    test_df = dedupe_with_log(test_df)

    # Step 3 — outliers (fences computed on train only; see outliers.md)
    # ... apply compute_iqr_fences / apply_fences from section 3 above

    # Step 4 — anomalies, if relevant (see anomaly-detection.md) — usually flagged, not auto-removed

    logger.info(f"Final: {len(train_df)} train / {len(test_df)} test")
    return train_df, test_df


if __name__ == "__main__":
    df = pd.read_csv("raw_data.csv")
    train_clean, test_clean = clean(df)
    train_clean.to_csv("train_clean.csv", index=False)
    test_clean.to_csv("test_clean.csv", index=False)
```

## Rules of thumb when generating cleaning code for a user

1. Default to the train/test-split-first structure above unless the task is explicitly
   exploratory (a one-off `df.isnull().sum()` to look, not to fix).
2. State which missingness mechanism (MCAR/MAR/MNAR) and which outlier handling option you're
   assuming for each column, even briefly — don't apply mean imputation everywhere by default.
3. Always print or return a before/after report (`cleaning_report`) rather than silently
   overwriting `df` in place with no visibility into what changed.
4. Never fit any imputer, scaler, or outlier-fence calculation on data that includes the test
   set — this is the single most common bug in generated cleaning code.
5. When dropping >50-60% of a column or >70% of a row, surface that decision to the user instead
   of silently executing it.
