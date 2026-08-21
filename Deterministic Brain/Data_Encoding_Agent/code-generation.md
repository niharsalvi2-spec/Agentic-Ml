# Code Generation — Leakage-Safe Encoding Pipeline Skeleton

Every encoding script this skill produces should follow fit-on-train / transform-both, and
should explicitly decide an unseen-category strategy, rather than calling `pd.get_dummies` or a
raw `groupby().mean()` directly against the whole dataframe.

## 1. Split before fitting any encoder

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

## 2. Mixed-encoding ColumnTransformer (one-hot + ordinal together)

```python
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.pipeline import Pipeline

nominal_cols = ["city", "color"]          # low cardinality nominal -> one-hot
ordinal_cols = ["education"]              # true order -> ordinal
education_order = [["School", "Graduate", "Postgraduate", "PhD"]]

preprocessor = ColumnTransformer([
    ("onehot", OneHotEncoder(handle_unknown="ignore", drop="first"), nominal_cols),
    ("ordinal", OrdinalEncoder(categories=education_order,
                                handle_unknown="use_encoded_value", unknown_value=-1), ordinal_cols),
])

preprocessor.fit(X_train)                      # fit ONCE, on train only
X_train_enc = preprocessor.transform(X_train)
X_test_enc = preprocessor.transform(X_test)     # transform only, never re-fit
```

## 3. Target encoding — K-fold, leakage-safe, with smoothing (reusable function)

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

def fit_kfold_target_encoder(train_df, cat_col, target_col, n_splits=5, smoothing=100, seed=42):
    """Returns (encoded_train_series, final_map, global_mean) for use at inference."""
    global_mean = train_df[target_col].mean()
    encoded = pd.Series(index=train_df.index, dtype=float)

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for tr_idx, val_idx in kf.split(train_df):
        fold_train = train_df.iloc[tr_idx]
        stats = fold_train.groupby(cat_col)[target_col].agg(["mean", "count"])
        smoothed = (stats["count"] * stats["mean"] + smoothing * global_mean) / (stats["count"] + smoothing)
        encoded.iloc[val_idx] = train_df.iloc[val_idx][cat_col].map(smoothed).fillna(global_mean)

    # final map for transforming test/inference data — built from ALL training data
    full_stats = train_df.groupby(cat_col)[target_col].agg(["mean", "count"])
    final_map = (full_stats["count"] * full_stats["mean"] + smoothing * global_mean) / (full_stats["count"] + smoothing)

    return encoded, final_map, global_mean

def apply_target_encoder(df, cat_col, final_map, global_mean):
    return df[cat_col].map(final_map).fillna(global_mean)   # unseen category -> global mean

# Usage
train_encoded, city_map, global_mean = fit_kfold_target_encoder(train_df, "city", "target")
train_df["city_enc"] = train_encoded
test_df["city_enc"] = apply_target_encoder(test_df, "city", city_map, global_mean)
```

## 4. Frequency encoding — fit on train, apply to both, handle unseen

```python
def fit_frequency_encoder(train_df, cat_col):
    return train_df[cat_col].value_counts(normalize=True)

def apply_frequency_encoder(df, cat_col, freq_map):
    return df[cat_col].map(freq_map).fillna(0)   # unseen category -> 0

freq_map = fit_frequency_encoder(train_df, "city")
train_df["city_freq"] = apply_frequency_encoder(train_df, "city", freq_map)
test_df["city_freq"] = apply_frequency_encoder(test_df, "city", freq_map)
```

## 5. Before/after audit — what got encoded, into how many columns

```python
def encoding_report(original_df, encoded_df, cat_cols):
    rows = []
    for col in cat_cols:
        rows.append({
            "column": col,
            "cardinality": original_df[col].nunique(),
            "columns_after_encoding": sum(1 for c in encoded_df.columns if str(c).startswith(col)),
        })
    return pd.DataFrame(rows)
```

Print or return this after encoding so a sudden 500-column blowup from an accidental one-hot on
a high-cardinality field is visible immediately, not discovered downstream.

## 6. Full skeleton

```python
"""
Generic encoding template — copy per dataset, fill in variable classifications.
"""
import logging
import pandas as pd
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("encoder")

def encode(df: pd.DataFrame, target_col: str,
           nominal_low_card: list, nominal_high_card: list, ordinal_specs: dict):
    """
    nominal_low_card:  one-hot
    nominal_high_card: target or frequency encoding (choose per column based on cardinality/signal)
    ordinal_specs:     {col: [ordered category list]}
    """
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

    # one-hot for low-cardinality nominal
    train_df = pd.get_dummies(train_df, columns=nominal_low_card, drop_first=True)
    test_df = pd.get_dummies(test_df, columns=nominal_low_card, drop_first=True)
    test_df = test_df.reindex(columns=train_df.columns, fill_value=0)  # align unseen dummy cols

    # target encoding for high-cardinality nominal
    for col in nominal_high_card:
        encoded, final_map, global_mean = fit_kfold_target_encoder(train_df, col, target_col)
        train_df[f"{col}_enc"] = encoded
        test_df[f"{col}_enc"] = apply_target_encoder(test_df, col, final_map, global_mean)

    # ordinal
    from sklearn.preprocessing import OrdinalEncoder
    for col, order in ordinal_specs.items():
        enc = OrdinalEncoder(categories=[order], handle_unknown="use_encoded_value", unknown_value=-1)
        train_df[f"{col}_enc"] = enc.fit_transform(train_df[[col]])
        test_df[f"{col}_enc"] = enc.transform(test_df[[col]])

    logger.info(f"Encoded: {train_df.shape[1]} train columns, {test_df.shape[1]} test columns")
    return train_df, test_df
```

## Rules of thumb when generating encoding code for a user

1. Always name the variable classification (nominal/ordinal/binary) and cardinality you diagnosed
   for each column before choosing a method — don't silently default to one-hot for everything.
2. Fit every encoder on the training split only; never on the full dataset.
3. For target encoding, always use K-fold or leave-one-out + smoothing — never a bare
   `groupby().mean()` reused as the training feature.
4. Explicitly handle unseen categories (`handle_unknown=`, `.fillna(global_mean)`, or a reindex
   after `pd.get_dummies`) rather than letting a shape mismatch surface as a runtime error later.
5. Report cardinality-to-columns-after-encoding so an accidental dimensionality blowup is
   visible immediately.
