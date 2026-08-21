# Target / Mean Encoding (and Leave-One-Out)

## What it is

Replace each category with the mean of the target variable for that category. Also called mean
encoding or likelihood encoding.

```
City        Rows    Mean Purchase
Mumbai      300     5200
Delhi       250     4800
Bangalore   200     6100
Pune        150     3900
```

Directly encodes the relationship between category and target in a single numeric column — a
strong predictive signal, especially useful for high-cardinality features and particularly
effective with gradient boosting models.

## The target leakage problem — the most critical issue in this file

Target encoding uses target values to build a feature. If computed on the full dataset
(including rows that will later be train/test evaluated together, or worse, on data that
includes the row itself), the row's own target leaks into its own feature value.

```
Row 1: City=Mumbai, Target=8000
Row 2: City=Mumbai, Target=5000
Row 3: City=Mumbai, Target=6000

Naive mean for Mumbai using all 3 rows = 6333
Used as Row 1's feature during training:
  Row 1's OWN target (8000) influenced its own feature value.
  Model learns "when Mumbai_encoded is high, target is high" —
  but that's only true because the target was used to build the encoding.
```

This produces an inflated, falsely optimistic training signal that doesn't generalize.

## Correct solution 1 — K-fold out-of-fold target encoding

```
1. Split training data into K folds.
2. For each fold:
     Compute category means using all OTHER folds only (out-of-fold).
     Encode the current fold using those means.
3. Every row is encoded using means computed WITHOUT that row.
```

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

def kfold_target_encode(df, cat_col, target_col, n_splits=5, smoothing=100, global_mean=None):
    df = df.copy()
    global_mean = global_mean if global_mean is not None else df[target_col].mean()
    encoded = pd.Series(index=df.index, dtype=float)

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    for train_idx, val_idx in kf.split(df):
        fold_train = df.iloc[train_idx]
        stats = fold_train.groupby(cat_col)[target_col].agg(["mean", "count"])
        # smoothing formula (see below) applied using only this fold's training data
        smoothed = (stats["count"] * stats["mean"] + smoothing * global_mean) / (stats["count"] + smoothing)
        encoded.iloc[val_idx] = df.iloc[val_idx][cat_col].map(smoothed).fillna(global_mean)

    return encoded, global_mean
```

At **inference/test time**, use the full-training-set smoothed means (computed once on all
training data) — there's no "held-out fold" concept once training is finished.

## Correct solution 2 — Leave-One-Out (LOO) encoding

A variant of target encoding: for row `i`, compute the mean target of all *other* rows with the
same category, explicitly excluding row `i` itself.

```
LOO_encoding(i) = (sum of targets for category c - target_i) / (count_c - 1)
```

```python
def leave_one_out_encode(df, cat_col, target_col):
    sums = df.groupby(cat_col)[target_col].transform("sum")
    counts = df.groupby(cat_col)[target_col].transform("count")
    return (sums - df[target_col]) / (counts - 1)
```

**Advantage over K-fold:** no explicit fold structure needed, each row automatically gets an
out-of-fold-equivalent encoding, simpler to implement.
**Disadvantage:** still needs smoothing for rare categories; at test time (no target available)
you fall back to the full training-set mean per category.

## Smoothing — for rare categories

A category with 2 samples has an unreliable mean (high variance). Pull rare categories toward
the global mean; let frequent categories keep their own mean.

```
mu_smoothed = (n * mu_category + m * mu_global) / (n + m)

n          = number of rows for this category
mu_category = mean target for this category
mu_global   = overall mean target
m           = smoothing factor (typically 10-300)
```

Worked examples:

```
Mumbai: n=300, mean=5200, global_mean=5000, m=100
mu_smoothed = (300*5200 + 100*5000) / (300+100) = 5150

Latur: n=2, mean=8000, global_mean=5000, m=100
mu_smoothed = (2*8000 + 100*5000) / (2+100) = 5059   <- pulled strongly toward global mean
```

```python
def smoothed_mean(group_mean, group_count, global_mean, m=100):
    return (group_count * group_mean + m * global_mean) / (group_count + m)
```

## When to use target encoding

High-cardinality nominal features; gradient boosting models (XGBoost, LightGBM, CatBoost all
have built-in support); when a strong category-target relationship genuinely exists. **Always**
pair with K-fold or leave-one-out + smoothing.

## When NOT to use

Small datasets (insufficient samples per category for a reliable mean); categories with very few
samples and no smoothing (high-variance mean estimate); or any implementation that skips leakage
prevention — this produces inflated accuracy that will not hold up outside of training.
