# Common Mistakes in Encoding

Five specific bugs account for most broken encoding pipelines. Check generated code against
this list before finalizing.

## Mistake 1 — Fitting the encoder on the full dataset

```python
# WRONG — leaks test-set categories/statistics into training
encoder.fit(full_df)

# CORRECT
encoder.fit(X_train)
X_train_enc = encoder.transform(X_train)
X_test_enc = encoder.transform(X_test)
```

Applies to every stateful encoder here — one-hot (which categories exist), frequency (counts),
target (means), ordinal — not just target encoding.

## Mistake 2 — Label encoding nominal data for a linear model

```python
# WRONG — City has no order; linear regression will treat these integers as meaningful magnitudes
city_map = {"Bangalore": 0, "Delhi": 1, "Mumbai": 2}
df["city_enc"] = df["city"].map(city_map)
LinearRegression().fit(df[["city_enc"]], y)

# CORRECT — one-hot encode nominal data for linear/KNN/SVM/NN models
df_onehot = pd.get_dummies(df, columns=["city"], drop_first=True)
```

## Mistake 3 — One-hot encoding a high-cardinality column

```python
# WRONG — 50,000 unique user IDs -> 50,000 new columns
pd.get_dummies(df, columns=["user_id"])

# CORRECT — frequency or target encoding for high cardinality
df["user_id_freq"] = df["user_id"].map(df["user_id"].value_counts(normalize=True))
```

## Mistake 4 — Target encoding without leakage prevention

```python
# WRONG — mean computed on the full training set, including the row itself, then reused as its own feature
city_means = df.groupby("city")["target"].mean()
df["city_enc"] = df["city"].map(city_means)

# CORRECT — K-fold out-of-fold encoding + smoothing (see target-encoding.md)
```

## Mistake 5 — Not handling unseen categories at test time

```
Problem: "Nagpur" appears in test but never in training.
  One-hot        -> no column exists for Nagpur, encoder errors or silently drops it
  Target encoding -> no mean exists for Nagpur
  Ordinal/label   -> encoder errors on an unmapped category
```

**Solutions, by method:**

```python
# One-hot: let sklearn's OneHotEncoder handle unknowns explicitly
from sklearn.preprocessing import OneHotEncoder
enc = OneHotEncoder(handle_unknown="ignore")   # unseen category -> all-zero row

# Target encoding: fall back to the global mean for unseen categories
df["city_enc"] = df["city"].map(city_means).fillna(global_mean)

# Frequency encoding: assign 0 (or the minimum observed frequency) for unseen categories
df["city_freq"] = df["city"].map(freq_map).fillna(0)

# Ordinal encoding: sklearn's OrdinalEncoder supports an explicit unknown_value
from sklearn.preprocessing import OrdinalEncoder
enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
```

Decide and implement the unseen-category strategy **before** deployment, not after the first
production error — a category that never appeared in training data is a near-certainty in any
real deployment with an open-ended nominal field (city, product name, merchant name, etc.).
