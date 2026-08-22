# One-Hot Encoding

## What it is

Creates one new binary column per unique category; each column answers "is this row that
category?" (1 = yes, 0 = no). Also called dummy encoding.

```
Original:          One-hot result:
City               City_Bangalore  City_Delhi  City_Mumbai  City_Pune
Bangalore    ->          1               0           0           0
Delhi        ->          0               1           0           0
Mumbai       ->          0               0           1           0
Pune         ->          0               0           0           1
```

```python
import pandas as pd
df_onehot = pd.get_dummies(df, columns=["city"], drop_first=False)

# or, sklearn (preferred inside a leakage-safe pipeline — see code-generation.md)
from sklearn.preprocessing import OneHotEncoder
enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
```

## Mathematical properties — why it's correct for nominal data

Euclidean distance between any two distinct one-hot rows is the same:

```
d(Bangalore, Delhi)  = sqrt((1-0)^2 + (0-1)^2) = sqrt(2)
d(Bangalore, Mumbai) = sqrt((1-0)^2 + (0-1)^2) = sqrt(2)
d(Bangalore, Pune)   = sqrt((1-0)^2 + (0-1)^2) = sqrt(2)
```

All categories are equidistant from each other — no false ordering, correct for nominal data.
Each row has exactly one 1 and the rest 0s (mutually exclusive categories), so full information
is preserved and the model can learn any relationship between category and target independently.

## The dummy variable trap

With `k` categories you get `k` columns, but they're perfectly multicollinear:

```
City_Bangalore + City_Delhi + City_Mumbai + City_Pune = 1   (always true)
```

Knowing any `k-1` columns tells you the `k`th exactly. For linear regression this is perfect
multicollinearity — the design matrix becomes singular and can't be inverted to compute
coefficients.

**Fix — drop one column (the reference category):**

```python
pd.get_dummies(df, columns=["city"], drop_first=True)
```

```
City_Delhi  City_Mumbai  City_Pune
    0           0           0        <- this is Bangalore (implicit reference)
    1           0           0        <- Delhi
    0           1           0        <- Mumbai
    0           0           1        <- Pune
```

| Model type | Drop one column? |
|---|---|
| Linear regression, logistic regression | **Always drop** — avoids singular matrix |
| Tree-based models | Not necessary — no matrix inversion |
| Neural networks | Not necessary |

## The curse of dimensionality

```
City with 500 unique values -> 500 (or 499) new columns
1000 rows x 500 columns, each column ~99.8% zeros -> extremely sparse
```

Problems: memory blows up; distances in very high-dimensional sparse space become less
meaningful; training needs exponentially more data to fill the space; overfitting risk rises
with the feature-to-sample ratio.

**Rule of thumb:** one-hot is practical for cardinality up to roughly 10-15. Above that, prefer
frequency, target, or binary encoding (see the other reference files).

## When one-hot is correct

Nominal data for linear models, KNN, SVM, or neural networks, at low-to-medium cardinality
(<15 unique values), when you want the model to independently learn each category's
relationship with the target.

## When one-hot is wrong

High-cardinality columns (city, ZIP code, user ID) — creates a sparse, high-dimensional,
memory-intensive feature space for little benefit; use frequency/target/binary/hashing encoding
instead.
