# Missing Values

## What is missing data?

A value not recorded, stored, or transmitted for a particular observation/feature. In pandas:
`NaN`, `None`, `NaT` (datetime), or an empty string. Missing data is not random noise — it has a
mechanism, and that mechanism determines the correct fix.

## The three mechanisms — the most important theory here

### MCAR — Missing Completely At Random

`P(missing | observed, unobserved) = P(missing)` — missingness depends on nothing.

- **Example:** a survey form randomly fails to save 5% of responses due to a server bug.
- **If ignored:** dataset gets smaller but stays representative — no bias.
- **Detection:** t-test between "rows with X missing" vs "rows without" on other variables; no
  significant difference → likely MCAR.
- **Handling:** drop or any imputation — all valid, no bias introduced.

### MAR — Missing At Random

`P(missing | observed, unobserved) = P(missing | observed)` — missingness is explainable by
*other columns you already have*, not by the missing value itself.

- **Example:** older people skip the income field (age observed, income missing).
- **If ignored:** dropping rows biases the dataset — you lose disproportionately many of one
  group (e.g. all older respondents).
- **Handling:** imputation using other observed variables — KNN or MICE, because they use other
  columns to predict the missing one.

### MNAR — Missing Not At Random

`P(missing | observed, unobserved) ≠ P(missing | observed)` — missingness depends on the missing
value itself. The fact that it's missing *is* information.

- **Example:** high earners don't report income (income missing *because* it's high); patients
  who got worse stop attending follow-ups (outcome missing *because* it's bad).
- **If ignored:** severe bias — the model is wrong in exactly the cases that matter most.
- **Handling:** cannot be fixed statistically alone. Requires domain knowledge, modeling the
  missingness itself as a feature, better data collection, or an indicator column
  (`income_missing = 1/0`).

## Patterns of missingness

| Pattern | Description |
|---|---|
| Univariate | Only one variable has missing values — simplest case |
| Multivariate | Multiple variables missing simultaneously, often correlated |
| Monotone | Once a variable is missing, all subsequent variables are too (e.g. trial dropout) |
| Non-monotone (arbitrary) | No particular pattern — needs full imputation methods like MICE |

## Detection

```python
import pandas as pd

df.isnull().sum()                         # count per column
(df.isnull().sum() / len(df)) * 100        # percentage per column
df.isnull().any()                          # any missing per column

# visual — missing value heatmap
import missingno as msno
msno.matrix(df)
msno.heatmap(df)   # are columns missing together? (multivariate pattern)
```

## Drop vs impute

**Drop when:**

| Situation | Action | Reason |
|---|---|---|
| Column >50-60% missing | Drop column | Too little info left to impute reliably |
| Row missing >70% of values | Drop row | Row carries almost no information |
| MCAR + small % missing | Drop rows | No bias introduced |
| Column irrelevant anyway | Drop column | Missing data is moot |

**Impute when:**

| Situation | Action |
|---|---|
| Column <5% missing | Any simple imputation |
| Column 5-50% missing | Careful imputation based on distribution |
| MAR | Model-based imputation (KNN, MICE) |
| MNAR | Add indicator column + impute |

## Imputation methods

### Mean

`x̄ = Σxᵢ / n`. Preserves mean; **shrinks variance** and **weakens correlations**; creates an
unnatural spike at the mean.

- Use when: roughly normal distribution, no outliers, <5% missing, feature not strongly
  correlated with others.
- Never use when: outliers present, skewed distribution, MAR/MNAR (mean ignores *why* it's
  missing).

```python
df["col"].fillna(df["col"].mean(), inplace=True)
```

### Median

Robust to outliers (doesn't move with extremes); still shrinks variance, but better than mean
for skewed data (income, house prices). Default choice for most numeric columns.

```python
df["col"].fillna(df["col"].median(), inplace=True)
```

### Mode

`argmax P(x = xᵢ)`. Only valid basic method for categorical columns, or numeric columns with
discrete values (e.g. a 1-5 rating). Risk: creates artificial dominance of the mode value; ties
are resolved arbitrarily.

```python
df["col"].fillna(df["col"].mode()[0], inplace=True)
```

### Forward fill / backward fill

Propagate the last (or next) known value. Use for ordered/time-series/sensor data where
"yesterday's value" is a reasonable estimate. Never use on unordered data, or when the gap is too
large to propagate a stale value across.

```python
df["col"].ffill(inplace=True)   # forward fill
df["col"].bfill(inplace=True)   # backward fill
```

### KNN Imputation

Find the K most similar rows (Euclidean distance over non-missing dimensions), take a
distance-weighted average of their values for the missing column:

```
d(a,b) = sqrt(Σᵢ (aᵢ - bᵢ)²)        (over non-missing dims only)
x̂_missing = Σₖ(wₖ·xₖ) / Σₖwₖ,  wₖ = 1/d(row, neighbor_k)
```

Preserves local structure and correlations. Expensive (O(n²)) and scale-sensitive — **normalize
features first**. Choosing K matters: small K overfits to local noise, large K over-smooths.

```python
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
import pandas as pd

numeric_cols = df.select_dtypes(include="number").columns
scaler = StandardScaler()
scaled = scaler.fit_transform(df[numeric_cols])

imputer = KNNImputer(n_neighbors=5, weights="distance")
imputed_scaled = imputer.fit_transform(scaled)
df[numeric_cols] = scaler.inverse_transform(imputed_scaled)
```

Use when features are correlated, dataset is small-medium (<100k rows), missingness is MAR.

### MICE — Multiple Imputation by Chained Equations

Treats each missing column as a regression problem, predicted from all other columns, iterated:

```
1. Initial fill: mean/median (rough start)
2. For each column with missing values:
     a. Reset that column's imputed values back to missing
     b. Use all other columns as features (X)
     c. Fit a regression/classification model on observed rows
     d. Predict and fill the missing values
3. Repeat step 2 for all missing columns = one iteration ("chain")
4. Repeat for max_iter iterations until imputed values stop changing significantly
```

"Chained" = each column's imputation uses the latest updated estimates of the other columns.
"Multiple" (full MICE) = repeat the whole process M times with different seeds, producing M
complete datasets, and pool results to capture imputation uncertainty.

```python
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor

imputer = IterativeImputer(
    estimator=RandomForestRegressor(n_estimators=50, random_state=42),
    max_iter=10, random_state=42,
)
df[numeric_cols] = imputer.fit_transform(df[numeric_cols])
```

Use when: dataset is complex with interrelated features, MAR mechanism, highest accuracy needed,
dataset not too large (<50k rows practical limit — it's slow).

### Indicator variable method — for MNAR

The fact a value is missing *is* the information; preserve it as a feature instead of discarding it.

```python
df["income_missing"] = df["income"].isnull().astype(int)
df["income"] = df["income"].fillna(df["income"].median())
```

Now the model can learn e.g. "when `income_missing=1`, this person is more likely a high earner"
instead of that signal being erased by imputation.

## Data leakage in imputation — critical

**Rule: fit the imputer on training data only; transform both train and test with those training
statistics.**

```python
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

X_train, X_test = train_test_split(df, test_size=0.2, random_state=42)

imputer = SimpleImputer(strategy="median")
imputer.fit(X_train[numeric_cols])                       # fit on train ONLY

X_train[numeric_cols] = imputer.transform(X_train[numeric_cols])
X_test[numeric_cols]  = imputer.transform(X_test[numeric_cols])  # transform, never re-fit
```

Fitting on the full dataset lets test-set statistics leak into training — the model implicitly
sees "future" information, and evaluation metrics come out optimistically biased. This applies to
every imputation method above, not just `SimpleImputer`.
