# Label Encoding & Ordinal Encoding

## Label encoding — what it is

Assigns a unique integer to each category, typically by sorting alphabetically: 0, 1, 2, 3...
Simplest encoding — one column in, one column out.

```python
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
df["city_encoded"] = le.fit_transform(df["city"])
# Bangalore -> 0, Delhi -> 1, Mumbai -> 2, Pune -> 3 (alphabetical)
```

## The mathematical problem — false ordinal assumption

Assigning integers 0,1,2,3 implicitly creates:

```
Bangalore(0) < Delhi(1) < Mumbai(2) < Pune(3)
Mumbai - Delhi = Delhi - Bangalore    (equal-gap assumption)
Pune = 2 x Delhi                       (ratio assumption)
```

None of these are true for city names. Linear models, KNN, and SVM use these numbers
mathematically:

```
Distance(Bangalore, Delhi) = |0-1| = 1
Distance(Bangalore, Pune)  = |0-3| = 3
```

The model concludes Bangalore and Delhi are 3x more similar than Bangalore and Pune — false
information with no basis in the actual data.

## When label encoding IS correct

**Tree-based models.** Decision trees split on thresholds (`City <= 1 -> left`); the tree finds
whatever split is optimal regardless of what the integer "means." Random Forest, XGBoost,
LightGBM, CatBoost all handle this correctly — label encoding is valid for tree models
regardless of whether the variable is nominal or ordinal.

**True ordinal variables**, where the order should be preserved (see ordinal encoding below —
technically a different thing from alphabetical label encoding, but the mechanism is the same).

**Binary variables.** `Male=0, Female=1` implies no ordering with only two values —
mathematically equivalent to one-hot for the binary case.

## When label encoding is WRONG

Nominal data fed to linear models, KNN, SVM, or neural networks — these models treat the
integers as having real mathematical meaning, so the false ordinal relationship corrupts what
the model learns.

## Ordinal encoding — the fix for genuinely ordered categories

Unlike label encoding (alphabetical, arbitrary), ordinal encoding **manually assigns integers
that respect a real natural order** you define from domain knowledge.

```
Label encoding (alphabetical):    Ordinal encoding (logical):
Graduate     -> 0                 School       -> 0
PhD          -> 1                 Graduate     -> 1
Postgraduate -> 2                 Postgraduate -> 2
School       -> 3                 PhD          -> 3
```

Label encoding loses the order; ordinal encoding preserves it.

```python
from sklearn.preprocessing import OrdinalEncoder

order = [["School", "Graduate", "Postgraduate", "PhD"]]
enc = OrdinalEncoder(categories=order, handle_unknown="use_encoded_value", unknown_value=-1)
df["education_encoded"] = enc.fit_transform(df[["education"]])
```

## When ordinal encoding is valid for linear models — the equal-gap question

If the gap between categories is genuinely roughly equal — e.g. `School=0, Graduate=1,
Postgraduate=2, PhD=3`, and each step up in education adds roughly the same incremental effect —
a linear model's single coefficient on this column is a reasonable approximation.

If the gap is **not** equal — e.g. a subjective pain scale `No Pain=0, Mild=1, Moderate=2,
Severe=3, Extreme=4` where "Severe minus Moderate" isn't obviously equal to "Moderate minus
Mild" — ordinal encoding for a linear model is only an approximation of the truth. Tree-based
models are valid regardless of whether the gap is equal, since they threshold rather than assume
linearity.

## When to use which

| Situation | Method |
|---|---|
| Variable has genuine natural order | Ordinal encoding, manual mapping |
| Tree-based model, any variable type | Label encoding (order doesn't matter to trees) |
| Linear model, gaps approximately equal | Ordinal encoding, manual mapping |
| Linear model, nominal (no order) | **Not label/ordinal encoding** — use one-hot (see `onehot-encoding.md`) |
