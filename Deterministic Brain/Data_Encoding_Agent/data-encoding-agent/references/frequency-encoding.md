# Frequency / Count Encoding

## What it is

Replace each category with how frequently it appears in the dataset. "Count encoding" uses raw
counts; "frequency encoding" uses proportions.

```
City        Count    Frequency
Mumbai      300      0.30
Delhi       250      0.25
Bangalore   200      0.20
Pune        150      0.15
Others      100      0.10
```

```python
freq = df["city"].value_counts(normalize=True)   # proportions; drop normalize=True for raw counts
df["city_freq"] = df["city"].map(freq)
```

**Leakage note:** compute `freq` on the training set only, then `.map()` it onto both train and
test (see `code-generation.md`) — the same fit-on-train rule as every other encoder here.

## What signal it captures

Implicitly encodes the *popularity/prevalence* of each category. The underlying hypothesis: more
frequent categories may behave differently from rare ones (e.g. common cities like Mumbai/Delhi
may have different purchase rates than tier-3 cities).

## Mathematical properties — the collision problem

Two categories with the same frequency get the same encoded value:

```
City_A appears 200 times -> encoded 200
City_B appears 200 times -> encoded 200
```

The model cannot distinguish City_A from City_B after encoding. This is a real problem when two
different categories with equal frequency have genuinely different relationships to the target.
It is *not* a problem when frequency itself is the meaningful signal (e.g. a rarer category
inherently implies less reliable data, and that unreliability is what you want captured).

## Advantages

- Handles high cardinality with no new columns — one column in, one column out.
- Captures a popularity/prevalence signal that's often genuinely predictive.
- Works with any model type.
- Memory efficient.

## Disadvantages

- Loses category identity for equal-frequency categories.
- Frequency may simply not be predictive, depending on the domain.
- Sensitive to dataset size/composition — frequencies shift if you add or remove data.
- Leakage risk if computed on the full dataset instead of training data only.

## When to use

High-cardinality nominal features where popularity/prevalence is plausibly predictive, memory is
a constraint, or you're feeding a tree-based model (frequency encoding works well there too).
