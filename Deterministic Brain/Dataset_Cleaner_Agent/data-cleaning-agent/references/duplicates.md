# Duplicate Values

## What are duplicates?

Rows identical or near-identical to another row — exact (every column matches) or partial (only
key columns match). Caused by data entry errors, merging datasets, scraping the same page twice,
or system retries.

## Types

| Type | Description |
|---|---|
| Exact | Every column value identical — easiest to detect |
| Partial | Key columns match, other columns differ (e.g. same customer, two emails — which is real?) — needs domain knowledge |
| Near / fuzzy | Similar but not identical due to typos/formatting: "Mumbai" vs "mumbai" vs "Bombay" — needs string similarity metrics |

## Why duplicates are harmful

**In ML models:**
- Data leakage — the same row in both train and test means the model memorizes rather than
  generalizes.
- Biased training — duplicated rows get artificially higher effective weight.
- Inflated evaluation — accuracy looks better than it actually is.
- Skewed distributions — duplicated rows shift the mean/distribution.

**In analysis:** count statistics and frequency analysis both come out wrong (e.g. "1000 rows"
but only 700 unique customers).

## Detection

### Exact

```python
df.duplicated().sum()                 # count of exact duplicate rows
df[df.duplicated(keep=False)]         # view all duplicate rows (both copies)
```

### Partial (key columns)

```python
key_cols = ["customer_id", "email"]
df[df.duplicated(subset=key_cols, keep=False)]
```

### Fuzzy — string similarity metrics

**Levenshtein distance (edit distance):** minimum single-character edits (insert/delete/
substitute) to transform one string into another.

```
"kitten" -> "sitting": k->s, e->i, insert g  = 3 edits = Levenshtein distance 3
```

```python
# pip install rapidfuzz
from rapidfuzz.distance import Levenshtein
Levenshtein.distance("kitten", "sitting")   # 3
```

**Jaccard similarity:** `J(A,B) = |A ∩ B| / |A ∪ B|` over character n-gram sets.

```
"Mumbai" -> {'Mu','um','mb','ba','ai'}
"Mumbay" -> {'Mu','um','mb','ba','ay'}
J = 4/6 = 0.67
```

**Cosine similarity:** TF-IDF vectorize the strings, measure the angle between vectors (0-1,
higher = more similar) — good for longer text fields, not just short names.

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 3))
tfidf = vec.fit_transform(df["name"])
sim_matrix = cosine_similarity(tfidf)  # sim_matrix[i][j] close to 1 -> likely duplicate
```

```python
# quick fuzzy match with rapidfuzz for pairwise candidate generation
from rapidfuzz import process, fuzz

candidates = process.extract("Jon Smith", df["name"].tolist(), scorer=fuzz.token_sort_ratio, limit=5)
```

Fuzzy matches should generally be **flagged for human review**, not auto-merged — "John Smith" vs
"Jon Smith" might be the same person or might not; that's a domain call.

## Handling strategy

### Keep first vs keep last

```python
df.drop_duplicates(keep="first")   # assume first entry is the original
df.drop_duplicates(keep="last")    # assume most recent entry is most up to date
```

This is a domain/business-logic decision, not a statistical one — state the assumption explicitly.

### Subset-based deduplication, keeping the most complete row

```python
df["_completeness"] = df.notnull().sum(axis=1)
df = (df.sort_values("_completeness", ascending=False)
        .drop_duplicates(subset=["customer_id"], keep="first")
        .drop(columns="_completeness"))
```

### Aggregate instead of delete

When "duplicates" actually represent multiple valid events for the same entity:

```python
# multiple orders for the same customer -> sum order value instead of dropping rows
agg = df.groupby("customer_id", as_index=False).agg(total_spend=("order_value", "sum"))

# multiple sensor readings -> average instead of picking one arbitrarily
agg = df.groupby(["sensor_id", "timestamp"], as_index=False).agg(reading=("reading", "mean"))
```
