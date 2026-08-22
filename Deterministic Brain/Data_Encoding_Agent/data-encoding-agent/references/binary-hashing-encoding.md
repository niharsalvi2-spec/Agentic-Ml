# Binary Encoding & Feature Hashing

## Binary encoding

Converts label-encoded integers into binary representation, then splits the binary digits into
separate columns. A compromise between label encoding (too few columns, false ordinality) and
one-hot (too many columns).

```
Step 1 - label encode:
Mumbai->1, Delhi->2, Bangalore->3, Pune->4, Chennai->5, Kolkata->6, Hyderabad->7, Jaipur->8

Step 2 - convert to binary:
1->001, 2->010, 3->011, 4->100, 5->101, 6->110, 7->111, 8->1000

Step 3 - each binary digit becomes a column:
City       Bit3  Bit2  Bit1
Mumbai       0     0     1
Delhi        0     1     0
Bangalore    0     1     1
Pune         1     0     0
```

```python
# pip install category_encoders
import category_encoders as ce

encoder = ce.BinaryEncoder(cols=["city"])
df_binary = encoder.fit_transform(df["city"])   # fit on train only; .transform() on test
```

**Dimensionality efficiency:**

```
One-hot: k categories -> k columns
Binary:  k categories -> ceil(log2(k)) columns

For 1000 categories: one-hot = 1000 columns, binary ≈ 10 columns -> ~100x reduction
```

**Tradeoff:** more compact than one-hot, and doesn't impose a false ordinal relationship like
label encoding — but individual binary columns have no interpretable meaning; the model has to
learn from abstract bit patterns, which is harder than one-hot's directly interpretable columns.

**When to use:** medium-high cardinality (50-1000 unique values), a memory constraint but need
for better structure than plain label encoding, tree-based models (which handle the binary
columns well).

## Feature hashing (the "hashing trick")

Applies a hash function to each category, mapping it into a **fixed** number of output columns
regardless of how many unique categories exist.

```
h("Mumbai")    -> 4
h("Delhi")     -> 7
h("Bangalore") -> 2

With n_components=8: creates 8 columns, column[h(city)] = 1, rest = 0
```

```python
from sklearn.feature_extraction import FeatureHasher

hasher = FeatureHasher(n_features=16, input_type="string")
hashed = hasher.transform(df["city"].astype(str).apply(lambda x: [x]))
```

**Hash collisions:**

```
h("Mumbai")  = 4
h("Kolkata") = 4   <- collision, both map to the same column
```

Two different categories get the same encoding and become indistinguishable to the model. More
output columns (`n_features`) reduces collision probability but costs more memory — this is a
direct dial between compactness and information loss.

**When to use:** very high cardinality (>10,000 categories), online/streaming learning where new
categories can appear at prediction time (hashing needs no "known categories" list, unlike
one-hot/target encoding), or when memory is extremely constrained and some information loss via
collisions is an acceptable tradeoff.

## Binary vs. hashing vs. one-hot at a glance

| | One-hot | Binary | Hashing |
|---|---|---|---|
| Columns for k categories | k | ~log2(k) | fixed (you choose) |
| Handles unseen categories at inference | No (needs "unknown" bucket) | No | Yes, naturally |
| Interpretable columns | Yes | No | No |
| Collision risk | None | None | Yes |
| Best cardinality range | <15 | 50-1000 | >10,000 or streaming |
