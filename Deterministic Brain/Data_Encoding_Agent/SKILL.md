---
name: categorical-encoding
description: Use this skill whenever the user needs to convert categorical/text columns into numeric form for ML — label encoding, one-hot encoding, ordinal encoding, frequency/count encoding, target/mean encoding, binary encoding, hashing (feature hashing), or leave-one-out encoding. Trigger for phrases like "encode this column", "one-hot encode", "how do I handle categorical variables", "high cardinality feature", "target leakage in encoding", "dummy variable trap", or "build an encoding agent". This skill picks the correct encoding based on variable type (nominal/ordinal/binary), cardinality, and target model family — and prevents the two most common encoding bugs: false ordinal relationships and target leakage.
compatibility: python3, pandas, scikit-learn (OrdinalEncoder/OneHotEncoder); category_encoders for target/binary/hashing/leave-one-out encoders (pip install category_encoders).
---

# Categorical Encoding Agent

Converts categorical data into numeric form without corrupting its meaning. ML algorithms are
mathematical functions — they can't compute a distance, dot product, or gradient on "Mumbai" or
"Male" directly. But the *wrong* encoding actively teaches the model false relationships: label
encoding `small=1, medium=2, large=3` silently asserts `large - medium = medium - small`, which
is only true if that gap really is equal.

## How to use this skill

1. **Classify the variable** — binary / ordinal / nominal — and its **cardinality** (unique
   category count). Both determine the correct method; skipping this step is how "just one-hot
   everything" bugs get shipped.
2. **Check the target model family** — linear/KNN/SVM/NN need mathematically meaningless category
   distances (one-hot); tree-based models tolerate arbitrary integers (label encoding is often
   fine); gradient boosting benefits most from target encoding.
3. **Open the relevant reference file** for the method(s) that fit.
4. **Use `references/code-generation.md`** for the leakage-safe fit-on-train-only pattern —
   this applies to every encoder, not just target encoding.
5. **Check `references/common-mistakes.md`** before finalizing — five specific bugs account for
   most broken encoding pipelines in practice.

## Variable classification (do this first)

| Type | Definition | Example |
|---|---|---|
| Nominal | No natural order | City, Gender, Color |
| Ordinal | Natural order exists, gaps may be unequal | Education (School<Graduate<Postgrad), Rating (Bad<Average<Good) |
| Binary | Exactly 2 categories | Yes/No, Male/Female, Spam/Not Spam |

## Cardinality — the other critical axis

| Cardinality | Example | Impact |
|---|---|---|
| Low (2-10) | Gender, Rating, Blood Group | Easy — one-hot is fine |
| Medium (10-50) | Indian states, product category | Manageable — one-hot borderline, consider alternatives |
| High (50-1000) | City, job title | One-hot creates too many columns |
| Very high (>1000) | User ID, product ID, ZIP code | One-hot impractical — needs frequency/target/hashing |

High cardinality is the single biggest encoding challenge in real-world data.

## Decision tree

```
Categorical variable
│
├── Binary (2 categories)?
│   └── Label encode 0/1 — works for all models → references/label-ordinal-encoding.md
│
├── Ordinal (natural order exists)?
│   └── Ordinal encoding, manual order mapping → references/label-ordinal-encoding.md
│
├── Nominal — check cardinality
│   │
│   ├── Low (<15 categories)
│   │   ├── Linear / KNN / SVM / NN → One-hot → references/onehot-encoding.md
│   │   └── Tree-based → label or one-hot, either works
│   │
│   ├── Medium (15-50)
│   │   ├── Tree-based → label encoding
│   │   ├── Linear / NN → binary or frequency encoding → references/binary-hashing-encoding.md
│   │   └── Strong target relationship → target encoding w/ K-fold → references/target-encoding.md
│   │
│   └── High (>50)
│       ├── Frequency itself predictive → frequency/count encoding → references/frequency-encoding.md
│       ├── Strong target relationship → target encoding w/ smoothing → references/target-encoding.md
│       ├── Memory-constrained → binary or hashing encoding → references/binary-hashing-encoding.md
│       └── Online/streaming, new categories at inference → hashing encoding → references/binary-hashing-encoding.md
```

## Reference files

| File | Covers |
|---|---|
| `references/label-ordinal-encoding.md` | Label encoding, the false-ordinal-assumption problem, when it's actually fine (trees, binary, true ordinal), manual ordinal mapping |
| `references/onehot-encoding.md` | One-hot mechanics, dummy variable trap, curse of dimensionality, when to drop a column |
| `references/frequency-encoding.md` | Count/frequency encoding, what signal it captures and loses |
| `references/target-encoding.md` | Target/mean encoding, target leakage (the most critical issue in this whole skill), K-fold out-of-fold encoding, smoothing formula, leave-one-out encoding |
| `references/binary-hashing-encoding.md` | Binary encoding (log₂(k) columns), feature hashing, hash collisions, when each fits |
| `references/model-recommendations.md` | Per-model-family recommended encoding, with reasoning |
| `references/common-mistakes.md` | The five most common encoding bugs, each with wrong vs correct code |
| `references/code-generation.md` | Leakage-safe fit-on-train-only pattern for every encoder type, unseen-category handling at inference |

## Non-negotiable defaults

- **Never fit any encoder (including `OneHotEncoder`) on the full dataset.** Fit on train,
  transform train and test separately — this is not just a target-encoding rule, it's true of
  every stateful encoder including one-hot (categories seen at fit time), frequency (counts), and
  ordinal.
- **Never label-encode nominal data for a linear/KNN/SVM/NN model.** State explicitly which
  model family the encoding is for before picking a method — the same column needs different
  encodings for XGBoost vs. logistic regression.
- **Target encoding always needs K-fold (or leave-one-out) + smoothing**, never a single
  full-training-set mean. This is the single highest-leakage-risk method in this skill.
- **Always define a strategy for unseen categories at inference time** before shipping — a
  category unseen during training breaks naive one-hot and target encoding alike unless handled.
- **State the cardinality and variable type you diagnosed** before recommending a method, so the
  choice is auditable rather than a default reflex.
