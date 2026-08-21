---
name: data-cleaning
description: Use this skill whenever the user has a raw or messy dataset and needs to detect and fix missing values, duplicates, outliers, or anomalies before modeling — or asks about NaN/null handling, imputation (mean/median/KNN/MICE), MCAR/MAR/MNAR, deduplication, outlier detection (Z-score/IQR/Isolation Forest/LOF), Winsorization, or anomaly detection (point/contextual/collective, DBSCAN, time-series). Trigger for phrases like "clean this data", "handle missing values", "remove duplicates", "detect outliers", "why is my model performing badly on dirty data", or "build a data cleaning agent". This skill diagnoses *why* data is dirty (not just where), picks a handling strategy backed by the underlying statistical mechanism, and writes leakage-safe cleaning code.
compatibility: python3, pandas, numpy, scipy; specific methods need extra packages (scikit-learn for KNNImputer/IterativeImputer/IsolationForest/LOF/DBSCAN, missingno for visualization, fuzzywuzzy/rapidfuzz for fuzzy dedup).
---

# Data Cleaning Agent

Turns a raw dataset into a modeling-ready one — the step that consumes 60-80% of real project time and the one no algorithm can compensate for if skipped. A model trained on dirty data learns dirty patterns; garbage in, garbage out is not a cliché here, it's the default outcome.

## Why data is dirty — check the root cause first

| Cause | Example |
|---|---|
| Human error | Typing "25yrs" instead of 25 |
| System error | Sensor malfunction → missing reading |
| Integration error | Merging two databases with different formats |
| Data decay | Customer moved city — old address still stored |
| Design flaw | Form allowed free text instead of a dropdown |
| Transmission error | Network failure → incomplete record saved |

Knowing *why* a value is wrong or missing changes the correct fix — this is why the missing-value
section below leads with mechanism (MCAR/MAR/MNAR) before methods. Don't jump straight to
"just impute it" without diagnosing the mechanism; the wrong imputation method silently
introduces bias that is invisible until the model underperforms in production.

## How to use this skill

1. **Run the four-step diagnostic** below on the incoming dataset, in order — missingness before
   duplicates before outliers before anomalies, because deduping and outlier-handling behave
   differently once missing values have (or haven't) been resolved.
2. **Open the matching reference file** for the theory + methods needed at each step; don't hold
   every method in context if only one is relevant.
3. **Use `references/code-generation.md`** for the reusable, leakage-safe pipeline skeleton
   (fit-on-train-only, `sklearn.Pipeline`/`ColumnTransformer` patterns) rather than writing
   imputation calls ad hoc.
4. **Report what was found and what was done**, not just clean output — see "What to hand back" below.

## The four-step diagnostic

```
Received new dataset
│
├── Step 1: Missing Values           → references/missing-values.md
│   ├── Detect: isnull, heatmap, % missing per column
│   ├── Classify mechanism: MCAR / MAR / MNAR (this determines the fix)
│   ├── >50-60% missing column        → drop column
│   ├── MCAR + small % missing        → drop rows or simple impute
│   ├── MAR                            → KNN or MICE (model-based)
│   └── MNAR                           → add indicator column + impute
│
├── Step 2: Duplicates                → references/duplicates.md
│   ├── Exact                          → drop (keep first/last — domain decision)
│   ├── Partial (key columns match)    → resolve by key columns, keep most complete row
│   └── Fuzzy (typos/case/formatting)  → string similarity → flag for review
│
├── Step 3: Outliers                  → references/outliers.md
│   ├── Detect: Z-score (normal-ish data) / IQR (skewed) / Isolation Forest (multivariate)
│   ├── Data entry error               → remove or impute as missing
│   ├── Genuine extreme value          → cap (Winsorize) or transform (log/sqrt/Box-Cox)
│   ├── The outlier IS the signal (fraud, disease) → keep, do not touch
│   └── Model is tree-based            → usually keep; linear/KNN/NN → cap or transform
│
└── Step 4: Anomalies                 → references/anomaly-detection.md
    ├── Point anomaly                  → Z-score, IQR, Isolation Forest, LOF
    ├── Contextual anomaly             → add context features (time, location) before detecting
    ├── Collective anomaly             → sequence/pattern-level detection, not row-level
    └── Time series                    → moving average residual, seasonal decomposition residual
```

## Reference files

| File | Covers |
|---|---|
| `references/missing-values.md` | MCAR/MAR/MNAR theory + detection, drop-vs-impute rules, mean/median/mode/ffill-bfill/KNN/MICE/indicator-variable methods, data leakage in imputation |
| `references/duplicates.md` | Exact/partial/fuzzy duplicate detection (Levenshtein, Jaccard, cosine/TF-IDF), keep-first vs keep-last, aggregation instead of deletion |
| `references/outliers.md` | Point/contextual/collective outliers, Z-score, IQR (Tukey), Modified Z-score (MAD), Isolation Forest, LOF, remove/cap/transform/keep/impute-as-missing decision guide |
| `references/anomaly-detection.md` | Outlier-vs-anomaly distinction, 3-sigma, Grubbs test, DBSCAN for anomaly detection, time-series anomaly detection |
| `references/code-generation.md` | Leakage-safe cleaning pipeline skeleton: fit-on-train-only, `ColumnTransformer`, before/after distribution diffing, cleaning report generator |

## Non-negotiable defaults (apply to every cleaning task)

- **Never impute or scale using statistics from the full dataset.** Fit on the training split
  only, then transform train and test with those fitted statistics. Fitting on everything is
  data leakage and silently inflates evaluation metrics — see `code-generation.md`.
- **Diagnose before you fix.** State the missingness mechanism (or outlier type) you believe
  applies and why, don't silently mean-impute everything or silently drop every duplicate.
- **Never drop >50-60% of a column or >70% of a row's values without flagging it to the user** —
  that's a modeling decision (drop the feature entirely vs. try to salvage it), not something to
  decide unilaterally.
- **Preserve traceability.** When you cap, impute, or drop, keep enough of a record (indicator
  columns, a before/after row count, a cleaning log) that someone auditing the pipeline later can
  see what changed and why — not just a final "cleaned_data.csv" with no paper trail.
- **Don't confuse "outlier" with "wrong."** A fraud case, a disease case, or a genuine extreme
  value is data working as intended — removing it can delete the exact signal the model needs.
