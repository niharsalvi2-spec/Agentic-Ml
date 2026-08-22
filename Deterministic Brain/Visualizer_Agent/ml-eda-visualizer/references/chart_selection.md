# Chart Selection — Flat Decision Framework

Answer "what chart do I use" directly, no theory reading required. Follow to the matching reference file for the math/interpretation once the chart is picked.

```
What are you trying to show?

├── Distribution of ONE variable
│   ├── Numeric                          → Histogram + KDE overlay          [univariate.md]
│   ├── Numeric, need outliers flagged   → Boxplot (never alone — pair w/ hist/KDE for bimodality) [univariate.md]
│   └── Categorical                      → Bar chart (not pie, unless ≤3 near-equal-weight cats)   [univariate.md]
│
├── Comparison ACROSS GROUPS
│   ├── Numeric distribution per group   → Violin (default) or Boxplot (if decluttering many groups) [bivariate.md]
│   ├── Need raw points, n is small      → Strip plot                       [bivariate.md]
│   └── Count per group                  → Bar chart                       [univariate.md]
│
├── Relationship between TWO variables
│   ├── Numeric vs Numeric               → Scatter plot (+ check heteroscedasticity) [bivariate.md]
│   ├── Numeric vs Categorical           → Violin / Strip / Boxplot        [bivariate.md]
│   ├── Over time / ordered index        → Line chart                     [bivariate.md]
│   └── Part-of-whole composition        → Pie ONLY if 2–3 categories, else Bar [univariate.md]
│
├── Relationships among MANY variables
│   ├── Pairwise correlations            → Correlation heatmap (flag |r|>0.85) [bivariate.md]
│   ├── All pairs + per-feature shape    → Pairplot (≤6–8 features, color by target) [multivariate.md]
│   ├── All features at once, patterns   → Parallel coordinates            [multivariate.md]
│   └── Exactly 3 features               → 3D scatter, interactive only — else 3× 2D projections [multivariate.md]
│
└── TARGET vs FEATURES (pre-modeling EDA)
    ├── Numeric feature, regression target      → Scatter + regression line
    ├── Numeric feature, classification target  → KDE per class, check overlap
    ├── Categorical feature, either target       → Bar chart of mean/count target per category
    └── Everything together                      → Pairplot colored by target
```

If still ambiguous after this table, default to: numeric+numeric → scatter, numeric+categorical → violin, categorical+categorical → grouped bar, single numeric → histogram+KDE, single categorical → bar. These five cover the large majority of real requests.
