# Distance & Proximity Metrics

code: `code/distance_metrics.py`

Foundation underneath K-Means, KNN, hierarchical clustering, and DBSCAN:
before you can group or compare records, you need to define what "similar"
means.

## Central Tendency

| Measure | Formula / definition | Robust to outliers? | Use when |
|---|---|---|---|
| Mean | `sum(x) / n` | No | Symmetric distribution, no outliers |
| Median | middle value when sorted | Yes | Skewed distribution or outliers present |
| Mode | most frequent value | Yes (for its purpose) | Nominal/categorical data — the only valid measure |
| Midrange | `(min + max) / 2` | No (uses only 2 points) | Quick rough estimate |

## Dispersion

| Measure | Formula | Robust to outliers? |
|---|---|---|
| Range | `max - min` | No |
| IQR | `Q3 - Q1` | Yes (ignores extreme 25% each side) |
| Five-number summary | `min, Q1, median, Q3, max` | Mostly (what a boxplot draws) |
| Variance | `mean((x - mean(x))^2)` | No |
| Standard deviation | `sqrt(variance)` | No, but same units as data |

**Empirical Rule** (normal distributions): ~68% of data within ±1 SD, ~95%
within ±2 SD, ~99.7% within ±3 SD of the mean — the basis for 3-SD Z-score
outlier detection.

## Proximity for Nominal Attributes — Simple Matching Coefficient

```
sim(p,q) = m / M          dissim(p,q) = (M - m) / M
```
`m` = number of attributes where `p` and `q` match, `M` = total attributes.
Only checks equality — no notion of "how far apart" two nominal values are.

## Proximity for Binary Attributes

Build a 2x2 contingency table for two binary vectors `p`, `q`:

```
              q=1   q=0
    p=1        a     b
    p=0        c     d
```
`a` = both 1, `b` = p=1/q=0, `c` = p=0/q=1, `d` = both 0.

| Coefficient | Formula | Use when |
|---|---|---|
| Simple Matching (SMC) | `(a+d) / (a+b+c+d)` | **Symmetric** attributes — both values equally meaningful (e.g. a yes/no survey answer) |
| Jaccard | `a / (a+b+c)` | **Asymmetric** attributes — presence (1) is meaningful/rare, absence (0) is common/uninformative (e.g. market-basket purchases, disease-positive tests) |

**Why Jaccard drops `d`:** in a 1000-item market basket, ~997 items neither
customer bought. Including that huge `d` term in SMC would make every pair
of customers look ~99.7% similar regardless of what they actually bought —
useless. Jaccard only counts what was actually meaningfully present.

## Dissimilarity for Numeric Data

| Distance | Formula | Notes |
|---|---|---|
| Euclidean | `sqrt(sum((p_i - q_i)^2))` | Straight-line distance. Sensitive to feature scale — **standardize first** (`standardize()`), or large-scale features dominate. Squaring amplifies outliers. |
| Manhattan (City Block) | `sum(\|p_i - q_i\|)` | Sum of axis-aligned travel, no diagonal shortcut. More robust to outliers (no squaring) and to high-dimensional/sparse data than Euclidean. |
| Minkowski | `(sum(\|p_i - q_i\|^r))^(1/r)` | Generalizes both: `r=1` → Manhattan, `r=2` → Euclidean, `r=inf` → Chebyshev (only the single largest coordinate difference matters). |

Euclidean distance is always <= Manhattan distance between the same two
points (straight line is the shortest path).

**Connection to clustering (Phase 6):** K-Means uses Euclidean distance for
its assignment step, which is exactly why it produces *spherical* clusters —
points equidistant from a centroid under Euclidean distance form a circle/
sphere. Under Manhattan distance, equidistant points form a diamond instead.
KNN's predictions similarly depend entirely on which distance metric is
chosen — it's a real hyperparameter, not an afterthought.

## Function Reference

```python
mean(x); median(x); mode(x); midrange(x)
data_range(x); quartiles(x); iqr(x); five_number_summary(x)
variance(x, ddof=0); std_dev(x, ddof=0)

simple_matching_coefficient_nominal(p, q)   # -> (similarity, dissimilarity)
simple_matching_coefficient_binary(p, q)    # symmetric binary
jaccard_coefficient(p, q)                   # asymmetric binary

euclidean_distance(p, q)
manhattan_distance(p, q)
minkowski_distance(p, q, r)
standardize(X)   # z-score per column, run before distance calcs on mixed-scale features
```
