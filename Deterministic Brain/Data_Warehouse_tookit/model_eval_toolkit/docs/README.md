# Model Evaluation Toolkit — Package Guide

AGENT-READABLE INDEX
=====================
Pure-numpy, from-scratch implementations of the standard classification,
regression, and clustering evaluation metrics, plus an `EvaluationAgent` that
recommends which metrics to use for a task (before you compute anything) and
flags common evaluation mistakes.

```
/code
  classification_metrics.py   confusion matrix, accuracy/precision/recall/specificity,
                               F1/F-beta, ROC curve+AUC, PR curve+AUC, multiclass
                               macro/micro/weighted averaging, classification_report()
  regression_metrics.py       MAE, MSE, RMSE, R2, Adjusted R2, regression_report()
  clustering_metrics.py       Silhouette, Davies-Bouldin, Inertia (internal);
                               Adjusted Rand Index, NMI (external); clustering_report()
  evaluation_agent.py         EvaluationAgent — recommend_metrics(), evaluate_*(),
                               check_common_mistakes()
  distance_metrics.py         Central tendency & dispersion (mean/median/mode/variance/
                               std/IQR); proximity measures (Simple Matching Coefficient,
                               Jaccard); Euclidean/Manhattan/Minkowski distance; standardize()
  association_rules.py        Apriori algorithm (frequent itemset mining), Support/
                               Confidence/Lift, generate_rules()
  data_cube.py                DataCube — Roll-Up, Drill-Down, Slice, Dice, Pivot over a
                               flat/denormalized fact table with dimension hierarchies
  dw_advisor.py                DataWarehouseAdvisor-style functions: recommend_schema()
                               (Star/Snowflake/Fact Constellation), recommend_olap_
                               implementation() (ROLAP/MOLAP/HOLAP), recommend_operation(),
                               recommend_storage() (Warehouse/Lake/Lakehouse), plus
                               FactTable/DimensionTable/StarSchema/SnowflakeSchema/
                               FactConstellationSchema dataclasses
/docs
  README.md                   this file — decision guide
  classification_metrics.md   theory, formulas, when to use each metric
  regression_metrics.md
  clustering_metrics.md
  evaluation_agent.md
  distance_metrics.md         central tendency/dispersion, nominal/binary proximity,
                               Euclidean/Manhattan/Minkowski theory
  association_rules.md        Apriori algorithm, Support/Confidence/Lift theory
  data_cube.md                data cube / OLAP operations theory + worked example
  dw_advisor.md                schema / OLAP implementation / operation / storage decision guide
/examples
  example_classification_metrics.py
  example_regression_metrics.py
  example_clustering_metrics.py
  example_evaluation_agent.py
  example_distance_metrics.py
  example_association_rules.py
  example_data_cube.py
  example_dw_advisor.py
```

Every metric function has been validated against `sklearn.metrics` on
synthetic data (bit-exact to floating-point precision, including
`average_precision_score`, ROC-AUC, multiclass macro/micro/weighted
precision/recall/F1, Silhouette, Davies-Bouldin, ARI, and NMI — see
`/examples` for the comparison scripts).

## The Golden Rule of Evaluation

**Never evaluate on training data** — the model has already memorized it.
Always hold out a test set (touched once) and, ideally, a separate validation
set for tuning. Use K-Fold cross-validation when data is limited.

```
Training set (60-70%)   -> model learns from this
Validation set (15-20%) -> tune hyperparameters, select model/threshold
Test set (15-20%)       -> final honest evaluation, touch only ONCE
```

## Complete Decision Guide — Choosing a Metric

```
CLASSIFICATION task:
|
+-- Balanced classes?
|   `-- Accuracy + F1 Score
|
+-- Imbalanced classes (common)?
|   +-- Rare event detection (fraud, disease):
|   |   `-- Recall (catch all positive cases) + PR-AUC
|   +-- False alarm cost is high (spam filter):
|   |   `-- Precision
|   `-- Balance both concerns:
|       `-- F1 Score or F-Beta Score
|
+-- Need threshold-independent evaluation?
|   +-- Balanced classes   -> ROC-AUC
|   `-- Imbalanced classes -> PR-AUC (use when positive class < 10% of data)
|
`-- Always report a Confusion Matrix alongside any single metric.

REGRESSION task:
|
+-- Outliers in target?          -> MAE (robust), report RMSE too
+-- Large errors much worse?     -> RMSE
+-- Want percentage/fit summary? -> R2
+-- Comparing different feature sets? -> Adjusted R2
`-- Complete reporting: MAE + RMSE + R2 together
    (MAE = typical error, RMSE = outlier sensitivity signal, R2 = fit quality)

CLUSTERING task:
|
+-- No ground truth -> Internal metrics:
|   +-- Choosing K            -> Elbow (Inertia) + Silhouette Score
|   `-- Comparing algorithms  -> Silhouette Score / Davies-Bouldin Index
|
`-- Ground truth available -> External metrics:
    +-- ARI -> chance-corrected agreement
    `-- NMI -> information-theoretic, robust to differing cluster/class counts
```

## Metric Comparison Tables

### Classification
| Metric | Range | Better | Sensitive to Imbalance | Use When |
|---|---|---|---|---|
| Accuracy | 0-1 | higher | Yes (misleading) | Balanced classes |
| Precision | 0-1 | higher | - | False positives costly |
| Recall | 0-1 | higher | - | False negatives costly |
| Specificity | 0-1 | higher | - | Negative-class detection matters |
| F1 | 0-1 | higher | Somewhat | Balance precision & recall |
| ROC-AUC | 0-1 | higher | Yes (can look inflated) | Balanced, threshold-free ranking |
| PR-AUC | 0-1 | higher | No - designed for it | Imbalanced, positive class < 10% |

### Regression
| Metric | Units | Outlier Sensitive | Use When |
|---|---|---|---|
| MAE | same as target | No (robust) | Outliers exist, want interpretable |
| MSE | squared units | Yes (heavily) | Large errors must be penalized |
| RMSE | same as target | Yes | Standard metric, penalize large errors |
| R2 | unitless (0-1) | Yes | Explain fit quality as a percentage |
| Adjusted R2 | unitless (0-1) | Yes | Comparing models with different features |

### Clustering
| Metric | Range | Better | Needs Ground Truth |
|---|---|---|---|
| Silhouette | -1 to 1 | higher | No |
| Davies-Bouldin | >= 0 | lower | No |
| Inertia (WCSS) | >= 0 | lower (with Elbow only) | No, K-Means only |
| Adjusted Rand Index | -1 to 1 | higher | Yes |
| NMI | 0 to 1 | higher | Yes |

## Distance / Proximity Quick Reference

| Data type | Measure | Notes |
|---|---|---|
| Nominal | Simple Matching Coefficient | `m / M`, equality-only check |
| Binary (symmetric) | Simple Matching Coefficient | both 0 and 1 equally meaningful |
| Binary (asymmetric) | Jaccard Coefficient | `a / (a+b+c)`, ignores "both absent" — use for market baskets, disease tests |
| Numeric, general | Euclidean | standardize features first |
| Numeric, high-dim/sparse/outlier-robust | Manhattan | no squaring, no diagonal shortcuts |
| Numeric, tunable | Minkowski (r=1 Manhattan, r=2 Euclidean, r=inf Chebyshev) | generalizes both |

See `docs/distance_metrics.md` for full theory.

## Association Rule Mining Quick Reference

```
Support(X)      = P(X occurs)
Confidence(X->Y)= P(Y | X) = Support(X∪Y) / Support(X)
Lift(X->Y)      = Confidence(X->Y) / Support(Y)   # >1 genuinely meaningful, =1 independent, <1 substitutes
```
Mine frequent itemsets with `apriori(transactions, min_support)` (prunes any
candidate itemset with an infrequent subset), then turn them into ranked
rules with `generate_rules(frequent_itemsets, transactions, min_confidence)`.
See `docs/association_rules.md` for the full worked example and algorithm
walkthrough.

## Data Warehouse Quick Reference

```
OLTP  = run the business right now (short transactions, normalized, current data)
OLAP  = analyze the business over time (complex aggregations, denormalized, historical data)

Schema:     Star (default) | Snowflake (storage/redundancy constrained) | Fact Constellation (multiple fact tables)
Implementation: ROLAP (large/flexible) | MOLAP (fast, dense/small) | HOLAP (hybrid, most common default)
Storage:    Data Warehouse (structured, known queries) | Data Lake (raw, ML/exploratory) | Lakehouse (both)

OLAP operations:
  Roll-Up       -> summarize, climb hierarchy up      (fewer rows)
  Drill-Down    -> detail, climb hierarchy down        (more rows)
  Slice         -> fix ONE dimension to one value       (drops that dimension from view)
  Dice          -> filter MULTIPLE dimensions to values (smaller sub-cube, same dim count)
  Pivot         -> rotate rows/columns                  (same data, new orientation)
  Drill-Across  -> jump to a related fact table via shared dimensions
```
`code/data_cube.py` implements all five in-view operations over an
in-memory cube; `code/dw_advisor.py` implements the four decision guides
above as callable functions, plus dataclasses to build the actual Star/
Snowflake/Fact-Constellation table shapes. See `docs/data_cube.md` and
`docs/dw_advisor.md` for full theory and worked examples.

## Common Evaluation Mistakes (checked by `EvaluationAgent.check_common_mistakes`)

1. **Evaluating on training data** — always use a held-out test set.
2. **Using accuracy on imbalanced data** — a majority-class model can score high while being useless.
3. **Tuning threshold on the test set** — turns your test set into a validation set; tune on validation, report test once.
4. **Ignoring class costs** — missing a cancer case vs. a false alarm are not equally bad; use F-beta or cost-sensitive evaluation.
5. **Reporting a single metric** — high R2/AUC can hide systematic bias; always check a confusion matrix or residual plot too.
6. **Data leakage in preprocessing** — fit scalers/encoders on train only, then transform test with those parameters.
7. **Not checking train vs. validation gap** — a large gap signals overfitting.
