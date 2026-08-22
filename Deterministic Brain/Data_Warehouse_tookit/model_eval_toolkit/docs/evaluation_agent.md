# EvaluationAgent

code: `code/evaluation_agent.py`

`EvaluationAgent` sits on top of `classification_metrics`,
`regression_metrics`, and `clustering_metrics`. It answers the question that
usually comes *before* any metric computation: **which metrics should I even
look at for this task?** — and it flags the classic evaluation mistakes that
quietly invalidate a "good" score.

## `recommend_metrics(task, **flags)`

`task` is one of `"classification"`, `"regression"`, `"clustering"`.

Returns a dict: `{"primary": [...], "also_report": [...], "reasoning": str}`.

### Classification flags
| Flag | Meaning |
|---|---|
| `imbalanced` | classes are not roughly equal |
| `positive_rate` | fraction of samples that are the positive class (0–1) |
| `fn_costly` | missing a real positive is expensive (disease, fraud) |
| `fp_costly` | a false alarm is expensive (spam, legal accusation) |

Logic mirrors the standard decision guide: balanced classes → accuracy + F1;
imbalanced → recall or precision depending on which error is costly, plus
ROC-AUC (or PR-AUC if `positive_rate < 0.10`).

### Regression flags
| Flag | Meaning |
|---|---|
| `outliers_in_target` | target has extreme values |
| `large_errors_worse` | big misses are disproportionately bad |
| `comparing_feature_sets` | comparing models with different numbers of features |

### Clustering flags
| Flag | Meaning |
|---|---|
| `has_ground_truth` | true labels are available for evaluation |
| `choosing_k` | selecting the number of clusters |
| `comparing_algorithms` | comparing different clustering methods |

### Example

```python
from evaluation_agent import EvaluationAgent

agent = EvaluationAgent()
agent.recommend_metrics(
    "classification", imbalanced=True, positive_rate=0.05, fn_costly=True
)
# -> {"primary": ["recall", "pr_auc"],
#     "also_report": ["confusion_matrix", "fbeta(beta=2)"],
#     "reasoning": "Classes are imbalanced ... Missing positives is costly "
#                  "-> prioritize recall. Positive class < 10% of data -> "
#                  "prefer PR-AUC over ROC-AUC."}
```

## `evaluate_classification/regression/clustering(...)`

Thin pass-throughs to `classification_report`, `regression_report`, and
`clustering_report` respectively — convenient when you want recommendation
and computation from a single object.

## `check_common_mistakes(task, **context)`

Pass whatever you know about how the evaluation was actually done; it
returns a list of warning strings for each mistake pattern it detects.
Unrecognized/omitted keys are simply ignored, so partial context is fine.

| Context key | Flags |
|---|---|
| `evaluated_on_training_data` | model scored on data it memorized |
| `used_only_accuracy` + `class_balance` | accuracy-only reporting on imbalanced data |
| `threshold_tuned_on_test_set` | test set silently became a validation set |
| `used_equal_error_costs` + `costs_actually_asymmetric` | ignoring that FN/FP costs differ |
| `metrics_reported` (list, len ≤ 1) | single-metric reporting hides bias |
| `scaler_fit_on == "train+test"` | preprocessing data leakage |
| `train_score`, `val_score` | overfitting gap (flagged if > 0.15) |

### Example

```python
agent.check_common_mistakes(
    "classification",
    used_only_accuracy=True,
    class_balance=0.95,
    train_score=0.99,
    val_score=0.65,
    metrics_reported=["accuracy"],
)
# -> [
#   "Mistake: relying on accuracy alone with a majority class at ~95%. ...",
#   "Mistake: reporting a single metric. ...",
#   "Possible overfitting: training score (0.990) is much higher than
#    validation score (0.650), a gap of 0.340. ...",
# ]
```

## The Seven Mistakes This Checks For

1. Evaluating on training data.
2. Using accuracy alone on imbalanced data.
3. Tuning the decision threshold on the test set.
4. Ignoring asymmetric error costs (FN vs. FP).
5. Reporting a single metric.
6. Data leakage from fitting preprocessors on train+test combined.
7. Not checking the train-vs-validation performance gap (overfitting).
