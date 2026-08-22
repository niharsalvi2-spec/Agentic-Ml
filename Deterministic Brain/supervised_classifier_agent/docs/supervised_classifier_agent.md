# Supervised Classifier Agent (Orchestrator)

code: `code/supervised_classifier_agent.py` — class `SupervisedClassifierAgent`

## Purpose

The single entry point an agent should import. It registers all 14 classifier
implementations (7 classifiers × sklearn/from-scratch) behind one interface
and provides three capabilities:

1. **`recommend(...)`** — rule-based scoring of every registered model against
   stated dataset characteristics, returned WITHOUT training anything.
2. **`compare_all(...)`** — actually trains + evaluates a set of models on
   real data and ranks them empirically (accuracy, precision, recall, F1, timing).
3. **`get_model(name)`** / **`describe_model(name)`** — direct access to a
   single model instance or its metadata.

## Registry Keys

```
logistic_regression_sklearn   logistic_regression_scratch
naive_bayes_sklearn           naive_bayes_scratch
knn_sklearn                   knn_scratch
decision_tree_sklearn         decision_tree_scratch
random_forest_sklearn         random_forest_scratch
boosting_sklearn              boosting_scratch
svm_sklearn                   svm_scratch
```

## `recommend()` — How Scoring Works

Each registered model's `METADATA` dict is scored against the characteristics
you pass in (interpretability need, suspected nonlinearity, outliers, class
imbalance, dataset size/dimensionality, inference-speed need). Points are
awarded per matching characteristic (see inline comments in the source for
exact weights). The top `top_k` models are returned with a plain-English
`reasons` list — this mirrors the decision guide in `docs/README.md` but lets
an agent call it programmatically instead of parsing markdown.

```python
agent.recommend(
    n_samples=5000, n_features=40,
    need_interpretability=True, need_proba=True,
    suspect_nonlinear=True, has_outliers=False,
    is_imbalanced=False, need_fast_inference=True,
)
# -> [{"model": "decision_tree_sklearn", "score": 7.5, "reasons": [...]}, ...]
```

## `compare_all()` — How Benchmarking Works

Trains every requested model on `(X_train, y_train)`, predicts on `X_test`,
and reports accuracy / weighted precision / weighted recall / weighted F1 /
train time / inference time. Wrap `X` in a scaler (e.g.
`sklearn.preprocessing.StandardScaler`) before calling this — several models
(KNN, SVM, Logistic/Linear-based) are scale-sensitive; tree-based models are
not but scaling never hurts them.

```python
results = agent.compare_all(X_train, y_train, X_test, y_test)
# results is a list of dicts sorted by accuracy, descending
```

Any model that throws during fit/predict is captured in the results list with
`"error": "<message>"` instead of crashing the whole benchmark run — useful
when e.g. `ScratchBoosting`/`ScratchSVM` are called on multiclass data (they
only support binary classification, this is documented in their METADATA).

## Design Notes for Extending the Agent

- To add a new classifier: create `Sklearn<Name>`/`Scratch<Name>` in a new
  file under `/code` following `base_classifier.BaseClassifier`'s contract
  and `METADATA` schema, then add both to `self.registry` in
  `SupervisedClassifierAgent.__init__`.
- The `METADATA` schema is intentionally flat and boolean/enum-heavy so an
  agent (or simple rule engine) can reason over it without an LLM call.
