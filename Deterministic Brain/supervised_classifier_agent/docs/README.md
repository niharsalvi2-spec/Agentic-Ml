# Supervised Classifier Agent — Package Guide

AGENT-READABLE INDEX
=====================
This package gives an agent (or a human) a consistent, swappable set of
classifiers, each available in two forms:
- **`Sklearn<Name>`** — production wrapper around scikit-learn
- **`Scratch<Name>`** — dependency-light, from-scratch numpy implementation

Every class shares one contract (see `code/base_classifier.py`):
`fit(X, y)`, `predict(X)`, `predict_proba(X)`, `score(X, y)`, and a class-level
`METADATA` dict the agent can read WITHOUT training anything, to decide which
model fits a task.

```
/code
  base_classifier.py               shared interface + METADATA schema
  logistic_regression.py           SklearnLogisticRegression, ScratchLogisticRegression
  naive_bayes.py                   SklearnNaiveBayes, ScratchNaiveBayes
  knn.py                           SklearnKNN, ScratchKNN
  decision_tree.py                 SklearnDecisionTree, ScratchDecisionTree
  random_forest.py                 SklearnRandomForest, ScratchRandomForest
  boosting.py                      SklearnBoosting, ScratchBoosting
  svm.py                           SklearnSVM, ScratchSVM
  supervised_classifier_agent.py   SupervisedClassifierAgent (orchestrator)
/docs
  README.md                        this file — decision guide
  logistic_regression.md ... svm.md   one reference doc per classifier
/examples
  example_<classifier>.py          standalone runnable usage example, one per classifier
  example_agent_recommend.py       ask the agent which model to use
  example_agent_compare_all.py     benchmark every model on one dataset
```

## Complete Decision Guide — Supervised Classification

```
What matters most for your task?
│
├── INTERPRETABILITY (need to explain "why")
│   ├── Data is roughly linear     → Logistic Regression
│   └── Data has clear rule splits → Decision Tree
│
├── RAW ACCURACY on tabular data
│   ├── Want it with minimal tuning        → Random Forest
│   └── Willing to tune, want best possible → Boosting (Gradient Boosting / AdaBoost)
│
├── HIGH-DIMENSIONAL data (features >> samples: text, genomics)
│   ├── Want interpretability → Logistic Regression (with regularization)
│   └── Want max separation   → SVM (linear or RBF kernel)
│
├── SMALL DATA / need a fast, cheap baseline
│   ├── Want calibrated-ish class priors     → Naive Bayes
│   └── Want to exploit local neighborhoods  → KNN (low-dim only)
│
├── NONLINEAR decision boundary suspected
│   ├── No feature engineering desired → Decision Tree / Random Forest / Boosting
│   └── Want a smooth margin           → SVM with RBF kernel
│
├── IMBALANCED classes
│   └── Logistic Regression / Random Forest / SVM with class_weight='balanced',
│       or resample the data before training any model
│
└── LARGE dataset (>100k rows)
    ├── Need fast train + fast inference → Logistic Regression, Naive Bayes
    └── Need fast train, ok w/ slower inference → Random Forest (n_jobs=-1)
        (avoid: plain KNN, kernel SVM, scratch implementations — all O(n^2)+ or worse)
```

## Model Comparison Table

| Classifier          | Linear/Nonlinear | Needs Scaling | Interpretable | Handles High-Dim | Train Speed | Inference Speed | Native Multiclass |
|----------------------|-------------------|----------------|----------------|-------------------|-------------|------------------|--------------------|
| Logistic Regression   | Linear            | Yes            | Yes            | Yes               | Fast        | Fast             | Yes                |
| Naive Bayes            | Nonlinear (mild)   | No             | Yes            | Yes               | Fast        | Fast             | Yes                |
| KNN                    | Nonlinear          | Yes            | Locally        | No                | Fast (lazy) | Slow             | Yes                |
| Decision Tree           | Nonlinear          | No             | Yes            | No                | Fast        | Fast             | Yes                |
| Random Forest           | Nonlinear          | No             | No             | No                | Medium      | Medium           | Yes                |
| Boosting (GBM/AdaBoost) | Nonlinear          | No             | No             | No                | Slow        | Medium/Fast      | Yes (GBM)          |
| SVM                     | Both (kernel-dep.) | Yes            | Kernel-dep.    | Yes               | Slow        | Medium           | Via OvO/OvR        |

## How the Agent Picks a Model

`SupervisedClassifierAgent.recommend(...)` scores every registered model
against the characteristics you provide (sample size, dimensionality,
interpretability needs, suspected nonlinearity, outliers, imbalance, inference
speed requirement) using the same rules as the decision tree above, and
returns the top-k candidates with human-readable reasons — no training
required to get a recommendation.

`SupervisedClassifierAgent.compare_all(...)` actually trains + evaluates a set
of models (or all of them) on your data and ranks them by accuracy/F1/timing,
for when you want an empirical answer instead of (or in addition to) the
rule-based recommendation.

## Evaluation Metrics Reference

- **Accuracy**: fraction correct. Misleading on imbalanced data.
- **Precision**: of predicted positives, how many were correct. Prioritize when false positives are costly.
- **Recall**: of actual positives, how many were found. Prioritize when false negatives are costly.
- **F1**: harmonic mean of precision and recall. Good single number for imbalanced data.
- Always check a confusion matrix, not just one aggregate metric, before shipping a model.
