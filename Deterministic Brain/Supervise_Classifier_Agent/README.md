# Supervised Classifier Agent — Reference Repo

A complete, agent-readable reference for classification models: theory
docs, dual implementations (from-scratch NumPy + scikit-learn wrapper),
and runnable examples for every model. Built from Phase 5 classification
theory (Logistic Regression → SVM).

## Structure
```
supervised_classifier_agent/
├── README.md                          ← you are here
├── code/                              ← implementations, both versions per model
│   ├── utils.py                       shared: split, StandardScaler, metrics
│   ├── logistic_regression.py         LogisticRegressionScratch / LogisticRegressionSklearn
│   ├── naive_bayes.py                 GaussianNBScratch / NaiveBayesSklearn (gaussian|multinomial|bernoulli)
│   ├── knn.py                         KNNClassifierScratch / KNNSklearn
│   ├── decision_tree.py               DecisionTreeScratch / DecisionTreeSklearn
│   ├── random_forest.py               RandomForestScratch / RandomForestSklearn
│   ├── boosting.py                    GradientBoostingScratch / BoostingSklearn (sklearn|xgboost|lightgbm)
│   └── svm.py                         LinearSVMScratch / SVMSklearn
├── docs/                              ← theory reference, one file per model
│   ├── 01_logistic_regression.md
│   ├── 02_naive_bayes.md
│   ├── 03_knn.md
│   ├── 04_decision_tree.md
│   ├── 05_random_forest.md
│   ├── 06_boosting.md
│   ├── 07_svm.md
│   └── 08_model_selection_guide.md    ← START HERE for "which model do I use"
└── examples/                          ← runnable end-to-end demos
    ├── example_logistic_regression.py
    ├── example_naive_bayes.py
    ├── example_knn.py
    ├── example_decision_tree.py
    ├── example_random_forest.py
    ├── example_boosting.py
    ├── example_svm.py
    └── example_run_all.py             ← trains/evaluates all 7 models side by side
```

## For an AI agent reading this repo
1. Read `docs/08_model_selection_guide.md` first — it routes any
   classification task to the right model and the right code file.
2. Every "Scratch" class and its "Sklearn" counterpart share the exact
   same API: `fit(X, y)`, `predict(X)`, `predict_proba(X)`. They are
   drop-in replacements for each other — pick Scratch to show/verify
   the math, pick Sklearn for production use.
3. `code/utils.py` has no external dependencies beyond NumPy; the
   "Scratch" classes only need NumPy too. The "Sklearn" classes need
   `scikit-learn` (and optionally `xgboost` / `lightgbm` for
   `boosting.py`'s alternate backends).
4. Each `docs/0N_*.md` file mirrors its `code/*.py` file 1:1, so you can
   pull in exactly the theory needed to explain or extend a given model
   without loading the whole repo into context.

## Quick start
```bash
pip install scikit-learn --break-system-packages   # only needed for *Sklearn classes
python3 examples/example_run_all.py                 # trains & compares all 7 models
python3 examples/example_svm.py                     # or run any single model demo
```

## Models covered
Logistic Regression · Naive Bayes (Gaussian/Multinomial/Bernoulli) ·
KNN · Decision Tree · Random Forest · Gradient Boosting
(sklearn/XGBoost/LightGBM-compatible) · SVM (linear + kernel)

## Design notes
- **Binary classification focus.** Multiclass extensions (softmax,
  one-vs-rest) are documented in each `docs/0N_*.md` file; the sklearn
  wrapper classes handle multiclass automatically, the Scratch classes
  are intentionally kept binary for clarity — extend them with softmax
  per `docs/01_logistic_regression.md` if you need multiclass from scratch.
- **Feature scaling**: mandatory before KNN and SVM (`utils.StandardScaler`),
  optional but harmless for the rest.
- **Class imbalance**: handled via `class_weight` in Random Forest /
  sklearn boosting, or `scale_pos_weight` for XGBoost — see
  `docs/05_random_forest.md` and `docs/06_boosting.md`.
