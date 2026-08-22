# Dimensionality Reduction Agent

A multi-skill agent for Machine Learning Dimensionality Reduction —
covers **Feature Selection** (Filter / Wrapper / Embedded) and
**Feature Extraction** (Linear / Nonlinear / Neural / Domain-specific),
matching the complete tree structure.

Each skill file below is self-contained: theory in 1 line + ready-to-run
Python code (mostly `scikit-learn`, plus a few specialized libs).

## Skill Map

| File | Covers |
|---|---|
| `01_feature_selection_filter.md` | Variance, Correlation, Chi2, ANOVA, MI, Fisher, VIF, mRMR, Relief |
| `02_feature_selection_wrapper.md` | Forward/Backward/Stepwise, RFE/RFECV, Exhaustive, Genetic/PSO |
| `03_feature_selection_embedded.md` | Lasso/Ridge/ElasticNet, Tree Importance, Permutation, SHAP |
| `04_feature_extraction_linear.md` | PCA, LDA, SVD, ICA, Factor Analysis, Random Projection |
| `05_feature_extraction_nonlinear.md` | t-SNE, UMAP, Isomap, LLE, MDS, Kernel PCA, Autoencoders, VAE |
| `06_feature_extraction_domain.md` | Image (HOG/SIFT), Text (TF-IDF/LSA), Time Series (FFT/Wavelet), Graph (Node2Vec) |

## How to use this agent

1. Pick the skill file matching your data type / goal.
2. Copy the snippet — every snippet assumes `X` (features, DataFrame/array)
   and `y` (target, for supervised methods) already exist.
3. Install once:

```bash
pip install scikit-learn scipy statsmodels umap-learn mlxtend shap \
            skrebate xgboost lightgbm gensim tsfresh --break-system-packages
```

## Decision shortcut

```
Need to KEEP original features (interpretability)?
 └── YES → FEATURE SELECTION
       ├── No model available / want it fast     → FILTER   (01)
       ├── Have a model, want best subset         → WRAPPER  (02)
       ├── Model trains once, selection is free   → EMBEDDED (03)
 └── NO, ok to create new combined features?
       ├── Need linear, interpretable axes        → LINEAR EXTRACTION (04)
       ├── Data lies on curved manifold            → NONLINEAR EXTRACTION (05)
       ├── Domain-specific (image/text/ts/graph)   → DOMAIN EXTRACTION (06)
```
