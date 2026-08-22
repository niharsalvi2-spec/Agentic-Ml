# Skill: Regression — Theory Notes (Plain Language)
Companion to the code files — the *why*, in simple terms, for quick recall before exams or interviews.

## Universal Learning Loop
Every model: **predict → measure error (loss) → adjust params → repeat.**
- **MSE** squares errors → punishes big mistakes harder, sensitive to outliers.
- **MAE** — treats all errors equally → robust to outliers.
- Rule: MSE when big errors are catastrophic; MAE when one outlier shouldn't dominate.

## A.1 Linear Family

**Linear Regression** — draws the best straight line/hyperplane. Weight `wᵢ` = "how much target moves per unit of feature i, others held constant." Solved exactly via Normal Equation `w=(XᵀX)⁻¹Xᵀy`, or iteratively via Gradient Descent when p is huge.

**5 assumptions to check, in order:**
1. Linearity — scatter plot straight, not curved
2. No multicollinearity — VIF < 5–10
3. Homoscedasticity — residual spread constant across predictions
4. Independent errors — no pattern in residuals-vs-row-index (fails on time series)
5. Normal residuals — Q-Q plot straight line (matters less at large n, CLT)

**Ridge (L2)** — tug-of-war between "fit data" and "keep weights small." Never zeroes weights (circle constraint, no corners) → keeps all features, fixes instability from correlated features by making `XᵀX+λI` always invertible.

**Lasso (L1)** — same idea but diamond-shaped constraint → corners → some weights snap to exactly zero → automatic feature selection. Weakness: with correlated features, picks one arbitrarily and drops the rest (unstable).

**Elastic Net** — L1 (sparsity) + L2 (stability). Fixes Lasso's correlated-feature weakness via the "grouping effect": correlated, relevant features are kept together instead of one being arbitrarily dropped.

**Quick picker:**
| Situation | Use |
|---|---|
| Few features, no correlation | Linear Regression |
| Correlated features, keep all | Ridge |
| Many irrelevant features | Lasso |
| Correlated *groups* of relevant features | Elastic Net |

## A.2 Tree-Based

**Decision Tree** — repeatedly asks yes/no questions that split data into the most *similar* groups (lowest variance). Leaf prediction = average of samples in that leaf. Left unconstrained → memorizes training data (overfits); control with `max_depth`, `min_samples_leaf`.

**Random Forest** — many trees, each on a bootstrap sample, each split considers only a random feature subset. Averaging reduces variance; the *feature subsampling* is what decorrelates trees so averaging actually helps (fully correlated trees don't benefit from averaging). OOB error = free validation score from the ~37% of data each tree never saw.

**Gradient Boosting** — trees built *sequentially*, each new tree predicts the *residuals* (mistakes) of the current ensemble, added in with a small `learning_rate`. Lower learning rate + more trees = better generalization, but unlike Random Forest, too many trees *can* overfit — use early stopping on validation error.

**XGBoost** — Gradient Boosting + second-order (Hessian) info for more precise leaf values, plus explicit regularization (`γ` penalizes leaf count, `λ` shrinks leaf weights) and system-level speed tricks (histogram binning, column subsampling, parallel split search).

**LightGBM** — grows leaf-wise (always split the single most error-reducing leaf) instead of level-wise → fewer splits for same accuracy, but risk of overfitting on small data. GOSS keeps high-error samples, subsamples low-error ones. EFB bundles mutually-exclusive sparse features (e.g. one-hot columns) into one. Fastest on large tabular data.

**CatBoost** — fixes target leakage in categorical encoding via *ordered target statistics* (a row's encoding only uses rows that came before it in a random permutation). Pass raw categorical columns directly — no manual encoding needed. Best when you have many/high-cardinality categoricals.

**Picking among boosted trees:** small-medium data & want stability → XGBoost. Large data & need speed → LightGBM. Heavy categoricals → CatBoost.

## A.3 Other Methods

**SVR** — builds an `ε`-wide tube around the prediction line; errors *inside* the tube cost nothing, only points outside (support vectors) matter. `C` controls how strictly violations are punished, `ε` controls tube width. Kernel trick (RBF, poly) lets a linear method fit curved relationships without explicitly transforming data.

**KNN Regressor** — no training, just memorizes data; prediction = average of K nearest neighbors (optionally distance-weighted). K=1 overfits, K=n underfits. Always standardize features first (distance is scale-sensitive). Breaks down above ~20 dimensions (curse of dimensionality — all points become equidistant) — reduce dimensions first.

**Bayesian Regression** — instead of one best weight vector, keeps a full probability distribution (prior → posterior) over weights, so predictions come with calibrated uncertainty (`prediction ± interval`). Most valuable with small datasets or when you need to know *how much to trust* a given prediction; converges to ordinary regression as data grows.

## End-to-End Decision Guide

```
Small data (<500 rows)
 ├── correlated features        → Ridge
 ├── need uncertainty           → Bayesian Regression
 └── want feature selection     → Lasso / Elastic Net

Medium data (500–50k rows)
 ├── want interpretability       → Linear / Ridge / Lasso
 ├── nonlinear, best accuracy    → XGBoost / LightGBM
 ├── heavy categoricals          → CatBoost
 └── general-purpose default     → Random Forest

Large data (>50k rows)
 ├── need raw speed              → LightGBM
 ├── many categoricals           → CatBoost
 └── need explainability         → Random Forest + SHAP

Special cases
 ├── need prediction uncertainty → Bayesian Regression
 ├── small + nonlinear           → SVR (RBF)
 └── zero training time budget   → KNN

Not sure? → baseline Random Forest → compare vs XGBoost/LightGBM →
            check if a linear model is competitive (prefer simpler if so)
```
