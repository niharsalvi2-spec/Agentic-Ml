# Pre-Modeling EDA Checklist

Run this in order. Don't skip to modeling until all applicable steps are done — each one changes a downstream modeling decision (imputation strategy, transform, feature drop, metric choice).

## Per feature
1. **Histogram + KDE** — distribution shape, skewness. Determines imputation strategy (mean vs median) and whether a log/power transform is needed before a linear model.
2. **Boxplot** — outliers, IQR. Determines whether to cap, transform, or leave outliers (tree-based models tolerate them; linear/distance-based models don't).
3. **Bar chart** (if categorical) — cardinality, imbalance within the feature itself. High-cardinality categoricals need encoding strategy (target encoding, hashing) instead of naive one-hot.

## Per feature pair
4. **Correlation heatmap** — multicollinearity, |r|>0.85 threshold. Determines which redundant feature to drop before a linear model (tree models are more tolerant but it still hurts interpretability).
5. **Scatter plots of the high-correlation pairs found in step 4** — verify it's actually linear and not a heatmap number driven by a different relationship shape or an outlier (recall Anscombe's Quartet — same r, different truth).

## Target variable
6. **Bar chart** (classification) — class balance. Determines metric choice (accuracy is misleading under imbalance — prefer F1/precision-recall/AUC) and whether resampling (SMOTE, class weights) is needed. Do this before anything else once you know the task is classification.
7. **Histogram** (regression) — target distribution shape. A heavily skewed target often benefits from a log-transform before fitting, and changes which error metric (MAE vs RMSE) is appropriate.

## Everything together
8. **Pairplot colored by target** — which features visually separate classes / correlate with a regression target. Fast pre-model signal on feature usefulness, cheap compared to running a model.
9. **Parallel coordinates** — multi-feature patterns invisible in any pairwise view; useful when no single feature or pair separates classes but a combination might.

## Reporting format
For each step, state three things, not just the chart: **(a)** what was observed, **(b)** what it implies for modeling, **(c)** the concrete action (transform / drop / resample / investigate collection / none needed). A chart with no (b) and (c) is incomplete EDA.
