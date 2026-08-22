---
name: ml-eda-visualizer
description: Perform exploratory data analysis (EDA) and generate ML-ready visualizations for a dataset — histograms, KDE, boxplots, violin plots, scatter plots, correlation heatmaps, pairplots, parallel coordinates. Use this skill whenever the user shares a dataset (CSV/DataFrame/DB table) and asks to "explore the data," "visualize," "check distributions," "find outliers," "check multicollinearity," "understand feature relationships," "prep for modeling," or asks any EDA question before training an ML model — even if they don't say "EDA" or "visualization" explicitly. Also trigger when the user asks which chart to use for a given variable type, or asks you to interpret a distribution/scatter/correlation matrix they've shown you.
---

# ML EDA Visualizer

Agent skill for exploratory data analysis before ML modeling. Two jobs, always both: **(1) pick the right chart, (2) state what it means for modeling** — a chart without an interpretation is not EDA, it's decoration.

## Core principle

Numbers alone hide structure (Anscombe's Quartet: four datasets, identical mean/variance/correlation/regression line, four different shapes — one is linear, one is curved and breaks linear regression, two are outlier-driven). Never report summary statistics without a visual, and never show a visual without stating the modeling consequence (transform needed, feature to drop, assumption violated, etc.).

## Workflow

1. **Profile first.** Get dtypes, shape, null rates, cardinality per column before choosing anything. Don't guess column types.
2. **Route each column** through the decision tree below.
3. **Generate the chart** using `scripts/eda_codegen.py` (import and call — don't hand-roll matplotlib calls that duplicate what's already there) or produce equivalent code inline if the user wants a copy-pasteable script instead of an executed result.
4. **Interpret, don't just render.** For every chart, state: shape/pattern observed → what it implies for the model (transform? drop? investigate collection artifact? feature is/isn't discriminative?).
5. **Run the full checklist** in `references/eda_checklist.md` before declaring EDA done, if the user wants a complete pass rather than one chart.

## Decision tree (read the matching reference file, don't guess)

```
One numeric variable, distribution?          → references/univariate.md (histogram + KDE)
One numeric variable, outliers?               → references/univariate.md (boxplot)
One categorical variable, frequency/imbalance?→ references/univariate.md (bar chart — NEVER pie beyond 3 categories)
Numeric vs numeric relationship?              → references/bivariate.md (scatter)
Numeric vs categorical (compare groups)?      → references/bivariate.md (violin > boxplot if bimodality suspected; strip if n small)
Ordered/time-indexed variable?                → references/bivariate.md (line chart)
Pairwise correlation / multicollinearity?     → references/bivariate.md (heatmap)
All numeric features vs each other + target?  → references/multivariate.md (pairplot, ≤8 features)
All features simultaneously, class patterns?  → references/multivariate.md (parallel coordinates)
"Which chart do I use for X?"                 → references/chart_selection.md (decision framework, no reading required first)
"Is this EDA complete before I model?"        → references/eda_checklist.md
```

## Hard rules (violating these produces wrong or misleading EDA)

- Never use a pie chart with >3 categories or near-equal slices — use a bar chart. Humans can't compare angles.
- Never trust Pearson correlation alone for "no relationship" — r=0 can hide a perfect nonlinear (e.g., U-shaped) relationship. Always scatter-check.
- Never bin a histogram by eye — use Freedman-Diaconis (`scripts/eda_codegen.py:recommend_bins`), it's robust to outliers unlike Scott's rule.
- Never use a static 3D scatter plot as a final deliverable — projection hides depth and misleads. Show 3 paired 2D projections instead, or use it only for live/interactive PCA exploration.
- Never rely on a boxplot alone to rule out bimodality — it cannot show two peaks. Pair it with a KDE or histogram.
- Flag multicollinearity at |r| > 0.85 between features and say which one you'd drop and why (the one with the weaker feature-target relationship, ties broken by domain relevance the user states).
- For a classification target: always show class balance (bar chart of target) before anything else — it changes every later modeling decision (metric choice, resampling).

## Files in this skill

- `scripts/eda_codegen.py` — importable Python module. Functions for every chart type below, each returns (fig, interpretation_dict). Use this instead of writing matplotlib/seaborn calls from scratch.
- `references/univariate.md` — histogram, KDE, boxplot, bar, pie: math, when to use, what shape means, bin-count rules, skew/kurtosis formulas.
- `references/bivariate.md` — scatter, strip, violin, line, correlation heatmap: math, heteroscedasticity, multicollinearity thresholds.
- `references/multivariate.md` — pairplot, parallel coordinates, 3D scatter: construction, reading patterns, limitations.
- `references/chart_selection.md` — flat decision framework, answers "what chart do I use" without reading the theory files.
- `references/eda_checklist.md` — the 9-step pre-modeling EDA checklist, feature-by-feature and target-by-target.
