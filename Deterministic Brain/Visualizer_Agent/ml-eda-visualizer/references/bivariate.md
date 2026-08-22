# Bivariate Visualization Reference

Purpose: relationships and dependencies between two variables.

## Scatter plot (numeric vs numeric)

Pearson `r = Σ(xᵢ-x̄)(yᵢ-ȳ) / √[Σ(xᵢ-x̄)² Σ(yᵢ-ȳ)²]`, range [-1,1], **linear relationships only**.

**Always scatter-check even when r≈0** — a U-shape or other nonlinear relationship gives r near 0 but is clearly visible and clearly modelable (with the right features/transform). Correlation hides what scatter reveals: nonlinearity, outliers, clusters, heteroscedasticity, floor/ceiling truncation.

**Heteroscedasticity check (matters for linear regression validity):** constant residual spread across x = homoscedastic (assumption holds). Funnel/cone shape = heteroscedastic (violates linear regression assumption) → log-transform y, weighted regression, or a robust regression method.

**Overplotting (large n):** use alpha transparency, jitter, hexbin, or 2D KDE contours instead of a solid blob.

## Strip plot (numeric vs categorical, small–medium n)

Raw points per category, optionally jittered to avoid vertical stacking. Shows sample size per group directly (unlike boxplot). Overplots badly at large n — switch to violin/box past a few hundred points per group.

## Violin plot (numeric vs categorical, preferred default over boxplot)

Boxplot (inner) + mirrored KDE (outer). Width at a height = density there.

**Use violin instead of boxplot whenever bimodality is plausible** — a violin shows two bulges where a boxplot shows nothing unusual. Works fine at large n (unlike strip plot). This is the general-purpose "compare a numeric variable across groups" chart; drop to boxplot only when you specifically need to declutter many groups side by side.

## Line chart (ordered/time-indexed only)

Connects points in sequence — implies values exist between them, so **only use when x has a real order** (time, sequence). Never for unordered categories (connecting "Mumbai" to "Delhi" implies a journey that doesn't exist — use bar chart for unordered categoricals).

Reveals: trend (long-term direction), seasonality (regular repeating pattern), sudden structural breaks, stationarity. Multiple lines: crossing = one group overtook another; parallel = same trend different level; diverging = groups splitting apart.

## Correlation heatmap

`R[i,j]` = Pearson r between feature i and j, symmetric, diagonal = 1.

**Multicollinearity threshold: |R[i,j]| > 0.85.** Consequence for linear models: unstable coefficients (small data changes → large coefficient swings), inflated standard errors, unreliable t-tests, model can't attribute effect to either feature individually. **Action: drop one of the pair** — prefer dropping the one with weaker |correlation to target| if target is in the matrix; otherwise use domain relevance the user supplies.

Feature-target row/column: high |r| → linear signal, worth keeping; low |r| → weak *linear* signal only (nonlinear relationship not ruled out — cross-check with scatter or KDE-by-class before dropping a feature on correlation alone).

**Limitation:** Pearson is linear-only. A perfectly nonlinear relationship can show r=0 in the same matrix that flags a spurious linear pair — never drop a feature from the heatmap alone without a scatter/KDE sanity check on the important ones.
