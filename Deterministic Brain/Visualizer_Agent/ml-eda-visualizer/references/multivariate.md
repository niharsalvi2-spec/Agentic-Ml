# Multivariate Visualization Reference

## Pairplot (scatter matrix)

p×p grid: diagonal = univariate distribution per feature (hist/KDE), off-diagonal = pairwise scatter, color by target class if classification.

- **Hard limit: ≤6–8 features.** Beyond that it's unreadable — run feature selection or pick the top-correlated/most-important subset first, don't dump every column in.
- What to look for: diagonal shape (skew per feature), off-diagonal linear band (candidate multicollinearity pair — cross-check against the correlation heatmap), color separation in a scatter cell (that feature pair is discriminative for classification), tight diagonal scatter line (near-perfect collinearity, stronger signal than a heatmap number).

## Parallel coordinates

p vertical axes (one per feature, independently scaled), each observation = one polyline across them, colored by class.

- Reads patterns across **all** features at once — something no pairwise view (scatter, heatmap, pairplot) can do.
- Lines crossing between two adjacent axes → negative correlation between those two features; staying parallel → positive correlation.
- One color's lines clustering in a band on one axis → that feature discriminates that class.
- A single line diverging from all others → outlier observation (not just an outlier value on one axis).
- **Axis order isn't unique** — there's no single correct ordering, and apparent crossing patterns depend on it. Don't over-read one arrangement; try feature groupings that make domain sense.
- Gets cluttered at large n — use alpha transparency, or subsample for the plot (keep full data for modeling).

## 3D scatter — use with real caution

Three numeric axes, one point per observation.

- Only genuinely useful after dimensionality reduction (first 3 PCA components) or when 2D projections have already failed to separate visible clusters, and ideally kept interactive (rotatable), not static.
- **A static 3D scatter in a report is close to always misleading** — projection onto a 2D image hides depth, points occlude each other, viewers misjudge distance. **Default to 3 separate 2D projections (XY, XZ, YZ)** for anything that has to be a fixed image (a report, a slide, a saved PNG). Reserve true 3D for a live/interactive session.
