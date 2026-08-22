# Univariate Visualization Reference

Purpose: shape, spread, and outliers of one variable. Read this when routed here from SKILL.md.

## Histogram

`height = count in bin` (frequency) or `count / (n × width)` (density, area = 1).

**Bin count — use Freedman-Diaconis, not eyeballing:**
```
width = 2 × IQR / n^(1/3)
```
Robust to outliers (unlike Scott's `3.5σ / n^(1/3)`, which std-based rules aren't). `scripts/eda_codegen.py:recommend_bins` implements this.

**Shape → action:**
| Shape | Meaning | Action |
|---|---|---|
| Symmetric bell | Normal | Mean imputation OK |
| Right skew | Few very high values | Log transform; use median for imputation |
| Left skew | Few very low values | Reflect + log; use median |
| Bimodal | Two subgroups | Segment before modeling, don't average over them |
| Uniform | No dominant value | — |
| Spike at one value | Mode dominance or capped/collection artifact | Investigate collection, don't trust at face value |

**Skewness:** `(1/n) Σ[(xᵢ-x̄)/σ]³`. Positive → mean>median>mode, long right tail (income, prices). Negative → mean<median<mode.
**Kurtosis (excess):** `(1/n) Σ[(xᵢ-x̄)/σ]⁴ - 3`. >0 leptokurtic (heavy tails — financial returns), <0 platykurtic (light tails).

**Limitation:** bin width choice changes apparent shape — always report the rule used, not just the picture.

## KDE (Kernel Density Estimate)

`f̂(x) = (1/nh) Σ K((x-xᵢ)/h)`, Gaussian kernel `K(u) = (1/√2π) exp(-u²/2)`.

Solves histogram's bin-dependency problem — smooth curve, no bin edges. Bandwidth `h` via Silverman: `h = 0.9 × min(σ, IQR/1.35) × n^(-1/5)`.

- Small `h` → spiky/noisy (undersmoothed). Large `h` → flat, hides structure (oversmoothed).
- **Overlay KDE per class** to check if a feature is discriminative: heavy overlap → weak feature; clean separation → strong feature. This is a fast pre-model feature-usefulness check, faster than running a model.
- KDE can assign density to impossible ranges (e.g. negative values for a strictly-positive variable) — clip/note this when the variable has a hard floor.

## Boxplot

5-number summary: min, Q1, median, Q3, max, plus outliers as separate points.
```
IQR = Q3 - Q1
Lower fence = Q1 - 1.5×IQR   Upper fence = Q3 + 1.5×IQR
```
Points beyond fences are plotted individually — those are your outlier candidates, not the whisker ends.

**Reading skew from the box:** median off-center toward the bottom + longer upper whisker → right skew. Symmetric box + equal whiskers → symmetric.

**Critical weakness: cannot show bimodality.** A two-peaked distribution looks identical to a uniform one in a boxplot. Never present a boxplot alone as "the distribution" — pair with histogram/KDE, always.

## Bar chart (categorical)

Frequency (count per category) or aggregate (mean/sum per category), grouped/stacked for sub-category comparison.

- Use for class-imbalance checks on a classification target — do this **before** any other EDA step once the target is categorical, since imbalance changes metric choice (accuracy misleads on imbalanced classes) and whether you need resampling.
- Missing expected category with zero bar → data collection gap, investigate before modeling.

## Pie chart — default to NOT using it

Angle `θᵢ = 360° × (nᵢ/N)`. Humans compare bar lengths accurately; angle/area comparison is poor (35% vs 32% is unreadable as a pie slice, obvious as a bar). Use only if: 2–3 categories, clearly different sizes, showing a whole. Otherwise convert to bar chart — this applies almost universally.
