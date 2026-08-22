"""
ml-eda-visualizer / scripts / eda_codegen.py

Importable EDA code-generation module. Every plotting function returns
(fig, interpretation) — interpretation is a dict of plain findings, not
just a picture, so the calling agent always has something to report.

Usage:
    from eda_codegen import eda_report
    findings = eda_report(df, target="churn")

Dependencies: pandas, numpy, matplotlib, seaborn, scipy
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats


# ---------------------------------------------------------------------------
# Bin / bandwidth rules
# ---------------------------------------------------------------------------

def recommend_bins(data: pd.Series) -> int:
    """Freedman-Diaconis bin count. Robust to outliers (uses IQR, not std)."""
    data = data.dropna().to_numpy()
    n = len(data)
    if n < 2:
        return 1
    iqr = np.subtract(*np.percentile(data, [75, 25]))
    if iqr == 0:
        return max(1, int(np.sqrt(n)))  # fallback: square-root rule
    width = 2 * iqr / (n ** (1 / 3))
    if width == 0:
        return max(1, int(np.sqrt(n)))
    bins = int(np.ceil((data.max() - data.min()) / width))
    return max(1, bins)


def skew_kurtosis(data: pd.Series) -> dict:
    data = data.dropna()
    return {
        "skewness": float(stats.skew(data)),
        "kurtosis_excess": float(stats.kurtosis(data)),  # already excess (fisher=True default)
    }


def _skew_label(skew: float) -> str:
    if skew > 0.5:
        return "right-skewed (long tail high) — consider log transform, use median for imputation"
    if skew < -0.5:
        return "left-skewed (long tail low) — consider reflect+log, use median for imputation"
    return "approximately symmetric — mean imputation is reasonable"


# ---------------------------------------------------------------------------
# Univariate
# ---------------------------------------------------------------------------

def plot_univariate_numeric(df: pd.DataFrame, col: str):
    """Histogram + KDE + boxplot stacked, with skew/kurtosis/outlier interpretation."""
    data = df[col].dropna()
    bins = recommend_bins(data)
    sk = skew_kurtosis(data)

    fig, axes = plt.subplots(2, 1, figsize=(7, 6), sharex=True,
                              gridspec_kw={"height_ratios": [3, 1]})
    sns.histplot(data, bins=bins, kde=True, ax=axes[0])
    axes[0].set_title(f"{col} — distribution (bins={bins}, Freedman-Diaconis)")
    sns.boxplot(x=data, ax=axes[1])
    axes[1].set_xlabel(col)
    fig.tight_layout()

    q1, q3 = np.percentile(data, [25, 75])
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    outliers = data[(data < lower) | (data > upper)]

    interpretation = {
        "n": len(data),
        "bins_used": bins,
        **sk,
        "shape": _skew_label(sk["skewness"]),
        "outlier_count": int(len(outliers)),
        "outlier_pct": round(100 * len(outliers) / max(1, len(data)), 2),
        "iqr_fences": (round(lower, 4), round(upper, 4)),
    }
    return fig, interpretation


def plot_univariate_categorical(df: pd.DataFrame, col: str, top_n: int = 20):
    """Bar chart of category frequency + imbalance flag. Never a pie chart."""
    counts = df[col].value_counts().head(top_n)
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.barplot(x=counts.values, y=counts.index.astype(str), ax=ax, orient="h")
    ax.set_title(f"{col} — category frequency")
    ax.set_xlabel("count")
    fig.tight_layout()

    total = df[col].dropna().shape[0]
    top_share = counts.iloc[0] / total if total else 0
    interpretation = {
        "cardinality": int(df[col].nunique(dropna=True)),
        "top_category": str(counts.index[0]) if len(counts) else None,
        "top_category_share": round(float(top_share), 4),
        "imbalanced": bool(top_share > 0.6),
        "note": "high cardinality — consider target/hash encoding over one-hot"
                if df[col].nunique(dropna=True) > 20 else None,
    }
    return fig, interpretation


# ---------------------------------------------------------------------------
# Bivariate
# ---------------------------------------------------------------------------

def plot_scatter(df: pd.DataFrame, x: str, y: str):
    """Scatter + Pearson r + heteroscedasticity flag via residual spread."""
    sub = df[[x, y]].dropna()
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.regplot(data=sub, x=x, y=y, ax=ax, scatter_kws={"alpha": 0.5}, line_kws={"color": "red"})
    ax.set_title(f"{x} vs {y}")
    fig.tight_layout()

    r, p = stats.pearsonr(sub[x], sub[y])
    # crude heteroscedasticity check: split x into terciles, compare residual variance
    slope, intercept = np.polyfit(sub[x], sub[y], 1)
    resid = sub[y] - (slope * sub[x] + intercept)
    terciles = pd.qcut(sub[x], 3, duplicates="drop")
    var_by_tercile = resid.groupby(terciles, observed=True).var()
    hetero_ratio = float(var_by_tercile.max() / max(var_by_tercile.min(), 1e-9))

    interpretation = {
        "pearson_r": round(float(r), 4),
        "p_value": round(float(p), 6),
        "linear_signal": "strong" if abs(r) > 0.5 else ("weak" if abs(r) > 0.2 else "near-zero (check for nonlinear pattern visually — r=0 does not mean no relationship)"),
        "heteroscedasticity_variance_ratio": round(hetero_ratio, 2),
        "likely_heteroscedastic": bool(hetero_ratio > 4),
    }
    return fig, interpretation


def plot_numeric_by_category(df: pd.DataFrame, num_col: str, cat_col: str, kind: str = "violin"):
    """kind: 'violin' (default, catches bimodality), 'box', or 'strip' (small n)."""
    fig, ax = plt.subplots(figsize=(7, 5))
    if kind == "violin":
        sns.violinplot(data=df, x=cat_col, y=num_col, ax=ax)
    elif kind == "strip":
        sns.stripplot(data=df, x=cat_col, y=num_col, ax=ax, jitter=True, alpha=0.6)
    else:
        sns.boxplot(data=df, x=cat_col, y=num_col, ax=ax)
    ax.set_title(f"{num_col} by {cat_col} ({kind})")
    plt.xticks(rotation=45, ha="right")
    fig.tight_layout()

    group_stats = df.groupby(cat_col, observed=True)[num_col].agg(["mean", "median", "std", "count"])
    interpretation = {"per_group_stats": group_stats.round(4).to_dict(orient="index")}
    return fig, interpretation


def plot_correlation_heatmap(df: pd.DataFrame, target: str | None = None, threshold: float = 0.85):
    """Correlation heatmap over numeric columns. Flags pairs above `threshold`."""
    numeric_df = df.select_dtypes(include=[np.number])
    corr = numeric_df.corr(method="pearson")

    fig, ax = plt.subplots(figsize=(max(6, 0.5 * len(corr)), max(5, 0.5 * len(corr))))
    sns.heatmap(corr, annot=len(corr) <= 15, fmt=".2f", cmap="coolwarm", center=0,
                vmin=-1, vmax=1, ax=ax)
    ax.set_title("Correlation heatmap")
    fig.tight_layout()

    flagged = []
    cols = corr.columns.tolist()
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            r = corr.iloc[i, j]
            if abs(r) > threshold:
                pair = {"feature_1": cols[i], "feature_2": cols[j], "r": round(float(r), 4)}
                if target and target in corr.columns:
                    t1 = abs(corr.loc[cols[i], target]) if cols[i] != target else 1
                    t2 = abs(corr.loc[cols[j], target]) if cols[j] != target else 1
                    pair["suggest_drop"] = cols[i] if t1 < t2 else cols[j]
                flagged.append(pair)

    interpretation = {
        "multicollinear_pairs": flagged,
        "threshold_used": threshold,
        "note": "Pearson only — nonlinear relationships not captured. Scatter-check anything important before dropping.",
    }
    return fig, interpretation


# ---------------------------------------------------------------------------
# Multivariate
# ---------------------------------------------------------------------------

def plot_pairplot(df: pd.DataFrame, target: str | None = None, max_features: int = 8):
    """Pairplot over numeric columns, capped at max_features (readability limit)."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if target and target in numeric_cols:
        numeric_cols.remove(target)
    if len(numeric_cols) > max_features:
        # keep the features most correlated with target if available, else first N
        if target and target in df.columns:
            corr_to_target = df[numeric_cols + [target]].corr()[target].drop(target).abs()
            numeric_cols = corr_to_target.sort_values(ascending=False).head(max_features).index.tolist()
        else:
            numeric_cols = numeric_cols[:max_features]

    plot_cols = numeric_cols + ([target] if target else [])
    g = sns.pairplot(df[plot_cols].dropna(), hue=target if target else None, corner=True)
    interpretation = {
        "features_used": numeric_cols,
        "truncated": len(df.select_dtypes(include=[np.number]).columns) > max_features,
        "note": f"capped at {max_features} features for readability" if len(df.select_dtypes(include=[np.number]).columns) > max_features else None,
    }
    return g.fig, interpretation


def plot_parallel_coordinates(df: pd.DataFrame, target: str, max_features: int = 10):
    """Parallel coordinates, features min-max scaled so axes are comparable."""
    from pandas.plotting import parallel_coordinates

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if target in numeric_cols:
        numeric_cols.remove(target)
    numeric_cols = numeric_cols[:max_features]

    sub = df[numeric_cols + [target]].dropna().copy()
    for c in numeric_cols:
        rng = sub[c].max() - sub[c].min()
        sub[c] = (sub[c] - sub[c].min()) / rng if rng else 0.0

    fig, ax = plt.subplots(figsize=(max(8, len(numeric_cols)), 5))
    parallel_coordinates(sub, target, ax=ax, alpha=0.4)
    ax.set_title("Parallel coordinates (min-max scaled)")
    plt.xticks(rotation=45, ha="right")
    fig.tight_layout()

    interpretation = {"features_used": numeric_cols, "note": "axis order is arbitrary — try regrouping features if patterns are unclear"}
    return fig, interpretation


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def eda_report(df: pd.DataFrame, target: str | None = None, save_dir: str | None = None):
    """
    Runs the full checklist (references/eda_checklist.md) automatically:
    per-feature univariate, correlation heatmap, target distribution, pairplot.
    Returns a dict of findings; saves figures to save_dir if given.
    """
    findings = {"n_rows": len(df), "n_cols": df.shape[1], "null_rate": df.isnull().mean().round(4).to_dict()}
    figs = {}

    for col in df.columns:
        if col == target:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            fig, interp = plot_univariate_numeric(df, col)
        else:
            fig, interp = plot_univariate_categorical(df, col)
        figs[col] = fig
        findings[col] = interp

    if df.select_dtypes(include=[np.number]).shape[1] >= 2:
        fig, interp = plot_correlation_heatmap(df, target=target)
        figs["_correlation_heatmap"] = fig
        findings["_multicollinearity"] = interp

    if target is not None:
        if pd.api.types.is_numeric_dtype(df[target]):
            fig, interp = plot_univariate_numeric(df, target)
        else:
            fig, interp = plot_univariate_categorical(df, target)
        figs["_target"] = fig
        findings["_target_distribution"] = interp

        fig, interp = plot_pairplot(df, target=target)
        figs["_pairplot"] = fig
        findings["_pairplot"] = interp

    if save_dir:
        import os
        os.makedirs(save_dir, exist_ok=True)
        for name, fig in figs.items():
            fig.savefig(os.path.join(save_dir, f"{name}.png".replace("/", "_")), dpi=150, bbox_inches="tight")

    return findings
