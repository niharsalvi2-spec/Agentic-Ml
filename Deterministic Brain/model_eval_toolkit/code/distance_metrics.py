"""
distance_metrics.py
---------------------
Pure-numpy implementations of central tendency / dispersion statistics and
proximity (similarity/dissimilarity) measures used throughout clustering,
KNN, and data mining: Simple Matching Coefficient, Jaccard Coefficient,
Euclidean/Manhattan/Minkowski distance.

No external dependencies besides numpy.
"""

import numpy as np


# --------------------------------------------------------------------------
# Central tendency
# --------------------------------------------------------------------------

def mean(x):
    """Arithmetic average. Sensitive to outliers. Use on symmetric, outlier-free data."""
    return float(np.mean(np.asarray(x, dtype=float)))


def median(x):
    """Middle value when sorted. Robust to outliers. Use on skewed data / outliers present."""
    return float(np.median(np.asarray(x, dtype=float)))


def mode(x):
    """
    Most frequently occurring value(s). The only central-tendency measure
    valid for nominal data. Returns a list (may have >1 value if multimodal).
    """
    x = np.asarray(x)
    values, counts = np.unique(x, return_counts=True)
    max_count = counts.max()
    return [v.item() for v, c in zip(values, counts) if c == max_count]


def midrange(x):
    """(min + max) / 2. Quick estimate, heavily influenced by outliers (uses only 2 points)."""
    x = np.asarray(x, dtype=float)
    return float((x.min() + x.max()) / 2)


# --------------------------------------------------------------------------
# Dispersion
# --------------------------------------------------------------------------

def data_range(x):
    """max - min. Simplest spread measure; ignores everything between the extremes."""
    x = np.asarray(x, dtype=float)
    return float(x.max() - x.min())


def quartiles(x):
    """Returns (Q1, Q2/median, Q3) using linear interpolation (numpy default)."""
    x = np.asarray(x, dtype=float)
    q1, q2, q3 = np.percentile(x, [25, 50, 75])
    return float(q1), float(q2), float(q3)


def iqr(x):
    """Interquartile range = Q3 - Q1. Spread of the middle 50%, robust to outliers."""
    q1, _, q3 = quartiles(x)
    return q3 - q1


def five_number_summary(x):
    """(min, Q1, median, Q3, max) -- exactly what a boxplot visualizes."""
    x = np.asarray(x, dtype=float)
    q1, q2, q3 = quartiles(x)
    return float(x.min()), q1, q2, q3, float(x.max())


def variance(x, ddof=0):
    """
    Average of squared deviations from the mean. ddof=0 for population variance
    (default here, matches the theory formula); use ddof=1 for sample variance.
    """
    x = np.asarray(x, dtype=float)
    return float(np.var(x, ddof=ddof))


def std_dev(x, ddof=0):
    """Square root of variance. Same units as the data, directly interpretable."""
    return float(np.sqrt(variance(x, ddof=ddof)))


# --------------------------------------------------------------------------
# Proximity for nominal attributes
# --------------------------------------------------------------------------

def simple_matching_coefficient_nominal(p, q):
    """
    For nominal attribute vectors p, q (any hashable values, e.g. strings):
    m = number of attributes where p and q match.
    Returns (similarity, dissimilarity) = (m/M, (M-m)/M).
    """
    p = list(p)
    q = list(q)
    if len(p) != len(q):
        raise ValueError("p and q must have the same number of attributes")
    M = len(p)
    m = sum(1 for a, b in zip(p, q) if a == b)
    sim = m / M if M else 0.0
    return sim, 1 - sim


# --------------------------------------------------------------------------
# Proximity for binary attributes
# --------------------------------------------------------------------------

def _binary_contingency(p, q):
    """Returns (a, b, c, d) counts for two 0/1 vectors p, q."""
    p = np.asarray(p).astype(int)
    q = np.asarray(q).astype(int)
    if p.shape != q.shape:
        raise ValueError("p and q must be the same length")
    a = int(np.sum((p == 1) & (q == 1)))   # both 1
    b = int(np.sum((p == 1) & (q == 0)))   # p=1, q=0
    c = int(np.sum((p == 0) & (q == 1)))   # p=0, q=1
    d = int(np.sum((p == 0) & (q == 0)))   # both 0
    return a, b, c, d


def simple_matching_coefficient_binary(p, q):
    """
    SMC = (a + d) / (a + b + c + d).
    Use for SYMMETRIC binary attributes, where both values (0 and 1) are
    equally meaningful (e.g. gender coded as 0/1, a yes/no survey answer
    with no rare/important side).
    """
    a, b, c, d = _binary_contingency(p, q)
    total = a + b + c + d
    return (a + d) / total if total else 0.0


def jaccard_coefficient(p, q):
    """
    J = a / (a + b + c). Ignores the "both absent" (d) term.
    Use for ASYMMETRIC binary attributes, where presence (1) is the
    meaningful/rare outcome and absence (0) is common/uninformative
    (e.g. market-basket "item purchased", disease test "positive").
    Using Simple Matching here would be dominated by the huge d term
    (everyone "matches" on not having most things) and hide real structure.
    """
    a, b, c, _ = _binary_contingency(p, q)
    denom = a + b + c
    return a / denom if denom else 0.0


# --------------------------------------------------------------------------
# Dissimilarity for numeric data
# --------------------------------------------------------------------------

def euclidean_distance(p, q):
    """
    d(p,q) = sqrt(sum((p_i - q_i)^2)). Straight-line distance (Pythagorean
    theorem generalized to n dimensions). Sensitive to feature scale -- always
    standardize features first, or large-scale features (e.g. income) will
    dominate small-scale ones (e.g. age).
    """
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    return float(np.sqrt(np.sum((p - q) ** 2)))


def manhattan_distance(p, q):
    """
    d(p,q) = sum(|p_i - q_i|). Also called City Block / Taxicab distance --
    sum of horizontal + vertical travel, no diagonal shortcuts. Not amplified
    by squaring, so more robust to outliers and to the curse of
    dimensionality in high-dimensional / sparse data than Euclidean distance.
    """
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    return float(np.sum(np.abs(p - q)))


def minkowski_distance(p, q, r):
    """
    d(p,q) = (sum(|p_i - q_i|^r))^(1/r) -- generalizes Euclidean and Manhattan.
    r=1 -> Manhattan, r=2 -> Euclidean, r=inf -> Chebyshev (max coordinate
    difference; pass r=np.inf to get this case, handled directly below since
    the general power-mean formula is numerically unstable at r=inf).
    As r increases, larger per-coordinate differences are increasingly
    amplified and dominate the result.
    """
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    diffs = np.abs(p - q)
    if r == np.inf:
        return float(np.max(diffs))
    return float(np.sum(diffs ** r) ** (1 / r))


def standardize(X):
    """
    Z-score standardization: (x - mean) / std per column. Run this before
    computing Euclidean/Manhattan/Minkowski distances on features with
    different scales/units, or the largest-scale feature will dominate.
    X: array-like, shape (n_samples, n_features). Returns the standardized array.
    """
    X = np.asarray(X, dtype=float)
    means = X.mean(axis=0)
    stds = X.std(axis=0)
    stds_safe = np.where(stds == 0, 1.0, stds)  # avoid divide-by-zero on constant columns
    return (X - means) / stds_safe


if __name__ == "__main__":
    print("mean/median/mode/midrange:", mean([48, 49, 50, 51, 52]),
          median([48, 49, 50, 51, 52]), mode([1, 2, 2, 3]), midrange([10, 20, 200]))
    print("variance/std:", variance([2, 4, 6, 8, 10]), std_dev([2, 4, 6, 8, 10]))
    print("SMC nominal:", simple_matching_coefficient_nominal(
        ["Mumbai", "Male", "Engineer", "Bachelor's"],
        ["Mumbai", "Male", "Doctor", "Bachelor's"]))
    print("Jaccard:", jaccard_coefficient([1, 1, 1, 0], [1, 1, 0, 1]))
    print("Euclidean/Manhattan:", euclidean_distance([2, 3], [5, 7]),
          manhattan_distance([2, 3], [5, 7]))
