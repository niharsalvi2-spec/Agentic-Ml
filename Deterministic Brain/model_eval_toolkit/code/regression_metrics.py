"""
regression_metrics.py
----------------------
Pure-numpy implementations of standard regression evaluation metrics:
MAE, MSE, RMSE, R2, Adjusted R2, and a combined regression_report().

No external dependencies besides numpy.
"""

import numpy as np


def mean_absolute_error(y_true, y_pred):
    """
    MAE = mean(|y_true - y_pred|). Same units as target. Robust to outliers
    (an error of 100 contributes 10x an error of 10, i.e. linearly) but not
    differentiable at zero error.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(y_true - y_pred)))


def mean_squared_error(y_true, y_pred):
    """
    MSE = mean((y_true - y_pred)^2). Squared units. Punishes large errors
    much more than small ones (an error of 20 contributes 4x an error of 10),
    which is why it's sensitive to outliers but mathematically smooth for
    gradient-based optimization.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean((y_true - y_pred) ** 2))


def root_mean_squared_error(y_true, y_pred):
    """
    RMSE = sqrt(MSE). Same units as target, but like MSE it penalizes large
    errors more than MAE does. RMSE >> MAE signals outlier predictions exist;
    RMSE ~= MAE signals consistent errors across all predictions.
    """
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def r2_score(y_true, y_pred):
    """
    R2 = 1 - SS_residual / SS_total. Fraction of variance in the target
    explained by the model, relative to always predicting the mean.
    R2=1 perfect, R2=0 no better than predicting the mean, R2<0 worse than
    predicting the mean.

    Caveat: never decreases when adding features (even useless ones), and
    is sensitive to outliers in y (they inflate SS_total). Use Adjusted R2
    to compare models with different numbers of features.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0:
        return 1.0 if ss_res == 0 else 0.0
    return float(1 - ss_res / ss_tot)


def adjusted_r2_score(y_true, y_pred, n_features):
    """
    Adjusted R2 = 1 - (1 - R2) * (n - 1) / (n - p - 1)

    Penalizes adding features that don't genuinely improve fit. Adding a
    useful feature increases Adjusted R2; adding a useless one can decrease
    it (correctly). Always prefer this over plain R2 when comparing models
    with different numbers of features.

    n_features (p) must satisfy n - p - 1 > 0.
    """
    y_true = np.asarray(y_true, dtype=float)
    n = len(y_true)
    p = n_features
    if n - p - 1 <= 0:
        raise ValueError("n_features too large relative to number of samples "
                          "(need n - p - 1 > 0)")
    r2 = r2_score(y_true, y_pred)
    return float(1 - (1 - r2) * (n - 1) / (n - p - 1))


def regression_report(y_true, y_pred, n_features=None):
    """
    Returns a dict with MAE, MSE, RMSE, R2, and (if n_features given)
    Adjusted R2. Reporting MAE + RMSE + R2 together is the recommended
    "complete" summary: MAE = typical error, RMSE = outlier-sensitivity
    signal (compare to MAE), R2 = overall fit quality.
    """
    report = {
        "mae": mean_absolute_error(y_true, y_pred),
        "mse": mean_squared_error(y_true, y_pred),
        "rmse": root_mean_squared_error(y_true, y_pred),
        "r2": r2_score(y_true, y_pred),
    }
    if n_features is not None:
        report["adjusted_r2"] = adjusted_r2_score(y_true, y_pred, n_features)
        report["rmse_vs_mae_ratio"] = (
            report["rmse"] / report["mae"] if report["mae"] else float("nan")
        )
    return report


if __name__ == "__main__":
    y_true = [50, 100, 150, 200]
    y_pred = [55, 90, 160, 190]
    print(regression_report(y_true, y_pred, n_features=2))
