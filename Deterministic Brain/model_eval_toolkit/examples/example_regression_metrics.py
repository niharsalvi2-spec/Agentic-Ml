"""
Example: regression_metrics.py

Run with:  python3 example_regression_metrics.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "code"))

import numpy as np
from regression_metrics import (
    mean_absolute_error, mean_squared_error, root_mean_squared_error,
    r2_score, adjusted_r2_score, regression_report,
)


def house_price_example():
    print("=" * 70)
    print("Example 1: basic MAE / MSE / RMSE / R2 (house prices, lakhs)")
    print("=" * 70)

    y_true = [50, 100, 150, 200]
    y_pred = [55, 90, 160, 190]

    print("MAE: ", mean_absolute_error(y_true, y_pred))
    print("MSE: ", mean_squared_error(y_true, y_pred))
    print("RMSE:", root_mean_squared_error(y_true, y_pred))
    print("R2:  ", r2_score(y_true, y_pred))
    print()


def outlier_sensitivity_example():
    print("=" * 70)
    print("Example 2: MAE vs RMSE -- spotting an outlier prediction")
    print("=" * 70)

    consistent_errors = np.array([5, 5, 5, 5, 5])
    one_outlier = np.array([1, 1, 1, 1, 21])

    for name, errors in [("consistent", consistent_errors), ("one outlier", one_outlier)]:
        y_true = np.zeros_like(errors, dtype=float)
        y_pred = errors.astype(float)  # errors directly as |y_true - y_pred|
        mae = mean_absolute_error(y_true, y_pred)
        rmse = root_mean_squared_error(y_true, y_pred)
        print(f"{name:12s}  MAE={mae:.2f}  RMSE={rmse:.2f}  RMSE/MAE={rmse/mae:.2f}")

    print("-> Same MAE in both cases, but RMSE >> MAE flags the outlier case.\n")


def r2_and_adjusted_r2_example():
    print("=" * 70)
    print("Example 3: R2 vs Adjusted R2 when adding a useless feature")
    print("=" * 70)

    rng = np.random.default_rng(1)
    n = 200
    x = rng.normal(0, 1, n)
    y_true = 3 * x + rng.normal(0, 1, n)

    # "Model A": good fit using 1 real feature
    y_pred_a = 3 * x
    r2_a = r2_score(y_true, y_pred_a)
    adj_r2_a = adjusted_r2_score(y_true, y_pred_a, n_features=1)

    # "Model B": same predictions, but pretend we added 20 useless extra features
    r2_b = r2_a  # R2 doesn't change just because more features exist in the model spec
    adj_r2_b = adjusted_r2_score(y_true, y_pred_a, n_features=21)

    print(f"Model A (1 feature):   R2={r2_a:.4f}  Adjusted R2={adj_r2_a:.4f}")
    print(f"Model B (21 features, same predictions): R2={r2_b:.4f}  "
          f"Adjusted R2={adj_r2_b:.4f}")
    print("-> R2 is identical, but Adjusted R2 drops once you account for the "
          "20 extra (useless) features -- this is the correct penalty.\n")


def full_report_example():
    print("=" * 70)
    print("Example 4: regression_report() end to end")
    print("=" * 70)

    rng = np.random.default_rng(3)
    y_true = rng.normal(50, 10, 100)
    y_pred = y_true + rng.normal(0, 3, 100)

    report = regression_report(y_true, y_pred, n_features=4)
    for k, v in report.items():
        print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")


if __name__ == "__main__":
    house_price_example()
    outlier_sensitivity_example()
    r2_and_adjusted_r2_example()
    full_report_example()
