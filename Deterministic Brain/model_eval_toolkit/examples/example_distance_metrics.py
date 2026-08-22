"""
Example: distance_metrics.py

Run with:  python3 example_distance_metrics.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "code"))

import numpy as np
from distance_metrics import (
    mean, median, mode, midrange, data_range, iqr, five_number_summary,
    variance, std_dev, simple_matching_coefficient_nominal,
    simple_matching_coefficient_binary, jaccard_coefficient,
    euclidean_distance, manhattan_distance, minkowski_distance, standardize,
)


def central_tendency_and_dispersion_example():
    print("=" * 70)
    print("Example 1: same mean, different spread")
    print("=" * 70)

    a = [48, 49, 50, 51, 52]
    b = [10, 30, 50, 70, 90]

    for name, data in [("Dataset A (tight)", a), ("Dataset B (wide)", b)]:
        print(f"{name}: mean={mean(data):.1f}  median={median(data):.1f}  "
              f"range={data_range(data):.1f}  iqr={iqr(data):.1f}  "
              f"std={std_dev(data):.2f}")

    print("-> Identical mean (50) but very different spread -- dispersion "
          "measures capture what the mean alone hides.\n")


def nominal_proximity_example():
    print("=" * 70)
    print("Example 2: Simple Matching Coefficient (nominal attributes)")
    print("=" * 70)

    customer_a = ["Mumbai", "Male", "Engineer", "Bachelor's"]
    customer_b = ["Mumbai", "Male", "Doctor", "Bachelor's"]

    sim, dissim = simple_matching_coefficient_nominal(customer_a, customer_b)
    print(f"Similarity={sim:.2f}  Dissimilarity={dissim:.2f}")
    print("-> 3 of 4 attributes match (City, Gender, Education) -> 75% similar.\n")


def binary_smc_vs_jaccard_example():
    print("=" * 70)
    print("Example 3: SMC (symmetric) vs Jaccard (asymmetric) on binary data")
    print("=" * 70)

    # Symmetric case: survey preference, both values equally meaningful
    patient_a = [1, 0, 1, 1]
    patient_b = [1, 1, 1, 0]
    print("Symmetric (survey pref) SMC:",
          simple_matching_coefficient_binary(patient_a, patient_b))

    # Asymmetric case: market basket, presence(1) is meaningful, absence(0) is not
    # Out of 1000 possible items, both customers didn't buy 997 of them.
    n_items = 1000
    a_bought = {"Bread", "Milk", "Eggs"}
    b_bought = {"Bread", "Milk", "Butter"}
    all_items = list(a_bought | b_bought) + [f"item_{i}" for i in range(n_items - 4)]
    a_vec = [1 if item in a_bought else 0 for item in all_items]
    b_vec = [1 if item in b_bought else 0 for item in all_items]

    smc = simple_matching_coefficient_binary(a_vec, b_vec)
    jac = jaccard_coefficient(a_vec, b_vec)
    print(f"Market basket (1000 items) SMC={smc:.3f}  Jaccard={jac:.3f}")
    print("-> SMC is dominated by the ~997 items neither bought (looks ~99.7% "
          "similar regardless of actual taste). Jaccard ignores that and "
          "correctly shows 50% overlap in what they actually bought.\n")


def euclidean_manhattan_minkowski_example():
    print("=" * 70)
    print("Example 4: Euclidean vs Manhattan vs Minkowski, and why to standardize")
    print("=" * 70)

    p, q = [2, 3], [5, 7]
    print(f"Euclidean:  {euclidean_distance(p, q):.2f}")
    print(f"Manhattan:  {manhattan_distance(p, q):.2f}")
    for r in [1, 2, 4, np.inf]:
        print(f"Minkowski r={r}: {minkowski_distance(p, q, r):.2f}")
    print("-> r=1 matches Manhattan, r=2 matches Euclidean, and as r grows the "
          "largest single coordinate difference dominates (Chebyshev at r=inf).\n")

    # unscaled feature dominance problem
    person_a = np.array([25, 50000])   # age, income
    person_b = np.array([30, 55000])
    print("Unscaled (age, income):", euclidean_distance(person_a, person_b))

    X = np.vstack([person_a, person_b])
    X_std = standardize(X)
    print("Standardized distance:  ", euclidean_distance(X_std[0], X_std[1]))
    print("-> Income's raw scale swamps age before standardizing; after "
          "z-scoring, both features contribute meaningfully.\n")


if __name__ == "__main__":
    central_tendency_and_dispersion_example()
    nominal_proximity_example()
    binary_smc_vs_jaccard_example()
    euclidean_manhattan_minkowski_example()
