"""
example_mean_shift.py
Two blobs -> Mean Shift (Scratch + Sklearn) -> compare number of
clusters discovered automatically (no K specified).
Run: python3 examples/example_mean_shift.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "code"))

import numpy as np
from mean_shift import MeanShiftScratch, MeanShiftSklearn
from utils import StandardScaler

def make_dataset(seed=0):
    rng = np.random.RandomState(seed)
    X = np.vstack([rng.randn(40, 2) * 0.6 + [5, 5], rng.randn(40, 2) * 0.6 + [-5, -5]])
    return X

if __name__ == "__main__":
    X = make_dataset()
    X_scaled = StandardScaler().fit_transform(X)

    scratch = MeanShiftScratch().fit(X_scaled)
    print(f"[Scratch]  clusters found={len(scratch.cluster_centers_)}  bandwidth used={scratch.bandwidth_used_:.3f}")

    sk = MeanShiftSklearn().fit(X_scaled)
    print(f"[Sklearn]  clusters found={len(sk.cluster_centers_)}")
