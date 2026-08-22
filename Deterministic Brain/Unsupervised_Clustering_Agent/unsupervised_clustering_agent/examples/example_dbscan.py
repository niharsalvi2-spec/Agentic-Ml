"""
example_dbscan.py
Two dense blobs + scattered noise -> DBSCAN (Scratch + Sklearn) ->
compare number of clusters found and noise points detected.
Run: python3 examples/example_dbscan.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "code"))

import numpy as np
from dbscan import DBSCANScratch, DBSCANSklearn
from utils import StandardScaler, k_distance_plot_values

def make_dataset(seed=0):
    rng = np.random.RandomState(seed)
    X = np.vstack([
        rng.randn(30, 2) * 0.5 + [5, 5],
        rng.randn(30, 2) * 0.5 + [-5, -5],
        rng.uniform(-10, 10, size=(6, 2)),  # noise
    ])
    return X

if __name__ == "__main__":
    X = make_dataset()
    X_scaled = StandardScaler().fit_transform(X)

    # k-distance plot values can guide eps selection (look for the "knee")
    kd = k_distance_plot_values(X_scaled, k=5)
    print("k-distance sample (top 5, for eps tuning):", np.round(kd[:5], 2))

    scratch = DBSCANScratch(eps=0.5, min_pts=5).fit(X_scaled)
    print(f"\n[Scratch]  clusters={len(set(scratch.labels_) - {-1})}  noise={np.sum(scratch.labels_ == -1)}")

    sk = DBSCANSklearn(eps=0.5, min_pts=5).fit(X_scaled)
    print(f"[Sklearn]  clusters={len(set(sk.labels_) - {-1})}  noise={np.sum(sk.labels_ == -1)}")
