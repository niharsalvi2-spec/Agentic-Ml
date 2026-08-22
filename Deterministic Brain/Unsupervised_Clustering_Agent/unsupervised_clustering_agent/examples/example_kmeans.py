"""
example_kmeans.py
Synthetic 3-blob dataset -> fit K-Means (Scratch + Sklearn) -> compare
inertia and silhouette score. Also demonstrates the elbow method.
Run: python3 examples/example_kmeans.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "code"))

import numpy as np
from kmeans import KMeansScratch, KMeansSklearn
from utils import StandardScaler, silhouette_score, elbow_curve

def make_dataset(seed=0):
    rng = np.random.RandomState(seed)
    X = np.vstack([
        rng.randn(60, 2) + [5, 5],
        rng.randn(60, 2) + [-5, -5],
        rng.randn(60, 2) + [5, -5],
    ])
    return X

if __name__ == "__main__":
    X = make_dataset()
    X_scaled = StandardScaler().fit_transform(X)

    print("Elbow method (K vs WCSS):")
    for k, wcss in elbow_curve(KMeansScratch, X_scaled, k_range=range(1, 7)):
        print(f"  K={k}: WCSS={wcss:.1f}")

    scratch = KMeansScratch(k=3).fit(X_scaled)
    print(f"\n[Scratch]  inertia={scratch.inertia_:.2f}  silhouette={silhouette_score(X_scaled, scratch.labels_):.3f}")

    sk = KMeansSklearn(k=3).fit(X_scaled)
    print(f"[Sklearn]  inertia={sk.inertia_:.2f}  silhouette={silhouette_score(X_scaled, sk.labels_):.3f}")
