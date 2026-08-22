"""
example_hierarchical.py
Synthetic 2-blob dataset -> Agglomerative clustering (Scratch + Sklearn)
with Ward linkage -> compare cluster assignments.
Run: python3 examples/example_hierarchical.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "code"))

import numpy as np
from hierarchical import AgglomerativeScratch, AgglomerativeSklearn
from utils import StandardScaler, silhouette_score

def make_dataset(seed=0):
    rng = np.random.RandomState(seed)
    X = np.vstack([rng.randn(25, 2) + [5, 5], rng.randn(25, 2) + [-5, -5]])
    return X

if __name__ == "__main__":
    X = make_dataset()
    X_scaled = StandardScaler().fit_transform(X)

    scratch = AgglomerativeScratch(n_clusters=2, linkage="ward").fit(X_scaled)
    print(f"[Scratch]  sizes={np.bincount(scratch.labels_)}  silhouette={silhouette_score(X_scaled, scratch.labels_):.3f}")

    sk = AgglomerativeSklearn(n_clusters=2, linkage="ward").fit(X_scaled)
    print(f"[Sklearn]  sizes={np.bincount(sk.labels_)}  silhouette={silhouette_score(X_scaled, sk.labels_):.3f}")
