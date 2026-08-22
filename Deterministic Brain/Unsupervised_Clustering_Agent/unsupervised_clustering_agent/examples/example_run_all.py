"""
example_run_all.py
Runs every clustering algorithm in the repo (both Scratch and Sklearn
versions where applicable) on the same synthetic dataset and prints a
comparison of clusters found + silhouette score.
Run: python3 examples/example_run_all.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "code"))

import numpy as np
from utils import StandardScaler, silhouette_score

from kmeans import KMeansScratch, KMeansSklearn
from hierarchical import AgglomerativeScratch, AgglomerativeSklearn
from dbscan import DBSCANScratch, DBSCANSklearn
from mean_shift import MeanShiftScratch, MeanShiftSklearn


def make_dataset(seed=0):
    rng = np.random.RandomState(seed)
    X = np.vstack([
        rng.randn(50, 2) + [5, 5],
        rng.randn(50, 2) + [-5, -5],
        rng.randn(50, 2) + [5, -5],
    ])
    return X


if __name__ == "__main__":
    X = make_dataset()
    X_scaled = StandardScaler().fit_transform(X)

    runs = [
        ("K-Means", KMeansScratch(k=3), KMeansSklearn(k=3)),
        ("Hierarchical (Ward)", AgglomerativeScratch(n_clusters=3, linkage="ward"), AgglomerativeSklearn(n_clusters=3, linkage="ward")),
        ("DBSCAN", DBSCANScratch(eps=0.6, min_pts=5), DBSCANSklearn(eps=0.6, min_pts=5)),
        ("Mean Shift", MeanShiftScratch(), MeanShiftSklearn()),
    ]

    print(f"{'Model':<22}{'Scratch #clusters':<20}{'Scratch silh.':<16}{'Sklearn #clusters':<20}{'Sklearn silh.':<16}")
    print("-" * 94)
    for name, scratch_model, sklearn_model in runs:
        scratch_labels = scratch_model.fit_predict(X_scaled)
        sklearn_labels = sklearn_model.fit_predict(X_scaled)

        n_scratch = len(set(scratch_labels) - {-1})
        n_sklearn = len(set(sklearn_labels) - {-1})
        s_scratch = silhouette_score(X_scaled, scratch_labels)
        s_sklearn = silhouette_score(X_scaled, sklearn_labels)

        print(f"{name:<22}{n_scratch:<20}{s_scratch:<16.3f}{n_sklearn:<20}{s_sklearn:<16.3f}")
