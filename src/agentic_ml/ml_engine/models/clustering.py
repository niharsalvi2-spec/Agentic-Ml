"""
Unsupervised Clustering Model Families.
Provides standard clustering initializers and fit helpers.
"""

from typing import Dict, Any
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering, MeanShift


def get_clustering_models(n_clusters: int = 3, random_state: int = 42) -> Dict[str, Any]:
    """Instantiates standard clustering algorithms."""
    return {
        "KMeans": KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10),
        "DBSCAN": DBSCAN(eps=0.5, min_samples=5),
        "Hierarchical": AgglomerativeClustering(n_clusters=n_clusters),
        "MeanShift": MeanShift(),
    }
