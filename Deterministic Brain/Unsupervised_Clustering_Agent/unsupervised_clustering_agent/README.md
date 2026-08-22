# Unsupervised Clustering Agent — Reference Repo

A complete, agent-readable reference for clustering: theory docs, dual
implementations (from-scratch NumPy + scikit-learn wrapper), and
runnable examples for every model. Built from Phase 6 unsupervised
learning theory (K-Means → Mean Shift).

## Structure
```
unsupervised_clustering_agent/
├── README.md                            ← you are here
├── code/                                ← implementations, both versions per model
│   ├── utils.py                         shared: StandardScaler, pairwise_distances,
│   │                                     inertia, elbow_curve, silhouette_score,
│   │                                     k_distance_plot_values
│   ├── kmeans.py                        KMeansScratch / KMeansSklearn
│   ├── hierarchical.py                  AgglomerativeScratch / AgglomerativeSklearn
│   │                                     (single|complete|average|ward linkage)
│   ├── dbscan.py                        DBSCANScratch / DBSCANSklearn
│   └── mean_shift.py                    MeanShiftScratch / MeanShiftSklearn
├── docs/                                ← theory reference, one file per model
│   ├── 01_kmeans.md
│   ├── 02_hierarchical.md
│   ├── 03_dbscan.md
│   ├── 04_mean_shift.md
│   └── 05_clustering_selection_guide.md ← START HERE for "which algorithm do I use"
└── examples/                            ← runnable end-to-end demos
    ├── example_kmeans.py                includes the elbow method demo
    ├── example_hierarchical.py
    ├── example_dbscan.py                includes k-distance plot for eps tuning
    ├── example_mean_shift.py
    └── example_run_all.py               runs all 4 algorithms side by side
```

## For an AI agent reading this repo
1. Read `docs/05_clustering_selection_guide.md` first — it routes any
   clustering task (spherical vs arbitrary shape, known vs unknown K,
   outlier detection needed or not) to the right model and code file.
2. Every `*Scratch` class and its `*Sklearn` counterpart share the same
   API: `fit(X)`, `fit_predict(X)`, and expose `.labels_`. They're
   drop-in replacements — pick Scratch to show/verify the math, pick
   Sklearn for production use.
3. `code/utils.py` has no dependencies beyond NumPy; the `*Scratch`
   classes only need NumPy too. The `*Sklearn` classes need
   `scikit-learn`.
4. Each `docs/0N_*.md` file mirrors its `code/*.py` file 1:1 — pull in
   only the theory needed for a given algorithm rather than loading
   the whole repo into context.
5. **All four algorithms are distance-based** — always run
   `utils.StandardScaler` on features first, or large-range features
   will dominate the distance calculation and produce meaningless
   clusters.

## Quick start
```bash
pip install scikit-learn --break-system-packages   # only needed for *Sklearn classes
python3 examples/example_run_all.py                 # runs & compares all 4 algorithms
python3 examples/example_kmeans.py                   # or run any single model demo
```

## Models covered
K-Means (with K-Means++ init, elbow method, silhouette score) ·
Hierarchical/Agglomerative (single/complete/average/ward linkage,
dendrogram merge history) · DBSCAN (core/border/noise points,
k-distance plot for eps tuning) · Mean Shift (automatic K via density
peaks, bandwidth estimation)

## Choosing an algorithm — the short version
| Question | Answer |
|---|---|
| Know K, want speed, spherical clusters? | **K-Means** |
| Want the full merge hierarchy / dendrogram? | **Hierarchical** |
| Irregular shapes, need outlier detection? | **DBSCAN** |
| Unknown K, varying cluster density, small data? | **Mean Shift** |

Full decision tree and comparison table: `docs/05_clustering_selection_guide.md`.

## Design notes
- **K-Means++** initialization is implemented from scratch (not plain
  random init) to match sklearn's default behavior and avoid bad local
  minima.
- **Hierarchical** stores the full merge history (`self.merges_`) even
  though `fit()` returns labels cut at `n_clusters` — an agent can
  re-cut at any level without refitting the distances.
- **DBSCAN** labels noise as `-1`, matching sklearn's convention, so
  downstream code (like `utils.silhouette_score`) can filter it out
  automatically.
- **Mean Shift** bandwidth is auto-estimated via a median-pairwise-
  distance heuristic if not provided, mirroring sklearn's
  `estimate_bandwidth()`.
- All four `*Scratch` implementations were smoke-tested against their
  `*Sklearn` counterparts on the same synthetic data and produce
  matching cluster counts and silhouette scores.
