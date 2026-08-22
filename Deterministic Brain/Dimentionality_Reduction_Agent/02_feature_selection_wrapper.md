# Skill: Feature Selection — Wrapper Methods
Uses a trained model's performance as the selection criterion. Expensive but accurate.

## 2.1 Sequential Search (Forward / Backward / Stepwise / Floating)

```python
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(max_iter=1000)

# Forward Selection
sfs_fwd = SFS(model, k_features=10, forward=True, floating=False,
              scoring='accuracy', cv=5)
sfs_fwd = sfs_fwd.fit(X, y)
selected = list(sfs_fwd.k_feature_names_)

# Backward Elimination
sfs_bwd = SFS(model, k_features=10, forward=False, floating=False,
              scoring='accuracy', cv=5)
sfs_bwd = sfs_bwd.fit(X, y)

# Stepwise / Floating (SFFS, SFBS) — floating=True adds "conditional exclusion"
sffs = SFS(model, k_features=10, forward=True, floating=True, scoring='accuracy', cv=5)
sffs = sffs.fit(X, y)
```

## 2.2 Recursive Methods

```python
from sklearn.feature_selection import RFE, RFECV
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=200, random_state=42)

# RFE — fixed number of features
rfe = RFE(estimator=model, n_features_to_select=10, step=1)
rfe.fit(X, y)
selected = X.columns[rfe.support_]

# RFECV — auto-finds optimal number via cross-validation
rfecv = RFECV(estimator=model, step=1, cv=5, scoring='accuracy')
rfecv.fit(X, y)
print("Optimal n_features:", rfecv.n_features_)
```

## 2.3 Exhaustive Search

```python
from mlxtend.feature_selection import ExhaustiveFeatureSelector as EFS
from sklearn.tree import DecisionTreeClassifier

efs = EFS(DecisionTreeClassifier(), min_features=2, max_features=4,
          scoring='accuracy', cv=5)   # only feasible for small p (<15-20)
efs = efs.fit(X, y)
best_subset = efs.best_feature_names_
```

## 2.4 Heuristic / Metaheuristic Search

```python
# Genetic Algorithm Feature Selection — pip install sklearn-genetic-opt
from sklearn_genetic import GAFeatureSelectionCV
from sklearn.ensemble import RandomForestClassifier

evolved = GAFeatureSelectionCV(
    estimator=RandomForestClassifier(),
    cv=5, scoring="accuracy",
    population_size=30, generations=20,
)
evolved.fit(X, y)
selected_ga = X.columns[evolved.best_features_]

# Particle Swarm Optimization — pip install pyswarms
import numpy as np
import pyswarms as ps
from sklearn.model_selection import cross_val_score

def fitness(mask_matrix):
    scores = []
    for mask in mask_matrix:
        cols = X.columns[mask > 0.5]
        if len(cols) == 0:
            scores.append(1.0)  # penalize empty subset
            continue
        acc = cross_val_score(RandomForestClassifier(), X[cols], y, cv=3).mean()
        scores.append(1 - acc)  # PSO minimizes
    return np.array(scores)

optimizer = ps.discrete.BinaryPSO(n_particles=20, dimensions=X.shape[1],
                                   options={'c1': 2, 'c2': 2, 'w': 0.9, 'k': 5, 'p': 2})
best_cost, best_mask = optimizer.optimize(fitness, iters=30)
selected_pso = X.columns[best_mask > 0.5]

# Simulated Annealing — conceptual sketch (no single standard lib)
# Accept worse subsets with probability exp(-delta/T), decreasing T over iterations,
# to escape local optima that greedy forward/backward search gets stuck in.
```
