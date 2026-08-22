# Skill: Feature Extraction — Linear Methods
Creates NEW features as linear combinations of originals.

```python
from sklearn.decomposition import PCA, TruncatedSVD, FastICA, FactorAnalysis
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.random_projection import GaussianRandomProjection, SparseRandomProjection
from sklearn.preprocessing import StandardScaler

X_scaled = StandardScaler().fit_transform(X)  # PCA/ICA/FA require standardization

# PCA — maximize variance, orthogonal components
pca = PCA(n_components=0.95)   # keep components till 95% variance explained
X_pca = pca.fit_transform(X_scaled)
print("Explained variance ratio:", pca.explained_variance_ratio_)

# LDA — supervised, maximizes class separability (classification only)
lda = LDA(n_components=min(len(set(y)) - 1, X.shape[1]))
X_lda = lda.fit_transform(X_scaled, y)

# SVD (Truncated) — works on sparse/non-square matrices, e.g. text TF-IDF
svd = TruncatedSVD(n_components=50)
X_svd = svd.fit_transform(X)   # no need to center/scale for sparse input

# ICA — statistically independent components (signal separation: EEG/audio)
ica = FastICA(n_components=10, random_state=42)
X_ica = ica.fit_transform(X_scaled)

# Factor Analysis — assumes latent factors + noise, good with noisy features
fa = FactorAnalysis(n_components=10, random_state=42)
X_fa = fa.fit_transform(X_scaled)

# Random Projection — very fast, approximate distance preservation (Johnson-Lindenstrauss)
grp = GaussianRandomProjection(n_components=50, random_state=42)
X_grp = grp.fit_transform(X_scaled)

srp = SparseRandomProjection(n_components=50, random_state=42)  # faster, sparse matrix
X_srp = srp.fit_transform(X_scaled)
```

## Choosing k for PCA (scree plot)

```python
import matplotlib.pyplot as plt
pca_full = PCA().fit(X_scaled)
plt.plot(range(1, len(pca_full.explained_variance_ratio_)+1),
         pca_full.explained_variance_ratio_.cumsum(), marker='o')
plt.axhline(0.95, color='r', linestyle='--')
plt.xlabel("Number of components"); plt.ylabel("Cumulative explained variance")
```
