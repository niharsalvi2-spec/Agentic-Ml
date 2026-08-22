# Skill: Feature Extraction — Nonlinear / Manifold + Neural Network Methods

## Manifold Learning

```python
from sklearn.manifold import TSNE, Isomap, LocallyLinearEmbedding, MDS, SpectralEmbedding
from sklearn.decomposition import KernelPCA

# t-SNE — visualization ONLY (2D/3D), preserves local structure, non-deterministic
tsne = TSNE(n_components=2, perplexity=30, random_state=42)
X_tsne = tsne.fit_transform(X)

# UMAP — preserves local + global structure, CAN be used as ML features (pip install umap-learn)
import umap
reducer = umap.UMAP(n_components=10, n_neighbors=15, min_dist=0.1, random_state=42)
X_umap = reducer.fit_transform(X)          # fit once
X_new_umap = reducer.transform(X_test)      # transform new points (t-SNE can't do this)

# Isomap — preserves geodesic distance, good for curved manifolds (Swiss roll)
isomap = Isomap(n_neighbors=10, n_components=2)
X_isomap = isomap.fit_transform(X)

# LLE — Locally Linear Embedding (+ variants via `method=`)
lle = LocallyLinearEmbedding(n_neighbors=10, n_components=2, method='standard')
X_lle = lle.fit_transform(X)
# method options: 'standard', 'modified', 'hessian', 'ltsa'

# MDS — preserves pairwise distances
mds = MDS(n_components=2, metric=True, random_state=42)   # metric=False -> non-metric MDS
X_mds = mds.fit_transform(X)

# Kernel PCA — nonlinear PCA via kernel trick
kpca = KernelPCA(n_components=10, kernel='rbf', gamma=0.1)  # kernel: rbf/poly/sigmoid/cosine
X_kpca = kpca.fit_transform(X_scaled)

# Spectral Embedding — Laplacian Eigenmaps
se = SpectralEmbedding(n_components=2, affinity='nearest_neighbors')
X_se = se.fit_transform(X)

# PHATE — visualizes trajectories/branches (pip install phate)
import phate
phate_op = phate.PHATE(n_components=2)
X_phate = phate_op.fit_transform(X)
```

## Autoencoders (PyTorch)

```python
import torch, torch.nn as nn

class Autoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim=10):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64), nn.ReLU(),
            nn.Linear(64, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64), nn.ReLU(),
            nn.Linear(64, input_dim)
        )
    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z), z

model = Autoencoder(input_dim=X.shape[1], latent_dim=10)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()
X_t = torch.tensor(X_scaled, dtype=torch.float32)

for epoch in range(100):
    recon, latent = model(X_t)
    loss = criterion(recon, X_t)                       # Vanilla Autoencoder
    optimizer.zero_grad(); loss.backward(); optimizer.step()

with torch.no_grad():
    X_encoded = model.encoder(X_t).numpy()              # <- new extracted features

# Denoising: add noise to input before encoding, still reconstruct clean X_t
# noisy = X_t + 0.1*torch.randn_like(X_t); recon, z = model(noisy); loss = criterion(recon, X_t)

# Sparse: add L1 penalty on latent activations
# loss = criterion(recon, X_t) + 1e-3 * latent.abs().mean()

# Contractive: penalize sensitivity of latent to input (Jacobian norm)
```

## VAE (sketch)

```python
class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim=10):
        super().__init__()
        self.fc_mu = nn.Linear(64, latent_dim)
        self.fc_logvar = nn.Linear(64, latent_dim)
        self.enc = nn.Sequential(nn.Linear(input_dim, 64), nn.ReLU())
        self.dec = nn.Sequential(nn.Linear(latent_dim, 64), nn.ReLU(), nn.Linear(64, input_dim))
    def forward(self, x):
        h = self.enc(x)
        mu, logvar = self.fc_mu(h), self.fc_logvar(h)
        std = torch.exp(0.5 * logvar)
        z = mu + std * torch.randn_like(std)      # reparameterization trick
        return self.dec(z), mu, logvar

# Loss = reconstruction_loss + KL_divergence(mu, logvar)
# Beta-VAE: multiply KL term by beta > 1 for more disentangled latent space
# Conditional VAE: concatenate class label to encoder/decoder input
```
