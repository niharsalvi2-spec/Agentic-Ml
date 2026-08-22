# Skill: Feature Extraction — Domain Specific

## Image

```python
from skimage.feature import hog, local_binary_pattern
import cv2

# HOG — Histogram of Oriented Gradients
features, hog_image = hog(gray_img, orientations=9, pixels_per_cell=(8, 8),
                           cells_per_block=(2, 2), visualize=True)

# SIFT — scale invariant keypoints/descriptors
sift = cv2.SIFT_create()
keypoints, descriptors = sift.detectAndCompute(gray_img, None)

# ORB — free alternative to SIFT/SURF (SURF is patented, not in modern OpenCV)
orb = cv2.ORB_create()
kp, des = orb.detectAndCompute(gray_img, None)

# LBP — Local Binary Patterns (texture)
lbp = local_binary_pattern(gray_img, P=8, R=1, method='uniform')

# Gabor Filters — texture/edge at specific orientation & frequency
gabor_kernel = cv2.getGaborKernel((21, 21), sigma=5, theta=0, lambd=10, gamma=0.5)
filtered = cv2.filter2D(gray_img, cv2.CV_8UC3, gabor_kernel)

# CNN Transfer Learning Features (ResNet/VGG/EfficientNet)
import torch, torchvision.models as models
resnet = models.resnet50(pretrained=True)
resnet.fc = torch.nn.Identity()  # strip classification head -> use penultimate layer as features
resnet.eval()
with torch.no_grad():
    features = resnet(image_batch_tensor)

# Vision Transformer features
from transformers import ViTModel, ViTImageProcessor
processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
vit = ViTModel.from_pretrained('google/vit-base-patch16-224')
inputs = processor(images=pil_image, return_tensors="pt")
vit_features = vit(**inputs).last_hidden_state[:, 0, :]   # CLS token embedding
```

## Text / NLP

```python
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, HashingVectorizer
from sklearn.decomposition import TruncatedSVD, LatentDirichletAllocation

# Bag of Words
bow = CountVectorizer(max_features=5000)
X_bow = bow.fit_transform(corpus)

# TF-IDF (uni/bi/n-gram via ngram_range)
tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))  # (1,2) = uni+bigrams
X_tfidf = tfidf.fit_transform(corpus)

# LSA — Latent Semantic Analysis == SVD on TF-IDF matrix
lsa = TruncatedSVD(n_components=100)
X_lsa = lsa.fit_transform(X_tfidf)

# LDA — Latent Dirichlet Allocation (topic modeling, NOT the same LDA as classification)
lda_topics = LatentDirichletAllocation(n_components=10, random_state=42)
X_topics = lda_topics.fit_transform(X_bow)

# Hashing Vectorizer — memory-efficient, no vocabulary storage
hv = HashingVectorizer(n_features=2**14)
X_hash = hv.fit_transform(corpus)

# Word2Vec (Skip-Gram / CBOW)
from gensim.models import Word2Vec
w2v = Word2Vec(sentences=tokenized_corpus, vector_size=100, sg=1, window=5)  # sg=1 skip-gram, sg=0 CBOW

# Sentence Transformers (BERT-based sentence embeddings)
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(sentences)
```

## Time Series

```python
import numpy as np
from scipy.fft import fft
from scipy.signal import stft
import pywt

# Statistical features
def stat_features(series):
    return {
        'mean': series.mean(), 'std': series.std(),
        'skew': series.skew(), 'kurt': series.kurt(),
        'min': series.min(), 'max': series.max(), 'range': series.max() - series.min(),
        'p25': series.quantile(.25), 'p75': series.quantile(.75),
    }

# FFT — Fast Fourier Transform (frequency domain)
freq_domain = np.abs(fft(series.values))

# STFT — Short-Time Fourier Transform (time-varying frequency)
f, t, Zxx = stft(series.values, fs=1.0, nperseg=256)

# Wavelet Transform (DWT / CWT)
coeffs = pywt.wavedec(series.values, 'db4', level=4)   # DWT
cwt_coeffs, freqs = pywt.cwt(series.values, np.arange(1, 128), 'morl')  # CWT

# tsfresh — automated extraction of 700+ time series features
from tsfresh import extract_features
extracted = extract_features(long_format_df, column_id='id', column_sort='time')

# Catch22 — 22 canonical, highly discriminative TS features (pip install pycatch22)
import pycatch22
features_c22 = pycatch22.catch22_all(series.values)
```

## Graph / Network

```python
# Node2Vec — pip install node2vec
from node2vec import Node2Vec
n2v = Node2Vec(graph, dimensions=64, walk_length=30, num_walks=200)
model = n2v.fit(window=10, min_count=1)
node_embedding = model.wv['node_id']

# DeepWalk — same idea as Node2Vec with pure random walks (p=q=1 special case)

# GraphSAGE / GCN — pip install torch_geometric
from torch_geometric.nn import SAGEConv, GCNConv
import torch.nn.functional as F

class GCN(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, out_dim)
    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        return self.conv2(x, edge_index)   # node embeddings as output
```

## Automated / Learned (general purpose)

```python
from sklearn.cluster import FeatureAgglomeration
from sklearn.decomposition import NMF, DictionaryLearning, MiniBatchDictionaryLearning

# Feature Agglomeration — hierarchical clustering of correlated features into groups
fa = FeatureAgglomeration(n_clusters=20)
X_agg = fa.fit_transform(X_scaled)

# NMF — Non-negative Matrix Factorization (all values must be >= 0, e.g. TF-IDF, images)
nmf = NMF(n_components=10, init='nndsvda', random_state=42)
X_nmf = nmf.fit_transform(X_nonneg)

# Dictionary Learning — sparse coding basis
dl = DictionaryLearning(n_components=15, alpha=1)
X_dict = dl.fit_transform(X_scaled)

mbdl = MiniBatchDictionaryLearning(n_components=15, alpha=1, batch_size=200)
X_mbdict = mbdl.fit_transform(X_scaled)
```
