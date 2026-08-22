# Synthetic Data Generation

Use when real data is unavailable, private/restricted (GDPR/HIPAA/DPDP), too imbalanced, or too expensive to label. Synthetic data is artificially generated data that mimics real data statistically, contains no real personal information, can be generated in any quantity, and can be balanced to fix class-imbalance problems.

Always name synthetic outputs clearly (`synthetic_*`) and run them through `validation-quality.md` before treating them as a stand-in for real data.

## Method map — pick by situation

```
Need fake names, emails, addresses for testing        → Faker
Practicing ML algorithms from scratch                  → sklearn generators
Imbalanced dataset (fraud, disease, churn)
  ├── Simple imbalance          → SMOTE
  ├── Hard boundary cases       → BorderlineSMOTE
  ├── Adaptive sampling         → ADASYN
  └── Mixed numeric+categorical → SMOTENC
Have real data, need a privacy-safe statistical copy
  ├── Fast, mostly numeric       → GaussianCopula (SDV)
  ├── Complex mixed data, best quality → CTGAN
  ├── Faster than CTGAN, continuous → TVAE
  └── Multiple related tables    → HMA Synthesizer (SDV)
Image data
  ├── Basic synthesis            → DCGAN
  ├── High quality                → StyleGAN3
  ├── Unpaired translation (MRI→CT) → CycleGAN
  └── Best quality / text-to-image → Diffusion models
Time series                       → statistical (numpy) or TimeGAN
Text / NLP                        → template-based or LLM-based
No data exists + need a pipeline test right now → statistical generation (numpy)
```

## 1. Statistical / distribution-based generation

Fastest, full control, no ML required. Use when you know the real distribution or just need to unblock pipeline testing.

```python
import numpy as np
import pandas as pd

np.random.seed(42)
n = 1000

df = pd.DataFrame({
    "age":             np.random.normal(35, 10, n).clip(18, 70).astype(int),
    "salary":          np.random.normal(50000, 15000, n).clip(15000, 200000).round(2),
    "test_score":      np.random.uniform(0, 100, n).round(1),        # uniform
    "income":          np.random.exponential(scale=30000, size=n).round(2),  # right-skewed
    "purchased":       np.random.binomial(1, 0.3, n),                # yes/no
    "daily_visits":    np.random.poisson(lam=5, size=n),             # counts
    "gender":          np.random.choice(["Male", "Female", "Other"], n, p=[0.49, 0.49, 0.02]),
    "conversion_rate": np.random.beta(2, 5, n).round(4),             # 0-1 rates
    "stock_price":     np.random.lognormal(mean=4, sigma=0.5, size=n).round(2),
})
```

Add a manual correlation between two features:

```python
age = np.random.normal(35, 10, n).clip(18, 70)
salary = age * 1500 + np.random.normal(0, 5000, n)  # salary correlated with age
```

## 2. Faker — realistic fake PII

```python
# pip install faker
from faker import Faker
import pandas as pd, random

fake = Faker("en_IN")  # locale-aware; swap for other regions
Faker.seed(42)

records = []
for _ in range(1000):
    records.append({
        "name": fake.name(), "email": fake.email(), "phone": fake.phone_number(),
        "address": fake.address(), "city": fake.city(), "pincode": fake.postcode(),
        "dob": fake.date_of_birth(minimum_age=18, maximum_age=65),
        "salary": round(fake.random.uniform(20000, 200000), 2),
        "company": fake.company(), "job_title": fake.job(),
        "ip_address": fake.ipv4(),
    })
df = pd.DataFrame(records)
```

Provider categories: Person (`name`, `first_name`), Address (`address`, `city`), Internet
(`email`, `url`, `ipv4`), Finance (`bban`, `credit_card_number`), Date/Time (`date_of_birth`,
`date_between`), Company (`company`, `job`), Misc (`uuid4`, `boolean`, `numerify`).

## 3. Scikit-learn dataset generators

For algorithm testing/benchmarking — returns `X, y` directly.

```python
from sklearn.datasets import make_classification, make_regression, make_blobs, make_moons

X, y = make_classification(
    n_samples=1000, n_features=10, n_informative=5, n_redundant=2,
    n_classes=2, weights=[0.7, 0.3], flip_y=0.01, random_state=42,
)
X, y = make_regression(n_samples=1000, n_features=8, noise=15.0, random_state=42)
X, y = make_blobs(n_samples=500, centers=4, cluster_std=1.0, random_state=42)
X, y = make_moons(n_samples=500, noise=0.1, random_state=42)  # non-linear boundary test
```

## 4. SMOTE family — class imbalance

Creates synthetic minority-class samples by interpolating between real minority points — it does not duplicate rows or add random noise. Apply **before** train/test split is finalized, and only to the training fold.

```python
# pip install imbalanced-learn
from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE, SVMSMOTE, SMOTENC
from imblearn.combine import SMOTETomek, SMOTEENN
from collections import Counter

print("Before:", Counter(y))  # e.g. {0: 950, 1: 50}

smote = SMOTE(sampling_strategy="auto", k_neighbors=5, random_state=42)
X_res, y_res = smote.fit_resample(X, y)
print("After SMOTE:", Counter(y_res))

# BorderlineSMOTE — generates only near the decision boundary
X_res, y_res = BorderlineSMOTE(kind="borderline-1", random_state=42).fit_resample(X, y)

# ADASYN — generates more where the classifier struggles most
X_res, y_res = ADASYN(random_state=42).fit_resample(X, y)

# SMOTENC — when some columns are categorical (pass their column indices)
# X_res, y_res = SMOTENC(categorical_features=[2, 4], random_state=42).fit_resample(X_mixed, y)
```

How it works internally: pick minority sample A → find its k nearest minority neighbors →
pick random neighbor B → new point = `A + random(0,1) * (B - A)`.

## 5. SDV (Synthetic Data Vault) — statistical modeling

Learns the real dataset's statistical structure (including correlations) and generates a
privacy-safe synthetic copy. Best open-source option for tabular data you have but can't share.

```python
# pip install sdv
from sdv.single_table import GaussianCopulaSynthesizer
from sdv.metadata import SingleTableMetadata
from sdv.evaluation.single_table import evaluate_quality

metadata = SingleTableMetadata()
metadata.detect_from_dataframe(real_df)
metadata.update_column("dob", sdtype="datetime", datetime_format="%Y-%m-%d")

synthesizer = GaussianCopulaSynthesizer(metadata)
synthesizer.fit(real_df)
synthetic_df = synthesizer.sample(num_rows=1000)

quality_report = evaluate_quality(real_df, synthetic_df, metadata)
print(quality_report.get_score())
```

Multi-table / relational data:

```python
from sdv.multi_table import HMASynthesizer
from sdv.metadata import MultiTableMetadata

metadata = MultiTableMetadata()
metadata.detect_from_dataframes({"customers": customers_df, "orders": orders_df})
metadata.add_relationship(
    parent_table_name="customers", parent_primary_key="customer_id",
    child_table_name="orders", child_foreign_key="customer_id",
)
synthesizer = HMASynthesizer(metadata)
synthesizer.fit({"customers": customers_df, "orders": orders_df})
synthetic = synthesizer.sample(scale=1.0)
```

## 6. CTGAN — Conditional Tabular GAN

Best when relationships between columns are non-linear and GaussianCopula quality isn't enough.

```python
from sdv.single_table import CTGANSynthesizer

synthesizer = CTGANSynthesizer(metadata, epochs=500, batch_size=500, verbose=True)
synthesizer.fit(real_df)
synthetic_df = synthesizer.sample(num_rows=1000)
```

Standalone (without SDV):

```python
# pip install ctgan
from ctgan import CTGAN

discrete_cols = ["gender", "city", "purchased", "category"]
model = CTGAN(epochs=300, verbose=True)
model.fit(real_df, discrete_cols)
synthetic_df = model.sample(1000)
```

## 7. TVAE — Tabular VAE

Often faster than CTGAN with comparable quality, especially for mostly-continuous data.

```python
from sdv.single_table import TVAESynthesizer

synthesizer = TVAESynthesizer(metadata, epochs=300, batch_size=500)
synthesizer.fit(real_df)
synthetic_df = synthesizer.sample(1000)
```

## 8. Image GANs (medical / vision data)

| Type | Use case |
|---|---|
| DCGAN | General image synthesis |
| StyleGAN2/3 | High quality face/texture generation |
| CycleGAN | Unpaired translation (e.g. MRI → CT) |
| Pix2Pix | Paired image translation |
| WGAN-GP | Stable training, better quality |

DCGAN generator/discriminator sketch (PyTorch) — use as a starting skeleton, not production code:

```python
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, latent_dim=100, img_channels=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 512, 4, 1, 0, bias=False), nn.BatchNorm2d(512), nn.ReLU(True),
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False), nn.BatchNorm2d(256), nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False), nn.BatchNorm2d(128), nn.ReLU(True),
            nn.ConvTranspose2d(128, img_channels, 4, 2, 1, bias=False), nn.Tanh(),
        )
    def forward(self, z):
        return self.net(z.view(*z.shape, 1, 1))
```

## 9. Diffusion models (best current image quality)

```python
# pip install diffusers transformers
from diffusers import StableDiffusionPipeline
import torch

pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16
).to("cuda")

images = pipe("brain MRI scan, T1 weighted, axial view, medical imaging",
              num_images_per_prompt=4).images
for i, img in enumerate(images):
    img.save(f"synthetic_brain_{i}.png")
```

Note: synthetic medical images for training/testing must be clearly labeled as synthetic and
never presented as real patient data.

## 10. Time series

```python
import numpy as np, pandas as pd

n = 365
dates = pd.date_range("2024-01-01", periods=n, freq="D")
trend = np.linspace(100, 200, n)
seasonal = 20 * np.sin(2 * np.pi * np.arange(n) / 30)
noise = np.random.normal(0, 5, n)
df = pd.DataFrame({"date": dates, "sales": trend + seasonal + noise})
```

For complex sequential patterns, `TimeGAN` (via `ydata-synthetic`) learns temporal dynamics
directly from a real series instead of hand-specifying trend/seasonality.

## 11. Text / NLP synthetic data

```python
import random

templates = [
    "The {product} was {adjective} and {adjective2}.",
    "I {verb} this {product}. {sentiment} recommendation.",
]
products = ["laptop", "phone", "headphones", "tablet"]
adjectives = ["excellent", "terrible", "average", "outstanding"]

reviews = [random.choice(templates).format(
    product=random.choice(products), adjective=random.choice(adjectives),
    adjective2=random.choice(adjectives), verb="loved", sentiment="Highly recommend",
) for _ in range(100)]
```

For higher-quality, varied text, generate via an LLM API with an explicit anonymization
instruction (e.g. "anonymize all names") rather than templating.

## Comparison table

| Method | Data type | Quality | Speed | Privacy safe | Needs real data |
|---|---|---|---|---|---|
| Statistical (numpy) | Tabular | Low | Fastest | Yes | No |
| Faker | PII/Text | Medium | Fast | Yes | No |
| sklearn generators | Tabular | Low | Fastest | Yes | No |
| SMOTE / variants | Tabular | Medium | Fast | Partial | Yes |
| GaussianCopula | Tabular | Medium-High | Fast | Yes | Yes |
| CTGAN | Tabular | High | Slow | Yes | Yes |
| TVAE | Tabular | High | Medium | Yes | Yes |
| DCGAN | Images | Medium | Slow | Yes | Yes |
| StyleGAN3 | Images | Very High | Very Slow | Yes | Yes |
| Diffusion | Images | Best | Very Slow | Yes | Yes |
| TimeGAN | Time series | High | Slow | Yes | Yes |
| LLM-based | Text | Very High | Medium | Yes | No |

Next step for anything generated here: `validation-quality.md`.
