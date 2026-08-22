# Public Datasets, Marketplaces & Crowdsourcing

Try this before scraping or generating synthetic data — someone may have already solved the problem.

## Public dataset sources

| Platform | Best for |
|---|---|
| Kaggle (kaggle.com/datasets) | ML competitions, tabular data |
| UCI ML Repository (archive.ics.uci.edu) | Classic ML datasets |
| Google Dataset Search | Any domain |
| Hugging Face (huggingface.co/datasets) | NLP, image datasets |
| data.gov.in | Indian government data |
| World Bank (data.worldbank.org) | Economic, global data |
| NASA (data.nasa.gov) | Space, climate data |

```python
# Kaggle API
# pip install kaggle
import kaggle
kaggle.api.dataset_download_files("username/dataset-name", path="./data", unzip=True)
```

## Data marketplaces (paid, licensed)

Use when the domain data is specific and you have budget:

| Marketplace | Speciality |
|---|---|
| AWS Data Exchange | Cloud-ready datasets |
| Snowflake Marketplace | Enterprise data sharing |
| Bloomberg / Refinitiv | Financial / market data |
| Dun & Bradstreet | Business / company data |

## Crowdsourcing (for labels, or data collected by humans)

Essential for supervised learning when ground-truth labels don't exist yet.

| Platform | Use case |
|---|---|
| Amazon Mechanical Turk | Large scale labeling tasks |
| Scale AI | ML training data, annotation |
| Labelbox | Image / video annotation |
| Appen | NLP, speech, image labeling |
| Roboflow | Computer vision dataset creation |

```python
# After getting labeled data back — resolve multi-annotator disagreement by majority vote
import pandas as pd

df = pd.read_csv("labeled_data.csv")
df["final_label"] = df[["label1", "label2", "label3"]].mode(axis=1)[0]
```

## Image / audio collection

See `web-scraping.md` for `icrawler` (images) and `yt-dlp` (audio) patterns.

## Decision guide

```
Already exists publicly
├── On a website          → web-scraping.md
├── Via API                → api-collection.md
├── Ready-made dataset     → this file (Kaggle/UCI/HuggingFace)
└── Government data        → this file (data.gov.in / World Bank)

Exists but restricted
├── Buy it                 → this file (marketplaces)
├── Cannot buy              → synthetic-data.md
└── Needs labels            → this file (crowdsourcing)
```
