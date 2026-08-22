---
name: data-collector
description: Use this skill whenever the user needs to acquire, gather, scrape, request, query, or generate data for an ML/analytics project — from the web, an API, a database, files, IoT/streams, public dataset repositories, data marketplaces, crowdsourcing platforms, or synthetically when real data is unavailable, private, imbalanced, or too expensive to label. Trigger for phrases like "collect data", "scrape", "pull from API", "get data from the database", "I don't have a dataset", "generate synthetic/fake data", "fix class imbalance", "need more training data", or "build a data collector agent". This skill routes to the right collection method, writes working collection code, and validates the result before handing off to cleaning/EDA.
compatibility: python3, pandas; specific sub-methods need extra packages (requests, beautifulsoup4, scrapy, selenium, sqlalchemy, faker, scikit-learn, imbalanced-learn, sdv, ctgan, kafka-python, paho-mqtt) — installed per-method, not all at once.
---

# Data Collector Agent

A single skill that turns "I need data" into working collection code, for any source, including the case where no real data exists at all.

## Why this skill exists

Every ML pipeline actually starts one phase earlier than most tutorials admit:

```
Phase 0 → Data Collection / Synthetic Generation   ← this skill
Phase 1 → Data Cleaning
Phase 2 → Encoding
Phase 3 → EDA / Visualization
Phase 4 → Dimensionality Reduction
Phase 5 → Model Building (Supervised)
Phase 6 → Model Building (Unsupervised)
Phase 7 → Model Evaluation
Phase 8 → MLOps / Deployment
```

Model quality is bounded by data quality. "Garbage in, garbage out" is the single most common reason real projects fail before they ever reach modeling — so treat this phase with the same rigor as training.

## How to use this skill

1. **Diagnose the situation** using the decision tree below.
2. **Open the matching reference file** — don't try to hold every method in context at once, load only what's relevant.
3. **Generate collection code** using the patterns in `references/code-generation.md` (error handling, retries, logging, incremental saves — every collector needs these regardless of source).
4. **Validate what you collected** using `references/validation-quality.md` before handing off to the cleaning phase.

## Decision tree — which method do I need?

```
What data do you need?
│
├── Already exists publicly
│   ├── On a website (no API)     → references/web-scraping.md
│   ├── Behind an API             → references/api-collection.md
│   ├── A ready-made dataset      → references/public-datasets.md
│   └── Government / open data    → references/public-datasets.md
│
├── Exists but restricted
│   ├── Can be purchased          → references/public-datasets.md (marketplaces)
│   ├── Cannot be bought/shared   → references/synthetic-data.md
│   └── Needs human labels        → references/public-datasets.md (crowdsourcing)
│
├── Lives in your organization
│   ├── In a database             → references/database-collection.md
│   └── In files (CSV/JSON/Excel/PDF) → references/file-handling.md
│
├── Real-time / continuous
│   ├── High volume streams       → references/realtime-streams.md (Kafka)
│   └── IoT devices               → references/realtime-streams.md (MQTT)
│
└── Does not exist at all, or is unusable as-is
    ├── Privacy/legal restricted (GDPR/HIPAA/DPDP) → references/synthetic-data.md
    ├── Rare-event / class imbalance (fraud, disease) → references/synthetic-data.md (SMOTE family)
    ├── Need realistic PII for testing → references/synthetic-data.md (Faker)
    ├── Need a statistical clone of a real dataset → references/synthetic-data.md (SDV/CTGAN)
    └── Need to test a pipeline right now → references/synthetic-data.md (statistical generation)
```

If more than one branch applies (common — e.g. "scrape a site AND fix class imbalance after"), open each relevant reference file and chain the steps.

## Reference files

| File | Covers |
|---|---|
| `references/web-scraping.md` | BeautifulSoup (static), Scrapy (large-scale), Selenium (JS-rendered), robots.txt / legal etiquette |
| `references/api-collection.md` | REST with `requests`, auth/API keys, pagination, rate limits, HTTP status codes |
| `references/database-collection.md` | SQLAlchemy, psycopg2, core SQL patterns for extraction |
| `references/file-handling.md` | CSV, JSON (incl. nested), Excel, PDF text/table extraction |
| `references/realtime-streams.md` | Kafka (high volume), MQTT (IoT) |
| `references/public-datasets.md` | Kaggle/UCI/HuggingFace/gov data, data marketplaces, crowdsourcing, image/audio scraping |
| `references/synthetic-data.md` | Statistical generation, Faker, sklearn generators, SMOTE family, SDV, CTGAN/TVAE, image GANs/diffusion, time series, text/LLM generation |
| `references/validation-quality.md` | How to check synthetic (or scraped) data is actually usable: KS test, TSTR, correlation preservation, privacy leakage check |
| `references/code-generation.md` | Reusable code skeleton every collector should follow — retries, rate limiting, incremental checkpointing, logging, config-driven sources |

## Non-negotiable defaults (apply regardless of source)

- **Never hand back naked data-pulling code.** Every script must include error handling, retries with backoff, and incremental saving — see `code-generation.md`.
- **Check legality/ethics before scraping or buying.** `robots.txt`, ToS, GDPR/HIPAA/DPDP applicability. Flag this to the user rather than silently proceeding on grey-area sources.
- **Never fabricate synthetic PII as if it were real people.** Synthetic data must be clearly synthetic in code comments/output naming (`synthetic_*`), and privacy-validated before being treated as a GDPR-safe substitute (see `validation-quality.md`).
- **Always validate before handoff.** A collector that returns data without a quality/shape check is unfinished — run at least a schema check and (for synthetic data) a distribution comparison.
- **Prefer the least invasive method that satisfies the need.** API > scraping; public dataset > synthetic; statistical generation > GAN, unless the simpler method demonstrably doesn't meet the quality bar.
