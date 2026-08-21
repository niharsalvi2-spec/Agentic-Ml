# Code Generation — Reusable Collector Skeleton

Every collector script this skill produces — scraper, API client, DB extractor, synthetic
generator — should be built on top of these patterns rather than a bare loop calling
`requests.get`/`cursor.execute` directly. This is what turns a one-off script into something
that survives a flaky network, a rate limit, or a crash halfway through a 10,000-row pull.

## 1. Retry with exponential backoff (wrap every network/DB call)

```python
import time
import random
import logging
from functools import wraps

logger = logging.getLogger("collector")

def with_retry(max_attempts=5, base_delay=1.0, max_delay=30.0, retry_on=(Exception,)):
    """Decorator: retries a function with exponential backoff + jitter."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            attempt = 0
            while True:
                try:
                    return func(*args, **kwargs)
                except retry_on as e:
                    attempt += 1
                    if attempt >= max_attempts:
                        logger.error(f"{func.__name__} failed after {attempt} attempts: {e}")
                        raise
                    delay = min(max_delay, base_delay * (2 ** (attempt - 1)))
                    delay += random.uniform(0, delay * 0.1)  # jitter avoids thundering herd
                    logger.warning(f"{func.__name__} failed ({e}); retry {attempt}/{max_attempts} in {delay:.1f}s")
                    time.sleep(delay)
        return wrapper
    return decorator
```

Usage:

```python
import requests

@with_retry(max_attempts=5, retry_on=(requests.RequestException,))
def fetch_page(url):
    resp = requests.get(url, timeout=10)
    resp.raise_for_status()
    return resp
```

## 2. Rate limiting (respect the source, avoid 429s)

```python
import time

class RateLimiter:
    def __init__(self, calls_per_second=2):
        self.min_interval = 1.0 / calls_per_second
        self._last_call = 0.0

    def wait(self):
        elapsed = time.monotonic() - self._last_call
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)
        self._last_call = time.monotonic()

limiter = RateLimiter(calls_per_second=2)
for url in urls:
    limiter.wait()
    fetch_page(url)
```

## 3. Incremental checkpointing (never lose 9,000 rows because row 9,001 crashed)

```python
import pandas as pd
import os

def collect_with_checkpoint(items, fetch_fn, out_path, batch_size=100):
    """fetch_fn(item) -> dict. Appends to out_path every batch_size items."""
    done_path = out_path + ".done"
    already_done = set()
    if os.path.exists(done_path):
        with open(done_path) as f:
            already_done = set(line.strip() for line in f)

    buffer = []
    for item in items:
        key = str(item)
        if key in already_done:
            continue
        try:
            buffer.append(fetch_fn(item))
            already_done.add(key)
        except Exception as e:
            logger.error(f"Skipping {key}: {e}")
            continue

        if len(buffer) >= batch_size:
            _flush(buffer, out_path, done_path, already_done)
            buffer = []

    if buffer:
        _flush(buffer, out_path, done_path, already_done)

def _flush(buffer, out_path, done_path, done_keys):
    header = not os.path.exists(out_path)
    pd.DataFrame(buffer).to_csv(out_path, mode="a", index=False, header=header)
    with open(done_path, "a") as f:
        f.write("\n".join(str(k) for k in list(done_keys)[-len(buffer):]) + "\n")
```

Rerunning the same script after a crash resumes instead of re-collecting from scratch.

## 4. Structured logging

```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[logging.FileHandler("collector.log"), logging.StreamHandler()],
)
logger = logging.getLogger("collector")
```

Log at minimum: start/end of run, item counts collected vs skipped/failed, and every retry —
not just final totals. A collector that only logs "Done, 4213 rows" gives no way to diagnose
why it's short of the 5000 expected.

## 5. Config-driven sources (so the same script handles many endpoints/tables)

```python
from dataclasses import dataclass

@dataclass
class SourceConfig:
    name: str
    kind: str          # "api" | "scrape" | "db" | "file"
    endpoint: str
    rate_limit: float = 2.0
    max_attempts: int = 5

SOURCES = [
    SourceConfig(name="countries", kind="api", endpoint="https://restcountries.com/v3.1/all"),
    SourceConfig(name="products", kind="scrape", endpoint="https://example.com/products", rate_limit=1.0),
]

for cfg in SOURCES:
    logger.info(f"Starting collection: {cfg.name}")
    # dispatch to the right collector function based on cfg.kind
```

## 6. Minimal end-to-end template (assemble the above into one script)

```python
"""
Generic collector template — copy per source, fill in `fetch_one` and `items`.
"""
import logging
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("collector")

limiter = RateLimiter(calls_per_second=2)

@with_retry(max_attempts=5)
def fetch_one(item):
    limiter.wait()
    # replace with requests.get / cursor.execute / soup.find / synthesizer.sample etc.
    raise NotImplementedError

def main():
    items = []  # e.g. list of URLs, IDs, page numbers, or table names
    collect_with_checkpoint(items, fetch_one, out_path="output.csv", batch_size=100)
    df = pd.read_csv("output.csv")
    logger.info(f"Collected {len(df)} rows -> output.csv")
    # hand off to validation-quality.md checks here

if __name__ == "__main__":
    main()
```

## Rules of thumb when generating collector code for a user

1. Default to this skeleton unless the task is a single quick pull of a few known rows.
2. Always parametrize secrets (API keys, DB passwords) via environment variables, never inline.
3. Prefer `timeout=` on every network call — an unbounded hang is worse than a clean failure.
4. Log counts, not just a success message — "collected 812/1000, 188 skipped (see log)" is
   actionable; "done" is not.
5. End every generated collector script with a call into the checks from
   `validation-quality.md` (at minimum shape/dtype/null checks) rather than leaving that as a
   manual follow-up step for the user.
