"""
collector_utils.py
Reusable building blocks for any data-collection script (scraping, API, DB, synthetic).
See ../references/code-generation.md for usage patterns and rationale.
"""
import logging
import os
import random
import time
from functools import wraps

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
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
                    delay += random.uniform(0, delay * 0.1)
                    logger.warning(
                        f"{func.__name__} failed ({e}); retry {attempt}/{max_attempts} in {delay:.1f}s"
                    )
                    time.sleep(delay)
        return wrapper
    return decorator


class RateLimiter:
    """Caps calls to N per second so you don't get rate-limited or banned."""

    def __init__(self, calls_per_second=2):
        self.min_interval = 1.0 / calls_per_second
        self._last_call = 0.0

    def wait(self):
        elapsed = time.monotonic() - self._last_call
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)
        self._last_call = time.monotonic()


def collect_with_checkpoint(items, fetch_fn, out_path, batch_size=100, key_fn=str):
    """
    Collect `items` via `fetch_fn(item) -> dict`, flushing to `out_path` (CSV) every
    `batch_size` items and tracking completed keys in `<out_path>.done` so a rerun after
    a crash resumes instead of re-collecting from scratch.
    """
    done_path = out_path + ".done"
    already_done = set()
    if os.path.exists(done_path):
        with open(done_path) as f:
            already_done = set(line.strip() for line in f if line.strip())

    buffer = []
    collected, skipped = 0, 0

    def flush():
        nonlocal buffer
        if not buffer:
            return
        header = not os.path.exists(out_path)
        pd.DataFrame(buffer).to_csv(out_path, mode="a", index=False, header=header)
        with open(done_path, "a") as f:
            f.write("\n".join(key_fn(b.get("_key", "")) for b in buffer) + "\n")
        buffer = []

    for item in items:
        key = key_fn(item)
        if key in already_done:
            continue
        try:
            record = fetch_fn(item)
            record["_key"] = key
            buffer.append(record)
            already_done.add(key)
            collected += 1
        except Exception as e:
            logger.error(f"Skipping {key}: {e}")
            skipped += 1
            continue

        if len(buffer) >= batch_size:
            flush()

    flush()
    logger.info(f"Collection complete: {collected} collected, {skipped} skipped -> {out_path}")
    return collected, skipped


def quick_quality_check(df: pd.DataFrame) -> None:
    """Minimum viable check before handing data off to cleaning/EDA."""
    logger.info(f"shape={df.shape}")
    logger.info(f"dtypes=\n{df.dtypes}")
    nulls = df.isnull().sum()
    logger.info(f"nulls=\n{nulls[nulls > 0] if nulls.any() else 'none'}")
    dupes = df.duplicated().sum()
    if dupes:
        logger.warning(f"{dupes} duplicate rows found")
