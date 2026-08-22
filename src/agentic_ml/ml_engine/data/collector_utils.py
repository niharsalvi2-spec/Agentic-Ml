"""
Data collection utilities.
Provides resilient ingestion with exponential backoff retries, rate limiting,
and batch checkpointing.
"""

import logging
import os
import random
import time
from functools import wraps
from typing import Callable, Any, Tuple, List, Dict, Optional
import pandas as pd

logger = logging.getLogger("agentic_ml.data.collector")


def with_retry(max_attempts: int = 5, base_delay: float = 1.0, max_delay: float = 30.0, retry_on=(Exception,)):
    """Decorator: retries a function with exponential backoff + jitter."""
    def decorator(func: Callable) -> Callable:
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
    """Caps calls to N per second to prevent rate-limiting or service throttling."""

    def __init__(self, calls_per_second: float = 2.0):
        self.min_interval = 1.0 / max(0.001, calls_per_second)
        self._last_call = 0.0

    def wait(self) -> None:
        elapsed = time.monotonic() - self._last_call
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)
        self._last_call = time.monotonic()


def collect_with_checkpoint(
    items: List[Any],
    fetch_fn: Callable[[Any], Dict[str, Any]],
    out_path: str,
    batch_size: int = 100,
    key_fn: Callable[[Any], str] = str
) -> Tuple[int, int]:
    """
    Collect items via fetch_fn(item) -> dict, flushing to out_path (CSV) every
    batch_size items and tracking completed keys in <out_path>.done so a rerun after
    a crash resumes instead of re-collecting from scratch.
    """
    done_path = out_path + ".done"
    already_done = set()
    if os.path.exists(done_path):
        with open(done_path, "r", encoding="utf-8") as f:
            already_done = set(line.strip() for line in f if line.strip())

    buffer: List[Dict[str, Any]] = []
    collected, skipped = 0, 0

    def flush():
        nonlocal buffer
        if not buffer:
            return
        header = not os.path.exists(out_path)
        pd.DataFrame(buffer).to_csv(out_path, mode="a", index=False, header=header)
        with open(done_path, "a", encoding="utf-8") as f:
            f.write("\n".join(str(b.get("_key", "")) for b in buffer) + "\n")
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


def quick_quality_check(df: pd.DataFrame) -> Dict[str, Any]:
    """Minimum viable quality and shape check before handing data off to cleaning/EDA."""
    nulls = df.isnull().sum()
    dupes = int(df.duplicated().sum())
    report = {
        "n_rows": int(df.shape[0]),
        "n_columns": int(df.shape[1]),
        "dtypes": {str(c): str(t) for c, t in df.dtypes.items()},
        "null_counts": {str(k): int(v) for k, v in nulls.items() if v > 0},
        "duplicate_rows": dupes,
        "is_empty": bool(df.empty)
    }
    return report
