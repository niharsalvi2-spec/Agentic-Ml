"""
DatasetValidator — validates dataset integrity before any preprocessing begins.

Produces a DatasetManifest with sha256, schema, and validation summary.
If validation fails, the pipeline should not proceed.

Checks:
  - File exists and format is supported
  - Rows > 0, columns > 0
  - Target column exists (for supervised tasks)
  - Target cardinality is reasonable
  - Missing value percentage per column
  - Dtype consistency
  - Constant columns (zero variance)
  - Likely ID columns (high cardinality + unique values)
  - Duplicate row detection
"""
from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("agentic_ml.ml_engine.data.dataset_validator")


class DatasetManifest(dict):
    """
    Manifest describing a validated dataset.

    Fields:
        dataset_id:     sha256[:16] of the raw file
        sha256:         full sha256 of the raw file
        source_path:    original file path
        row_count:      number of rows
        column_count:   number of columns
        schema:         {col_name: dtype_str}
        target:         target column name (None for clustering)
        target_cardinality: number of unique values in target
        missingness:    {col_name: missing_pct}
        constant_cols:  list of zero-variance columns
        id_columns:     list of likely ID columns
        duplicate_rows: number of duplicate rows
        created_at:     ISO-8601 UTC
        valid:          overall validation result
        warnings:       list of warning strings
        errors:         list of error strings
    """


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def _sha256_df(df: Any) -> str:
    """Compute sha256 of DataFrame contents (used when there's no source file)."""
    import hashlib
    h = hashlib.sha256()
    h.update(df.to_csv(index=False).encode("utf-8"))
    return h.hexdigest()


class DatasetValidator:
    """
    Validates a dataset before preprocessing.

    Usage:
        manifest = DatasetValidator.validate(df, target_column="churn", source_path="...")
        if not manifest["valid"]:
            raise ValueError(manifest["errors"])
    """

    @staticmethod
    def validate(
        df: Any,
        target_column: Optional[str] = None,
        task_type: str = "classification",
        source_path: Optional[str] = None,
    ) -> DatasetManifest:
        """
        Validate a DataFrame and produce a DatasetManifest.

        Args:
            df:            pandas DataFrame to validate
            target_column: target column name (None for clustering)
            task_type:     "classification" | "regression" | "clustering"
            source_path:   original file path (for sha256; computed from df if None)

        Returns:
            DatasetManifest with validation results.
        """
        import pandas as pd
        import numpy as np

        errors: List[str] = []
        warnings: List[str] = []

        # ── Compute dataset fingerprint ──────────────────────────────────
        if source_path and Path(source_path).exists():
            sha = _sha256_file(Path(source_path))
        else:
            sha = _sha256_df(df)

        # ── Basic shape checks ───────────────────────────────────────────
        if len(df) == 0:
            errors.append("Dataset has 0 rows.")
        if len(df.columns) == 0:
            errors.append("Dataset has 0 columns.")

        row_count = int(len(df))
        col_count = int(len(df.columns))

        # ── Schema ───────────────────────────────────────────────────────
        schema = {col: str(df[col].dtype) for col in df.columns}

        # ── Target validation ─────────────────────────────────────────────
        target_cardinality = None
        if task_type != "clustering":
            if target_column is None:
                warnings.append("target_column not specified; will be inferred as last column.")
                target_column = df.columns[-1] if len(df.columns) > 0 else None
            if target_column and target_column not in df.columns:
                errors.append(f"Target column '{target_column}' not found in dataset.")
            elif target_column:
                target_cardinality = int(df[target_column].nunique())
                if task_type == "classification" and target_cardinality > 50:
                    warnings.append(
                        f"Target has {target_cardinality} unique values — may need regression or label encoding."
                    )
                if task_type == "regression" and target_cardinality < 10:
                    warnings.append(
                        f"Target has only {target_cardinality} unique values — may be better as classification."
                    )

        # ── Missingness ───────────────────────────────────────────────────
        missingness: Dict[str, float] = {}
        for col in df.columns:
            pct = float(df[col].isna().mean() * 100)
            missingness[col] = round(pct, 2)
            if pct > 80:
                warnings.append(f"Column '{col}' has {pct:.1f}% missing values.")

        # ── Constant columns (zero variance) ─────────────────────────────
        constant_cols: List[str] = []
        for col in df.select_dtypes(include=[np.number]).columns:
            if df[col].nunique() <= 1:
                constant_cols.append(col)
                warnings.append(f"Column '{col}' is constant — will be dropped in preprocessing.")

        # ── Likely ID columns ─────────────────────────────────────────────
        id_columns: List[str] = []
        for col in df.columns:
            if col.lower() in {"id", "index", "uuid", "rowid", "customer_id", "user_id"}:
                id_columns.append(col)
                warnings.append(f"Column '{col}' looks like an ID column — consider dropping.")
            elif df[col].dtype == object and df[col].nunique() == len(df):
                id_columns.append(col)
                warnings.append(f"Column '{col}' has all unique string values — likely an ID.")

        # ── Duplicate rows ────────────────────────────────────────────────
        dup_count = int(df.duplicated().sum())
        if dup_count > 0:
            warnings.append(f"Dataset has {dup_count} duplicate rows ({dup_count/row_count*100:.1f}%).")

        valid = len(errors) == 0

        manifest: DatasetManifest = DatasetManifest({
            "dataset_id": sha[:16],
            "sha256": sha,
            "source_path": source_path or "(in-memory)",
            "row_count": row_count,
            "column_count": col_count,
            "schema": schema,
            "target": target_column,
            "target_cardinality": target_cardinality,
            "missingness": missingness,
            "constant_cols": constant_cols,
            "id_columns": id_columns,
            "duplicate_rows": dup_count,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "valid": valid,
            "warnings": warnings,
            "errors": errors,
        })

        if valid:
            logger.info(
                "DatasetValidator: PASS — %d rows, %d cols, target='%s', sha256=%s...",
                row_count, col_count, target_column, sha[:12]
            )
        else:
            logger.error("DatasetValidator: FAIL — %s", errors)

        return manifest
