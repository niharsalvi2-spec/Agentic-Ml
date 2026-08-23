"""
TargetResolver — Canonical target column inference for datasets.
"""
from dataclasses import dataclass
from typing import Optional, List
import pandas as pd

COMMON_TARGET_NAMES: List[str] = [
    "target", "label", "churn", "y", "class", "status", "price", "is_churn", "fraud"
]


@dataclass
class TargetResolution:
    column: Optional[str]
    method: str  # "explicit" | "common_name" | "last_column" | "none"
    confidence: float


def resolve_target(df: pd.DataFrame, requested_target: Optional[str] = None) -> TargetResolution:
    """
    Deterministically resolve target column from user request or dataset inspection.
    """
    if requested_target:
        if requested_target in df.columns:
            return TargetResolution(
                column=requested_target,
                method="explicit",
                confidence=1.0,
            )
        # If requested target is missing, check case-insensitive match
        for col in df.columns:
            if str(col).lower() == requested_target.lower():
                return TargetResolution(
                    column=str(col),
                    method="explicit",
                    confidence=0.98,
                )

    # Search for known domain target column names case-insensitively
    lower_map = {str(c).lower(): str(c) for c in df.columns}
    for candidate in COMMON_TARGET_NAMES:
        if candidate in lower_map:
            return TargetResolution(
                column=lower_map[candidate],
                method="common_name",
                confidence=0.90,
            )

    # Fallback to last column if dataframe is non-empty
    if len(df.columns) > 0:
        return TargetResolution(
            column=str(df.columns[-1]),
            method="last_column",
            confidence=0.60,
        )

    return TargetResolution(
        column=None,
        method="none",
        confidence=0.0,
    )
