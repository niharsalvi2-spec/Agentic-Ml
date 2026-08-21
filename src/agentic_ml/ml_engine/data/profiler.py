import pandas as pd
from typing import Dict, Any, List

class DataProfiler:
    """Extracts statistical metadata and shapes from raw data."""
    
    @staticmethod
    def profile(df: pd.DataFrame, target_col: str) -> Dict[str, Any]:
        return {
            "n_rows": int(df.shape[0]),
            "n_columns": int(df.shape[1]),
            "feature_columns": [c for c in df.columns if c != target_col],
            "target_column": target_col,
            "missing_values": df.isnull().sum().to_dict(),
            "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
            "numerical_features": df.select_dtypes(include=["number"]).columns.tolist(),
            "categorical_features": df.select_dtypes(include=["object", "category"]).columns.tolist()
        }
