import pandas as pd
from typing import Dict, Any

class EDAEngine:
    """Computes descriptive and correlation statistics."""
    
    @staticmethod
    def analyze(df: pd.DataFrame) -> Dict[str, Any]:
        numeric_df = df.select_dtypes(include=["number"])
        return {
            "summary_stats": numeric_df.describe().to_dict(),
            "correlations": numeric_df.corr().to_dict() if not numeric_df.empty else {}
        }
