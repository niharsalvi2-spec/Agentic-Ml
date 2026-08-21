import pandas as pd
from typing import Tuple
from sklearn.preprocessing import StandardScaler

class DeterministicPreprocessor:
    """Handles missing value imputation and scaling."""
    
    def __init__(self):
        self.scaler = StandardScaler()
        
    def fit_transform(self, df: pd.DataFrame, target_col: str) -> Tuple[pd.DataFrame, pd.Series]:
        # Drop duplicates
        cleaned = df.drop_duplicates().copy()
        
        # Split X and y
        X = cleaned.drop(columns=[target_col])
        y = cleaned[target_col]
        
        # Simple imputation for numeric columns
        num_cols = X.select_dtypes(include=["number"]).columns
        for col in num_cols:
            if X[col].isnull().sum() > 0:
                X[col] = X[col].fillna(X[col].median())
                
        # Scale numeric features
        if len(num_cols) > 0:
            X[num_cols] = self.scaler.fit_transform(X[num_cols])
            
        return X, y
