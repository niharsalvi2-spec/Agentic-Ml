import pandas as pd
from typing import Dict, Any, Tuple
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, Ridge

class ModelTrainer:
    """Trains and compares baseline candidate models."""
    
    @staticmethod
    def train_candidates(X: pd.DataFrame, y: pd.Series, task_type: str = "classification") -> Dict[str, Any]:
        models = {}
        if task_type == "classification":
            models["RandomForest"] = RandomForestClassifier(n_estimators=50, random_state=42)
            models["GradientBoosting"] = GradientBoostingClassifier(n_estimators=50, random_state=42)
            models["LogisticRegression"] = LogisticRegression(max_iter=200, random_state=42)
        else:
            models["RandomForest"] = RandomForestRegressor(n_estimators=50, random_state=42)
            models["GradientBoosting"] = GradientBoostingRegressor(n_estimators=50, random_state=42)
            models["Ridge"] = Ridge()
            
        trained = {}
        for name, model in models.items():
            model.fit(X, y)
            trained[name] = model
            
        return trained
