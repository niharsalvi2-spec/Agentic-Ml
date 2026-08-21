import pandas as pd
from typing import Dict, Any, Tuple
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, f1_score, r2_score, mean_squared_error

class ModelEvaluator:
    """Evaluates and cross-validates trained models."""
    
    @staticmethod
    def evaluate(models: Dict[str, Any], X: pd.DataFrame, y: pd.Series, task_type: str = "classification") -> Tuple[str, Dict[str, float]]:
        best_model_name = ""
        best_score = -float("inf")
        results = {}
        
        scoring = "accuracy" if task_type == "classification" else "r2"
        
        for name, model in models.items():
            scores = cross_val_score(model, X, y, cv=3, scoring=scoring)
            mean_score = float(scores.mean())
            results[name] = mean_score
            
            if mean_score > best_score:
                best_score = mean_score
                best_model_name = name
                
        return best_model_name, results
