"""
Supervised Regression Model Families.
Provides standard regression initializers with metadata.
"""

from typing import Dict, Any
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor


def get_regression_models(random_state: int = 42) -> Dict[str, Any]:
    """Instantiates standard regression candidate models."""
    return {
        "LinearRegression": LinearRegression(),
        "Ridge": Ridge(alpha=1.0, random_state=random_state),
        "Lasso": Lasso(alpha=0.1, random_state=random_state),
        "ElasticNet": ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=random_state),
        "DecisionTree": DecisionTreeRegressor(max_depth=5, random_state=random_state),
        "RandomForest": RandomForestRegressor(n_estimators=100, random_state=random_state),
        "GradientBoosting": GradientBoostingRegressor(n_estimators=100, random_state=random_state),
        "SVR": SVR(),
        "KNN": KNeighborsRegressor(n_neighbors=5),
    }
