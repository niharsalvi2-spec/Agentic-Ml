"""
Supervised Classification Model Families.
Provides standard classifier initializers with metadata.
"""

from typing import Dict, Any
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC


def get_classification_models(random_state: int = 42) -> Dict[str, Any]:
    """Instantiates standard classification candidate models."""
    return {
        "LogisticRegression": LogisticRegression(max_iter=500, random_state=random_state),
        "NaiveBayes": GaussianNB(),
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "DecisionTree": DecisionTreeClassifier(max_depth=5, random_state=random_state),
        "RandomForest": RandomForestClassifier(n_estimators=100, random_state=random_state),
        "GradientBoosting": GradientBoostingClassifier(n_estimators=100, random_state=random_state),
        "SVM": SVC(probability=True, random_state=random_state),
    }
