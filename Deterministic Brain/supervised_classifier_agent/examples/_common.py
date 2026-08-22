"""Shared helper to build a small synthetic dataset for all examples."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "code"))

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def get_dataset(n_samples=500, n_features=10, n_classes=2, random_state=42):
    X, y = make_classification(
        n_samples=n_samples, n_features=n_features, n_informative=max(4, n_features // 2),
        n_classes=n_classes, random_state=random_state,
    )
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=random_state)
    scaler = StandardScaler().fit(X_train)
    return scaler.transform(X_train), scaler.transform(X_test), y_train, y_test
