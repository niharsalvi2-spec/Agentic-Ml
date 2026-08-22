"""Runnable example: K-Nearest Neighbors, sklearn vs from-scratch."""
from _common import get_dataset
from knn import SklearnKNN, ScratchKNN

X_train, X_test, y_train, y_test = get_dataset()

sk_model = SklearnKNN(n_neighbors=7, weights="distance").fit(X_train, y_train)
print("sklearn  accuracy:", sk_model.score(X_test, y_test))

scratch_model = ScratchKNN(k=7, weights="distance").fit(X_train, y_train)
print("scratch  accuracy:", scratch_model.score(X_test, y_test))
