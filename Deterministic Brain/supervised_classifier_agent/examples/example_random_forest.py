"""Runnable example: Random Forest, sklearn vs from-scratch."""
from _common import get_dataset
from random_forest import SklearnRandomForest, ScratchRandomForest

X_train, X_test, y_train, y_test = get_dataset()

sk_model = SklearnRandomForest(n_estimators=200, max_depth=6).fit(X_train, y_train)
print("sklearn  accuracy:", sk_model.score(X_test, y_test))
print("sklearn  feature importances:", sk_model.feature_importances())

scratch_model = ScratchRandomForest(n_estimators=15, max_depth=5).fit(X_train, y_train)
print("scratch  accuracy:", scratch_model.score(X_test, y_test))
