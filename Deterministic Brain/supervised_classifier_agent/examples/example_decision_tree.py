"""Runnable example: Decision Tree, sklearn vs from-scratch."""
from _common import get_dataset
from decision_tree import SklearnDecisionTree, ScratchDecisionTree

X_train, X_test, y_train, y_test = get_dataset()

sk_model = SklearnDecisionTree(max_depth=5).fit(X_train, y_train)
print("sklearn  accuracy:", sk_model.score(X_test, y_test))
print("sklearn  feature importances:", sk_model.feature_importances())

scratch_model = ScratchDecisionTree(max_depth=5).fit(X_train, y_train)
print("scratch  accuracy:", scratch_model.score(X_test, y_test))
