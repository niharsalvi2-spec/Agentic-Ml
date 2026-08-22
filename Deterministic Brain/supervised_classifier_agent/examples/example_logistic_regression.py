"""Runnable example: Logistic Regression, sklearn vs from-scratch."""
from _common import get_dataset
from logistic_regression import SklearnLogisticRegression, ScratchLogisticRegression

X_train, X_test, y_train, y_test = get_dataset()

sk_model = SklearnLogisticRegression(C=1.0).fit(X_train, y_train)
print("sklearn  accuracy:", sk_model.score(X_test, y_test))

scratch_model = ScratchLogisticRegression(lr=0.5, n_iters=2000).fit(X_train, y_train)
print("scratch  accuracy:", scratch_model.score(X_test, y_test))

print("sklearn  proba[:3]:\n", sk_model.predict_proba(X_test[:3]))
print("scratch  proba[:3]:\n", scratch_model.predict_proba(X_test[:3]))
