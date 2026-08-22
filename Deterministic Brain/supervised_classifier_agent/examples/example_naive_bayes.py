"""Runnable example: Gaussian Naive Bayes, sklearn vs from-scratch."""
from _common import get_dataset
from naive_bayes import SklearnNaiveBayes, ScratchNaiveBayes

X_train, X_test, y_train, y_test = get_dataset()

sk_model = SklearnNaiveBayes().fit(X_train, y_train)
print("sklearn  accuracy:", sk_model.score(X_test, y_test))

scratch_model = ScratchNaiveBayes().fit(X_train, y_train)
print("scratch  accuracy:", scratch_model.score(X_test, y_test))
