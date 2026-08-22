"""Runnable example: SVM - sklearn (kernel-capable) vs from-scratch (linear only)."""
from _common import get_dataset
from svm import SklearnSVM, ScratchSVM

# note: ScratchSVM is binary-only (linear), so use n_classes=2
X_train, X_test, y_train, y_test = get_dataset(n_classes=2)

sk_model = SklearnSVM(C=1.0, kernel="rbf").fit(X_train, y_train)
print("sklearn  (RBF kernel) accuracy:", sk_model.score(X_test, y_test))

scratch_model = ScratchSVM(C=1.0, lr=0.001, n_iters=1000).fit(X_train, y_train)
print("scratch  (linear)     accuracy:", scratch_model.score(X_test, y_test))
