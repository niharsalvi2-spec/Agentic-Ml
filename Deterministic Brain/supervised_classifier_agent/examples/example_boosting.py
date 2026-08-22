"""Runnable example: Boosting - Gradient Boosting (sklearn) vs AdaBoost (from-scratch)."""
from _common import get_dataset
from boosting import SklearnBoosting, ScratchBoosting

# note: ScratchBoosting (AdaBoost) is binary-only, so use n_classes=2
X_train, X_test, y_train, y_test = get_dataset(n_classes=2)

sk_model = SklearnBoosting(n_estimators=150, learning_rate=0.1, max_depth=3).fit(X_train, y_train)
print("sklearn  (Gradient Boosting) accuracy:", sk_model.score(X_test, y_test))

scratch_model = ScratchBoosting(n_estimators=50, learning_rate=1.0).fit(X_train, y_train)
print("scratch  (AdaBoost) accuracy:", scratch_model.score(X_test, y_test))
