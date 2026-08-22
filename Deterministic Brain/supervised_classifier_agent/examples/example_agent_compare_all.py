"""Runnable example: train + evaluate every registered classifier on one
dataset and print a ranked leaderboard."""
from _common import get_dataset
from supervised_classifier_agent import SupervisedClassifierAgent

X_train, X_test, y_train, y_test = get_dataset(n_samples=600, n_features=12)

agent = SupervisedClassifierAgent()

print("=== sklearn versions (fast) ===")
sklearn_models = [m for m in agent.list_models() if m.endswith("_sklearn")]
results_sk = agent.compare_all(X_train, y_train, X_test, y_test, models=sklearn_models)

print("\n=== from-scratch versions (slower, educational) ===")
scratch_models = [m for m in agent.list_models() if m.endswith("_scratch")]
results_scratch = agent.compare_all(X_train, y_train, X_test, y_test, models=scratch_models)

print("\n=== Top 3 overall (sklearn) ===")
for r in results_sk[:3]:
    print(" ", r["model"], "-> accuracy:", r["accuracy"], "f1:", r["f1"])
