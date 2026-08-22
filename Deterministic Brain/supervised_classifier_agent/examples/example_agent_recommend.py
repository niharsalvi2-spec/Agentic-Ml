"""Runnable example: ask SupervisedClassifierAgent which model fits your task
BEFORE training anything, using its rule-based decision guide."""
import _common  # noqa: F401 - adds ../code to sys.path
from supervised_classifier_agent import SupervisedClassifierAgent

agent = SupervisedClassifierAgent()

print("Scenario 1: small, high-dim, need interpretability + probabilities")
for r in agent.recommend(
    n_samples=800, n_features=300, need_interpretability=True,
    need_proba=True, suspect_nonlinear=False,
):
    print(" ", r)

print("\nScenario 2: large tabular dataset, nonlinear, no interpretability needed")
for r in agent.recommend(
    n_samples=80000, n_features=25, need_interpretability=False,
    suspect_nonlinear=True, need_fast_inference=True,
):
    print(" ", r)

print("\nScenario 3: imbalanced classes, has outliers, want probabilities")
for r in agent.recommend(
    n_samples=3000, n_features=15, is_imbalanced=True,
    has_outliers=True, need_proba=True,
):
    print(" ", r)
