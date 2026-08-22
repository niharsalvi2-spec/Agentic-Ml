"""
Runnable example: registering multiple model versions, promoting one to
production, and comparing metrics across all versions — a lightweight
model registry backed by pkl_generator_agent.PKLGeneratorAgent.version_manager.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "code"))

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from pkl_generator_agent import PKLGeneratorAgent

X = np.random.RandomState(1).randn(400, 5)
y = (X[:, 0] + 0.5 * X[:, 1] > 0).astype(int)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

agent = PKLGeneratorAgent(save_dir="pkl_output_versions")

# Version 1: quick logistic regression baseline
model_v1 = LogisticRegression(max_iter=1000).fit(X_train, y_train)
agent.generate(
    pipeline_or_model=model_v1, task="classification", model_name="churn_model",
    feature_columns=[f"f{i}" for i in range(5)],
    metrics={"test_accuracy": float(model_v1.score(X_test, y_test))},
    description="Baseline logistic regression",
    register_version=True, verbose=False,
)

# Version 2: tuned random forest, better accuracy
model_v2 = RandomForestClassifier(n_estimators=300, max_depth=6, random_state=1).fit(X_train, y_train)
agent.generate(
    pipeline_or_model=model_v2, task="classification", model_name="churn_model",
    feature_columns=[f"f{i}" for i in range(5)],
    metrics={"test_accuracy": float(model_v2.score(X_test, y_test))},
    description="Tuned Random Forest",
    register_version=True, verbose=False,
)

# Compare all versions
agent.version_manager.compare_versions("churn_model")

# Promote v2 to production (assuming it scored higher — check the printed table above)
prod_path = agent.version_manager.promote_to_production("churn_model", "v2")
print(f"\nPromoted to production: {prod_path}")

# Load whatever is currently in production, without needing to know the version number
production_bundle = agent.version_manager.load_production("churn_model")
print(f"Production model metrics: {production_bundle['metrics']}")
