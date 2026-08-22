"""
Runnable example: the FULL pipeline this agent sits at the end of.

Simulates what your 11 upstream agents would hand off:
  - Dataset_Collector_Agent  -> raw dataframe + dataset_info
  - Dataset_Cleaner_Agent /
    Data_Encoding_Agent      -> numeric_cols / categorical_cols split
  - Supervised_Classifier_Agent (built earlier in this project) -> trained pipeline
  - model_eval_toolkit (built earlier in this project)          -> metrics report
  - PKLGeneratorAgent (this package)                             -> final .pkl

If the earlier packages aren't on your path, adjust the sys.path lines below
to point at wherever you saved `supervised_classifier_agent/code` and
`model_eval_toolkit/code`.
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "code"))
# adjust these two to your actual paths if running standalone:
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "supervised_classifier_agent", "code"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "model_eval_toolkit", "code"))

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

from pkl_generator_agent import PKLGeneratorAgent

# --- Dataset_Collector_Agent output (simulated) ---
np.random.seed(42)
n = 800
df = pd.DataFrame({
    "age": np.random.randint(18, 70, n),
    "salary": np.random.normal(50000, 15000, n).clip(15000, 200000),
    "experience": np.random.randint(0, 40, n),
    "city": np.random.choice(["Mumbai", "Delhi", "Pune", "Bangalore"], n),
    "department": np.random.choice(["IT", "HR", "Finance", "Sales"], n),
    "target": np.random.randint(0, 2, n),
})
dataset_info = {"source": "synthetic_hr_dataset", "n_rows": n, "collected_via": "Dataset_Collector_Agent"}

# --- Dataset_Cleaner_Agent / Data_Encoding_Agent output (simulated column split) ---
numeric_cols = ["age", "salary", "experience"]
categorical_cols = ["city", "department"]
target_column = "target"

X = df.drop(columns=[target_column])
y = df[target_column]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# --- Supervised_Classifier_Agent output: a trained pipeline ---
# (using a plain sklearn Pipeline here to keep this example self-contained;
#  swap in SupervisedClassifierAgent.get_model(...) from that package directly)
preprocessor = ColumnTransformer([
    ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), numeric_cols),
    ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                       ("encoder", OneHotEncoder(handle_unknown="ignore"))]), categorical_cols),
])
from sklearn.ensemble import RandomForestClassifier
full_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42)),
])
full_pipeline.fit(X_train, y_train)

# --- model_eval_toolkit output: full metric report ---
try:
    from evaluation_agent import EvaluationAgent
    eval_agent = EvaluationAgent()
    y_pred = full_pipeline.predict(X_test)
    y_score = full_pipeline.predict_proba(X_test)[:, 1]
    metrics = eval_agent.evaluate_classification(y_test.values, y_pred, y_score)
except ImportError:
    # fallback if model_eval_toolkit isn't on the path in this environment
    metrics = {"test_accuracy": float(full_pipeline.score(X_test, y_test))}

print("Evaluation metrics:", metrics)

# --- PKLGeneratorAgent: bundle everything into the final deliverable ---
agent = PKLGeneratorAgent(save_dir="pkl_output")
result = agent.generate(
    pipeline_or_model=full_pipeline,
    task="classification",
    model_name="hr_attrition_classifier",
    feature_columns=list(X.columns),
    numeric_cols=numeric_cols,
    categorical_cols=categorical_cols,
    target_column=target_column,
    classes=list(full_pipeline.classes_),
    metrics=metrics,
    dataset_info=dataset_info,
    description="Random Forest classifier, full pipeline with preprocessing baked in",
    register_version=True,
)
print("\nFinal PKL:", result["filepath"])
print("sha256:", result["sha256"])

# --- Downstream / serving usage ---
loader = PKLGeneratorAgent.load(result["filepath"])
print("\n" + loader.summary())

sample = X_test.head(3)
print("\npredictions:", loader.predict(sample))
print("probabilities:\n", loader.predict_proba(sample))
