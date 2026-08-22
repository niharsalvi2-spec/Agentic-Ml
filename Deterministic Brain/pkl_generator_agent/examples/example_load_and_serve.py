"""
Runnable example: the minimal snippet a serving app / API endpoint needs to
load a production .pkl and answer prediction requests.

Run example_pkl_generator_agent.py first to produce pkl_output/hr_attrition_classifier.pkl.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "code"))

import pandas as pd
from pkl_generator_agent import PKLGeneratorAgent

PKL_PATH = os.path.join(os.path.dirname(__file__), "pkl_output", "hr_attrition_classifier.pkl")

if not os.path.exists(PKL_PATH):
    raise SystemExit("Run example_pkl_generator_agent.py first to generate the .pkl file.")

# This is essentially your whole "model server" load step:
loader = PKLGeneratorAgent.load(PKL_PATH)   # verifies integrity hash automatically
print(loader.summary())

# Simulate an incoming request (e.g. from a JSON body)
request_payload = {
    "age": 34, "salary": 62000, "experience": 9,
    "city": "Bangalore", "department": "IT",
}
X = pd.DataFrame([request_payload])

prediction = loader.predict(X)[0]
probability = loader.predict_proba(X)[0]
print(f"\nprediction: {prediction}")
print(f"probability: {dict(zip(loader.bundle['classes'], probability))}")

# Missing-field requests fail loudly and clearly instead of silently misaligning columns
bad_payload = {"age": 34, "salary": 62000}  # missing experience/city/department
try:
    loader.predict(pd.DataFrame([bad_payload]))
except ValueError as e:
    print(f"\nrejected malformed request: {e}")
