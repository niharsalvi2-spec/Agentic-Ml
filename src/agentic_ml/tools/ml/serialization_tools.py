import joblib
def dump_model(model, filepath): joblib.dump(model, filepath)
def load_model(filepath): return joblib.load(filepath)
