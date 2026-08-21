from sklearn.metrics import accuracy_score, f1_score
def compute_classification_metrics(y_true, y_pred):
    return {"accuracy": float(accuracy_score(y_true, y_pred)), "f1": float(f1_score(y_true, y_pred, average='weighted'))}
