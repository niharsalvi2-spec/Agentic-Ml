from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
def get_baseline_classifiers():
    return {"RandomForest": RandomForestClassifier(random_state=42), "GradientBoosting": GradientBoostingClassifier(random_state=42)}
