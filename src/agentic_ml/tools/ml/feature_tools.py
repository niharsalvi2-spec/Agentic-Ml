from sklearn.feature_selection import SelectKBest, f_classif
def get_kbest_selector(k=5): return SelectKBest(f_classif, k=k)
