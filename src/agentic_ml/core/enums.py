from enum import Enum

class TaskType(str, Enum):
    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    CLUSTERING = "clustering"
    UNKNOWN = "unknown"

class ModelType(str, Enum):
    RANDOM_FOREST = "random_forest"
    LOGISTIC_REGRESSION = "logistic_regression"
    LINEAR_REGRESSION = "linear_regression"
    GRADIENT_BOOSTING = "gradient_boosting"
    DECISION_TREE = "decision_tree"

class PipelineStage(str, Enum):
    PROBLEM_ANALYSIS = "problem_analysis"
    DATA_COLLECTION = "data_collection"
    PREPROCESSING = "preprocessing"
    EDA = "eda"
    FEATURE_ENGINEERING = "feature_engineering"
    FEATURE_SELECTION = "feature_selection"
    MODEL_BUILDING = "model_building"
    TESTING = "testing"
    VALIDATION = "validation"
    DEPLOYMENT = "deployment"
