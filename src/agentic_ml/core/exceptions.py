class AgenticMLException(Exception):
    """Base exception for all Agentic ML errors."""
    pass

class DataNotFoundError(AgenticMLException):
    """Raised when required dataset is missing."""
    pass

class ModelTrainingError(AgenticMLException):
    """Raised when ML model training fails."""
    pass

class ValidationFailedError(AgenticMLException):
    """Raised when model does not meet performance requirements."""
    pass
