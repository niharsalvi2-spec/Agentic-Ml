"""
Dataset Schema and Data Contract Specifications.
"""

from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field


class ColumnSchema(BaseModel):
    name: str
    dtype: str
    is_nullable: bool = True
    cardinality: Optional[int] = None
    is_target: bool = False


class DatasetSchema(BaseModel):
    n_rows: int
    n_columns: int
    target_column: Optional[str] = None
    feature_columns: List[str] = Field(default_factory=list)
    numeric_columns: List[str] = Field(default_factory=list)
    categorical_columns: List[str] = Field(default_factory=list)
    columns: Dict[str, ColumnSchema] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def validate_features(self, feature_names: List[str]) -> bool:
        expected = set(self.feature_columns)
        actual = set(feature_names)
        return expected.issubset(actual)
