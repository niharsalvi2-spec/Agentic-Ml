"""
Data Warehouse and Storage Architecture Advisor.
Programmatic recommendations for schema modeling, OLAP implementation, operations, and storage.
"""

from typing import Dict, Any, List
from dataclasses import dataclass, field


def recommend_schema(
    query_speed_priority: bool = True,
    storage_constrained: bool = False,
    multiple_fact_tables: bool = False,
) -> Dict[str, str]:
    if multiple_fact_tables:
        return {
            "schema": "Fact Constellation (Galaxy)",
            "reasoning": "Multiple business processes sharing common dimensions."
        }
    if storage_constrained and not query_speed_priority:
        return {
            "schema": "Snowflake",
            "reasoning": "Normalized dimension hierarchy saves storage and enforces integrity."
        }
    return {
        "schema": "Star",
        "reasoning": "Denormalized dimensions for fastest query join performance."
    }


def recommend_storage(
    data_is_structured: bool = True,
    need_raw_ml_data: bool = False,
    need_both: bool = False
) -> Dict[str, str]:
    if need_both:
        return {
            "storage": "Data Lakehouse",
            "reasoning": "Combines raw object storage flexibility with ACID metadata and governance."
        }
    if not data_is_structured or need_raw_ml_data:
        return {
            "storage": "Data Lake",
            "reasoning": "Raw unstructured/semi-structured object storage for exploratory ML."
        }
    return {
        "storage": "Data Warehouse",
        "reasoning": "Schema-on-write structured storage optimized for analytical SQL BI reporting."
    }
