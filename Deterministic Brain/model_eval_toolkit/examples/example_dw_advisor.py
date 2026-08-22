"""
Example: dw_advisor.py

Run with:  python3 example_dw_advisor.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "code"))

from dw_advisor import (
    recommend_schema, recommend_olap_implementation, recommend_operation,
    recommend_storage, FactTable, DimensionTable, StarSchema,
    SnowflakeSchema, FactConstellationSchema,
)


def schema_recommendation_example():
    print("=" * 70)
    print("Example 1: schema recommendation across a few scenarios")
    print("=" * 70)

    scenarios = [
        ("Small new retail warehouse, speed matters",
         dict(query_speed_priority=True)),
        ("Huge product dimension, storage is tight, speed less critical",
         dict(query_speed_priority=False, storage_constrained=True,
              dimension_tables_large=True)),
        ("Enterprise warehouse with Sales + Inventory + Shipping",
         dict(multiple_fact_tables=True)),
    ]
    for label, flags in scenarios:
        rec = recommend_schema(**flags)
        print(f"\n{label}")
        print(f"  -> {rec['schema']}")
        print(f"     {rec['reasoning']}")
    print()


def olap_implementation_example():
    print("=" * 70)
    print("Example 2: ROLAP vs MOLAP vs HOLAP recommendation")
    print("=" * 70)

    scenarios = [
        ("Small, dense dataset, need instant dashboards",
         dict(need_extreme_speed=True)),
        ("Huge dataset, dimensions change often",
         dict(dataset_very_large=True, need_flexibility=True)),
        ("Huge dataset but ALSO need instant dashboards",
         dict(dataset_very_large=True, need_extreme_speed=True)),
    ]
    for label, flags in scenarios:
        rec = recommend_olap_implementation(**flags)
        print(f"\n{label}")
        print(f"  -> {rec['implementation']}")
        print(f"     {rec['reasoning']}")
    print()


def operation_recommendation_example():
    print("=" * 70)
    print("Example 3: which OLAP operation fits the analysis need")
    print("=" * 70)

    needs = [
        "see_bigger_picture",
        "investigate_a_summary_number",
        "focus_on_one_category",
        "compare_several_categories_at_once",
        "different_visual_perspective",
        "relate_across_business_processes",
    ]
    for need in needs:
        rec = recommend_operation(need)
        print(f"{need:38s} -> {rec['operation']}")
    print()


def storage_recommendation_example():
    print("=" * 70)
    print("Example 4: Data Warehouse vs Data Lake vs Lakehouse")
    print("=" * 70)

    scenarios = [
        ("Standard monthly sales BI reports", dict()),
        ("Raw video/log data for a future ML project",
         dict(data_is_structured=False, need_raw_ml_data=True)),
        ("Need both raw flexibility and fast structured reporting",
         dict(need_both=True)),
    ]
    for label, flags in scenarios:
        rec = recommend_storage(**flags)
        print(f"\n{label}")
        print(f"  -> {rec['storage']}")
        print(f"     {rec['reasoning']}")
    print()


def build_schema_shapes_example():
    print("=" * 70)
    print("Example 5: building and printing the actual schema shapes")
    print("=" * 70)

    fact_sales = FactTable(
        "FACT_SALES",
        dimension_keys=["Product_ID", "Time_ID", "Store_ID", "Customer_ID"],
        measures=["Quantity_Sold", "Sales_Amount", "Profit"],
    )

    print("\n-- Star Schema --")
    star = StarSchema(fact_table=fact_sales, dimension_tables=[
        DimensionTable("Dim_Product", ["Product_ID", "Product_Name", "Category", "Brand"]),
        DimensionTable("Dim_Time", ["Time_ID", "Date", "Month", "Quarter", "Year"]),
        DimensionTable("Dim_Store", ["Store_ID", "Store_Name", "City", "Region"]),
        DimensionTable("Dim_Customer", ["Customer_ID", "Customer_Name", "Age", "Segment"]),
    ])
    print(star.describe())

    print("\n-- Snowflake Schema (Product dimension normalized) --")
    snowflake = SnowflakeSchema(fact_table=fact_sales, normalized_dimensions={
        "product": [
            DimensionTable("Dim_Product", ["Product_ID", "Subcategory_ID", "Product_Name"]),
            DimensionTable("Dim_Subcategory", ["Subcategory_ID", "Category_ID", "Subcategory_Name"]),
            DimensionTable("Dim_Category", ["Category_ID", "Category_Name"]),
        ],
    })
    print(snowflake.describe())

    print("\n-- Fact Constellation Schema (Sales + Inventory sharing dimensions) --")
    fact_inventory = FactTable(
        "FACT_INVENTORY",
        dimension_keys=["Product_ID", "Time_ID", "Supplier_ID"],
        measures=["Stock_Level", "Reorder_Quantity"],
    )
    constellation = FactConstellationSchema(
        fact_tables=[fact_sales, fact_inventory],
        shared_dimensions=[
            DimensionTable("Dim_Time", ["Time_ID", "Date", "Month", "Quarter", "Year"]),
            DimensionTable("Dim_Product", ["Product_ID", "Product_Name", "Category"]),
        ],
    )
    print(constellation.describe())


if __name__ == "__main__":
    schema_recommendation_example()
    olap_implementation_example()
    operation_recommendation_example()
    storage_recommendation_example()
    build_schema_shapes_example()
