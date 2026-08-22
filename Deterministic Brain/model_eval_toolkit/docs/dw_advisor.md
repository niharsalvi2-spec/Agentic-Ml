# DataWarehouseAdvisor — Schema, OLAP Implementation & Operation Decision Guide

code: `code/dw_advisor.py`

Four independent decision-guide functions, plus dataclasses to actually
build and print the schema shapes described in the theory.

## `recommend_schema(...)` — Star vs Snowflake vs Fact Constellation

| Flag | Meaning |
|---|---|
| `query_speed_priority` | fast queries matter more than storage/redundancy |
| `storage_constrained` | storage/space is a real limitation |
| `dimension_tables_large` | dimension tables have significant redundancy at scale |
| `multiple_fact_tables` | multiple business processes share common dimensions |
| `dimension_hierarchy_changes_often` | hierarchies get restructured/updated often |

**Logic:** multiple fact tables always wins -> Fact Constellation (a single
star can't represent more than one fact table). Otherwise, if storage/size/
change-frequency concerns outweigh speed -> Snowflake (normalize dimensions
to cut redundancy, at the cost of extra joins). Otherwise -> **Star**, the
default recommendation for most new warehouses.

```python
recommend_schema(multiple_fact_tables=True)
# -> {"schema": "Fact Constellation (Galaxy)", "reasoning": "..."}
```

## `recommend_olap_implementation(...)` — ROLAP vs MOLAP vs HOLAP

| Flag | Meaning |
|---|---|
| `dataset_very_large` | pre-computing a full cube would be costly at this scale |
| `need_extreme_speed` | sub-second response for complex aggregations required |
| `need_flexibility` | dimensions/queries change often |
| `cube_would_be_sparse` | many dimension-value combos have no data (sparse cube problem) |

**Logic:** need extreme speed + small/dense data -> **MOLAP** (pre-computed
cube). Large/flexible/sparse data -> **ROLAP** (aggregate on the fly over
relational tables), or **HOLAP** if extreme speed is *also* required
alongside those constraints. No dominant constraint -> **HOLAP** by default
(what most modern systems use).

## `recommend_operation(need)` — which OLAP operation fits an analysis task

| `need` | Operation |
|---|---|
| `"see_bigger_picture"` | Roll-Up |
| `"investigate_a_summary_number"` | Drill-Down |
| `"focus_on_one_category"` | Slice |
| `"compare_several_categories_at_once"` | Dice |
| `"different_visual_perspective"` | Pivot |
| `"relate_across_business_processes"` | Drill-Across |

```python
recommend_operation("focus_on_one_category")
# -> {"operation": "Slice", "reasoning": "Fix one dimension to a single value..."}
```

## `recommend_storage(...)` — Data Warehouse vs Data Lake vs Lakehouse

| Flag | Meaning |
|---|---|
| `data_is_structured` | data is tabular/structured vs. mixed/unstructured |
| `query_patterns_known` | reporting queries are standard/known in advance |
| `need_raw_ml_data` | data is destined for ML/exploratory use |
| `need_both` | need raw flexibility AND structured query performance |

**Logic:** `need_both` -> **Lakehouse**. Unstructured / unknown query
patterns / ML-destined -> **Data Lake** (schema-on-read, store raw, transform
later). Otherwise -> **Data Warehouse** (schema-on-write, ETL before
storage, standard BI reporting).

## Schema-Shape Dataclasses

Build and print the actual table structures the theory describes, rather
than just naming the schema:

```python
from dw_advisor import FactTable, DimensionTable, StarSchema

fact = FactTable("FACT_SALES",
                  dimension_keys=["Product_ID", "Time_ID", "Store_ID", "Customer_ID"],
                  measures=["Quantity_Sold", "Sales_Amount", "Profit"])

star = StarSchema(fact_table=fact, dimension_tables=[
    DimensionTable("Dim_Product", ["Product_ID", "Product_Name", "Category", "Brand"]),
    DimensionTable("Dim_Time", ["Time_ID", "Date", "Month", "Quarter", "Year"]),
])
print(star.describe())
```

`SnowflakeSchema` takes `normalized_dimensions: {dim_name: [DimensionTable, ...]}`
— an ordered chain of tables from finest to coarsest level, mirroring how a
Snowflake schema splits `Dim_Product` into `Dim_Product -> Dim_Subcategory ->
Dim_Category`. `FactConstellationSchema` takes multiple `FactTable`s plus a
list of `shared_dimensions` referenced by more than one of them.

## Function Reference

```python
recommend_schema(query_speed_priority=True, storage_constrained=False,
                  dimension_tables_large=False, multiple_fact_tables=False,
                  dimension_hierarchy_changes_often=False)

recommend_olap_implementation(dataset_very_large=False, need_extreme_speed=False,
                               need_flexibility=True, cube_would_be_sparse=False)

recommend_operation(need)

recommend_storage(data_is_structured=True, query_patterns_known=True,
                   need_raw_ml_data=False, need_both=False)

DimensionTable(name, columns)
FactTable(name, dimension_keys, measures)
StarSchema(fact_table, dimension_tables)
SnowflakeSchema(fact_table, normalized_dimensions)
FactConstellationSchema(fact_tables, shared_dimensions)
```
