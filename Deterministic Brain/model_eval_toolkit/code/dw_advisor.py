"""
dw_advisor.py
---------------
DataWarehouseAdvisor: a decision-support layer for Phase 9 (Data Warehouse)
concepts, mirroring EvaluationAgent's role for Phase 7. It answers three
"which should I use" questions programmatically:

  1. recommend_schema(...)              -- Star vs Snowflake vs Fact Constellation
  2. recommend_olap_implementation(...) -- ROLAP vs MOLAP vs HOLAP
  3. recommend_operation(need)          -- which OLAP operation fits an analysis need
  4. recommend_storage(...)             -- Data Warehouse vs Data Lake vs Lakehouse

Also includes lightweight dataclasses (FactTable, DimensionTable, StarSchema,
SnowflakeSchema, FactConstellationSchema) so the schema shapes described in
the theory can actually be built and printed, not just recommended.
"""

from dataclasses import dataclass, field


# --------------------------------------------------------------------------
# 1. Schema recommendation
# --------------------------------------------------------------------------

def recommend_schema(query_speed_priority=True, storage_constrained=False,
                      dimension_tables_large=False, multiple_fact_tables=False,
                      dimension_hierarchy_changes_often=False):
    """
    Returns {"schema": str, "reasoning": str}.

    Flags:
      query_speed_priority             -- fast queries matter more than storage
      storage_constrained              -- storage/space is a real limitation
      dimension_tables_large           -- dimension tables have significant redundancy at scale
      multiple_fact_tables             -- multiple business processes sharing common dimensions
      dimension_hierarchy_changes_often-- hierarchies get restructured/updated often
    """
    if multiple_fact_tables:
        return {
            "schema": "Fact Constellation (Galaxy)",
            "reasoning": ("Multiple business processes (e.g. Sales, Inventory, Shipping) "
                          "need to share common dimensions like Time and Product. A single "
                          "star can't represent more than one fact table, so use a Fact "
                          "Constellation: several fact tables connected to shared dimension "
                          "tables."),
        }

    normalize_reasons = []
    if storage_constrained:
        normalize_reasons.append("storage is constrained")
    if dimension_tables_large:
        normalize_reasons.append("dimension tables are large with significant redundancy")
    if dimension_hierarchy_changes_often:
        normalize_reasons.append("dimension hierarchies change frequently, "
                                 "which is easier to update when normalized")

    if normalize_reasons and not query_speed_priority:
        return {
            "schema": "Snowflake",
            "reasoning": ("Normalizing dimension tables into their hierarchy levels "
                          "(Category -> Subcategory -> Product) reduces redundancy and "
                          "improves data integrity, at the cost of extra joins per query. "
                          "Worth it here because: " + "; ".join(normalize_reasons) + "."),
        }

    return {
        "schema": "Star",
        "reasoning": ("Denormalized dimension tables mean every query only needs to join "
                      "the fact table to a handful of flat dimension tables -- fewer joins, "
                      "faster queries, easier for business users to understand. This is the "
                      "default recommendation for most new warehouses; storage is cheap and "
                      "query speed/simplicity usually wins unless a specific constraint "
                      "(storage, very large dimensions, frequently changing hierarchies, "
                      "or multiple fact tables) says otherwise."),
    }


# --------------------------------------------------------------------------
# 2. OLAP implementation recommendation
# --------------------------------------------------------------------------

def recommend_olap_implementation(dataset_very_large=False, need_extreme_speed=False,
                                   need_flexibility=True, cube_would_be_sparse=False):
    """
    Returns {"implementation": str, "reasoning": str}.

    Flags:
      dataset_very_large   -- data volume is large enough that pre-computing a full cube is costly
      need_extreme_speed   -- sub-second response for complex aggregations is required
      need_flexibility     -- dimensions/queries change often, need to add dimensions easily
      cube_would_be_sparse -- many dimension-value combinations would have no data (sparse cube problem)
    """
    if need_extreme_speed and not (dataset_very_large or cube_would_be_sparse):
        return {
            "implementation": "MOLAP",
            "reasoning": ("Pre-computing and storing aggregations in an actual multidimensional "
                          "cube structure gives the fastest possible query response. This works "
                          "well here because the data isn't so large or sparse that the cube "
                          "would explode in size."),
        }

    if dataset_very_large or need_flexibility or cube_would_be_sparse:
        reasons = []
        if dataset_very_large:
            reasons.append("the dataset is very large (relational storage scales better than a full cube)")
        if cube_would_be_sparse:
            reasons.append("a full multidimensional cube would be mostly empty cells (the sparse cube problem)")
        if need_flexibility:
            reasons.append("dimensions/queries need to stay easy to change without rebuilding a cube")
        if need_extreme_speed:
            return {
                "implementation": "HOLAP",
                "reasoning": ("Both extreme speed AND large/flexible/sparse data are needed, which "
                              "a pure ROLAP or MOLAP approach can't satisfy alone. HOLAP stores "
                              "detailed data relationally (ROLAP, for flexibility/scale) and "
                              "pre-aggregates the common summary views as a cube (MOLAP, for speed) "
                              "-- most modern OLAP systems default to this hybrid."),
            }
        return {
            "implementation": "ROLAP",
            "reasoning": ("Storing data in standard relational tables (Star/Snowflake schema) and "
                          "aggregating on-the-fly with SQL handles this better than a pre-built cube, "
                          "because " + "; ".join(reasons) + "."),
        }

    return {
        "implementation": "HOLAP",
        "reasoning": ("No single overriding constraint pushes strongly toward pure ROLAP or pure "
                      "MOLAP, so a hybrid gives a reasonable balance of speed and flexibility -- "
                      "this is also what most modern enterprise/cloud OLAP systems default to."),
    }


# --------------------------------------------------------------------------
# 3. OLAP operation recommendation
# --------------------------------------------------------------------------

_OPERATION_GUIDE = {
    "see_bigger_picture": ("Roll-Up", "Climb up the dimension hierarchy to summarize "
                            "detailed rows into a coarser view (e.g. Month -> Quarter)."),
    "investigate_a_summary_number": ("Drill-Down", "Climb down the dimension hierarchy to "
                                       "see the detail behind a summary figure "
                                       "(e.g. Year -> Quarter -> Month)."),
    "focus_on_one_category": ("Slice", "Fix one dimension to a single value, removing it "
                                "from the view entirely (e.g. Region = Mumbai only)."),
    "compare_several_categories_at_once": ("Dice", "Filter multiple dimensions to specific "
                                             "value sets, producing a smaller sub-cube "
                                             "(e.g. Region in {Mumbai, Pune}, Month in {Jan, Feb})."),
    "different_visual_perspective": ("Pivot", "Rotate the cube's rows/columns to view the "
                                       "same data from a different orientation."),
    "relate_across_business_processes": ("Drill-Across", "Navigate from one fact table to a "
                                           "related fact table via their shared dimensions "
                                           "(e.g. Sales -> Inventory for the same Product)."),
}


def recommend_operation(need):
    """
    need: one of "see_bigger_picture", "investigate_a_summary_number",
          "focus_on_one_category", "compare_several_categories_at_once",
          "different_visual_perspective", "relate_across_business_processes".

    Returns {"operation": str, "reasoning": str}.
    """
    if need not in _OPERATION_GUIDE:
        raise ValueError(f"Unknown need '{need}'. Valid options: {list(_OPERATION_GUIDE)}")
    operation, reasoning = _OPERATION_GUIDE[need]
    return {"operation": operation, "reasoning": reasoning}


# --------------------------------------------------------------------------
# 4. Storage architecture recommendation
# --------------------------------------------------------------------------

def recommend_storage(data_is_structured=True, query_patterns_known=True,
                       need_raw_ml_data=False, need_both=False):
    """
    Returns {"storage": str, "reasoning": str}.
    """
    if need_both:
        return {
            "storage": "Data Lakehouse",
            "reasoning": ("Both raw-storage flexibility and structured query performance are "
                          "needed. A Lakehouse stores raw data like a lake but adds warehouse-like "
                          "structure/governance on top -- increasingly the default modern choice."),
        }
    if not data_is_structured or need_raw_ml_data or not query_patterns_known:
        return {
            "storage": "Data Lake",
            "reasoning": ("Data is unstructured/semi-structured, query patterns aren't fixed yet, "
                          "or the data is destined for ML/exploratory use -- store it raw "
                          "(schema-on-read) and decide the transformation later."),
        }
    return {
        "storage": "Data Warehouse",
        "reasoning": ("Data is structured with known, standard BI reporting query patterns -- "
                      "ETL/clean before storage (schema-on-write) for fast, reliable reporting."),
    }


# --------------------------------------------------------------------------
# Schema-shape dataclasses (build and print the structures, not just recommend them)
# --------------------------------------------------------------------------

@dataclass
class DimensionTable:
    name: str
    columns: list  # flat list for Star; for Snowflake, model each hierarchy level as its own DimensionTable

    def describe(self):
        return f"{self.name}({', '.join(self.columns)})"


@dataclass
class FactTable:
    name: str
    dimension_keys: list   # foreign key column names, one per linked dimension
    measures: list          # numeric measure column names

    def describe(self):
        cols = self.dimension_keys + self.measures
        return f"{self.name}({', '.join(cols)})"


@dataclass
class StarSchema:
    fact_table: FactTable
    dimension_tables: list  # list[DimensionTable], each denormalized (flat)

    def describe(self):
        lines = [f"FACT (center):   {self.fact_table.describe()}"]
        for dim in self.dimension_tables:
            lines.append(f"  DIMENSION:     {dim.describe()}")
        return "\n".join(lines)


@dataclass
class SnowflakeSchema:
    fact_table: FactTable
    # dimension_name -> ordered list of DimensionTable, finest first, each referencing the next
    normalized_dimensions: dict = field(default_factory=dict)

    def describe(self):
        lines = [f"FACT (center):   {self.fact_table.describe()}"]
        for dim_name, chain in self.normalized_dimensions.items():
            chain_desc = " -> ".join(t.describe() for t in chain)
            lines.append(f"  DIMENSION '{dim_name}' (normalized chain): {chain_desc}")
        return "\n".join(lines)


@dataclass
class FactConstellationSchema:
    fact_tables: list        # list[FactTable]
    shared_dimensions: list  # list[DimensionTable] referenced by more than one fact table

    def describe(self):
        lines = []
        for fact in self.fact_tables:
            lines.append(f"FACT: {fact.describe()}")
        lines.append("SHARED DIMENSIONS:")
        for dim in self.shared_dimensions:
            lines.append(f"  {dim.describe()}")
        return "\n".join(lines)


if __name__ == "__main__":
    print(recommend_schema(multiple_fact_tables=True))
    print(recommend_olap_implementation(need_extreme_speed=True, need_flexibility=False))
    print(recommend_operation("focus_on_one_category"))
    print(recommend_storage(need_raw_ml_data=True))
