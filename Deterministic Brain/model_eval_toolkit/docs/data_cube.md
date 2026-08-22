# Data Cube & OLAP Operations

code: `code/data_cube.py`

## Why Multidimensional Thinking

Business questions are naturally multidimensional: *"What were total sales,
broken down by Product, by Region, by Time?"* has one **measure** (Sales —
the number being analyzed) and several **dimensions** (Product, Region, Time
— the ways of slicing that number). A **Data Cube** generalizes a 2D table
into N dimensions: each axis is a dimension, each cell is the measure value
for that specific combination.

## Dimension Hierarchies

Dimensions usually have multiple levels of granularity, e.g.:

```
Time:     Year -> Quarter -> Month -> Day
Product:  Category -> Subcategory -> Brand -> Product
Location: Country -> State -> City -> Store
```

`DataCube` models a hierarchy as an ordered list of column names, **finest
level first**: `{"time": ["month", "quarter", "year"]}`. Every record must
carry a column for every level (exactly how a denormalized Star Schema
dimension table stores Category + Subcategory + Product all in one row) —
this is what lets the engine roll up or drill down instantly without
recomputing anything.

## Implementation Note

This engine stores flat rows and aggregates on demand for every query —
i.e. it behaves like a small **ROLAP** implementation, not a literal
pre-materialized N-dimensional array (MOLAP). The OLAP *operations* below
are identical either way; only the underlying storage/performance strategy
differs (see `dw_advisor.md` for ROLAP vs MOLAP vs HOLAP).

## The Operations

| Operation | What it does | Dimensionality change |
|---|---|---|
| `roll_up(dim, to_level=None)` | Climb UP the hierarchy — summarize (e.g. Month -> Quarter) | Fewer rows, same dimension count |
| `drill_down(dim, to_level=None)` | Climb DOWN the hierarchy — more detail (e.g. Year -> Month). Exact inverse of roll-up | More rows, same dimension count |
| `slice_op(dim, value)` | Fix ONE dimension to a single value, removing it from the displayed grouping | Reduces dimension count by 1 |
| `dice_op({dim: [values...]})` | Filter one or more dimensions to specific value sets, without removing them from the view | Smaller sub-cube, same dimension count |
| `pivot_table(row_col, col_col)` | Reshape the current query into a row x column matrix | Same data, different layout |

`reset()` returns the cube to its initial view (finest level everywhere, no
slice/dice). `unslice(dim)` / `undice(dim)` undo one operation at a time.

## Worked Example (matches the theory doc exactly)

```python
from data_cube import DataCube

cube = DataCube(records, hierarchies={
    "time":    ["month", "quarter", "year"],
    "region":  ["city", "state", "country"],
    "product": ["product", "category"],
}, measures=["sales"])

# Slice: "show me data where Region = Mumbai" -> 2D Product x Time table
cube.slice_op("region", "Mumbai")
rows, cols, matrix = cube.pivot_table("product", "month")
print(cube.format_pivot(rows, cols, matrix, row_header="product"))
cube.reset()

# Dice: Region in {Mumbai, Pune}, Time in {Jan, Feb}, Product in {Laptop, Phone}
cube.dice_op({"region": ["Mumbai", "Pune"], "time": ["Jan", "Feb"],
              "product": ["Laptop", "Phone"]})
print(cube.query())
cube.reset()

# Roll-up: city -> state (Mumbai + Pune both roll into Maharashtra)
cube.roll_up("region", to_level="state")
print(cube.query())
```

Slice vs Dice, in one line: **Slice fixes one dimension to a single value and
removes it from view; Dice filters multiple dimensions to value sets and
keeps them all in view.**

## Function / Method Reference

```python
DataCube(records, hierarchies, measures, agg="sum")

cube.roll_up(dimension, to_level=None)
cube.drill_down(dimension, to_level=None)
cube.slice_op(dimension, value, level=None)
cube.unslice(dimension)
cube.dice_op(filters, level=None)          # filters: {dim: [values]}
cube.undice(dimension=None)                # None clears all dice filters
cube.reset()

cube.query(agg=None)                       # -> list[dict], one row per grouping
cube.pivot_table(row_col, col_col, measure=None, agg=None)
                                            # -> (row_labels, col_labels, matrix)
DataCube.format_pivot(row_labels, col_labels, matrix, row_header="")
                                            # -> pretty-printed text table
```
