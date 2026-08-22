"""
Example: data_cube.py

Run with:  python3 example_data_cube.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "code"))

from data_cube import DataCube


def build_sales_cube():
    """Recreates the Product x Region x Time cube from the theory notes."""
    monthly_sales = {
        ("Laptop", "Mumbai"): {"Jan": 50000, "Feb": 55000, "Mar": 48000},
        ("Phone", "Mumbai"):  {"Jan": 30000, "Feb": 32000, "Mar": 35000},
        ("Tablet", "Mumbai"): {"Jan": 15000, "Feb": 16000, "Mar": 14000},
        ("Laptop", "Delhi"):  {"Jan": 40000, "Feb": 42000, "Mar": 38000},
        ("Phone", "Delhi"):   {"Jan": 25000, "Feb": 27000, "Mar": 26000},
        ("Tablet", "Delhi"):  {"Jan": 12000, "Feb": 13000, "Mar": 11000},
        ("Laptop", "Pune"):   {"Jan": 30000, "Feb": 33000, "Mar": 29000},
        ("Phone", "Pune"):    {"Jan": 20000, "Feb": 21000, "Mar": 22000},
        ("Tablet", "Pune"):   {"Jan": 8000,  "Feb": 9000,  "Mar": 7000},
    }
    state_of = {"Mumbai": "Maharashtra", "Pune": "Maharashtra", "Delhi": "Delhi"}

    records = []
    for (product, city), months in monthly_sales.items():
        for month, sales in months.items():
            records.append({
                "product": product, "category": "Electronics",
                "city": city, "state": state_of[city], "country": "India",
                "month": month, "quarter": "Q1", "year": 2024,
                "sales": sales,
            })

    return DataCube(
        records,
        hierarchies={
            "time": ["month", "quarter", "year"],
            "region": ["city", "state", "country"],
            "product": ["product", "category"],
        },
        measures=["sales"],
    )


def slice_example():
    print("=" * 70)
    print("Example 1: Slice -- fix Region=Mumbai, view Product x Time")
    print("=" * 70)

    cube = build_sales_cube()
    cube.slice_op("region", "Mumbai")
    rows, cols, matrix = cube.pivot_table("product", "month")
    print(cube.format_pivot(rows, cols, matrix, row_header="product"))
    print("-> Region has been removed from the view entirely -- this is a 2D "
          "cross-section of the original 3D cube.\n")


def dice_example():
    print("=" * 70)
    print("Example 2: Dice -- Region in {Mumbai,Pune}, Month in {Jan,Feb}, "
          "Product in {Laptop,Phone}")
    print("=" * 70)

    cube = build_sales_cube()
    cube.dice_op({
        "region": ["Mumbai", "Pune"],
        "time": ["Jan", "Feb"],
        "product": ["Laptop", "Phone"],
    })
    for r in cube.query():
        print(" ", r)
    print("-> All 3 dimensions stay in the view, just restricted to a smaller "
          "sub-cube of specific values (unlike Slice, which drops a dimension).\n")


def roll_up_drill_down_example():
    print("=" * 70)
    print("Example 3: Roll-Up (city -> state) then Drill-Down back to city")
    print("=" * 70)

    cube = build_sales_cube()
    cube.roll_up("region", to_level="state")
    print("Rolled up to state level:")
    totals = {}
    for r in cube.query():
        totals[r["state"]] = totals.get(r["state"], 0) + r["sales"]
    for state, total in sorted(totals.items()):
        print(f"  {state}: {total}")
    print("-> Mumbai and Pune (both Maharashtra) are merged into one summarized row.")

    cube.drill_down("region", to_level="city")
    print("\nDrilled back down to city level (first 3 rows):")
    for r in cube.query()[:3]:
        print(" ", r)
    print()


def pivot_example():
    print("=" * 70)
    print("Example 4: Pivot -- rotate Product x Month into Month x Product")
    print("=" * 70)

    cube = build_sales_cube()
    cube.slice_op("region", "Mumbai")

    rows, cols, matrix = cube.pivot_table("product", "month")
    print("Before pivot (Product as rows, Month as columns):")
    print(cube.format_pivot(rows, cols, matrix, row_header="product"))

    rows2, cols2, matrix2 = cube.pivot_table("month", "product")
    print("\nAfter pivot (Month as rows, Product as columns):")
    print(cube.format_pivot(rows2, cols2, matrix2, row_header="month"))
    print("-> Same underlying data, different orientation for spotting patterns.\n")


def full_cube_query_example():
    print("=" * 70)
    print("Example 5: querying the full cube at finest granularity")
    print("=" * 70)

    cube = build_sales_cube()
    all_rows = cube.query()
    print(f"{len(all_rows)} rows at full Product x City x Month granularity "
          f"(3 products x 3 cities x 3 months = 27).")


if __name__ == "__main__":
    slice_example()
    dice_example()
    roll_up_drill_down_example()
    pivot_example()
    full_cube_query_example()
