"""
data_cube.py
--------------
A small, from-scratch multidimensional "data cube" engine (pure Python,
stdlib only) that implements the standard OLAP navigation operations:
Roll-Up, Drill-Down, Slice, Dice, and Pivot.

Modeling choice: rather than pre-materializing a literal N-dimensional cube
(the MOLAP approach), records are stored as flat rows (like a denormalized
star-schema fact table joined to its dimensions) and every operation is
answered by aggregating on demand -- i.e. this engine behaves like a small
ROLAP implementation. Each dimension is defined by a hierarchy of levels
(finest first), and every record must carry a column for every level of
every hierarchy (exactly how a denormalized Star Schema dimension table
stores Category/Subcategory/Brand/Product all in one row).

Typical usage:

    cube = DataCube(
        records=sales_records,
        hierarchies={
            "time":    ["month", "quarter", "year"],
            "region":  ["city", "state", "country"],
            "product": ["product", "category"],
        },
        measures=["sales"],
    )
    cube.roll_up("region", to_level="state")
    cube.slice_op("region", "Mumbai")
    rows = cube.query()
"""

from collections import defaultdict


class DataCube:

    def __init__(self, records, hierarchies, measures, agg="sum"):
        """
        records: list of dicts. Each dict must contain every level column
                 named in `hierarchies` plus every column named in `measures`.
        hierarchies: dict {dimension_name: [level_column_finest, ..., level_column_coarsest]}
        measures: list of measure column names (numeric, summable).
        agg: default aggregation function name for query(): "sum", "avg",
             "count", "min", or "max".
        """
        self.records = [dict(r) for r in records]
        self.hierarchies = {dim: list(levels) for dim, levels in hierarchies.items()}
        self.measures = list(measures)
        self.default_agg = agg

        # cube "view" state
        self.current_level = {dim: levels[0] for dim, levels in self.hierarchies.items()}
        self.sliced = {}     # dim -> value it's fixed to (removed from grouping/display)
        self.filters = {}    # dim -> (level_column, set(allowed_values))  -- from dice_op

        self._validate_records()

    def _validate_records(self):
        required_cols = set(self.measures)
        for levels in self.hierarchies.values():
            required_cols |= set(levels)
        for i, r in enumerate(self.records):
            missing = required_cols - set(r.keys())
            if missing:
                raise ValueError(f"Record {i} is missing columns: {missing}")

    # ------------------------------------------------------------------
    # Roll-Up / Drill-Down: change the granularity of a dimension
    # ------------------------------------------------------------------

    def roll_up(self, dimension, to_level=None):
        """
        Climb UP the hierarchy (more summarized, fewer rows). If to_level is
        omitted, moves one level coarser than the current level.
        """
        levels = self.hierarchies[dimension]
        cur_idx = levels.index(self.current_level[dimension])
        target_idx = levels.index(to_level) if to_level else cur_idx + 1
        if target_idx <= cur_idx:
            raise ValueError(f"roll_up must move to a coarser level than "
                              f"'{self.current_level[dimension]}'")
        if target_idx >= len(levels):
            raise ValueError(f"'{dimension}' is already at its coarsest level "
                              f"('{levels[-1]}')")
        self.current_level[dimension] = levels[target_idx]
        return self

    def drill_down(self, dimension, to_level=None):
        """
        Climb DOWN the hierarchy (more detailed, more rows). If to_level is
        omitted, moves one level finer than the current level. Exact inverse
        of roll_up.
        """
        levels = self.hierarchies[dimension]
        cur_idx = levels.index(self.current_level[dimension])
        target_idx = levels.index(to_level) if to_level else cur_idx - 1
        if target_idx >= cur_idx:
            raise ValueError(f"drill_down must move to a finer level than "
                              f"'{self.current_level[dimension]}'")
        if target_idx < 0:
            raise ValueError(f"'{dimension}' is already at its finest level "
                              f"('{levels[0]}')")
        self.current_level[dimension] = levels[target_idx]
        return self

    # ------------------------------------------------------------------
    # Slice / Dice: restrict the cube to specific values
    # ------------------------------------------------------------------

    def slice_op(self, dimension, value, level=None):
        """
        Fix `dimension` to a single `value`, removing it from the displayed
        grouping entirely (reduces dimensionality by 1). level defaults to
        the dimension's current display level.
        """
        level = level or self.current_level[dimension]
        self.sliced[dimension] = (level, value)
        return self

    def unslice(self, dimension):
        """Undo a previous slice_op on `dimension`, restoring it to the view."""
        self.sliced.pop(dimension, None)
        return self

    def dice_op(self, filters, level=None):
        """
        Restrict one or more dimensions to a set of allowed values WITHOUT
        removing them from the displayed grouping (dimensionality unchanged,
        just a smaller sub-cube). filters: dict {dimension: [values, ...]}.
        level: optional dict {dimension: level_column} if you want to filter
        at a level other than the dimension's current display level.
        """
        level = level or {}
        for dim, values in filters.items():
            lvl = level.get(dim, self.current_level[dim])
            self.filters[dim] = (lvl, set(values))
        return self

    def undice(self, dimension=None):
        """Clear dice filters -- for one dimension, or all if none given."""
        if dimension is None:
            self.filters = {}
        else:
            self.filters.pop(dimension, None)
        return self

    def reset(self):
        """Reset the cube to its initial view: finest level, no slice/dice."""
        self.current_level = {dim: levels[0] for dim, levels in self.hierarchies.items()}
        self.sliced = {}
        self.filters = {}
        return self

    # ------------------------------------------------------------------
    # Query: aggregate given the current view state
    # ------------------------------------------------------------------

    def _passes_filters(self, record):
        for dim, (level, value) in self.sliced.items():
            if record[level] != value:
                return False
        for dim, (level, allowed) in self.filters.items():
            if record[level] not in allowed:
                return False
        return True

    def query(self, agg=None):
        """
        Aggregate the cube given the current roll-up/drill-down levels and
        any active slice/dice filters. Returns a list of dicts, one per
        distinct combination of (non-sliced) dimension values, each with the
        grouping columns plus every measure aggregated with `agg`
        ("sum"/"avg"/"count"/"min"/"max"; defaults to the cube's default_agg).
        """
        agg = agg or self.default_agg
        group_dims = [dim for dim in self.hierarchies if dim not in self.sliced]
        group_cols = [self.current_level[dim] for dim in group_dims]

        buckets = defaultdict(list)
        for r in self.records:
            if not self._passes_filters(r):
                continue
            key = tuple(r[col] for col in group_cols)
            buckets[key].append(r)

        results = []
        for key, rows in buckets.items():
            row = dict(zip(group_cols, key))
            for m in self.measures:
                values = [r[m] for r in rows]
                row[m] = self._aggregate(values, agg)
            results.append(row)

        results.sort(key=lambda row: tuple(str(row[c]) for c in group_cols))
        return results

    @staticmethod
    def _aggregate(values, agg):
        if agg == "sum":
            return sum(values)
        if agg == "avg":
            return sum(values) / len(values) if values else 0.0
        if agg == "count":
            return len(values)
        if agg == "min":
            return min(values)
        if agg == "max":
            return max(values)
        raise ValueError(f"Unknown aggregation '{agg}'")

    # ------------------------------------------------------------------
    # Pivot: reshape a 2-dimensional query result into a row x column table
    # ------------------------------------------------------------------

    def pivot_table(self, row_col, col_col, measure=None, agg=None):
        """
        Runs query() at the current view state, then reshapes it into a
        row_col x col_col matrix for the given measure (defaults to the
        first measure). Both row_col and col_col must be current display
        level columns for two of the (non-sliced) dimensions.

        Returns (row_labels, col_labels, matrix) where matrix[i][j] is the
        aggregated measure for (row_labels[i], col_labels[j]), or None if
        that combination doesn't appear in the data.
        """
        measure = measure or self.measures[0]
        rows = self.query(agg=agg)

        row_labels = sorted({r[row_col] for r in rows}, key=str)
        col_labels = sorted({r[col_col] for r in rows}, key=str)
        lookup = {(r[row_col], r[col_col]): r[measure] for r in rows}

        matrix = [[lookup.get((rl, cl)) for cl in col_labels] for rl in row_labels]
        return row_labels, col_labels, matrix

    @staticmethod
    def format_pivot(row_labels, col_labels, matrix, row_header=""):
        """Pretty-prints the output of pivot_table() as an aligned text table."""
        col_width = max([len(row_header)] + [len(str(c)) for c in col_labels] +
                         [len(str(v)) for row in matrix for v in row if v is not None]) + 2
        row_label_width = max([len(row_header)] + [len(str(r)) for r in row_labels]) + 2

        header = row_header.ljust(row_label_width) + "".join(
            str(c).rjust(col_width) for c in col_labels)
        lines = [header]
        for rl, row in zip(row_labels, matrix):
            line = str(rl).ljust(row_label_width) + "".join(
                (str(v) if v is not None else "-").rjust(col_width) for v in row)
            lines.append(line)
        return "\n".join(lines)


if __name__ == "__main__":
    records = [
        {"product": "Laptop", "category": "Electronics", "city": "Mumbai",
         "state": "Maharashtra", "country": "India",
         "month": "Jan", "quarter": "Q1", "year": 2024, "sales": 50000},
        {"product": "Laptop", "category": "Electronics", "city": "Mumbai",
         "state": "Maharashtra", "country": "India",
         "month": "Feb", "quarter": "Q1", "year": 2024, "sales": 55000},
    ]
    cube = DataCube(
        records,
        hierarchies={"time": ["month", "quarter", "year"],
                     "region": ["city", "state", "country"],
                     "product": ["product", "category"]},
        measures=["sales"],
    )
    print(cube.query())
