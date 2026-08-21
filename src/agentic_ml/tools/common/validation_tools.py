def validate_dataframe_non_empty(df):
    if df is None or df.empty:
        raise ValueError("DataFrame is empty or None.")
