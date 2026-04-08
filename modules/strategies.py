import pandas as pd
import numpy as np


def impute_other(df: pd.DataFrame, cols, strategy: str):
    """Baseline imputation strategies: mean, median, mode, MostFreq."""

    def _fill_int(s, val):
        if pd.api.types.is_integer_dtype(s.dtype):
            try:
                return s.fillna(int(round(float(val)))).astype(s.dtype)
            except Exception:
                return s.fillna(int(round(float(val)))).astype("Int64")
        return s.fillna(val)

    original = df.copy()
    cols = [c for c in cols if c in df.columns]

    if strategy == "mean":
        for c in cols:
            if pd.api.types.is_numeric_dtype(df[c]):
                df[c] = _fill_int(df[c], df[c].mean())

    elif strategy == "median":
        for c in cols:
            if pd.api.types.is_numeric_dtype(df[c]):
                df[c] = _fill_int(df[c], df[c].median())

    elif strategy in ("mode", "MostFreq"):
        for c in cols:
            mode_val = df[c].mode(dropna=True)
            if not mode_val.empty:
                df[c] = df[c].fillna(mode_val.iloc[0])

    mask = original.isna() & df.notna()
    return df, mask
