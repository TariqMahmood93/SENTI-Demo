import numpy as np
import pandas as pd

def inject_nulls(df: pd.DataFrame, columns=None, frac: float = 0.1, seed: int = 42):
    if not isinstance(df, pd.DataFrame):
        raise ValueError("inject_nulls expects a pandas DataFrame")
    if not columns:
        columns = list(df.columns)
    rng = np.random.default_rng(seed)
    n = len(df)
    k = int(round(frac * n))
    result = df.copy()
    for c in columns:
        if c not in result.columns:
            continue
        if k <= 0:
            continue
        idx = rng.choice(n, size=min(k, n), replace=False)
        result.loc[result.index[idx], c] = np.nan
    return result
