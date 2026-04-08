import numpy as np
import pandas as pd

GRAY   = "#c8dff0"   # imputed cells — vivid blue tint, clearly visible on light bg
YELLOW = "rgba(255, 247, 204, 0.5)"   # appended rows

_TABLE_STYLES = [
    {"selector": "table", "props": [("border-collapse", "collapse"), ("font-size", "0.95rem")]},
    {"selector": "th, td", "props": [("border", "1px solid #ddd"), ("padding", "6px 8px")]},
]


def style_imputed_and_appended(df: pd.DataFrame, mask=None, appended_tuples=None):
    """Return a Styler with gray cell highlights (imputed) and yellow row highlights (appended)."""
    nrows, ncols = df.shape
    styles = np.full((nrows, ncols), "", dtype=object)

    if appended_tuples is not None:
        if isinstance(appended_tuples, pd.Series):
            ar = appended_tuples.reset_index(drop=True)
            row_idx = np.where(ar.values if len(ar) == nrows else np.zeros(nrows, bool))[0]
        else:
            try:
                row_idx = np.where(appended_tuples)[0]
            except Exception:
                row_idx = []
        if len(row_idx):
            styles[row_idx, :] = f"background-color: {YELLOW}"

    if mask is not None:
        arr = mask.values if isinstance(mask, pd.DataFrame) else mask
        r, c = np.where(arr)
        for i, j in zip(r, c):
            styles[i, j] = f"background-color: {GRAY}"

    styler = df.style.set_table_styles(_TABLE_STYLES)
    styler = styler.apply(
        lambda _df: pd.DataFrame(styles, index=_df.index, columns=_df.columns),
        axis=None,
    )
    return styler


def style_similarity_bins(imputed_df: pd.DataFrame,
                           cell_scores: pd.DataFrame,
                           missing_mask: pd.DataFrame):
    """
    Color imputed cells by cosine-similarity bin:
      0.00 – 0.70  → soft red
      0.70 – 0.95  → soft orange
      0.95 – 1.00  → soft green
    """
    common = [c for c in imputed_df.columns
              if c in cell_scores.columns and c in missing_mask.columns]
    impA = imputed_df[common].copy()
    scA  = cell_scores.reindex_like(impA)
    mkA  = missing_mask.reindex_like(impA).astype(bool)

    nrows, ncols = impA.shape
    styles = np.full((nrows, ncols), "", dtype=object)

    for i in range(nrows):
        for j in range(ncols):
            if not bool(mkA.iloc[i, j]):
                continue
            try:
                x = float(scA.iloc[i, j])
            except Exception:
                continue
            if pd.isna(x):
                continue
            if x < 0.70:
                styles[i, j] = "background-color: rgba(220, 53, 69, 0.30)"
            elif x < 0.95:
                styles[i, j] = "background-color: rgba(255, 159, 67, 0.30)"
            else:
                styles[i, j] = "background-color: rgba(46, 204, 113, 0.30)"

    styler = impA.style.set_table_styles(_TABLE_STYLES)
    return styler.apply(
        lambda _df: pd.DataFrame(styles, index=_df.index, columns=_df.columns),
        axis=None,
    )
