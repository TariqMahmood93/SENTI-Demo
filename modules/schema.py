"""
modules/schema.py — FK detection and schema utilities.

Supports SQLite (auto-detection via PRAGMA) and multi-CSV
(user-declared FK relationships).
"""
import os
import sqlite3
import tempfile
from typing import Dict, List, Tuple, Optional
import pandas as pd

# FK tuple: (from_table, from_col, to_table, to_col)
FKRelation = Tuple[str, str, str, str]


# ── SQLite helpers ─────────────────────────────────────────────────────────────

def sqlite_list_tables(conn: sqlite3.Connection) -> List[str]:
    cur = conn.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
    return [r[0] for r in cur.fetchall()]


def sqlite_load_table(conn: sqlite3.Connection, table: str) -> pd.DataFrame:
    return pd.read_sql_query(f'SELECT * FROM "{table}"', conn)


def sqlite_detect_fks(conn: sqlite3.Connection, tables: List[str]) -> List[FKRelation]:
    fks: List[FKRelation] = []
    for table in tables:
        try:
            cur = conn.execute(f'PRAGMA foreign_key_list("{table}")')
            for row in cur.fetchall():
                _, _, to_table, from_col, to_col, *_ = row
                fks.append((table, from_col, to_table, to_col))
        except Exception:
            continue
    return fks


def sqlite_load_all(path_or_bytes) -> Tuple[Dict[str, pd.DataFrame], List[FKRelation]]:
    """Load all tables and FK relationships from a SQLite file or bytes."""
    if isinstance(path_or_bytes, (bytes, bytearray)):
        tmp = tempfile.NamedTemporaryFile(suffix=".sqlite", delete=False)
        tmp.write(path_or_bytes)
        tmp.flush()
        tmp.close()
        db_path, cleanup = tmp.name, True
    else:
        db_path, cleanup = path_or_bytes, False

    try:
        conn   = sqlite3.connect(db_path)
        tables = sqlite_list_tables(conn)
        dfs    = {t: sqlite_load_table(conn, t) for t in tables}
        fks    = sqlite_detect_fks(conn, tables)
        conn.close()
    finally:
        if cleanup:
            try:
                os.unlink(db_path)
            except Exception:
                pass
    return dfs, fks


# ── FK context helpers ─────────────────────────────────────────────────────────

def enrich_with_fk_context(df: pd.DataFrame, tables: Dict[str, pd.DataFrame],
                            fks: List[FKRelation], from_table: str) -> pd.DataFrame:
    """
    Left-join all referenced tables onto df (for FK columns originating from
    from_table), adding prefixed context columns used to enrich embeddings.
    """
    relevant = [(fc, tt, tc) for (ft, fc, tt, tc) in fks if ft == from_table]
    if not relevant:
        return df.copy()

    enriched = df.copy()
    for from_col, to_table, to_col in relevant:
        if to_table not in tables:
            continue
        prefix      = f"{to_table}__"
        ref_renamed = tables[to_table].rename(columns={c: f"{prefix}{c}"
                                                        for c in tables[to_table].columns})
        ref_key     = f"{prefix}{to_col}"
        try:
            enriched = enriched.merge(
                ref_renamed.drop_duplicates(subset=[ref_key]),
                left_on=from_col, right_on=ref_key,
                how="left", suffixes=("", f"_{to_table}"),
            )
        except Exception:
            continue
    return enriched


def validate_fk_constraints(imputed_df: pd.DataFrame, from_col: str,
                              ref_df: pd.DataFrame, ref_col: str) -> pd.Series:
    """Return a boolean Series: True = valid FK value (or null), False = violation."""
    valid_vals = set(ref_df[ref_col].dropna().astype(str).tolist())
    return imputed_df[from_col].apply(
        lambda v: True if pd.isna(v) else str(v) in valid_vals
    )
