from io import StringIO
import pandas as pd


def _demo_demographic() -> pd.DataFrame:
    csv = StringIO(
        "id,age,city,income,score\n"
        "1,34,Rome,,7\n"
        "2,,Milan,4200,6\n"
        "3,28,,3800,\n"
        "4,46,Naples,,\n"
        "5,,Rome,5200,8\n"
    )
    df = pd.read_csv(csv)
    for c in df.select_dtypes(include=["number"]).columns:
        df[c] = pd.to_numeric(df[c], errors="coerce").round(0).astype("Int64")
    return df


def _demo_health() -> pd.DataFrame:
    csv = StringIO(
        "id,age,diagnosis,bp,glucose\n"
        "1,55,Hypertension,140,\n"
        "2,,Diabetes,,180\n"
        "3,47,,128,95\n"
        "4,62,Hypertension,,110\n"
        "5,,Diabetes,150,\n"
    )
    df = pd.read_csv(csv)
    for c in df.select_dtypes(include=["number"]).columns:
        df[c] = pd.to_numeric(df[c], errors="coerce").round(0).astype("Int64")
    return df


def _demo_cargo() -> pd.DataFrame:
    """
    Cargo logistics dataset from the TIDI paper example (r0 — incomplete batch).
    Missing values (⊥ in the paper) are represented as NaN.
    Ground-truth completions (r0*):
        t2 Hub   → NYC
        t3 Country → USA
        t4 Status  → Delayed
    """
    csv = StringIO(
        "TupleID,CargoID,Status,Period,Country,Hub\n"
        "t1,Charlie-3,Active,Spring,France,Paris\n"
        "t2,Alpha-7,Delayed,Summer,USA,\n"
        "t3,Beta-7,Pending,Winter,,New York\n"
        "t4,Alpha-1,,Jul-Aug,USA,NYC\n"
    )
    return pd.read_csv(csv)


def load_demo_df(kind: str = "Demographic") -> pd.DataFrame:
    """Return a small demo dataset.  kind: 'Demographic' | 'Health' | 'Cargo'"""
    k = (kind or "").lower()
    if k.startswith("health"):
        return _demo_health()
    if k.startswith("cargo"):
        return _demo_cargo()
    return _demo_demographic()
