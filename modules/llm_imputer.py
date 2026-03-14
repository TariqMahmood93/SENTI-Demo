"""
modules/llm_imputer.py
──────────────────────
LLM-based imputation using the Groq API (Llama models).

Strategy
────────
- One prompt per incomplete row — all missing columns answered together as JSON.
- Low-cardinality columns (≤ LOW_CARD_THRESH unique values): domain list shown,
  model must pick from it exactly.
- High-cardinality columns: domain list omitted; 5 examples shown for style only.
- Numeric columns: observed range + mean shown.
- Response parsed as JSON with line-by-line fallback.
- Cells that cannot be parsed remain NaN.
"""
from __future__ import annotations

import re
import json
import time
import concurrent.futures
from typing import Callable, Optional

import requests
import pandas as pd
import numpy as np

GROQ_URL        = "https://api.groq.com/openai/v1/chat/completions"
DEFAULT_MODEL   = "llama-3.3-70b-versatile"
MAX_WORKERS     = 6
RETRY_WAIT      = 2.0
MAX_RETRIES     = 3
LOW_CARD_THRESH = 15
MAX_DOMAIN      = 15

_SESSION = requests.Session()
_SESSION.headers.update({
    "Content-Type": "application/json",
    "Accept":       "application/json",
    "User-Agent":   "groq-python/0.9.0",
})


# ── Groq HTTP ─────────────────────────────────────────────────────────────────

def _groq_chat(api_key: str, model: str, messages: list, temperature: float = 0.0) -> str:
    headers = {"Authorization": f"Bearer {api_key}"}
    payload = {"model": model, "messages": messages,
                "temperature": temperature, "max_tokens": 256}
    for attempt in range(MAX_RETRIES):
        try:
            resp = _SESSION.post(GROQ_URL, json=payload, headers=headers, timeout=30)
            if resp.status_code == 429:
                time.sleep(RETRY_WAIT * (attempt + 1))
                continue
            if not resp.ok:
                raise RuntimeError(f"Groq HTTP {resp.status_code}: {resp.text[:300]}")
            return resp.json()["choices"][0]["message"]["content"].strip()
        except RuntimeError:
            raise
        except Exception as e:
            if attempt == MAX_RETRIES - 1:
                raise RuntimeError(f"Groq request failed: {e}") from e
            time.sleep(RETRY_WAIT)
    raise RuntimeError("Max retries exceeded.")


def check_groq(api_key: str, model: str = DEFAULT_MODEL) -> tuple[bool, str]:
    try:
        _groq_chat(api_key, model, [{"role": "user", "content": "ping"}])
        return True, f"✓ Connected — model `{model}` is ready."
    except Exception as e:
        return False, str(e)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _col_is_numeric(series: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(series)

def _is_low_cardinality(series: pd.Series) -> bool:
    return series.nunique(dropna=True) <= LOW_CARD_THRESH


# ── Prompt builder ────────────────────────────────────────────────────────────

def _build_row_prompt(row: pd.Series, missing_cols: list[str],
                       all_series: dict[str, pd.Series], table_name: str,
                       fk_context: Optional[dict] = None) -> str:
    known = {k: v for k, v in row.items()
             if k not in missing_cols and pd.notna(v) and str(v).strip() != ""}
    known_str = "\n".join(f"  {k}: {v}" for k, v in known.items()) or "  (no other known values)"

    fk_block = ""
    if fk_context:
        for fk_tbl, fk_row in fk_context.items():
            fk_known = {k: v for k, v in fk_row.items() if pd.notna(v) and str(v).strip() != ""}
            if fk_known:
                fk_block += f"\nRelated row from table '{fk_tbl}':\n"
                fk_block += "\n".join(f"  {k}: {v}" for k, v in fk_known.items()) + "\n"

    col_instructions = []
    for col in missing_cols:
        ser = all_series.get(col)
        if ser is None:
            col_instructions.append(f'  "{col}": infer the most likely value from context')
            continue
        if _col_is_numeric(ser):
            nums = ser.dropna().tolist()
            if nums:
                lo, hi = min(nums), max(nums)
                mean_v = round(float(np.mean(nums)), 2)
                col_instructions.append(
                    f'  "{col}": numeric (range {lo}–{hi}, mean ~{mean_v}). '
                    f'Reply with only the number.'
                )
            else:
                col_instructions.append(f'  "{col}": numeric — infer from context')
        elif _is_low_cardinality(ser):
            domain = sorted(ser.dropna().unique().tolist(), key=str)[:MAX_DOMAIN]
            col_instructions.append(
                f'  "{col}": MUST be one of [{", ".join(repr(v) for v in domain)}]. '
                f'Pick the best match based on context.'
            )
        else:
            examples = ser.dropna().unique().tolist()[:5]
            ex_str = ", ".join(repr(v) for v in examples) if examples else "none available"
            col_instructions.append(
                f'  "{col}": free text — infer the most plausible value. '
                f'Examples (style only): {ex_str}'
            )

    schema = "{" + ", ".join(f'"{c}": ...' for c in missing_cols) + "}"
    return (
        f"You are an expert data imputation assistant for table '{table_name}'.\n\n"
        f"Known values in this row:\n{known_str}\n{fk_block}\n"
        f"Fill the following missing columns:\n" + "\n".join(col_instructions) + "\n\n"
        f"Rules:\n"
        f"- Reply with ONLY a valid JSON object. No explanation, no markdown.\n"
        f"- Use exactly these keys: {schema}\n"
        f"- For constrained columns, pick from the listed options exactly as written.\n\n"
        f"Reply:"
    )


# ── Response parser ───────────────────────────────────────────────────────────

def _coerce_value(val: str, col: str, series: Optional[pd.Series]) -> object:
    if series is None:
        return val
    if _col_is_numeric(series):
        nums = re.findall(r"[-+]?\d*\.?\d+", val)
        if nums:
            try:
                f = float(nums[0])
                return int(round(f)) if pd.api.types.is_integer_dtype(series.dtype) else f
            except ValueError:
                pass
        return None
    if _is_low_cardinality(series):
        domain = series.dropna().unique().tolist()
        val_l = val.lower()
        for dv in domain:
            if str(dv).lower() == val_l:
                return dv
        for dv in domain:
            if str(dv).lower() in val_l or val_l in str(dv).lower():
                return dv
    return val


def _parse_row_response(raw: str, missing_cols: list[str],
                         all_series: dict[str, pd.Series]) -> dict[str, object]:
    raw = re.sub(r"^```[a-z]*\n?", "", raw.strip(), flags=re.IGNORECASE)
    raw = re.sub(r"```$", "", raw).strip()
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            result = {}
            for col in missing_cols:
                val = parsed.get(col)
                if val is not None and str(val).strip() not in ("", "null", "None"):
                    result[col] = _coerce_value(str(val), col, all_series.get(col))
            return result
    except json.JSONDecodeError:
        pass
    # Line-by-line fallback
    result = {}
    for line in raw.splitlines():
        for col in missing_cols:
            m = re.search(
                rf'["\']?{re.escape(col)}["\']?\s*[:=]\s*["\']?([^"\',$\n]+)["\']?',
                line, re.IGNORECASE,
            )
            if m:
                val = m.group(1).strip().strip("\"'").strip()
                if val and val.lower() not in ("null", "none", "..."):
                    result[col] = _coerce_value(val, col, all_series.get(col))
    return result


# ── Main imputation function ──────────────────────────────────────────────────

def impute_llm(df: pd.DataFrame, cols: list, api_key: str,
               model: str = DEFAULT_MODEL, table_name: str = "table",
               tables: Optional[dict] = None, fks: Optional[list] = None,
               target_table: Optional[str] = None,
               progress_cb: Optional[Callable[[int, int], None]] = None,
               ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Impute missing values row-by-row using Groq Llama. One API call per incomplete row."""
    if not api_key:
        raise ValueError("Groq API key is required.")

    original    = df.copy()
    result      = df.copy()
    target_cols = [c for c in cols if c in df.columns and df[c].isna().any()]
    if not target_cols:
        return df, original.isna() & df.notna()

    all_series = {c: df[c] for c in target_cols}

    # Pre-build FK lookup: row_idx → {fk_table_name: Series}
    fk_lookup: dict[int, dict] = {}
    if fks and tables and target_table:
        relevant = [(fc, tt, tc) for (ft, fc, tt, tc) in fks if ft == target_table]
        for row_idx in range(len(df)):
            row_fk: dict = {}
            for fc, tt, tc in relevant:
                if tt not in tables or fc not in df.columns:
                    continue
                fk_val = df.iloc[row_idx][fc]
                if pd.isna(fk_val):
                    continue
                matches = tables[tt][tables[tt][tc] == fk_val]
                if not matches.empty:
                    row_fk[tt] = matches.iloc[0]
            if row_fk:
                fk_lookup[row_idx] = row_fk

    incomplete_rows = [
        (idx, [c for c in target_cols if pd.isna(df.loc[idx, c])])
        for idx in df.index
        if df.loc[idx, target_cols].isna().any()
    ]
    total = len(incomplete_rows)
    done  = 0

    def _impute_row(row_idx, missing):
        prompt = _build_row_prompt(
            df.loc[row_idx], missing, all_series, table_name, fk_lookup.get(row_idx, {})
        )
        try:
            raw    = _groq_chat(api_key, model, [{"role": "user", "content": prompt}])
            values = _parse_row_response(raw, missing, all_series)
        except Exception:
            values = {}
        return row_idx, values

    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(_impute_row, idx, missing): idx
                   for idx, missing in incomplete_rows}
        for future in concurrent.futures.as_completed(futures):
            row_idx, values = future.result()
            for col, val in values.items():
                if val is not None:
                    result.at[row_idx, col] = val
            done += 1
            if progress_cb:
                progress_cb(done, total)

    return result, original.isna() & result.notna()
