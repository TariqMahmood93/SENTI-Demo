from modules.state import ss
import streamlit as st

import numpy as np
import pandas as pd


def _normalize_pair(gt: pd.DataFrame, imp: pd.DataFrame, cols, method: str, params: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    gt = gt.copy()
    imp = imp.copy()
    for c in cols:
        if c not in gt.columns or c not in imp.columns:
            continue
        if not pd.api.types.is_numeric_dtype(gt[c]):
            continue
        if method == "None":
            continue
        elif method == "Min-Max [0,1]":
            gmin, gmax = gt[c].min(), gt[c].max()
            denom = (gmax - gmin) if (gmax - gmin) != 0 else 1.0
            gt[c] = (gt[c] - gmin) / denom
            imp[c] = (imp[c] - gmin) / denom
        elif method == "Z-score (mean/std)":
            mu = gt[c].mean()
            sd = gt[c].std(ddof=0)
            denom = sd if sd != 0 else 1.0
            gt[c] = (gt[c] - mu) / denom
            imp[c] = (imp[c] - mu) / denom
        elif method == "Robust (median/IQR)":
            med = gt[c].median()
            q1, q3 = gt[c].quantile(0.25), gt[c].quantile(0.75)
            iqr = (q3 - q1) if (q3 - q1) != 0 else 1.0
            gt[c] = (gt[c] - med) / iqr
            imp[c] = (imp[c] - med) / iqr
        elif method == "Divide by constant":
            k = float(params.get("divide_by", 1.0)) or 1.0
            gt[c] = gt[c] / k
            imp[c] = imp[c] / k
        elif method == "User min/max":
            lo = float(params.get("user_min", 0.0))
            hi = float(params.get("user_max", 1.0))
            denom = (hi - lo) if (hi - lo) != 0 else 1.0
            gt[c] = (gt[c] - lo) / denom
            imp[c] = (imp[c] - lo) / denom
    return gt, imp


def _rmse_mae_report(gt_df: pd.DataFrame, imp_df: pd.DataFrame, cols) -> tuple[pd.DataFrame, float, float]:
    per_col_rmse, per_col_mae, used = {}, {}, []
    for c in cols:
        if c in gt_df.columns and c in imp_df.columns and pd.api.types.is_numeric_dtype(gt_df[c]):
            diff = (imp_df[c] - gt_df[c]).astype(float)
            per_col_rmse[c] = float(np.sqrt(np.nanmean(np.square(diff))))
            per_col_mae[c] = float(np.nanmean(np.abs(diff)))
            used.append(c)
    out = pd.DataFrame({
        "Column": used,
        "RMSE": [per_col_rmse[c] for c in used],
        "MAE": [per_col_mae[c] for c in used],
    }).set_index("Column")
    overall_rmse = float(np.nanmean(list(per_col_rmse.values()))) if per_col_rmse else float("nan")
    overall_mae = float(np.nanmean(list(per_col_mae.values()))) if per_col_mae else float("nan")
    return out, overall_rmse, overall_mae


from modules.state import init_state, ss
from modules.demo_data import load_demo_df
from modules.strategies import impute_other
from modules.senti_backend import impute_senti
try:
    from modules.llm_imputer import DEFAULT_MODEL as _LLM_DEFAULT_MODEL
except Exception:
    _LLM_DEFAULT_MODEL = "llama-3.3-70b-versatile"
from modules.highlight import style_imputed_and_appended
from modules.eval_metrics import record_missing_positions, exact_match_at_positions, semantic_similarity_at_positions

st.set_page_config(page_title="SENTI — Data Imputation", layout="wide", page_icon="⬡")

# ── Splash / loading screen ───────────────────────────────────────────────────
st.markdown("""
<style>
#senti-splash {
    position: fixed; inset: 0; z-index: 999999;
    background: linear-gradient(160deg, #ffffff 0%, #e8f4fd 60%, #d0eafb 100%);
    display: flex; flex-direction: column;
    align-items: center; justify-content: center; gap: 28px;
    animation: splashFadeOut 0.8s ease 6s forwards;
    pointer-events: none;
}
@keyframes splashFadeOut {
    0%   { opacity: 1; pointer-events: none; }
    100% { opacity: 0; pointer-events: none; visibility: hidden; }
}
.splash-hex {
    width: 64px; height: 64px;
    background: linear-gradient(135deg, #5BB8F5, #0284C7);
    clip-path: polygon(50% 0%, 95% 25%, 95% 75%, 50% 100%, 5% 75%, 5% 25%);
    display: flex; align-items: center; justify-content: center;
    font-size: 1.6rem; font-weight: 700; color: #ffffff; font-family: serif;
    animation: splash-pulse 2s ease-in-out infinite;
}
@keyframes splash-pulse {
    0%, 100% { transform: scale(1); }
    50%       { transform: scale(1.08); box-shadow: 0 0 28px rgba(91,184,245,0.55); }
}
.splash-title {
    font-family: Georgia, serif; font-size: 2.4rem;
    font-weight: 700; color: #000000; letter-spacing: -0.02em; text-align: center;
}
.splash-title span { color: #0284C7; }
.splash-msg {
    font-family: Arial, sans-serif; font-size: 1.15rem;
    color: #333333; text-align: center; max-width: 420px; line-height: 1.6;
}
.splash-dots { display: flex; gap: 10px; }
.splash-dots span {
    width: 10px; height: 10px; border-radius: 50%; background: #5BB8F5;
    animation: splash-bounce 1.2s ease-in-out infinite;
}
.splash-dots span:nth-child(2) { animation-delay: 0.2s; }
.splash-dots span:nth-child(3) { animation-delay: 0.4s; }
@keyframes splash-bounce {
    0%, 80%, 100% { transform: translateY(0); opacity: 0.4; }
    40%            { transform: translateY(-10px); opacity: 1; }
}
.splash-sub {
    font-family: Arial, sans-serif; font-size: 0.82rem;
    color: #777777; letter-spacing: 0.08em; text-transform: uppercase;
}
.splash-note {
    display: flex; align-items: flex-start; gap: 10px;
    background: #fffbe6;
    border: 1px solid #f5c842;
    border-left: 4px solid #f5a800;
    border-radius: 8px;
    padding: 12px 16px;
    max-width: 460px;
    font-family: Arial, sans-serif; font-size: 0.88rem;
    color: #5a4500; line-height: 1.55; text-align: left;
}
.splash-note-icon {
    font-size: 1.1rem; margin-top: 1px; flex-shrink: 0;
}
.splash-note a {
    color: #0284C7; font-weight: 600; text-decoration: none;
    pointer-events: auto !important;
    position: relative; z-index: 9999999;
}
.splash-note a:hover { text-decoration: underline; }
</style>
<div id="senti-splash">
    <div class="splash-hex">ST</div>
    <div class="splash-title">Welcome to <span>SENTI</span></div>
    <div class="splash-msg">A Dynamic Data imputation tool powered by Pre-trained Language Models.<br>Please wait while the tool is loading&hellip;</div>
    <div class="splash-dots"><span></span><span></span><span></span></div>
    <div class="splash-sub">Initialising modules &amp; models</div>
    <div class="splash-note">
        <div class="splash-note-icon">⚠️</div>
        <div>
            This is a <em>CPU-only</em> version for demonstration purposes.
            For maximal efficiency and efficacy, please refer to our publicly available code on
            <a href="https://github.com/TariqMahmood93/SENTI-Demo.git" target="_blank">GitHub</a>,
            which is fully optimised for GPU usage.
        </div>
    </div>
</div>
""", unsafe_allow_html=True)
# ─────────────────────────────────────────────
#  MASTER CSS  — Full advanced redesign
# ─────────────────────────────────────────────
def _inject_master_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:ital,wght@0,300;0,400;0,500;0,700;1,400&family=Fraunces:ital,opsz,wght@0,9..144,300;0,9..144,700;1,9..144,400&display=swap');

    :root {
        --ink:      #eef3f8;
        --ink2:     #e4edf5;
        --ink3:     #dae6f0;
        --panel:    #ffffff;
        --panel2:   #f0f5fa;
        --rim:      rgba(74,127,165,0.18);
        --rim2:     rgba(74,127,165,0.32);
        --teal:     #4a7fa5;
        --teal2:    #3a6a8e;
        --teal3:    rgba(74,127,165,0.12);
        --teal4:    rgba(74,127,165,0.06);
        --amber:    #92600a;
        --amber2:   rgba(146,96,10,0.10);
        --rose:     #b91c3c;
        --rose2:    rgba(185,28,60,0.10);
        --lilac:    #5b3fa8;
        --lilac2:   rgba(91,63,168,0.10);
        --sky:      #0369a1;
        --sky2:     rgba(3,105,161,0.10);
        --green:    #166534;
        --text:     #1a2c3d;
        --text2:    #3a506a;
        --text3:    #6a8099;
        --r:        10px;
        --r2:       6px;
        --shadow:   0 4px 18px rgba(74,127,165,0.12);
        --mono:     'JetBrains Mono', monospace;
        --sans:     'Space Grotesk', sans-serif;
        --display:  'Fraunces', serif;

    /* ── Compatibility aliases for helper functions ── */
    --accent:    var(--teal);
    --accent2:   var(--teal);
    --accent3:   var(--teal);
    --border:    var(--rim);
    --border2:   var(--rim2);
    --font-mono: var(--mono);
    --gold:      var(--amber);
    --red:       var(--rose);
    --card:      var(--panel);
    --card2:     var(--panel2);
}

    /* ── Reset & Base ────────────────────────────────────── */
    html, body, [data-testid="stAppViewContainer"] {
        background: linear-gradient(160deg, #eef3f8 0%, #dde8f2 100%) !important;
        color: var(--text) !important;
        font-family: var(--sans) !important;
    }

    /* ── Global black text for ALL widget labels ────────── */
    [data-testid="stWidgetLabel"] p,
    [data-testid="stWidgetLabel"] span,
    [data-testid="stWidgetLabel"] label,
    [data-testid="stToggle"] label,
    [data-testid="stToggle"] p,
    [data-testid="stCheckbox"] label,
    [data-testid="stCheckbox"] p,
    [data-testid="stRadio"] label,
    [data-testid="stRadio"] p,
    [data-testid="stSelectbox"] label,
    [data-testid="stTextInput"] label,
    [data-testid="stNumberInput"] label,
    [data-testid="stSlider"] label,
    [data-testid="stMultiSelect"] label,
    [data-testid="stFileUploader"] label,
    [data-testid="stTextArea"] label,
    .stRadio label, .stCheckbox label, .stSelectbox label,
    div[class*="st-emotion"] p,
    div[class*="st-emotion"] span {
        color: #000000 !important;
    }
    .block-container {
        background: transparent !important;
        padding: 4rem 2rem 6rem 2rem !important;
        max-width: 1380px !important;
    }
    footer { visibility: hidden !important; }
    #MainMenu { visibility: hidden !important; }

    /* ── Grid background pattern ─────────────────────────── */
    [data-testid="stAppViewContainer"]::before {
        content: '';
        position: fixed; inset: 0; pointer-events: none; z-index: 0;
        background-image:
            linear-gradient(rgba(74,127,165,0.06) 1px, transparent 1px),
            linear-gradient(90deg, rgba(74,127,165,0.06) 1px, transparent 1px);
        background-size: 44px 44px;
    }

    /* ── Streamlit header ────────────────────────────────── */
    [data-testid="stHeader"] {
        background: var(--ink2) !important;
        border-bottom: 1px solid var(--rim) !important;
        height: 0px !important; min-height: 0px !important;
        overflow: hidden !important;
    }

    /* ── Sidebar ─────────────────────────────────────────── */
    [data-testid="stSidebar"] {
        background: #ffffff !important;
        border-right: 1px solid rgba(74,127,165,0.20) !important;
        padding-top: 0 !important;
        position: relative;
    }
    [data-testid="stSidebar"]::after {
        content: '';
        position: absolute; top: 0; right: 0; bottom: 0; width: 1px;
        background: linear-gradient(to bottom, transparent, rgba(74,127,165,0.15) 40%, rgba(74,127,165,0.15) 60%, transparent);
    }
    [data-testid="stSidebar"] * {
        color: var(--text) !important;
        font-family: var(--sans) !important;
        font-size: 1.2rem !important;
    }
    [data-testid="stSidebar"] [data-testid="stRadio"] label {
        font-size: 1.2rem !important;
    }
    [data-testid="stSidebar"] .stButton > button,
    [data-testid="stSidebar"] .stDownloadButton > button {
        font-size: 1.2rem !important;
    }
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] div {
        font-size: 1.2rem !important;
    }

    /* Sidebar radio nav */
    [data-testid="stSidebar"] [data-testid="stRadio"] [role="radiogroup"] {
        display: flex !important; flex-direction: column !important; gap: 3px !important;
    }
    [data-testid="stSidebar"] [data-testid="stRadio"] label {
        padding: 8px 12px !important;
        border-radius: var(--r2) !important;
        border: 1px solid transparent !important;
        font-size: 1.306rem !important; font-weight: 500 !important;
        cursor: pointer !important;
        transition: all 0.15s ease !important;
        background: transparent !important;
        color: var(--text) !important;
        display: flex; align-items: center; gap: 8px;
    }
    [data-testid="stSidebar"] [data-testid="stRadio"] label:hover {
        background: var(--ink2) !important;
        color: var(--text) !important;
        border-color: var(--rim) !important;
    }
    [data-testid="stSidebar"] [data-testid="stRadio"] label[data-checked="true"],
    [data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked) {
        background: var(--teal3) !important;
        border-color: var(--rim2) !important;
        color: var(--text) !important;
    }

    /* ── Main buttons ────────────────────────────────────── */
    .stButton > button, .stDownloadButton > button {
        background: var(--teal) !important;
        color: #ffffff !important;
        border: 1px solid var(--teal) !important;
        border-radius: var(--r2) !important;
        font-family: var(--sans) !important;
        font-weight: 600 !important;
        font-size: 1.274rem !important;
        padding: 0.45rem 1.2rem !important;
        letter-spacing: 0.01em !important;
        box-shadow: 0 2px 8px rgba(74,127,165,0.20) !important;
        transition: all 0.15s ease !important;
    }
    .stButton > button:hover, .stDownloadButton > button:hover {
        background: var(--teal2) !important;
        box-shadow: 0 4px 14px rgba(74,127,165,0.30) !important;
        transform: translateY(-1px) !important;
    }
    .stButton > button[kind="secondary"] {
        background: transparent !important;
        color: var(--text2) !important;
        border-color: var(--rim2) !important;
        box-shadow: none !important;
    }
    .stButton > button[kind="secondary"]:hover {
        background: var(--panel) !important;
        color: var(--text) !important;
        transform: none !important;
    }

    /* Sidebar buttons — ghost style */
    [data-testid="stSidebar"] .stButton > button,
    [data-testid="stSidebar"] .stDownloadButton > button {
        background: transparent !important;
        border-color: var(--rim) !important;
        color: var(--text2) !important;
        box-shadow: none !important;
        transform: none !important;
        width: 100% !important;
        text-align: left !important;
        justify-content: flex-start !important;
    }
    [data-testid="stSidebar"] .stButton > button:hover,
    [data-testid="stSidebar"] .stDownloadButton > button:hover {
        background: var(--ink2) !important;
        color: var(--text) !important;
        border-color: var(--rim2) !important;
        transform: none !important;
        box-shadow: none !important;
    }

    /* ── Inputs / Selects ─────────────────────────────────── */
    [data-testid="stTextInput"] input,
    [data-testid="stNumberInput"] input,
    [data-testid="stTextArea"] textarea {
        background: #ffffff !important;
        border: 1px solid var(--rim2) !important;
        border-radius: var(--r2) !important;
        color: var(--text) !important;
        font-family: var(--mono) !important;
        font-size: 1.274rem !important;
    }
    [data-testid="stTextInput"] input:focus,
    [data-testid="stNumberInput"] input:focus {
        border-color: var(--teal) !important;
        box-shadow: 0 0 0 3px rgba(74,127,165,0.10) !important;
    }
    div[data-baseweb="select"] > div {
        background: #ffffff !important;
        border: 1px solid var(--rim2) !important;
        border-radius: var(--r2) !important;
        color: var(--text) !important;
        font-size: 1.274rem !important;
        font-family: var(--mono) !important;
    }
    div[data-baseweb="select"] > div:hover { border-color: var(--teal) !important; }
    div[data-baseweb="popover"] { background: #ffffff !important; border: 1px solid var(--rim2) !important; box-shadow: var(--shadow) !important; }
    [data-baseweb="list-item"] { background: #ffffff !important; color: var(--text) !important; }
    [data-baseweb="list-item"]:hover { background: var(--ink2) !important; }

    /* Slider */
    [data-testid="stSlider"] [role="slider"] { background: var(--teal) !important; }

    /* File uploader */
    [data-testid="stFileUploader"] section {
        background: #ffffff !important;
        border: 2px dashed rgba(74,127,165,0.25) !important;
        border-radius: var(--r) !important;
        transition: all 0.18s !important;
    }
    [data-testid="stFileUploader"] section:hover {
        border-color: rgba(74,127,165,0.50) !important;
        background: var(--teal4) !important;
    }
    [data-testid="stFileUploader"] * { color: var(--text2) !important; }

    /* Toggle / Checkbox */
    [data-testid="stCheckbox"] label, [data-testid="stToggle"] label {
        color: #000000 !important;
        font-family: var(--sans) !important;
    }

    /* Dataframe */
    [data-testid="stDataFrame"] {
        border-radius: var(--r) !important;
        overflow: hidden !important;
        border: 1px solid var(--rim) !important;
    }
    [data-testid="stDataFrame"] * { color: var(--text) !important; font-family: var(--mono) !important; }
    .stDataFrame iframe { background: #ffffff !important; }

    /* Expander */
    [data-testid="stExpander"] {
        background: #ffffff !important;
        border: 1px solid var(--rim) !important;
        border-radius: var(--r) !important;
        margin-bottom: 0.75rem !important;
        margin-top: 1.2rem !important;
    }
    [data-testid="stExpander"] summary {
        color: var(--text) !important;
        font-weight: 600 !important;
        font-family: var(--sans) !important;
    }
    [data-testid="stExpander"] summary:hover { color: var(--teal) !important; }
    [data-testid="stExpander"] * { color: var(--text) !important; }

    /* Alerts */
    [data-testid="stAlert"] {
        border-radius: var(--r2) !important;
        border-left: 3px solid var(--teal) !important;
        background: var(--teal4) !important;
    }
    [data-testid="stSuccess"] {
        border-left-color: var(--green) !important;
        background: rgba(52,211,153,0.07) !important;
    }
    [data-testid="stWarning"] {
        border-left-color: var(--amber) !important;
        background: var(--amber2) !important;
    }
    [data-testid="stError"] {
        border-left-color: var(--rose) !important;
        background: var(--rose2) !important;
    }

    /* Metric */
    [data-testid="stMetric"] {
        background: #ffffff !important;
        border: 1px solid var(--rim) !important;
        border-radius: var(--r) !important;
        padding: 1rem 1.2rem !important;
    }
    [data-testid="stMetricLabel"] {
        color: var(--text3) !important; font-size: 1.075rem !important;
        text-transform: uppercase !important; letter-spacing: 0.06em !important;
        font-family: var(--mono) !important;
    }
    [data-testid="stMetricValue"] {
        color: var(--text) !important;
        font-family: var(--display) !important;
        font-size: 2.381rem !important;
    }

    /* Multiselect tags */
    [data-baseweb="tag"] {
        background: var(--sky2) !important;
        border: 1px solid rgba(56,189,248,0.22) !important;
        border-radius: 4px !important;
        color: var(--sky) !important;
        font-size: 1.152rem !important;
        font-family: var(--mono) !important;
    }

    /* Caption */
    [data-testid="stCaptionContainer"] {
        color: var(--text3) !important;
        font-size: 1.152rem !important;
        font-family: var(--mono) !important;
    }

    /* Radio pills */
    [data-testid="stRadio"] [role="radiogroup"] { gap: 6px !important; flex-wrap: wrap !important; }
    [data-testid="stRadio"] label {
        padding: 5px 16px !important;
        border: 1px solid var(--rim2) !important;
        border-radius: 20px !important;
        font-size: 1.229rem !important;
        cursor: pointer !important;
        transition: all 0.15s ease !important;
        color: var(--text2) !important;
        background: #ffffff !important;
        font-family: var(--sans) !important;
    }
    [data-testid="stRadio"] label:has(input:checked) {
        border-color: var(--teal) !important;
        color: var(--teal) !important;
        background: var(--teal3) !important;
    }

    /* Divider */
    hr { border-color: var(--rim) !important; margin: 1.5rem 0 !important; }

    /* Scrollbar */
    * { scrollbar-width: thin; scrollbar-color: var(--rim2) var(--ink2); }

    /* ── Custom components ─────────────────────────────────── */

    /* Topbar */
    .SENTI-topbar {
        position: sticky; top: 0; z-index: 1000;
        background: rgba(255,255,255,0.92);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border-bottom: 1px solid var(--rim);
        padding: 13px 2rem;
        display: flex; align-items: center; gap: 16px;
        margin: 0 -2rem;
    }
    .SENTI-topbar-logo {
        font-family: var(--display);
        font-size: 1.997rem; font-weight: 700; letter-spacing: -0.01em;
        color: var(--text) !important;
        display: flex; align-items: center; gap: 10px;
    }
    .SENTI-topbar-logo .hex {
        width: 30px; height: 30px;
        background: linear-gradient(135deg, var(--teal), #0088CC);
        clip-path: polygon(50% 0%, 95% 25%, 95% 75%, 50% 100%, 5% 75%, 5% 25%);
        display: flex; align-items: center; justify-content: center;
        font-size: 1.044rem; font-weight: 700; color: var(--ink);
        font-family: var(--display); flex-shrink: 0;
    }
    .SENTI-topbar-pill {
        background: var(--teal3);
        border: 1px solid var(--rim2);
        border-radius: 20px;
        font-family: var(--mono); font-size: 0.998rem;
        color: var(--teal); padding: 3px 10px;
        letter-spacing: 0.05em;
    }
    .SENTI-topbar-sub {
        font-family: var(--mono); font-size: 0.998rem;
        color: var(--text3); margin-left: auto;
        letter-spacing: 0.04em;
    }

    /* Stepper */
    .SENTI-stepper { display: flex; align-items: center; margin-left: auto; }
    .SENTI-step {
        display: flex; align-items: center; gap: 6px;
        font-size: 1.044rem; font-family: var(--mono);
        color: var(--text3); padding: 4px 10px;
        white-space: nowrap;
    }
    .SENTI-step.active { color: var(--text); }
    .SENTI-step.done { color: var(--teal); }
    .SENTI-step-num {
        width: 18px; height: 18px; border-radius: 50%;
        display: flex; align-items: center; justify-content: center;
        font-size: 0.922rem; font-weight: 700;
        background: var(--ink2); border: 1px solid var(--rim);
        color: var(--text3); flex-shrink: 0;
    }
    .SENTI-step.active .SENTI-step-num {
        background: var(--teal3);
        border-color: var(--rim2);
        color: var(--teal);
    }
    .SENTI-step.done .SENTI-step-num {
        background: var(--teal3);
        border-color: var(--rim2);
        color: var(--teal);
    }
    .SENTI-step-arrow { color: var(--text3); font-size: 0.922rem; margin: 0 -2px; }

    /* Section head */
    .SENTI-section-head {
        display: flex; align-items: center; gap: 10px;
        margin: 24px 0 12px;
        padding-bottom: 10px;
        border-bottom: 1px solid var(--rim);
    }
    .SENTI-section-head h3 {
        font-family: var(--display) !important;
        font-size: 1.536rem !important; font-weight: 700 !important;
        color: var(--text) !important; margin: 0 !important;
        letter-spacing: -0.01em;
    }
    .SENTI-section-badge {
        font-family: var(--mono);
        font-size: 0.89rem; text-transform: uppercase; letter-spacing: 0.1em;
        border-radius: 3px; padding: 2px 7px;
    }
    .badge-teal { background: var(--teal3); border: 1px solid var(--rim2); color: var(--teal); }
    .badge-amber { background: var(--amber2); border: 1px solid rgba(255,184,0,0.2); color: var(--amber); }
    .badge-rose { background: var(--rose2); border: 1px solid rgba(255,77,109,0.2); color: var(--rose); }
    .badge-lilac { background: var(--lilac2); border: 1px solid rgba(167,139,250,0.2); color: var(--lilac); }
    .badge-sky { background: var(--sky2); border: 1px solid rgba(56,189,248,0.2); color: var(--sky); }

    /* Cards */
    .SENTI-card {
        background: #ffffff;
        border: 1px solid var(--rim);
        border-radius: var(--r);
        padding: 16px 20px;
        margin-bottom: 14px;
        transition: border-color 0.2s;
        box-shadow: 0 1px 4px rgba(74,127,165,0.07);
    }
    .SENTI-card:hover { border-color: var(--rim2); }

    .SENTI-card-label {
        font-family: var(--mono); font-size: 0.922rem;
        color: var(--text3); text-transform: uppercase;
        letter-spacing: 0.12em; margin-bottom: 8px;
    }
    .SENTI-card h4 {
        font-family: var(--display) !important;
        font-size: 1.414rem !important; font-weight: 700 !important;
        color: var(--text) !important; margin: 0 0 5px !important;
    }
    .SENTI-card p {
        font-size: 1.26rem !important; color: var(--text2) !important;
        margin: 0 !important; line-height: 1.55 !important;
    }

    /* Stat row */
    .SENTI-stat-row { display: flex; gap: 12px; flex-wrap: wrap; margin-bottom: 16px; }
    .SENTI-stat {
        flex: 1; min-width: 120px;
        background: #ffffff; border: 1px solid var(--rim);
        border-radius: var(--r2); padding: 14px 16px;
        box-shadow: 0 1px 4px rgba(74,127,165,0.07);
    }
    .SENTI-stat .stat-label {
        font-family: var(--mono); font-size: 0.922rem;
        color: var(--text3); text-transform: uppercase;
        letter-spacing: 0.1em; margin-bottom: 5px;
    }
    .SENTI-stat .stat-val {
        font-family: var(--display); font-size: 2.381rem;
        font-weight: 700; color: var(--text); line-height: 1;
    }
    .SENTI-stat .stat-sub {
        font-family: var(--mono); font-size: 0.967rem;
        color: var(--text3); margin-top: 3px;
    }

    /* Delta card */
    .SENTI-delta-card {
        display: flex; gap: 16px; flex-wrap: wrap; align-items: center;
        background: #ffffff; border: 1px solid var(--rim);
        border-radius: var(--r); padding: 14px 20px; margin-bottom: 14px;
        box-shadow: 0 1px 4px rgba(74,127,165,0.07);
    }
    .SENTI-delta-item { display: flex; flex-direction: column; gap: 2px; }
    .SENTI-delta-label {
        font-family: var(--mono); font-size: 0.922rem;
        color: var(--text3); text-transform: uppercase; letter-spacing: 0.08em;
    }
    .SENTI-delta-val {
        font-family: var(--display); font-size: 2.15rem; font-weight: 700;
    }
    .SENTI-delta-arrow { font-size: 1.997rem; color: var(--text3); }
    .SENTI-delta-change { font-size: 1.198rem; color: var(--teal); font-weight: 600; }

    /* Pipeline */
    .SENTI-pipeline { display: flex; gap: 0; margin-bottom: 18px; }
    .SENTI-pipeline-step {
        flex: 1; padding: 8px 10px; text-align: center;
        background: #ffffff; border: 1px solid var(--rim);
        font-family: var(--mono); font-size: 0.998rem;
        font-weight: 700; letter-spacing: 0.05em; color: var(--text3);
        transition: all 0.2s;
    }
    .SENTI-pipeline-step:first-child { border-radius: 8px 0 0 8px; }
    .SENTI-pipeline-step:last-child { border-radius: 0 8px 8px 0; }
    .SENTI-pipeline-step.active {
        background: var(--teal3);
        border-color: var(--rim2);
        color: var(--teal);
    }
    .SENTI-pipeline-step.done {
        background: var(--ink2);
        color: var(--text3);
    }

    /* Iter badge */
    .SENTI-iter-badge {
        display: inline-flex; align-items: center; gap: 6px;
        background: var(--teal3); border: 1px solid var(--rim2);
        border-radius: 20px; padding: 3px 12px;
        font-family: var(--mono); font-size: 1.106rem; color: var(--teal);
    }
    .SENTI-iter-dot {
        width: 6px; height: 6px; border-radius: 50%;
        background: var(--teal); animation: SENTI-pulse 2s infinite;
    }
    @keyframes SENTI-pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.25; } }

    /* Empty state */
    .SENTI-empty {
        display: flex; flex-direction: column; align-items: center;
        justify-content: center; padding: 48px 24px; text-align: center;
        border: 2px dashed var(--rim); border-radius: var(--r); margin: 14px 0;
    }
    .SENTI-empty-icon { font-size: 3.072rem; margin-bottom: 12px; opacity: 0.4; }
    .SENTI-empty h4 {
        font-family: var(--display) !important; font-size: 1.459rem !important;
        font-weight: 700 !important; color: var(--text2) !important;
        margin: 0 0 6px !important;
    }
    .SENTI-empty p {
        font-size: 1.229rem !important; color: var(--text3) !important;
        margin: 0 !important; max-width: 300px; line-height: 1.5 !important;
    }

    /* Miss bars */
    .miss-bar-wrap { margin: 6px 0 14px; }
    .miss-bar-row {
        display: flex; align-items: center; gap: 8px;
        margin-bottom: 5px; font-family: var(--mono);
        font-size: 1.075rem; color: var(--text2);
    }
    .miss-bar-col {
        min-width: 100px; max-width: 120px;
        overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
        color: var(--text3) !important;
    }
    .miss-bar-track { flex: 1; height: 4px; background: var(--panel2); border-radius: 2px; overflow: hidden; }
    .miss-bar-fill { height: 100%; border-radius: 2px; }
    .miss-bar-pct { min-width: 32px; text-align: right; }

    /* FAISS monitor */
    .faiss-table-wrap { overflow-x: auto; }
    .faiss-table {
        width: 100%; border-collapse: collapse;
        font-family: var(--mono); font-size: 1.168rem;
    }
    .faiss-table th {
        padding: 5px 10px; text-align: left;
        color: var(--text3); font-size: 0.922rem;
        text-transform: uppercase; letter-spacing: 0.07em;
        background: var(--ink2); border-bottom: 1px solid var(--rim);
    }
    .faiss-table td { padding: 6px 10px; border-bottom: 1px solid var(--rim); }

    /* Dataset status sidebar */
    .ds-status-card {
        margin: 0 10px 10px;
        background: var(--teal3);
        border: 1px solid var(--rim2);
        border-radius: var(--r2); padding: 10px 12px;
    }
    .ds-status-label {
        font-family: var(--mono); font-size: 0.89rem;
        color: var(--text3) !important; text-transform: uppercase;
        letter-spacing: 0.1em; margin-bottom: 3px;
    }
    .ds-status-name {
        font-size: 1.26rem; font-weight: 600;
        color: var(--text) !important; margin-bottom: 5px;
        word-break: break-all;
    }
    .ds-status-pills { display: flex; gap: 4px; flex-wrap: wrap; }
    .ds-pill {
        font-family: var(--mono); font-size: 0.922rem;
        padding: 1px 6px; border-radius: 3px;
    }
    .ds-pill-rows { background: var(--sky2); color: var(--sky) !important; border: 1px solid rgba(3,105,161,0.2); }
    .ds-pill-miss { background: var(--amber2); color: var(--amber) !important; border: 1px solid rgba(146,96,10,0.2); }
    .ds-pill-done { background: var(--teal3); color: var(--teal) !important; border: 1px solid var(--rim2); }

    /* Sidebar brand */
    .sidebar-brand {
        padding: 20px 16px 16px;
        border-bottom: 1px solid var(--rim);
        margin-bottom: 10px;
    }
    .sidebar-brand-name {
        font-family: var(--display);
        font-size: 1.843rem; font-weight: 700; letter-spacing: -0.01em;
        color: var(--text) !important;
        display: flex; align-items: center; gap: 9px;
        margin-bottom: 3px;
    }
    .sidebar-brand-hex {
        width: 26px; height: 26px;
        background: linear-gradient(135deg, var(--teal), #0088CC);
        clip-path: polygon(50% 0%, 95% 25%, 95% 75%, 50% 100%, 5% 75%, 5% 25%);
    }
    .sidebar-brand-sub {
        font-family: var(--mono); font-size: 0.922rem;
        color: var(--text3) !important; text-transform: uppercase;
        letter-spacing: 0.1em; padding-left: 35px;
    }
    .sidebar-nav-label {
        font-family: var(--mono); font-size: 0.89rem;
        color: var(--text3) !important; text-transform: uppercase;
        letter-spacing: 0.12em; padding: 0 12px;
        margin: 8px 0 3px;
    }
    .sidebar-divider { height: 1px; background: var(--rim); margin: 8px 12px; }

    /* Drift monitor */
    .drift-row {
        display: flex; justify-content: space-between; align-items: center;
        padding: 8px 0; border-bottom: 1px solid var(--rim);
        font-size: 1.229rem;
    }
    .drift-row:last-child { border-bottom: none; }

    /* Mechanism cards */
    .mech-card-grid { display: flex; gap: 10px; flex-wrap: wrap; margin-bottom: 16px; }
    .mech-card {
        flex: 1; min-width: 180px;
        border: 1px solid var(--rim); border-radius: var(--r);
        padding: 14px 16px; cursor: pointer; transition: all 0.15s;
        background: #ffffff;
    }
    .mech-card:hover { border-color: var(--rim2); background: var(--ink2); }
    .mech-card.selected {
        border-color: var(--rim2) !important;
        background: var(--teal3) !important;
    }
    .mech-tag { font-family: var(--mono); font-size: 0.998rem; font-weight: 700; margin-bottom: 5px; }
    .mech-card h4 { font-family: var(--display) !important; font-size: 1.351rem !important; font-weight: 700 !important; color: var(--text) !important; margin-bottom: 3px !important; }
    .mech-card p { font-size: 1.168rem !important; color: var(--text2) !important; line-height: 1.4 !important; margin: 0 !important; }

    /* Doc steps */
    .doc-step {
        display: flex; gap: 14px; align-items: flex-start;
        padding: 14px 0; border-bottom: 1px solid var(--rim);
    }
    .doc-step:last-child { border-bottom: none; }
    .doc-num {
        min-width: 28px; height: 28px; border-radius: 50%;
        background: var(--teal3); border: 1px solid var(--rim2);
        display: flex; align-items: center; justify-content: center;
        font-family: var(--display); font-size: 1.26rem;
        font-weight: 700; color: var(--teal);
    }
    .doc-text h4 { font-family: var(--display) !important; font-size: 1.351rem !important; font-weight: 700 !important; color: var(--text) !important; margin-bottom: 4px !important; }
    .doc-text p { font-size: 1.229rem !important; color: var(--text2) !important; line-height: 1.5 !important; margin: 0 !important; }

    /* Column type badges */
    .col-N { display:inline-block; font-family:var(--mono); font-size: 0.86rem; padding:1px 4px; border-radius:3px; margin-right:4px; background:var(--sky2); color:var(--sky); border:1px solid rgba(3,105,161,0.2); }
    .col-C { display:inline-block; font-family:var(--mono); font-size: 0.86rem; padding:1px 4px; border-radius:3px; margin-right:4px; background:var(--amber2); color:var(--amber); border:1px solid rgba(146,96,10,0.2); }

    /* Table head label */
    .SENTI-table-head {
        font-family: var(--mono); font-size: 0.922rem;
        color: var(--text3); text-transform: uppercase;
        letter-spacing: 0.1em; margin-bottom: 5px;
    }

    /* Sim legend */
    .sim-legend {
        display: flex; gap: 14px; font-size: 1.106rem;
        color: var(--text2); margin: 4px 0 8px; flex-wrap: wrap;
    }
    .sim-legend span { display: flex; align-items: center; gap: 5px; }
    .sim-dot { width: 9px; height: 9px; border-radius: 2px; }

    /* Compare headers */
    .compare-head { display: flex; gap: 12px; margin-bottom: 5px; }
    .compare-head-item {
        flex: 1; font-family: var(--mono); font-size: 0.998rem;
        color: var(--text3); text-transform: uppercase; letter-spacing: 0.1em;
        padding-bottom: 4px; border-bottom: 2px solid var(--rim);
    }
    .compare-head-item.imputed { border-color: var(--teal); color: var(--teal) !important; }
    .compare-head-item.ground { border-color: var(--sky); color: var(--sky) !important; }

    /* Footer */
    .SENTI-footer {
        position: fixed; bottom: 0; left: 0; right: 0; height: 38px;
        display: flex; align-items: center; justify-content: center; gap: 8px;
        background: rgba(255,255,255,0.95); border-top: 1px solid var(--rim);
        font-family: var(--mono); font-size: 0.967rem;
        color: var(--text3); z-index: 999;
    }
    .SENTI-footer a { color: var(--teal); text-decoration: none; }
    .SENTI-footer a:hover { color: var(--teal); }
    .SENTI-footer-sep { color: var(--rim2); }

    /* Spiner override */
    .stSpinner { color: var(--teal) !important; }
    [data-testid="stSpinner"] svg { stroke: var(--teal) !important; }

    /* Code inline */
    code { background: var(--ink2) !important; border: 1px solid var(--rim) !important; color: var(--teal) !important; font-family: var(--mono) !important; }

    /* ── Global secondary buttons in main content: light blue pill ── */
    section[data-testid="stMain"] .stButton > button[kind="secondary"] {
        background: #b8d4e8 !important;
        color: #1a2c3d !important;
        border: 1.5px solid #8fb8d4 !important;
        border-radius: 22px !important;
        font-size: 0.92rem !important;
        font-weight: 600 !important;
        box-shadow: none !important;
        transition: all 0.15s ease !important;
    }
    section[data-testid="stMain"] .stButton > button[kind="secondary"]:hover {
        background: #9ac4de !important;
        border-color: #6aa8cc !important;
        transform: translateY(-1px) !important;
        box-shadow: 0 2px 8px rgba(74,127,165,0.20) !important;
        color: #1a2c3d !important;
    }

    </style>
    """, unsafe_allow_html=True)


def _render_topbar(active_page: str = "Imputation"):
    # Determine step states
    page_order = ["Inject nulls", "Imputation", "Evaluation"]
    def _step_class(name):
        idx_active = page_order.index(active_page) if active_page in page_order else 1
        idx_step   = page_order.index(name)
        if idx_step < idx_active:  return "done"
        if idx_step == idx_active: return "active"
        return ""
    steps_html = ""
    labels = {"Inject nulls": "Inject Nulls", "Imputation": "Imputation", "Evaluation": "Evaluation"}
    for i, p in enumerate(page_order):
        cls = _step_class(p)
        check = "✓" if cls == "done" else str(i + 1)
        if i > 0:
            steps_html += '<span class="SENTI-step-arrow">›</span>'
        steps_html += f'''<div class="SENTI-step {cls}">
            <div class="SENTI-step-num">{check}</div>
            {labels[p]}
        </div>'''

    st.markdown(f"""
    <div class="SENTI-topbar">
        <div class="SENTI-topbar-logo">
            <div class="hex">ST</div>
            SENTI
        </div>
        <div style="display:flex;flex-direction:column;justify-content:center;gap:1px;">
            <span style="font-family:var(--display);font-size:1.459rem;font-weight:700;
                color:var(--text);letter-spacing:-0.01em;line-height:1.2;">
            </span>
            <span style="font-family:var(--mono);font-size:0.922rem;color:var(--text3);
                letter-spacing:0.1em;text-transform:uppercase;">
            </span>
        </div>
        <span class="SENTI-topbar-pill">SENtence Transformer based data Imputation</span>
        <div class="SENTI-stepper">{steps_html}</div>
    </div>
    """, unsafe_allow_html=True)


def _render_footer():
    st.markdown("""
    <div class="SENTI-footer">
        <span>Developed at DIMES, University of Calabria</span>
        <span class="SENTI-footer-sep">·</span>
        <a href="https://dimes.unical.it" target="_blank">dimes.unical.it</a>
        <span class="SENTI-footer-sep">·</span>
        <a href="https://dottorato.dimes.unical.it/students/mahmood-tariq" target="_blank">Mahmood Tariq</a>
    </div>
    """, unsafe_allow_html=True)


def _section(title, badge=None):
    badge_html = f'<span class="SENTI-section-badge">{badge}</span>' if badge else ""
    st.markdown(f"""
    <div class="SENTI-section-head">
        <h3>{title}</h3>
        {badge_html}
    </div>
    """, unsafe_allow_html=True)


def _empty_state(icon: str, title: str, body: str):
    """Render a friendly empty-state placeholder."""
    st.markdown(f"""
    <div class="SENTI-empty">
        <div class="SENTI-empty-icon">{icon}</div>
        <h4>{title}</h4>
        <p>{body}</p>
    </div>
    """, unsafe_allow_html=True)


def _missingness_bars(df: pd.DataFrame):
    """Render per-column missingness bars."""
    n = len(df)
    if n == 0:
        return
    rows_html = ""
    for col in df.columns:
        miss = int(df[col].isna().sum())
        pct = miss / n
        pct_label = f"{pct*100:.0f}%"
        if pct == 0:
            fill_color = "var(--border2)"
        elif pct < 0.3:
            fill_color = "var(--gold)"
        else:
            fill_color = "var(--red)"
        rows_html += f"""
        <div class="miss-bar-row">
            <span class="miss-bar-col" title="{col}">{col}</span>
            <div class="miss-bar-track">
                <div class="miss-bar-fill" style="width:{pct*100:.1f}%; background:{fill_color};"></div>
            </div>
            <span class="miss-bar-pct">{pct_label}</span>
        </div>"""
    st.markdown(f'<div class="miss-bar-wrap">{rows_html}</div>', unsafe_allow_html=True)


def _imputation_delta_card(source_df: pd.DataFrame, imputed_df: pd.DataFrame, mask: pd.DataFrame):
    """Render a before/after missing-cell summary card."""
    before = int(source_df.isna().sum().sum())
    after  = int(imputed_df.isna().sum().sum())
    filled = before - after
    pct_resolved = f"{100*filled/before:.0f}%" if before else "—"
    st.markdown(f"""
    <div class="SENTI-delta-card">
        <div class="SENTI-delta-item">
            <span class="SENTI-delta-label">Missing Before</span>
            <span class="SENTI-delta-val" style="color:var(--gold)">{before}</span>
        </div>
        <div class="SENTI-delta-arrow">→</div>
        <div class="SENTI-delta-item">
            <span class="SENTI-delta-label">Missing After</span>
            <span class="SENTI-delta-val" style="color:var(--green)">{after}</span>
        </div>
        <div class="SENTI-delta-item" style="margin-left:auto;">
            <span class="SENTI-delta-label">Cells Filled</span>
            <span class="SENTI-delta-val" style="color:var(--green)">{filled}</span>
            <span class="SENTI-delta-change">▲ {pct_resolved} resolved</span>
        </div>
    </div>
    """, unsafe_allow_html=True)


def _sidebar_dataset_status():
    """Render a compact dataset status card in the sidebar."""
    df = ss.get("working_df")
    imputed = ss.get("imputed_df")
    if df is None:
        return
    name = ss.get("_last_upload_name") or ss.get("demo_kind") or "Dataset"
    miss = int(df.isna().sum().sum())
    total_cells = df.shape[0] * df.shape[1]
    miss_pct = f"{100*miss/total_cells:.0f}%" if total_cells else "0%"
    done_pill = '<span class="ds-pill ds-pill-done">✓ Imputed</span>' if imputed is not None else ""
    st.markdown(f"""
    <div class="ds-status-card">
        <div class="ds-status-label">Active Dataset</div>
        <div class="ds-status-name">{name}</div>
        <div class="ds-status-pills">
            <span class="ds-pill ds-pill-rows">{df.shape[0]}×{df.shape[1]}</span>
            <span class="ds-pill ds-pill-miss">{miss} missing ({miss_pct})</span>
            {done_pill}
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_sidebar():
    with st.sidebar:
        base_options = ["Inject nulls", "Imputation", "Evaluation"]
        display_options = ["Inject Nulls", "Imputation", "Evaluation"]

        current_page = getattr(ss, "page", "Imputation")
        # While docs are open keep radio on the last real page, not "Guide"
        _last_real = ss.get("_last_real_page") or "Imputation"
        radio_page = _last_real if getattr(ss, "_doc", False) else current_page
        if radio_page not in base_options:
            radio_page = "Imputation"
        current_index = base_options.index(radio_page)

        selected_index = st.radio(
            label="",
            label_visibility="collapsed",
            options=range(len(base_options)),
            format_func=lambda i: display_options[i],
            index=current_index,
            key="page_radio",
        )

        new_page = base_options[selected_index]

        if new_page != radio_page:
            ss._doc = False
            ss._last_real_page = new_page
            ss.page = new_page
            st.rerun()

        st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)

        if st.button("📄  Guide", key="btn_nav_docs"):
            real = getattr(ss, "page", "Imputation")
            if real != "Guide":
                ss._last_real_page = real
            ss._doc = True
            ss.page = "Guide"
            st.rerun()

        if st.button("↺  Reset", key="btn_reset_app_sidebar"):
            st.session_state.clear()
            st.rerun()


_SUPPORTED_TYPES = ["csv", "json", "jsonl", "xlsx", "xls", "parquet", "tsv"]

def _read_uploaded_file(uploaded_file) -> pd.DataFrame:
    """Read a Streamlit UploadedFile into a DataFrame.
    Supports CSV, TSV, JSON, JSONL, Excel (xlsx/xls), Parquet.
    """
    name = getattr(uploaded_file, "name", "") or ""
    ext  = name.rsplit(".", 1)[-1].lower() if "." in name else ""
    try:
        if ext in ("csv", "tsv", ""):
            sep = "\t" if ext == "tsv" else ","
            return pd.read_csv(uploaded_file, sep=sep)
        elif ext in ("json", "jsonl"):
            import io
            raw = uploaded_file.read()
            uploaded_file.seek(0)
            # Try normal JSON first, then JSON Lines
            try:
                return pd.read_json(io.BytesIO(raw))
            except Exception:
                return pd.read_json(io.BytesIO(raw), lines=True)
        elif ext in ("xlsx", "xls"):
            return pd.read_excel(uploaded_file)
        elif ext == "parquet":
            return pd.read_parquet(uploaded_file)
        else:
            # Fallback: try CSV
            return pd.read_csv(uploaded_file)
    except Exception as e:
        raise ValueError(f"Could not read file '{name}' as {ext or 'CSV'}: {e}")


def _bootstrap_df(df: pd.DataFrame):
    ss.raw_df = df.copy()
    ss.working_df = df.copy()
    ss.selected_cols = []
    ss.imputed_df = None
    ss.imputed_mask = None
    ss.append_highlight = None
    ss.flow_state = "idle"
    ss.iter_k = 1
    ss.last_imputed_iter = 0
    ss.source_snapshot = None
    ss.pre_append_snapshot = None
    ss.append_history = []
    ss.inc_log = []
    ss._last_appended_file_id = None
    ss.faiss_history = []
    ss.faiss_log = []
    st.success(f"Dataset loaded — {df.shape[0]} rows × {df.shape[1]} columns")


def _df_stats_html(df: pd.DataFrame) -> str:
    missing = int(df.isna().sum().sum())
    total = df.shape[0] * df.shape[1]
    miss_pct = f"{100*missing/total:.1f}%" if total else "—"
    num_cols = sum(pd.api.types.is_numeric_dtype(df[c]) for c in df.columns)
    cat_cols = df.shape[1] - num_cols
    return f"""
    <div class="SENTI-stat-row">
      <div class="SENTI-stat">
        <span class="stat-label">Rows</span>
        <span class="stat-val">{df.shape[0]}</span>
      </div>
      <div class="SENTI-stat">
        <span class="stat-label">Columns</span>
        <span class="stat-val">{df.shape[1]}</span>
        <span class="stat-sub">{num_cols} num · {cat_cols} cat</span>
      </div>
      <div class="SENTI-stat">
        <span class="stat-label">Missing</span>
        <span class="stat-val" style="color:var(--gold)">{missing}</span>
        <span class="stat-sub">{miss_pct} of cells</span>
      </div>
    </div>
    """


def run_imputation(df, context_df=None):
    try:
        if df is None or len(df) == 0:
            st.warning("Please upload or select a dataset first.")
            return None, None
        strategy = getattr(ss, "strategy", "SENTI")
        cols = (ss.selected_cols if getattr(ss, "selected_cols", None) else list(df.columns))
        if strategy == "SENTI":
            imputed, mask, faiss_stats = impute_senti(
                df, cols,
                getattr(ss, "transformer_choice", "LaBSE"),
                context_df=context_df,
            )
            if not isinstance(ss.get("faiss_log"), list):
                ss.faiss_log = []
            ss.faiss_log.append({
                "batch":  (ss.iter_k or 1),
                "n_new":  len(df),
                "stats":  faiss_stats,
            })
        elif strategy == "custom":
            adapter_code = ss.get("custom_adapter_code", "")
            repo_path    = ss.get("custom_repo_path", None)
            if not adapter_code or not repo_path:
                st.error("Custom imputer: no adapter ready. Complete the setup in the Custom panel above.")
                return None, None
            from pathlib import Path as _Path
            from modules.custom_imputer import run_adapter
            with st.spinner("Running custom imputer in subprocess …"):
                imputed, mask, log = run_adapter(adapter_code, df, cols, _Path(repo_path))
            if imputed is None:
                st.error("Custom imputer failed.")
                with st.expander("Error log", expanded=True):
                    st.markdown(
                        f'<pre style="background:#fff0f3;color:#b91c3c;font-size:1.136rem;'
                        f'line-height:1.6;padding:14px 16px;border-radius:8px;'
                        f'overflow-x:auto;white-space:pre-wrap;border:1px solid rgba(185,28,60,0.25)">'
                        f'{log}</pre>', unsafe_allow_html=True)
                return None, None
            if log.strip():
                with st.expander("Custom imputer log", expanded=False):
                    st.markdown(
                        f'<pre style="background:#f5f8fb;color:#3a506a;font-size:1.136rem;'
                        f'line-height:1.6;padding:14px 16px;border-radius:8px;'
                        f'overflow-x:auto;white-space:pre-wrap;border:1px solid rgba(74,127,165,0.2)">'
                        f'{log}</pre>', unsafe_allow_html=True)
        elif strategy in ("knn", "ffill"):
            imputed = df.copy()
            original = df.copy()
            target_cols = [c for c in cols if c in df.columns]
            if strategy == "knn":
                try:
                    from sklearn.impute import KNNImputer
                    num_cols_knn = [c for c in target_cols if pd.api.types.is_numeric_dtype(df[c])]
                    if num_cols_knn:
                        knn_imp = KNNImputer(n_neighbors=5)
                        filled = knn_imp.fit_transform(df[num_cols_knn])
                        for i, c in enumerate(num_cols_knn):
                            imputed[c] = filled[:, i]
                except Exception:
                    for c in target_cols:
                        if pd.api.types.is_numeric_dtype(df[c]):
                            imputed[c] = df[c].fillna(df[c].median())
            else:
                for c in target_cols:
                    imputed[c] = df[c].ffill().bfill()
            mask = original.isna() & imputed.notna()
        else:
            other_strat = getattr(ss, "other_strategy", "mean")
            if other_strat not in ("mean", "median", "mode", "MostFreq"):
                other_strat = "mean"
            imputed, mask = impute_other(df, cols, other_strat)
        return imputed, mask
    except Exception as e:
        st.error("Imputation failed.")
        st.exception(e)
        return None, None


def render_source_preview(df, iter_k: int):
    if iter_k > 0:
        st.markdown(f"""
        <div class="SENTI-section-head" style="margin-top:1rem;">
            <h3>Source Dataset</h3>
            <span class="SENTI-iter-badge"><span class="dot"></span> Iteration {iter_k}</span>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown('<div class="SENTI-table-head">Source Dataset</div>', unsafe_allow_html=True)

    if df is not None:
        st.markdown(unsafe_allow_html=True, body=_df_stats_html(df))
        miss_any = df.isna().any().any()
        if miss_any:
            with st.expander("📊 Per-column missingness", expanded=False):
                _missingness_bars(df)
    # Show only first 5 rows, replace nan with empty string for display
    preview = df.head(5).copy() if df is not None else pd.DataFrame()
    total_rows = len(df) if df is not None else 0
    # Convert NA/nan to empty string for clean display
    display_preview = preview.astype(object).where(preview.notna(), other="")
    try:
        styled = style_imputed_and_appended(display_preview, mask=None, appended_tuples=None)
        st.markdown(styled.to_html(), unsafe_allow_html=True)
    except Exception:
        st.markdown(display_preview.to_html(index=True, na_rep=""), unsafe_allow_html=True)
    st.markdown(
        f'<div style="font-size:1.106rem;color:var(--text3);font-family:var(--font-mono);margin-top:0.25rem;">'
        f'Showing first 5 of {total_rows} rows</div>',
        unsafe_allow_html=True
    )


def render_appended_preview(df, iter_k: int, appended_tuples):
    # Extract only the newly appended rows (now at the END of combined df)
    new_rows = pd.DataFrame()
    if appended_tuples is not None and df is not None:
        try:
            mask_arr = appended_tuples.reset_index(drop=True).values if isinstance(appended_tuples, pd.Series) else np.array(appended_tuples)
            n_new = int(mask_arr.sum())
            if n_new > 0:
                new_rows = df.iloc[-n_new:].copy()
        except Exception:
            pass

    if new_rows.empty:
        return  # Nothing to show yet

    st.markdown(f"""
    <div class="SENTI-section-head" style="margin-top:1rem;">
        <h3>New Batch Preview</h3>
        <span class="SENTI-iter-badge"><span class="dot"></span> Iteration {iter_k}</span>
    </div>
    """, unsafe_allow_html=True)
    total_new = len(new_rows)
    preview = new_rows.head(5).copy()
    display_preview = preview.astype(object).where(preview.notna(), other="")
    try:
        styled = style_imputed_and_appended(display_preview, mask=None, appended_tuples=None)
        st.markdown(styled.to_html(), unsafe_allow_html=True)
    except Exception:
        st.markdown(display_preview.to_html(index=True, na_rep=""), unsafe_allow_html=True)
    st.markdown(
        f'<div style="font-size:1.106rem;color:var(--text3);font-family:var(--font-mono);margin-top:0.25rem;">'
        f'Showing first 5 of {total_new} new rows</div>',
        unsafe_allow_html=True
    )


def _render_faiss_panel(latest_only: bool = False):
    """
    Render the FAISS Index Monitor — one expandable section per batch,
    with a sub-row for each of the 3 imputation phases showing:
      CONTEXT (before) = cumulative vectors from all previously imputed rows
      + NEW BATCH      = vectors from the current new batch
      TOTAL (after)    = CONTEXT + NEW  — grows cumulatively each iteration
      Δ CHECK          = must equal NEW BATCH rows to confirm correctness

    Confirms the FAISS DB grows only by new batch rows each iteration,
    and that previously imputed rows enrich neighbor search without re-imputation.
    Only shown when strategy == SENTI and faiss_log is non-empty.
    """
    if ss.strategy != "SENTI":
        return
    log = ss.get("faiss_log") or []
    if not log:
        return

    display_log = [log[-1]] if latest_only else log
    n_hidden    = len(log) - 1 if latest_only else 0

    # ── Legend bar ────────────────────────────────────────────────────
    prev_note_faiss = (
        f'<span style="font-size:1.106rem;color:var(--text3);margin-left:8px">'
        f'(+ {n_hidden} previous batch(es) saved to JSON)</span>'
        if n_hidden > 0 else ""
    )
    legend = (
        '<div style="display:flex;gap:18px;flex-wrap:wrap;font-size:1.106rem;'
        'color:var(--text3);margin-bottom:10px;line-height:1.8">'
        '<span><strong style="color:var(--text2)">CONTEXT&nbsp;(before)</strong>'
        ' = cumulative vectors from all prior imputed batches</span>'
        '<span><strong style="color:var(--accent3)">+NEW&nbsp;BATCH</strong>'
        ' = rows in this batch</span>'
        '<span><strong style="color:var(--green)">TOTAL&nbsp;(after)</strong>'
        ' = context + new batch (cumulative index size)</span>'
        '<span><strong style="color:var(--text)">Δ&nbsp;CHECK</strong>'
        ' ✓ green = delta equals new rows exactly (correct incremental behaviour)</span>'
        '</div>'
    )

    # ── Per-batch blocks ──────────────────────────────────────────────
    PHASE_LABEL_COLOR = {
        "Phase 1 — categorical": "#60A5FA",
        "Phase 2 — numeric":     "#A78BFA",
        "Phase 3 — fallback":    "#36D399",
    }

    batches_html = ""
    for entry in display_log:
        batch_no = entry["batch"]
        n_new    = entry["n_new"]
        stats    = entry.get("stats") or {}
        phases   = stats.get("phases") or []
        n_ctx    = stats.get("n_context", 0)
        n_after  = stats.get("n_total_after", 0)
        dim      = stats.get("dim", "—")

        # Summary row (top of batch block)
        delta_ok  = (n_after - n_ctx) == n_new
        sum_color = "var(--green)" if delta_ok else "var(--accent3)"
        sum_icon  = "✓" if delta_ok else "△"

        batches_html += (
            f'<div style="margin-bottom:14px;border:1px solid var(--border);'
            f'border-radius:8px;overflow:hidden">'
            # Batch header
            f'<div style="background:rgba(74,127,165,0.04);padding:7px 12px;'
            f'display:flex;gap:20px;align-items:center;border-bottom:1px solid var(--border)">'
            f'<span style="color:var(--accent2);font-weight:700;font-size:1.306rem">Batch #{batch_no}</span>'
            f'<span style="font-size:1.198rem;color:var(--text3)">Context&nbsp;<strong style="color:var(--text2)">{n_ctx}</strong></span>'
            f'<span style="font-size:1.198rem;color:var(--accent3)">+New&nbsp;<strong>{n_new}</strong></span>'
            f'<span style="font-size:1.198rem;color:var(--green)">Total&nbsp;<strong>{n_after}</strong></span>'
            f'<span style="font-size:1.198rem;color:{sum_color};font-weight:700">{sum_icon} Δ={n_after - n_ctx}</span>'
            f'<span style="font-size:1.106rem;color:var(--text3);margin-left:auto">dim={dim}</span>'
            f'</div>'
            # Phase sub-table
            f'<table style="width:100%;border-collapse:collapse;font-size:1.198rem">'
            f'<thead><tr style="color:var(--text3);font-size:1.03rem;letter-spacing:0.07em;'
            f'border-bottom:1px solid var(--border);background:rgba(74,127,165,0.06)">'
            f'<th style="padding:4px 12px;text-align:left">PHASE</th>'
            f'<th style="padding:4px 12px;text-align:right">CONTEXT&nbsp;(before)</th>'
            f'<th style="padding:4px 12px;text-align:right">+NEW BATCH</th>'
            f'<th style="padding:4px 12px;text-align:right">TOTAL&nbsp;(after)</th>'
            f'<th style="padding:4px 12px;text-align:right">Δ&nbsp;CHECK</th>'
            f'<th style="padding:4px 12px;text-align:right">AVG NORM</th>'
            f'</tr></thead><tbody>'
        )

        for ph in phases:
            ph_label   = ph.get("phase", "—")
            ph_ctx     = ph.get("n_context", 0)
            ph_new     = ph.get("n_new", 0)
            ph_total   = ph.get("n_total", 0)
            ph_delta   = ph.get("delta", 0)
            ph_ok      = ph.get("ok", False)
            ph_norm    = ph.get("avg_norm", "—")
            ph_color   = PHASE_LABEL_COLOR.get(ph_label, "var(--text2)")
            d_color    = "var(--green)" if ph_ok else "var(--red)"
            d_icon     = "✓" if ph_ok else "✗"

            batches_html += (
                f'<tr style="border-bottom:1px solid rgba(74,127,165,0.10)">'
                f'<td style="padding:5px 12px;color:{ph_color};font-weight:500">{ph_label}</td>'
                f'<td style="padding:5px 12px;text-align:right;color:var(--text2)">{ph_ctx}</td>'
                f'<td style="padding:5px 12px;text-align:right;color:var(--accent3)">+{ph_new}</td>'
                f'<td style="padding:5px 12px;text-align:right;color:var(--green);font-weight:600">{ph_total}</td>'
                f'<td style="padding:5px 12px;text-align:right;color:{d_color};font-weight:700">{d_icon}&nbsp;+{ph_delta}</td>'
                f'<td style="padding:5px 12px;text-align:right;color:var(--text3);font-size:1.106rem">{ph_norm}</td>'
                f'</tr>'
            )

        batches_html += '</tbody></table></div>'

    # ── Summary footer ────────────────────────────────────────────────
    last      = log[-1]
    n_batches = len(log)
    final_vec = (last.get("stats") or {}).get("n_total_after", "—")
    footer_html = (
        f'<p style="font-size:1.136rem;color:var(--text3);margin:10px 0 0;line-height:1.7;'
        f'border-top:1px solid var(--border);padding-top:8px">'
        f'After <strong style="color:var(--text)">{n_batches}</strong> batch(es), the cumulative '
        f'FAISS context holds <strong style="color:var(--green)">{final_vec} vectors</strong>. '
        f'Each batch queries against <em>all</em> previously imputed rows as neighbors '
        f'(context), but only <span style="color:var(--accent3)">NEW rows</span> are imputed — '
        f'prior rows are never re-processed. '
        f'A <span style="color:var(--green)">✓ green Δ&nbsp;CHECK</span> on every phase '
        f'confirms the index grows by exactly the new-batch count each time.'
        f'</p>'
    )

    st.markdown(f"""
    <div class="SENTI-card" style="margin-top:1rem">
      <div class="SENTI-card-label">FAISS Index Monitor{prev_note_faiss}</div>
      {legend}
      {batches_html}
      {footer_html}
    </div>
    """, unsafe_allow_html=True)


def show_imputed(imputed, mask, title="Imputed Dataset"):
    if imputed is None:
        return
    _section(title, badge="OUTPUT")

    # Delta summary card (before → after)
    _snap = ss.get("source_snapshot")
    source = _snap if _snap is not None else ss.get("working_df")
    if source is not None:
        _imputation_delta_card(source, imputed, mask)

    # Show only first 5 rows, replace nan with empty string for display
    preview = imputed.head(5).copy()
    mask_preview = mask.head(5) if mask is not None else None
    display_preview = preview.astype(object).where(preview.notna(), other="")
    try:
        styled = style_imputed_and_appended(display_preview, mask_preview, appended_tuples=None)
        st.markdown(styled.to_html(), unsafe_allow_html=True)
    except Exception:
        st.markdown(display_preview.to_html(index=True, na_rep=""), unsafe_allow_html=True)
    total_rows = len(imputed)
    st.markdown(f"""
    <div style="margin-top:0.3rem; font-size:1.106rem; color:var(--text3); font-family:var(--font-mono);">
        Showing first 5 of {total_rows} rows &nbsp;·&nbsp; Gray cells = imputed &nbsp;·&nbsp; Yellow rows = appended tuples
    </div>
    """, unsafe_allow_html=True)

    csv = imputed.to_csv(index=False).encode("utf-8")
    counter = st.session_state.get("_dl_counter", 0) + 1
    st.session_state["_dl_counter"] = counter
    from datetime import datetime
    try:
        from zoneinfo import ZoneInfo
        _now = datetime.now(ZoneInfo("Europe/Rome"))
    except Exception:
        _now = datetime.now()
    _ts = _now.strftime("%Y-%m-%d_%H-%M-%S")
    dl_col, hint_col = st.columns([2, 5])
    with dl_col:
        st.download_button(
            "⬇  Download Imputed CSV",
            data=csv,
            file_name=f"imputed_{_ts}.csv",
            mime="text/csv",
            key=f"dl_imputed_{ss.mode.replace(' ', '_')}_{counter}"
        )
    with hint_col:
        st.markdown(
            f'<div style="padding-top:0.55rem; font-size:1.106rem; color:var(--text3);">'
            f'Saved as <code style="color:var(--accent3); font-size:1.075rem;">imputed_{_ts}.csv</code></div>',
            unsafe_allow_html=True
        )


def Incremental_append_ui():
    st.markdown('<div style="margin-top:1.5rem;"></div>', unsafe_allow_html=True)
    _section("Continue Incrementally?", badge="NEXT STEP")

    iter_tag = ss.last_imputed_iter or ss.iter_k or 1
    col_yes, col_no, _ = st.columns([1, 1, 4])
    with col_yes:
        yes = st.button("＋  Add More Tuples", key=f"btn_dyn_yes_{iter_tag}",
                        type="primary", use_container_width=True)
    with col_no:
        finish = st.button("✓  Finish", key=f"btn_dyn_no_finish_{iter_tag}",
                           type="secondary", use_container_width=True)

    if yes:
        if ss.imputed_df is not None:
            ss.working_df = ss.imputed_df.copy()
        # Reset append tracking for this new iteration
        ss.append_history = []
        ss.append_highlight = None
        ss._last_appended_file_id = None
        ss.pre_append_snapshot = ss.working_df.copy() if ss.working_df is not None else None
        ss.flow_state = "append_phase"
        st.rerun()

    if finish:
        ss.flow_state = "finished"
        return


def append_panel():
    _section("Append New Tuples", badge="INCREMENTAL")

    c_csv, c_manual = st.columns(2)

    with c_csv:
        st.markdown('<div class="SENTI-card-label">Upload CSV</div>', unsafe_allow_html=True)
        if ss.pre_append_snapshot is None:
            ss.pre_append_snapshot = ss.working_df.copy() if ss.working_df is not None else None
        if ss.append_history is None:
            ss.append_history = []
        up = st.file_uploader("Append via CSV", type=["csv","tsv","json","jsonl","xlsx","xls","parquet"], key=f"csv_append_{ss.iter_k}", label_visibility="collapsed")
        if up is not None:
            # Guard: only append once per unique file (name + size) per iteration
            _file_id = f"{ss.iter_k}::{up.name}::{up.size}"
            if ss.get("_last_appended_file_id") != _file_id:
                ss._last_appended_file_id = _file_id
                df_new = _read_uploaded_file(up)
                df_new = df_new.reindex(columns=ss.working_df.columns, fill_value=np.nan)
                combined = pd.concat([ss.working_df, df_new], ignore_index=True)
                n_existing = len(ss.working_df)
                mark = pd.Series([False] * n_existing + [True] * len(df_new), name="_appended")
                ss.append_history.append(df_new.copy())
                ss.working_df = combined
                ss.append_highlight = mark
                st.success(f"Appended {len(df_new)} tuples from CSV.")

    with c_manual:
        st.markdown('<div class="SENTI-card-label">Manual Entry</div>', unsafe_allow_html=True)
        if ss.pre_append_snapshot is None:
            ss.pre_append_snapshot = ss.working_df.copy() if ss.working_df is not None else None
        if ss.append_history is None:
            ss.append_history = []
        template = pd.DataFrame(columns=list(ss.working_df.columns))
        manual_df = st.data_editor(template, num_rows="dynamic", key=f"man_edit_{ss.iter_k}")
        if st.button("＋  Append Rows", key=f"btn_append_manual_{ss.iter_k}"):
            if manual_df is not None and not manual_df.empty:
                nonempty = manual_df.dropna(how="all")
                def _all_empty(row):
                    return all((str(v).strip() == "" or pd.isna(v)) for v in row.values)
                nonempty = nonempty[~nonempty.apply(_all_empty, axis=1)]
                if not nonempty.empty:
                    nonempty = nonempty.reindex(columns=ss.working_df.columns, fill_value=np.nan)
                    # Append new rows AFTER the existing imputed rows
                    combined = pd.concat([ss.working_df, nonempty], ignore_index=True)
                    n_existing = len(ss.working_df)
                    mark = pd.Series([False] * n_existing + [True] * len(nonempty), name="_appended")
                    ss.append_history.append(nonempty.copy())
                    ss.working_df = combined
                    ss.append_highlight = mark
                    st.success(f"Appended {len(nonempty)} manual tuples.")
                else:
                    st.info("No non-empty rows to append.")

    if ss.working_df is not None and ss.append_highlight is not None:
        render_appended_preview(ss.working_df, ss.iter_k, appended_tuples=ss.append_highlight)

    if ss.append_history:
        if st.button("↩  Undo Last Append", key=f"btn_undo_last_append_{ss.iter_k}_{len(ss.append_history)}"):
            last = ss.append_history.pop()
            n = len(last)
            if n > 0 and ss.working_df is not None:
                # Rows are at the END — remove from tail
                ss.working_df = ss.working_df.iloc[:-n].reset_index(drop=True)
                total_new = sum(len(x) for x in ss.append_history)
                if total_new > 0:
                    n_old = len(ss.working_df) - total_new
                    mark = pd.Series([False] * n_old + [True] * total_new, name="_appended")
                    ss.append_highlight = mark
                else:
                    ss.append_highlight = None
                    if ss.pre_append_snapshot is not None:
                        ss.working_df = ss.pre_append_snapshot.copy()
                        ss.pre_append_snapshot = None
                st.success("Last append undone.")
            st.rerun()

    if st.button("▶  Run Imputation", type="primary", key="btn_run_impute_Incremental"):
        n_new = int(sum(len(x) for x in (ss.append_history or [])))
        prev_imputed = ss.pre_append_snapshot  # already-imputed rows from all previous batches

        if n_new > 0 and prev_imputed is not None:
            # Impute ONLY the new rows at the end, not the previously imputed ones
            new_only_df = ss.working_df.iloc[-n_new:].copy().reset_index(drop=True)
            ss.source_snapshot = ss.working_df.copy()
            imputed_new, mask_new = run_imputation(new_only_df, context_df=prev_imputed)
            if imputed_new is not None:
                # Combine: all previous imputed rows + newly imputed new rows
                combined_imputed = pd.concat([prev_imputed, imputed_new], ignore_index=True)
                mask_old = pd.DataFrame(False, index=range(len(prev_imputed)), columns=prev_imputed.columns)
                combined_mask = pd.concat([mask_old, mask_new], ignore_index=True)
                imputed, mask = combined_imputed, combined_mask
            else:
                imputed, mask = None, None
        else:
            # First iteration: impute entire working_df (no prior context)
            ss.source_snapshot = ss.working_df.copy()
            imputed, mask = run_imputation(ss.working_df, context_df=None)

        if imputed is not None:
            ss.iter_k = (ss.iter_k or 0) + 1
            ss.last_imputed_iter = ss.iter_k
            ss.imputed_df = imputed
            ss.imputed_mask = mask
            ss.flow_state = "post_impute_prompt"
            # Count missing ONLY in the newly appended rows
            if n_new > 0:
                new_batch_df = ss.source_snapshot.iloc[-n_new:]
                nm_batch    = int(new_batch_df.isna().sum().sum())
                n_imp_batch = int(mask.iloc[-n_new:].sum().sum())
            else:
                nm_batch    = int(ss.source_snapshot.isna().sum().sum())
                n_imp_batch = int(mask.sum().sum())
            if not isinstance(ss.get("inc_log"), list):
                ss.inc_log = []
            ss.inc_log.append({
                "batch":      ss.iter_k,
                "new_rows":   n_new if n_new > 0 else len(ss.source_snapshot),
                "missing_in": nm_batch,
                "imputed":    n_imp_batch,
                "method":     ss.strategy,
            })
            try:
                ss.eval_missing_mask_k = ss.source_snapshot.isna()
            except Exception:
                ss.eval_missing_mask_k = ss.working_df.isna() if getattr(ss, 'working_df', None) is not None else None
            st.rerun()



def data_loader_ui():
   # _section("Dataset", badge="INPUT")

    left, right = st.columns([3, 1])
    with left:
        prev_demo = bool(ss.get("demo_toggle", False))
        ss.demo_toggle = st.toggle("Use demo dataset", value=prev_demo, key="toggle_demo_dataset")

        if ss.demo_toggle:
            _demo_opts = ["Demographic", "Health"]
            _demo_default = ss.get("demo_kind", "Demographic")
            _demo_idx = _demo_opts.index(_demo_default) if _demo_default in _demo_opts else 0
            ss.demo_kind = st.radio(
                "Demo category",
                options=_demo_opts,
                index=_demo_idx,
                horizontal=True,
                key="radio_demo_kind"
            )
            last_demo = ss.get("_last_demo_choice")
            if (last_demo != ss.demo_kind) or (ss.get("raw_df") is None):
                try:
                    df = load_demo_df(ss.demo_kind)
                    _bootstrap_df(df)
                    ss.dataset_label = "demo"
                    ss._last_demo_choice = ss.demo_kind
                    ss._last_upload_name = None
                    ss.null_injection_decision = None
                    ss.dataset_after_null_injection = df
                except Exception as e:
                    st.error(f"Failed to load demo dataset: {e}")
        else:
            if ss.get("_last_demo_choice") is not None and ss.get("raw_df") is not None:
                ss.raw_df = None
                ss.working_df = None
                ss.imputed_df = None
                ss.imputed_mask = None
                ss.append_highlight = None
                ss.flow_state = "idle"
                ss.iter_k = 1
                ss.last_imputed_iter = 0
                ss.source_snapshot = None
                ss.pre_append_snapshot = None
                ss.append_history = []
                ss.selected_cols = []
                ss._last_demo_choice = None
                ss.inc_log = []

            st.markdown('<div class="SENTI-card-label">Upload Incomplete CSV</div>', unsafe_allow_html=True)
            uploaded = st.file_uploader("Upload CSV", type=["csv","tsv","json","jsonl","xlsx","xls","parquet"], key="u_csv", label_visibility="collapsed")
            if uploaded is None and ss.get("raw_df") is None:
                _empty_state("📂", "No dataset loaded",
                    "Upload a CSV with missing values, or toggle the demo dataset above.")
            if uploaded is not None:
                last_name = ss.get("_last_upload_name")
                if (last_name != getattr(uploaded, "name", None)) or ss.get("raw_df") is None:
                    try:
                        df = _read_uploaded_file(uploaded)
                        _bootstrap_df(df)
                        ss.dataset_label = "uploaded"
                        ss._last_upload_name = getattr(uploaded, "name", None)
                        ss.null_injection_decision = None
                        ss.dataset_after_null_injection = df
                    except Exception as e:
                        st.error(f"Failed to read CSV: {e}")


def page_null_injection():
    import random as _random

    st.markdown("""
    <div class="SENTI-card">
        <h4>Controlled Missingness Injection</h4>
        <ul style="margin:0.4rem 0 0 1rem; line-height:1.9">
            <li><strong>SENTI built-in</strong> — fast, column-by-column injection with MCAR, MAR and MNAR
            mechanisms using a simple, interactive condition builder.</li>
            <li><strong>pyampute</strong> — a rigorous multivariate amputation library
            (<a href="https://github.com/RianneSchouten/pyampute" target="_blank">github.com/RianneSchouten/pyampute</a>)
            that lets the user define multiple simultaneous missing-data patterns, each with its own
            mechanism, frequency, and weighted-score function.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    # ── Engine selector ────────────────────────────────────────────
    engine = st.radio(
        "Injection engine",
        ["SENTI built-in", "pyampute"],
        horizontal=True,
        key="null_engine_select",
        label_visibility="collapsed",
    )

    st.markdown("---")

    if engine.startswith("pyampute"):
        _page_null_pyampute()
    else:
        _page_null_builtin()


# ─────────────────────────────────────────────────────────────────────────────
# SENTI built-in injection (original logic, extracted into helper)
# ─────────────────────────────────────────────────────────────────────────────
def _page_null_builtin():
    import random as _random

  #  st.markdown("""
   # <div class="SENTI-card" style="padding:10px 16px;">
     #   <strong style="color:var(--accent2);">MCAR</strong> — Missing Completely At Random: nulls injected uniformly at random, independent of any data values.<br>
     #   <strong style="color:var(--gold);">MAR</strong> — Missing At Random: missingness depends on <em>another observed column</em> via a threshold condition.<br>
      #  <strong style="color:var(--green);">MNAR</strong> — Missing Not At Random: missingness depends on the <em>column's own value</em> (highest or lowest values are made missing).
   # </div>
  #  """, unsafe_allow_html=True)

    uploaded = st.file_uploader("Upload complete CSV", type=["csv","tsv","json","jsonl","xlsx","xls","parquet"], key="null_injection_uploader")
    if uploaded is None:
        st.markdown('<p style="color:var(--text3); font-size:1.306rem;">No file selected yet.</p>', unsafe_allow_html=True)
        return

    try:
        df = _read_uploaded_file(uploaded)
    except Exception as e:
        st.error(f"Failed to read file: {e}")
        return

    _section("Original Dataset")
    st.markdown(unsafe_allow_html=True, body=_df_stats_html(df))
    st.dataframe(df, use_container_width=True)
    st.session_state["null_injection_original_df"] = df

    n_rows = len(df)

    # ── Mechanism ─────────────────────────────────────────────────
    _section("Missingness Mechanism", badge="CONFIG")
    null_mechanism = st.radio(
        "Null Mechanism",
        ["MCAR", "MAR", "MNAR"],
        key="cor_null_mech", label_visibility="collapsed", horizontal=True,
    )
    mech = null_mechanism.split(" — ")[0]

    # ── Disguised missing values ──────────────────────────────────
    st.markdown('<div class="SENTI-card-label" style="margin-top:0.8rem;">Disguised Missing Value</div>', unsafe_allow_html=True)
    disguise_opts = ["None (true NaN)", "N/A", "?", "-", "999", "0", "unknown", "Custom…"]
    disguise_sel = st.selectbox("Disguised missing sentinel", disguise_opts, key="cor_disguise", label_visibility="collapsed")
    if disguise_sel == "Custom…":
        disguise_val = st.text_input("Custom sentinel value", value="MISSING", key="cor_disguise_custom", label_visibility="collapsed")
    elif disguise_sel == "None (true NaN)":
        disguise_val = None
    else:
        disguise_val = disguise_sel
    if disguise_val is not None:
        st.warning(f'Disguised mode: instead of NaN, cells will be set to **"{disguise_val}"** — making them invisible to naive null-checks.')

    # ── Target columns ────────────────────────────────────────────
    st.markdown('<div class="SENTI-card-label" style="margin-top:0.8rem;">Target Columns</div>', unsafe_allow_html=True)
    null_cols = st.multiselect(
        "Columns to inject nulls into",
        df.columns.tolist(), key="cor_null",
        label_visibility="collapsed",
        placeholder="Select columns to inject nulls into…",
        default=list(df.columns),
    )

    # ── Mechanism-specific controls ───────────────────────────────
    mar_condition_col = None
    mar_condition_op  = ">"
    mar_condition_val = 0.0
    mnar_direction    = "Highest values"
    is_multivariate   = False

    if mech == "MAR":
        st.markdown('<div class="SENTI-card-label" style="margin-top:0.8rem;">MAR Conditions</div>', unsafe_allow_html=True)
        st.caption(
            "Define one or more conditions. Rows matching the combined condition will be the pool from which nulls are drawn. "
            "Use **AND** to narrow the pool (rows must satisfy ALL conditions), or **OR** to widen it (rows satisfying ANY condition)."
        )
        if "mar_conditions" not in st.session_state:
            st.session_state["mar_conditions"] = [{"col": df.columns[0], "op": ">", "val": None}]

        mar_combiner = "AND"
        if len(st.session_state["mar_conditions"]) > 1:
            _cb_col, _ = st.columns([1, 3])
            with _cb_col:
                mar_combiner = st.radio(
                    "Combine conditions with",
                    ["AND", "OR"], horizontal=True,
                    key="mar_combiner", label_visibility="collapsed",
                )

        def _render_condition(idx, cond):
            col_col, op_col, val_col, del_col = st.columns([2, 2, 3, 0.5])
            with col_col:
                chosen_col = st.selectbox(
                    "Column", df.columns.tolist(),
                    index=df.columns.tolist().index(cond["col"]) if cond["col"] in df.columns else 0,
                    key=f"mar_col_{idx}", label_visibility="collapsed"
                )
            cond_series_i = df[chosen_col].dropna()
            is_num_i = pd.api.types.is_numeric_dtype(df[chosen_col])
            with op_col:
                if is_num_i:
                    _ops = ["< (less than)", "<= (less than or equal)", "> (greater than)", ">= (greater than or equal)", "== (equal to)"]
                    _op_map_i = {"< (less than)": "<", "<= (less than or equal)": "<=", "> (greater than)": ">", ">= (greater than or equal)": ">=", "== (equal to)": "=="}
                    _rev = {v: k for k, v in _op_map_i.items()}
                    _cur_label = _rev.get(cond["op"], "> (greater than)")
                    op_label = st.selectbox("Operator", _ops, index=_ops.index(_cur_label) if _cur_label in _ops else 2, key=f"mar_op_{idx}", label_visibility="collapsed")
                    chosen_op = _op_map_i[op_label]
                else:
                    _ops_cat = ["== (is equal to)", "!= (is not equal to)"]
                    _op_map_cat_i = {"== (is equal to)": "==", "!= (is not equal to)": "!="}
                    _rev_cat = {v: k for k, v in _op_map_cat_i.items()}
                    _cur_label_cat = _rev_cat.get(cond["op"], "== (is equal to)")
                    op_label_cat = st.selectbox("Operator", _ops_cat, index=_ops_cat.index(_cur_label_cat) if _cur_label_cat in _ops_cat else 0, key=f"mar_op_{idx}", label_visibility="collapsed")
                    chosen_op = _op_map_cat_i[op_label_cat]
            with val_col:
                if is_num_i:
                    col_min_i = float(cond_series_i.min())
                    col_max_i = float(cond_series_i.max())
                    col_med_i = float(cond_series_i.median())
                    _step_i = (col_max_i - col_min_i) / 100 if (col_max_i - col_min_i) > 0 else 1.0
                    _is_int_i = pd.api.types.is_integer_dtype(df[chosen_col])
                    _default_i = cond["val"] if cond["val"] is not None else col_med_i
                    if _is_int_i:
                        chosen_val = st.slider("Value", int(col_min_i), int(col_max_i), int(_default_i) if _default_i is not None else int(col_med_i), key=f"mar_val_{idx}", label_visibility="collapsed")
                    else:
                        chosen_val = st.slider("Value", col_min_i, col_max_i, float(_default_i) if _default_i is not None else col_med_i, step=float(round(_step_i, 4)), key=f"mar_val_{idx}", label_visibility="collapsed")
                    st.caption(f"range {col_min_i:.4g} – {col_max_i:.4g}")
                else:
                    cat_vals_i = sorted(cond_series_i.unique().tolist(), key=str)
                    _cur_val = cond["val"] if cond["val"] in cat_vals_i else cat_vals_i[0]
                    chosen_val = st.selectbox("Value", cat_vals_i, index=cat_vals_i.index(_cur_val), key=f"mar_val_{idx}", label_visibility="collapsed")
                    st.caption(f"{len(cat_vals_i)} unique values")
            with del_col:
                st.markdown("<div style='margin-top:0.3rem'></div>", unsafe_allow_html=True)
                remove = st.button("✕", key=f"mar_del_{idx}", help="Remove this condition")
            return {"col": chosen_col, "op": chosen_op, "val": chosen_val}, remove

        updated_conditions = []
        remove_idx = None
        for i, cond in enumerate(st.session_state["mar_conditions"]):
            if i > 0:
                st.markdown(f'<div style="text-align:center;font-size:1.106rem;font-weight:700;color:var(--accent2);margin:2px 0;">{mar_combiner}</div>', unsafe_allow_html=True)
            new_cond, do_remove = _render_condition(i, cond)
            updated_conditions.append(new_cond)
            if do_remove:
                remove_idx = i
        st.session_state["mar_conditions"] = updated_conditions
        if remove_idx is not None and len(st.session_state["mar_conditions"]) > 1:
            st.session_state["mar_conditions"].pop(remove_idx)
            st.rerun()
        if st.button("＋ Add condition", key="mar_add_cond"):
            st.session_state["mar_conditions"].append({"col": df.columns[0], "op": ">", "val": None})
            st.rerun()

        mar_conditions_final = st.session_state["mar_conditions"]
        def _eval_condition(cond_dict):
            cs = df[cond_dict["col"]]
            v  = cond_dict["val"]
            if pd.api.types.is_numeric_dtype(cs):
                try: v = float(v)
                except Exception: pass
            op = cond_dict["op"]
            if op == ">":   return cs > v
            elif op == ">=": return cs >= v
            elif op == "<":  return cs < v
            elif op == "<=": return cs <= v
            elif op == "!=": return cs != v
            else:            return cs == v
        try:
            masks = [_eval_condition(c) for c in mar_conditions_final if c["val"] is not None]
            if masks:
                if mar_combiner == "AND":
                    combined = masks[0]
                    for m in masks[1:]: combined = combined & m
                else:
                    combined = masks[0]
                    for m in masks[1:]: combined = combined | m
                n_elig = int(combined.sum())
                pct_elig = 100 * n_elig / max(n_rows, 1)
                color = "var(--green)" if n_elig > 0 else "var(--red)"
                cond_strs = [f"{c['col']} {c['op']} {c['val']}" for c in mar_conditions_final if c["val"] is not None]
                cond_label = f" {mar_combiner} ".join(cond_strs)
                st.markdown(
                    f'<div class="SENTI-card" style="padding:8px 14px;margin-top:0.5rem">'
                    f'Combined condition <strong>{cond_label}</strong> '
                    f'matches <strong style="color:{color}">{n_elig} rows ({pct_elig:.1f}%)</strong>'
                    f' — nulls will be drawn from these rows only.</div>',
                    unsafe_allow_html=True
                )
        except Exception:
            pass

    elif mech == "MNAR":
        st.markdown('<div class="SENTI-card-label" style="margin-top:0.8rem;">MNAR Direction</div>', unsafe_allow_html=True)
        mnar_direction = st.radio("MNAR direction", ["Highest values", "Lowest values"], key="mnar_dir", label_visibility="collapsed", horizontal=True)
        st.info(f"MNAR: the **{'highest' if mnar_direction == 'Highest values' else 'lowest'}** values in each target column will be replaced with nulls.")

    # ── Null fraction + seed ──────────────────────────────────────
    _section("Injection Parameters", badge="CONFIG")
    s1, s2 = st.columns(2)
    with s1:
        st.markdown('<div class="SENTI-card-label">Null %</div>', unsafe_allow_html=True)
        null_pct = st.slider("NULL", 0.0, 100.0, 10.0, 0.5, key="cor_s1", label_visibility="collapsed")
    with s2:
        st.markdown('<div class="SENTI-card-label">Random Seed</div>', unsafe_allow_html=True)
        seed = st.number_input("Seed", min_value=0, max_value=1_000_000, value=42, step=1, key="null_injection_seed", label_visibility="collapsed")

    result_key = "null_injection_result_df"
    ba, bb, _ = st.columns([1.5, 1, 5])
    with ba:
        inject_btn = st.button("▶  Apply Null Injection", type="primary", key="btn_apply_null_injection_page", use_container_width=True)
    with bb:
        if st.button("↺ Reset", key="cor_reset_null", use_container_width=True):
            for k in [result_key, "null_injection_original_df"]:
                st.session_state.pop(k, None)
            st.rerun()

    if inject_btn:
        if not null_cols:
            st.warning("Select at least one column to inject nulls into.")
        else:
            _random.seed(int(seed))
            wdf = df.copy()
            log = []

            def apply_null(df_ref, r, col):
                if disguise_val is None:
                    df_ref.at[r, col] = np.nan
                else:
                    try:
                        df_ref.at[r, col] = type(df_ref.at[r, col])(disguise_val)
                    except Exception:
                        df_ref.at[r, col] = disguise_val

            if mech == "MCAR":
                for col in null_cols:
                    target = max(1, int(n_rows * null_pct / 100))
                    avail = list(range(n_rows))
                    for r in _random.sample(avail, min(target, len(avail))):
                        apply_null(wdf, r, col)
                        log.append((r, col, "NULL-MCAR"))
            elif mech == "MAR":
                _mar_conds = st.session_state.get("mar_conditions", [])
                _mar_comb  = st.session_state.get("mar_combiner", "AND")
                try:
                    def _eval_exec(cond_dict):
                        cs2 = df[cond_dict["col"]]
                        _v  = cond_dict["val"]
                        if pd.api.types.is_numeric_dtype(cs2):
                            try: _v = float(_v)
                            except Exception: pass
                        op = cond_dict["op"]
                        if op == ">":    return cs2 > _v
                        elif op == ">=": return cs2 >= _v
                        elif op == "<":  return cs2 < _v
                        elif op == "<=": return cs2 <= _v
                        elif op == "!=": return cs2 != _v
                        else:            return cs2 == _v
                    masks2 = [_eval_exec(c) for c in _mar_conds if c.get("val") is not None]
                    if masks2:
                        if _mar_comb == "AND":
                            combined2 = masks2[0]
                            for _m in masks2[1:]: combined2 = combined2 & _m
                        else:
                            combined2 = masks2[0]
                            for _m in masks2[1:]: combined2 = combined2 | _m
                        elig_rows = [r for r in combined2[combined2].index.tolist() if 0 <= r < n_rows]
                    else:
                        elig_rows = list(range(n_rows))
                except Exception:
                    elig_rows = list(range(n_rows))
                for col in null_cols:
                    target = max(1, int(len(elig_rows) * null_pct / 100))
                    avail = list(elig_rows)
                    for r in _random.sample(avail, min(target, len(avail))):
                        apply_null(wdf, r, col)
                        log.append((r, col, "NULL-MAR"))
            elif mech == "MNAR":
                for col in null_cols:
                    target = max(1, int(n_rows * null_pct / 100))
                    col_series = df[col].dropna()
                    if len(col_series) == 0:
                        continue
                    if pd.api.types.is_numeric_dtype(df[col]):
                        sorted_idx = col_series.sort_values(ascending=(mnar_direction == "Lowest values")).index.tolist()
                    else:
                        sorted_idx = col_series.astype(str).sort_values(ascending=(mnar_direction == "Lowest values")).index.tolist()
                    for r in sorted_idx[:min(target, len(sorted_idx))]:
                        apply_null(wdf, r, col)
                        log.append((r, col, "NULL-MNAR"))

            st.session_state[result_key] = wdf
            st.success(f"Injection complete — **{len(log)}** cells modified via **{mech}** mechanism.")

    df_result = st.session_state.get(result_key)
    if df_result is not None:
        _section("Incomplete Dataset", badge="RESULT")
        st.markdown(unsafe_allow_html=True, body=_df_stats_html(df_result))
        st.dataframe(df_result, use_container_width=True)
        try:
            csv_bytes = df_result.to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇  Download Incomplete CSV",
                data=csv_bytes,
                file_name="incomplete_dataset.csv",
                mime="text/csv",
                key="btn_download_incomplete_page",
            )
        except Exception as e:
            st.error(f"Failed to create download: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# pyampute injection engine
# ─────────────────────────────────────────────────────────────────────────────
def _page_null_pyampute():
    """
    Null injection using the pyampute library (MultivariateAmputation).

    How it works
    ────────────
    pyampute models missingness as a set of *patterns*.  Each pattern says:
      • which columns become missing (incomplete_vars)
      • what fraction of rows follow this pattern (freq)
      • what mechanism drives missingness: MCAR | MAR | MNAR
      • for MAR/MNAR: which observed columns influence the probability of
        a row being selected (weights)
      • the shape of the probability function (sigmoid-right / sigmoid-left /
        sigmoid-mid / sigmoid-tail)

    The user builds one or more patterns in the UI.  Under the hood SENTI calls:

        from pyampute.ampute import MultivariateAmputation
        ma = MultivariateAmputation(prop=prop, patterns=patterns, seed=seed)
        X_incomplete = ma.fit_transform(X_numeric)

    pyampute only accepts numeric arrays, so SENTI:
      1. Label-encodes any categorical columns before amputation.
      2. Runs MultivariateAmputation on the numeric matrix.
      3. Copies the NaN mask back onto the original (mixed-type) DataFrame.

    This means the structural missingness pattern (which cells are NaN) is
    fully controlled by pyampute, while the original dtypes/string values
    are preserved in the output CSV.
    """

    # ── Install check ────────────────────────────────────────────────
    try:
        from pyampute.ampute import MultivariateAmputation as _MA
        _pyampute_ok = True
    except ImportError:
        _pyampute_ok = False

    if not _pyampute_ok:
        st.error(
            "**pyampute is not installed** in this environment.  "
            "Ask your system admin to run:  `pip install pyampute`  "
            "or add it to `requirements.txt`."
        )
        st.code("pip install pyampute", language="bash")
        return

    # ── File upload ──────────────────────────────────────────────────
    uploaded = st.file_uploader(
        "Upload complete CSV",
        type=["csv", "tsv", "xlsx", "xls"],
        key="pyampute_uploader",
    )
    if uploaded is None:
        st.markdown('<p style="color:var(--text3); font-size:1.306rem;">No file selected yet.</p>', unsafe_allow_html=True)
        return

    try:
        df = _read_uploaded_file(uploaded)
    except Exception as e:
        st.error(f"Failed to read file: {e}")
        return

    _section("Original Dataset")
    st.markdown(unsafe_allow_html=True, body=_df_stats_html(df))
    st.dataframe(df, use_container_width=True)

    all_cols = df.columns.tolist()
    n_cols   = len(all_cols)

    # ── Injection mode explanation ───────────────────────────────────
    st.markdown("""
    <div class="SENTI-card" style="padding:10px 16px; margin-bottom:0.6rem; border-left:4px solid var(--gold);">
    <strong>⚠️ How pyampute works (important)</strong><br>
    pyampute operates <strong>row-by-row</strong>: it selects a fraction of rows (<em>prop</em>)
    and makes <strong>all columns listed in a pattern missing for those same rows</strong>.
    This is correct for simulation studies but can look like "entire rows wiped out".<br><br>
    To spread missingness across <em>different rows per column</em>, use
    <strong>Per-column mode</strong> below — SENTI will run pyampute once per column independently,
    so each column loses values in a different random subset of rows.
    </div>
    """, unsafe_allow_html=True)

    # ── Injection mode ───────────────────────────────────────────────
    pa_mode = st.radio(
        "Injection mode",
        ["🔲  Multivariate (one run, row-level patterns)",
         "🔳  Per-column (independent run per column)"],
        horizontal=True,
        key="pa_mode_radio",
        label_visibility="collapsed",
    )
    is_per_column = pa_mode.startswith("🔳")

    # ── Global proportion ────────────────────────────────────────────
    _section("Global Settings", badge="CONFIG")
    gc1, gc2 = st.columns(2)
    with gc1:
        if is_per_column:
            st.markdown('<div class="SENTI-card-label">Fraction of rows to null <em>per column</em> (prop)</div>', unsafe_allow_html=True)
            st.caption("Each column is amputed independently — prop controls what fraction of rows lose that column's value.")
        else:
            st.markdown('<div class="SENTI-card-label">Fraction of rows with ≥1 missing value (prop)</div>', unsafe_allow_html=True)
            st.caption("Fraction of all rows that will receive at least one missing value across all patterns combined.")
        prop = st.slider(
            "prop", 0.01, 0.99, 0.20, 0.01,
            key="pa_prop", label_visibility="collapsed",
        )
    with gc2:
        st.markdown('<div class="SENTI-card-label">Random seed</div>', unsafe_allow_html=True)
        pa_seed = st.number_input("seed", min_value=0, max_value=1_000_000, value=42, step=1,
                                  key="pa_seed", label_visibility="collapsed")

    if is_per_column:
        st.caption(
            f"**prop = {prop:.0%}** → roughly **{int(prop * len(df))}** of the {len(df)} rows "
            "will lose each column's value (independently per column)."
        )
    else:
        st.caption(
            f"**prop = {prop:.0%}** → roughly **{int(prop * len(df))}** of the {len(df)} rows "
            "will receive at least one missing value."
        )

    # ── Pattern builder ──────────────────────────────────────────────
    _section("Missing-Data Patterns", badge="PATTERNS")
    st.caption(
        "Add one or more patterns.  Each pattern defines *which columns* go missing, "
        "*how often* (relative frequency among all incomplete rows), and *why* (mechanism). "
        "All frequency values must sum to 1."
    )

    if "pa_patterns" not in st.session_state:
        # Start with one sensible default
        st.session_state["pa_patterns"] = [
            {"incomplete_vars": [], "freq": 1.0, "mechanism": "MCAR",
             "score_to_probability_func": "sigmoid-right", "weights": {}}
        ]

    # Add / remove pattern buttons
    padd_col, _ = st.columns([1, 5])
    with padd_col:
        if st.button("＋ Add pattern", key="pa_add_pattern"):
            n_existing = len(st.session_state["pa_patterns"])
            new_freq = round(1.0 / (n_existing + 1), 4)
            # Redistribute frequencies evenly
            for p in st.session_state["pa_patterns"]:
                p["freq"] = new_freq
            st.session_state["pa_patterns"].append(
                {"incomplete_vars": [], "freq": new_freq, "mechanism": "MCAR",
                 "score_to_probability_func": "sigmoid-right", "weights": {}}
            )
            st.rerun()

    patterns_state = st.session_state["pa_patterns"]
    remove_pat_idx = None

    MECH_COLORS = {"MCAR": "var(--accent2)", "MAR": "var(--gold)", "MNAR": "var(--green)"}

    for pi, pat in enumerate(patterns_state):
        mech_color = MECH_COLORS.get(pat.get("mechanism", "MCAR"), "var(--accent2)")
        st.markdown(
            f'<div style="border-left:4px solid {mech_color}; padding:4px 12px; '
            f'margin:0.6rem 0 0.2rem 0; font-weight:700; font-size:1.382rem;">'
            f'Pattern {pi + 1}</div>',
            unsafe_allow_html=True
        )

        p_c1, p_c2, p_c3, p_c4, p_del = st.columns([2.5, 1.2, 1.5, 1.8, 0.5])

        with p_c1:
            st.markdown('<div class="SENTI-card-label">Columns to make missing</div>', unsafe_allow_html=True)
            chosen_vars = st.multiselect(
                f"incomplete_vars_{pi}", all_cols,
                default=[c for c in pat.get("incomplete_vars", []) if c in all_cols],
                key=f"pa_ivars_{pi}", label_visibility="collapsed",
                placeholder="Pick columns…"
            )
            pat["incomplete_vars"] = chosen_vars

        with p_c2:
            st.markdown('<div class="SENTI-card-label">Frequency</div>', unsafe_allow_html=True)
            if is_per_column:
                st.caption("N/A in per-column mode")
                freq_val = 1.0
            else:
                freq_val = st.number_input(
                    f"freq_{pi}", min_value=0.0, max_value=1.0,
                    value=float(pat.get("freq", 1.0)), step=0.05,
                    key=f"pa_freq_{pi}", label_visibility="collapsed",
                    help="Relative fraction of incomplete rows following this pattern. All patterns must sum to 1."
                )
            pat["freq"] = round(freq_val, 4)

        with p_c3:
            st.markdown('<div class="SENTI-card-label">Mechanism</div>', unsafe_allow_html=True)
            mech_options = ["MCAR", "MAR", "MNAR"]
            mech_idx = mech_options.index(pat.get("mechanism", "MCAR"))
            chosen_mech = st.selectbox(
                f"mechanism_{pi}", mech_options, index=mech_idx,
                key=f"pa_mech_{pi}", label_visibility="collapsed"
            )
            pat["mechanism"] = chosen_mech

        with p_c4:
            st.markdown('<div class="SENTI-card-label">Probability shape</div>', unsafe_allow_html=True)
            func_options = ["sigmoid-right", "sigmoid-left", "sigmoid-mid", "sigmoid-tail"]
            func_help = {
                "sigmoid-right": "Rows with HIGH weighted scores → more likely to be missing",
                "sigmoid-left":  "Rows with LOW weighted scores → more likely to be missing",
                "sigmoid-mid":   "Rows with AVERAGE weighted scores → more likely to be missing",
                "sigmoid-tail":  "Rows with EXTREME weighted scores → more likely to be missing",
            }
            cur_func = pat.get("score_to_probability_func", "sigmoid-right")
            if cur_func not in func_options:
                cur_func = "sigmoid-right"
            chosen_func = st.selectbox(
                f"func_{pi}", func_options,
                index=func_options.index(cur_func),
                key=f"pa_func_{pi}", label_visibility="collapsed",
                help="\n".join(f"{k}: {v}" for k, v in func_help.items()),

            )
            pat["score_to_probability_func"] = chosen_func

        with p_del:
            st.markdown("<div style='margin-top:1.4rem'></div>", unsafe_allow_html=True)
            if st.button("✕", key=f"pa_del_{pi}", help="Remove this pattern"):
                remove_pat_idx = pi

        # Weights (MAR / MNAR only)
        if chosen_mech in ("MAR", "MNAR"):
            with st.expander(f"⚖️  Weights for Pattern {pi + 1} ({chosen_mech})", expanded=False):
                st.caption(
                    "Assign a weight to each column that **influences** the probability of a row "
                    "being selected for missingness.  "
                    "**Positive** weight → higher value = higher missingness probability.  "
                    "**Negative** weight → lower value = higher missingness probability.  "
                    "**0** → column plays no role.  "
                    "Leave all at 0 to use equal weights automatically."
                )
                weight_dict = pat.get("weights", {})
                new_weights = {}
                w_cols = st.columns(min(4, len(all_cols)))
                for wi, wcol_name in enumerate(all_cols):
                    with w_cols[wi % len(w_cols)]:
                        cur_w = float(weight_dict.get(wcol_name, 0.0))
                        new_w = st.number_input(
                            wcol_name, value=cur_w, step=0.5,
                            key=f"pa_w_{pi}_{wi}", label_visibility="visible",
                        )
                        new_weights[wcol_name] = new_w
                pat["weights"] = new_weights

    # Apply removals
    if remove_pat_idx is not None and len(patterns_state) > 1:
        patterns_state.pop(remove_pat_idx)
        st.rerun()

    # Always compute freq_sum; only show warning in multivariate mode
    freq_sum = sum(p.get("freq", 0.0) for p in patterns_state)
    if not is_per_column:
        if abs(freq_sum - 1.0) > 0.01:
            st.warning(
                f"Pattern frequencies sum to **{freq_sum:.3f}** — they must sum to **1.0**.  "
                "pyampute will raise an error otherwise."
            )
        else:
            st.success(f"✓ Frequencies sum to {freq_sum:.3f}")

    # ── Run button ───────────────────────────────────────────────────
    st.markdown("---")
    rb1, rb2, _ = st.columns([1.5, 1, 5])
    with rb1:
        run_btn = st.button("▶  Run pyampute", type="primary", key="pa_run_btn", use_container_width=True)
    with rb2:
        if st.button("↺ Reset", key="pa_reset_btn", use_container_width=True):
            for k in ["pa_result_df", "pa_patterns"]:
                st.session_state.pop(k, None)
            st.rerun()

    if run_btn:
        # Validate
        if not patterns_state:
            st.error("Add at least one pattern.")
            return
        for pi2, pat2 in enumerate(patterns_state):
            if not pat2.get("incomplete_vars"):
                st.error(f"Pattern {pi2 + 1}: select at least one column to make missing.")
                return
        if abs(freq_sum - 1.0) > 0.01:
            st.error(f"Pattern frequencies must sum to 1.0 (currently {freq_sum:.3f}).")
            return

        try:
            from pyampute.ampute import MultivariateAmputation as _MA
            from sklearn.preprocessing import LabelEncoder as _LE
            import warnings as _warnings

            # --- Encode categoricals so pyampute gets a pure numeric array ---
            df_enc = df.copy()
            for col in all_cols:
                if not pd.api.types.is_numeric_dtype(df_enc[col]):
                    le = _LE()
                    df_enc[col] = le.fit_transform(df_enc[col].astype(str))

            X = df_enc.values.astype(float)
            df_result = df.copy()

            with st.spinner("Running pyampute…"):
                with _warnings.catch_warnings():
                    _warnings.simplefilter("ignore")

                    if is_per_column:
                        # ── Per-column mode ─────────────────────────────────────
                        # Collect all unique columns referenced across all patterns
                        # and their mechanism/func settings, then run pyampute once
                        # per column independently so each column loses values in
                        # different rows.
                        col_settings = {}  # col → {mechanism, func, weights}
                        for pat3 in patterns_state:
                            mech3 = pat3.get("mechanism", "MCAR")
                            func3 = pat3.get("score_to_probability_func", "sigmoid-right")
                            wdict3 = pat3.get("weights", {})
                            for col in pat3.get("incomplete_vars", []):
                                if col in all_cols:
                                    col_settings[col] = {
                                        "mechanism": mech3,
                                        "score_to_probability_func": func3,
                                        "weights": wdict3,
                                    }

                        for ci_single, (col, settings) in enumerate(col_settings.items()):
                            col_idx = all_cols.index(col)
                            pat_single = {
                                "incomplete_vars": [col_idx],
                                "freq": 1.0,
                                "mechanism": settings["mechanism"],
                                "score_to_probability_func": settings["score_to_probability_func"],
                            }
                            wdict = settings.get("weights", {})
                            w_arr = [float(wdict.get(c, 0.0)) for c in all_cols]
                            if any(w != 0.0 for w in w_arr):
                                pat_single["weights"] = w_arr

                            ma = _MA(
                                prop=float(prop),
                                patterns=[pat_single],
                                seed=int(pa_seed) + ci_single,  # different seed per col
                            )
                            X_col = ma.fit_transform(X.copy())
                            col_mask = np.isnan(X_col[:, col_idx])
                            df_result.loc[col_mask, col] = np.nan

                    else:
                        # ── Multivariate mode (original) ────────────────────────
                        pa_patterns_built = []
                        for pat3 in patterns_state:
                            ivars_idx = [all_cols.index(c) for c in pat3["incomplete_vars"] if c in all_cols]
                            entry = {
                                "incomplete_vars": ivars_idx,
                                "freq": float(pat3["freq"]),
                                "mechanism": pat3["mechanism"],
                                "score_to_probability_func": pat3["score_to_probability_func"],
                            }
                            wdict2 = pat3.get("weights", {})
                            w_arr = [float(wdict2.get(c, 0.0)) for c in all_cols]
                            if any(w != 0.0 for w in w_arr):
                                entry["weights"] = w_arr
                            pa_patterns_built.append(entry)

                        ma = _MA(prop=float(prop), patterns=pa_patterns_built, seed=int(pa_seed))
                        X_incomplete = ma.fit_transform(X)
                        nan_mask = np.isnan(X_incomplete)
                        for ci, col in enumerate(all_cols):
                            df_result.loc[nan_mask[:, ci], col] = np.nan

            # Stats
            final_nan = df_result.isna()
            n_missing_cells = int(final_nan.sum().sum())
            n_missing_rows  = int(final_nan.any(axis=1).sum())
            pct_rows = 100 * n_missing_rows / max(len(df), 1)

            st.session_state["pa_result_df"] = df_result
            mode_label = "per-column" if is_per_column else "multivariate"
            st.success(
                f"✓ pyampute ({mode_label}) complete — **{n_missing_rows} rows** ({pct_rows:.1f}%) "
                f"received missing values, **{n_missing_cells} cells** total."
            )

        except Exception as e:
            st.error(f"pyampute failed: {e}")
            st.exception(e)

    df_result = st.session_state.get("pa_result_df")
    if df_result is not None:
        _section("Incomplete Dataset (pyampute)", badge="RESULT")
        st.markdown(unsafe_allow_html=True, body=_df_stats_html(df_result))
        st.dataframe(df_result, use_container_width=True)
        try:
            csv_bytes = df_result.to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇  Download Incomplete CSV",
                data=csv_bytes,
                file_name="incomplete_pyampute.csv",
                mime="text/csv",
                key="pa_download_btn",
            )
        except Exception as e:
            st.error(f"Download failed: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# Custom GitHub imputer panel
# ══════════════════════════════════════════════════════════════════════════════

def _custom_imputer_panel():
    """
    Full UI for the Custom (GitHub) imputer.
    State keys used:
      ss.custom_repo_url       – URL typed by user
      ss.custom_repo_path      – str path to cloned repo on disk
      ss.custom_repo_info      – dict from inspect_repo()
      ss.custom_adapter_code   – str, the editable adapter code
      ss.custom_deps_installed – bool
      ss.custom_ready          – bool, True once adapter is confirmed runnable
    """
    from modules.custom_imputer import (
        clone_repo, inspect_repo, make_template, install_deps,
    )
    from pathlib import Path

    # Shared CSS — all self-contained, no split open/close divs
    st.markdown("""
    <style>
    .cst-badge {
        display: inline-block;
        background: var(--accent2, #60A5FA);
        color: #fff !important;
        font-size: 0.84rem;
        font-weight: 700;
        padding: 3px 10px;
        border-radius: 20px;
        letter-spacing: 0.05em;
        margin-bottom: 4px;
    }
    .cst-warn {
        font-size: 0.984rem;
        color: #F59E0B !important;
        background: rgba(245,158,11,0.10);
        border-left: 3px solid #F59E0B;
        padding: 8px 14px;
        border-radius: 0 6px 6px 0;
        margin: 6px 0;
        line-height: 1.65;
    }
    .cst-ok {
        font-size: 0.984rem;
        color: #36D399 !important;
        background: rgba(52,211,153,0.10);
        border-left: 3px solid #36D399;
        padding: 8px 14px;
        border-radius: 0 6px 6px 0;
        margin: 6px 0;
        line-height: 1.65;
    }
    .cst-tag {
        display: inline-block;
        font-size: 0.864rem;
        font-weight: 600;
        padding: 2px 9px;
        border-radius: 10px;
        margin: 2px 3px 2px 0;
        border: 1px solid rgba(96,165,250,0.35);
        color: #93C5FD !important;
        background: rgba(96,165,250,0.08);
    }
    .cst-tag-purple {
        display: inline-block;
        font-size: 0.864rem;
        font-weight: 600;
        padding: 2px 9px;
        border-radius: 10px;
        margin: 2px 3px 2px 0;
        border: 1px solid rgba(167,139,250,0.35);
        color: #C4B5FD !important;
        background: rgba(167,139,250,0.08);
    }
    .cst-tag-gray {
        display: inline-block;
        font-size: 0.864rem;
        padding: 2px 9px;
        border-radius: 10px;
        margin: 2px 3px 2px 0;
        border: 1px solid rgba(148,163,184,0.25);
        color: var(--text3) !important;
        background: rgba(148,163,184,0.07);
    }
    .cst-label {
        font-size: 0.936rem;
        font-weight: 600;
        color: #CBD5E1 !important;
        margin: 10px 0 4px;
        letter-spacing: 0.03em;
    }
    .cst-hint {
        font-size: 0.96rem;
        color: #94A3B8 !important;
        line-height: 1.65;
        margin: 2px 0 8px;
    }
    .repo-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 10px;
        margin-top: 4px;
    }
    .repo-card {
        background: rgba(96,165,250,0.04);
        border: 1px solid rgba(96,165,250,0.15);
        border-radius: 8px;
        padding: 10px 13px;
    }
    .repo-card-name { font-size:1.26rem; font-weight:700; color:#E2E8F0 !important; margin-bottom:3px; }
    .repo-card-url  { font-size:1.106rem; color:#60A5FA !important; font-family:monospace;
                      word-break:break-all; margin-bottom:3px; }
    .repo-card-desc { font-size:1.136rem; color:var(--text3) !important; line-height:1.5; }
    </style>
    """, unsafe_allow_html=True)

    # ── Popular repos reference card ───────────────────────────────
   # with st.expander("📚  Known imputation repos you can paste directly", expanded=False):
    #    repos = [
    #        ("MissForest (Python)",     "https://github.com/epsilon-machine/missingpy",
    #         "Random-forest-based · sklearn API: MissForest().fit_transform(X)"),
    #        ("autoimpute",              "https://github.com/kearnz/autoimpute",
     #        "Multiple imputation strategies · Pandas-native API"),
     #       ("DataWig (AWS)",           "https://github.com/awslabs/datawig",
    #         "Deep-learning imputer · Supports categorical + numeric"),
    #        ("GAIN (TF)",               "https://github.com/jsyoon0823/GAIN",
     #        "GAN-based imputation · Requires TensorFlow"),
     #       ("HyperImpute",             "https://github.com/vanderschaarlab/hyperimpute",
    #         "Iterative plugin-based · sklearn-compatible"),
#            ("MIDA",                    "https://github.com/Oracen/MIDA",
 #            "Multiple Imputation via Denoising Autoencoders"),
#            ("fancyimpute",             "https://github.com/iskandr/fancyimpute",
  #           "SoftImpute · IterativeSVD · BiScaler · NuclearNormMinimization"),
  #          ("sklearn IterativeImputer","https://github.com/scikit-learn/scikit-learn",
  #           "MICE-like · Already installed · Use sklearn.impute.IterativeImputer"),
  #      ]
  #      cards_html = '<div class="repo-grid">'
#        for name, url, desc in repos:
#            cards_html += (
#                f'<div class="repo-card">'
#                f'<div class="repo-card-name">{name}</div>'
#                f'<div class="repo-card-url">{url}</div>'
#                f'<div class="repo-card-desc">{desc}</div>'
#                f'</div>'
#            )
#        cards_html += '</div>'
#        st.markdown(cards_html, unsafe_allow_html=True)
#
#    st.markdown("<br>", unsafe_allow_html=True)

    # ── Step 1 — URL input + Clone ─────────────────────────────────
    with st.container(border=True):
        st.markdown('<span class="cst-badge">STEP 1 — CLONE REPO</span>', unsafe_allow_html=True)
        st.markdown('<div class="cst-hint">Paste any GitHub repository URL. SENTI clones it locally, scans for impute-like functions, and auto-generates an adapter.</div>', unsafe_allow_html=True)

        url = st.text_input(
            "GitHub repository URL",
            value=ss.get("custom_repo_url", ""),
            placeholder="https://github.com/epsilon-machine/missingpy",
            key="custom_url_input",
        )
        ss.custom_repo_url = url

        col_clone, col_reclone, col_pad = st.columns([2, 2, 3])
        with col_clone:
            do_clone = st.button("🔽  Clone & Inspect", key="btn_custom_clone",
                                 disabled=not bool(url.strip()))
        with col_reclone:
            do_reclone = st.button("↺  Re-clone (force)", key="btn_custom_reclone",
                                   disabled=not bool(url.strip()))

    if do_clone or do_reclone:
        with st.spinner("Cloning repository …"):
            ok, msg, repo_path = clone_repo(url.strip(), force=do_reclone)
        if ok:
            ss.custom_repo_path      = str(repo_path)
            ss.custom_repo_info      = inspect_repo(repo_path)
            ss.custom_adapter_code   = make_template(ss.custom_repo_info, repo_path)
            ss.custom_deps_installed = False
            ss.custom_ready          = False
            st.success(f"✓ {msg}")
        else:
            st.error(msg)

    # ── Steps 2-4 — only shown after a successful clone ────────────
    info      = ss.get("custom_repo_info")
    repo_path = ss.get("custom_repo_path")

    if info and repo_path:

        # ── Step 2 — Repo summary ──────────────────────────────────
        with st.container(border=True):
            st.markdown('<span class="cst-badge">STEP 2 — REPO SUMMARY</span>', unsafe_allow_html=True)

            c1, c2, c3 = st.columns(3)
            c1.metric("Python files", len(info.get("py_files", [])))
            c2.metric("Impute functions", len(info.get("functions", [])))
            c3.metric("Impute classes", len(info.get("classes", [])))

            if info.get("functions"):
                fn_html = '<div class="cst-label">Detected functions</div>'
                for fn in info["functions"]:
                    args_str = ", ".join(fn["args"])
                    fn_html += (
                        f'<span class="cst-tag">{fn["name"]}({args_str})</span>'
                        f'<span style="font-size:1.075rem;color:#64748B;margin-right:10px"> in {fn["file"]}</span>'
                    )
                st.markdown(fn_html, unsafe_allow_html=True)

            if info.get("classes"):
                cls_html = '<div class="cst-label">Detected classes</div>'
                for cls in info["classes"]:
                    methods_str = ", ".join(m["name"] for m in cls["methods"])
                    cls_html += (
                        f'<span class="cst-tag-purple">{cls["class"]}</span>'
                        f'<span style="font-size:1.075rem;color:#64748B;margin-right:10px"> {methods_str} · {cls["file"]}</span>'
                    )
                st.markdown(cls_html, unsafe_allow_html=True)

            reqs = info.get("requirements", [])
            if reqs:
                req_html = f'<div class="cst-label">Requirements ({len(reqs)})</div>'
                req_html += "".join(
                    f'<span class="cst-tag-gray">{r}</span>'
                    for r in reqs[:14]
                )
                if len(reqs) > 14:
                    req_html += f'<span style="font-size:1.106rem;color:#64748B"> +{len(reqs)-14} more</span>'
                st.markdown(req_html, unsafe_allow_html=True)

            snippet = info.get("readme_snippet", "")
            if snippet:
                with st.expander("README preview", expanded=False):
                    st.markdown(
                        f'<div style="font-size:1.229rem;color:#CBD5E1;line-height:1.7;white-space:pre-wrap">{snippet[:600]}</div>',
                        unsafe_allow_html=True)

        # ── Step 3 — Install dependencies ─────────────────────────
        with st.container(border=True):
            st.markdown('<span class="cst-badge">STEP 3 — INSTALL DEPENDENCIES</span>', unsafe_allow_html=True)
            st.markdown(
                '<div class="cst-warn">⚠ This pip-installs the repo\'s packages into SENTI\'s '
                'Python environment. Only proceed if you trust this repository.</div>',
                unsafe_allow_html=True)

            if ss.get("custom_deps_installed"):
                st.markdown('<div class="cst-ok">✓ Dependencies already installed for this session.</div>',
                            unsafe_allow_html=True)

            if st.button("📦  Install dependencies", key="btn_custom_install"):
                with st.spinner("Installing … (may take a minute)"):
                    ok, log = install_deps(Path(repo_path))
                ss.custom_install_ok  = ok
                ss.custom_install_log = log
                if ok:
                    ss.custom_deps_installed = True

            # Always render the log if it exists (survives reruns)
            ok  = ss.get("custom_install_ok")
            log = ss.get("custom_install_log", "")
            if log:
                if ok:
                    st.markdown('<div class="cst-ok">✓ Installation completed successfully.</div>',
                                unsafe_allow_html=True)
                else:
                    # ── Extract key error lines ───────────────────
                    error_lines = []
                    for line in log.splitlines():
                        l = line.strip()
                        if any(kw in l.lower() for kw in [
                            "error:", "failed", "cannot", "could not",
                            "no module", "exit code", "traceback", "exception",
                            "not found", "permission denied",
                        ]):
                            error_lines.append(l)

                    if error_lines:
                        summary_html = (
                            '<div style="background:#fff0f3;border:1px solid rgba(185,28,60,0.25);border-radius:8px;'
                            'padding:12px 16px;margin:8px 0">'
                            '<div style="font-size:1.152rem;font-weight:700;color:#F87171;'
                            'margin-bottom:8px;letter-spacing:0.04em">⛔ INSTALLATION ERRORS</div>'
                        )
                        for el in error_lines[:10]:
                            summary_html += (
                                f'<div style="font-family:monospace;font-size:1.152rem;'
                                f'color:#b91c3c;padding:2px 0;line-height:1.6">{el}</div>'
                            )
                        if len(error_lines) > 10:
                            summary_html += (
                                f'<div style="font-size:1.121rem;color:#64748B;margin-top:4px">'
                                f'… and {len(error_lines)-10} more error lines</div>'
                            )
                        summary_html += '</div>'
                        st.markdown(summary_html, unsafe_allow_html=True)

                    # ── Smart diagnosis ───────────────────────────
                    log_lower = log.lower()

                    # Build list of (condition, title, explanation, fix_cmds)
                    _DIAGNOSES = [
                        (
                            "distutils.msvccompiler" in log_lower or "msvccompiler" in log_lower,
                            "C compiler / distutils conflict",
                            "The repo's <code>setup.py</code> uses <code>distutils.msvccompiler</code> "
                            "which was removed in Python 3.12. This is an outdated build system. "
                            "Try installing a pre-built wheel instead of building from source.",
                            ["pip install setuptools --break-system-packages --upgrade",
                             "pip install wheel --break-system-packages",
                             "# Then retry Install dependencies above"],
                        ),
                        (
                            "metadata-generation-failed" in log_lower or "pyproject.toml did not run" in log_lower,
                            "pyproject.toml build failed",
                            "The repo uses <code>pyproject.toml</code> but its build backend "
                            "(e.g. <code>hatchling</code>, <code>flit</code>, <code>meson</code>) "
                            "failed or is missing. Often caused by missing C headers or an old build tool.",
                            ["pip install build hatchling flit_core --break-system-packages",
                             "# Then retry Install dependencies above"],
                        ),
                        (
                            "no module named" in log_lower and "distutils" in log_lower,
                            "Missing distutils module",
                            "<code>distutils</code> was removed from Python 3.12+. "
                            "Install <code>setuptools</code> which provides a compatibility shim.",
                            ["pip install setuptools --break-system-packages --upgrade"],
                        ),
                        (
                            "no matching distribution" in log_lower or "could not find a version" in log_lower,
                            "Package not found on PyPI",
                            "One or more packages listed in <code>requirements.txt</code> do not exist "
                            "on PyPI under that exact name/version. Check the repo's README for "
                            "installation instructions — some packages are installed differently.",
                            ["# Check the repo README for the correct install command",
                             "# Some packages need: pip install git+https://github.com/..."],
                        ),
                        (
                            "permission denied" in log_lower,
                            "Permission error",
                            "pip cannot write to the Python environment. "
                            "This usually happens when running inside a system Python.",
                            ["pip install --user <package> --break-system-packages"],
                        ),
                        (
                            "ssl" in log_lower or "certificate" in log_lower,
                            "SSL / network error",
                            "pip could not reach PyPI due to an SSL or network issue. "
                            "Check internet connectivity or corporate proxy settings.",
                            ["pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org <pkg>"],
                        ),
                        (
                            ("gcc" in log_lower or "g++" in log_lower or "compiler" in log_lower)
                            and "command not found" in log_lower,
                            "C/C++ compiler not found",
                            "The package needs to compile native extensions but <code>gcc</code>/<code>g++</code> "
                            "is not installed on this system.",
                            ["# On Ubuntu/Debian:",
                             "sudo apt-get install build-essential python3-dev",
                             "# On macOS:",
                             "xcode-select --install"],
                        ),
                        (
                            "torch" in log_lower and ("cuda" in log_lower or "version" in log_lower),
                            "PyTorch version conflict",
                            "The repo requires a specific version of PyTorch. "
                            "SENTI already has PyTorch installed — version conflicts can occur. "
                            "Try installing the CPU-only wheel to avoid CUDA conflicts.",
                            ["pip install torch --index-url https://download.pytorch.org/whl/cpu "
                             "--break-system-packages"],
                        ),
                    ]

                    matched = [(t, e, c) for cond, t, e, c in _DIAGNOSES if cond]

                    if matched:
                        for title, explanation, fix_cmds in matched:
                            diag_html = (
                                '<div style="background:#0f1f2e;border:1px solid #1e40af;'
                                'border-radius:8px;padding:14px 16px;margin:10px 0">'
                                '<div style="font-size:1.152rem;font-weight:700;color:#60A5FA;'
                                'margin-bottom:6px;letter-spacing:0.04em">🔍 DIAGNOSIS — '
                                + title +
                                '</div>'
                                '<div style="font-size:1.229rem;color:#CBD5E1;line-height:1.7;margin-bottom:10px">'
                                + explanation +
                                '</div>'
                                '<div style="font-size:1.136rem;font-weight:600;color:var(--text3);'
                                'margin-bottom:4px">Suggested fix:</div>'
                                '<pre style="background:#0a0a1a;color:#86EFAC;font-size:1.136rem;'
                                'padding:10px 13px;border-radius:6px;margin:0;'
                                'white-space:pre-wrap;border:1px solid #14532d">'
                                + "\n".join(fix_cmds) +
                                '</pre></div>'
                            )
                            st.markdown(diag_html, unsafe_allow_html=True)
                    else:
                        st.markdown(
                            '<div style="background:#0f1f2e;border:1px solid #1e3a5f;'
                            'border-radius:8px;padding:10px 14px;margin:8px 0;'
                            'font-size:1.213rem;color:#94A3B8;line-height:1.65">'
                            '💡 No automatic diagnosis matched. Check the <strong>Full pip log</strong> '
                            'below and search the error message in the repo\'s GitHub Issues.'
                            '</div>',
                            unsafe_allow_html=True)

                with st.expander("Full pip log", expanded=False):
                    st.markdown(
                        f'<pre style="background:#0f172a;color:#94A3B8;font-size:1.136rem;'

                        f'line-height:1.6;padding:14px 16px;border-radius:8px;'
                        f'overflow-x:auto;white-space:pre-wrap;border:1px solid #1e293b">'
                        f'{log}</pre>',
                        unsafe_allow_html=True,
                    )

        # ── Step 4 — Adapter editor ────────────────────────────────
        with st.container(border=True):
            st.markdown('<span class="cst-badge">STEP 4 — ADAPTER CODE</span>', unsafe_allow_html=True)
            st.markdown(
                '<div class="cst-hint">SENTI auto-generated this adapter from the repo inspection. '
                'Edit the body of <code style="color:#93C5FD">impute(df, cols)</code> so it calls the '
                'repo correctly and returns a completed DataFrame. The function signature must not change.</div>',
                unsafe_allow_html=True)

            current_code = ss.get("custom_adapter_code", "")
            edited_code = st.text_area(
                "Adapter code",
                value=current_code,
                height=320,
                key="custom_adapter_editor",
                label_visibility="collapsed",
            )
            ss.custom_adapter_code = edited_code

            ca1, ca2, ca3 = st.columns([2, 2, 3])
            with ca1:
                if st.button("✅  Confirm adapter", key="btn_custom_confirm_adapter"):
                    ss.custom_ready = True
                    st.success("Adapter confirmed — press ▶ Run below.")
            with ca2:
                if st.button("↺  Regenerate template", key="btn_custom_regen_tmpl"):
                    ss.custom_adapter_code = make_template(info, Path(repo_path))
                    ss.custom_ready = False
                    st.rerun()
            with ca3:
                st.download_button(
                    "⬇  Download SENTI_adapter.py",
                    data=edited_code.encode(),
                    file_name="SENTI_adapter.py",
                    mime="text/x-python",
                    key="btn_dl_adapter",
                )

            if ss.get("custom_ready"):
                st.markdown(
                    '<div class="cst-ok">✓ Adapter ready — press ▶ Run Imputation below.</div>',
                    unsafe_allow_html=True)


def page_senti():
    # ── Toy Example ────────────────────────────────────────────────
    with st.expander("SENTI Toy Example (click to expand)", expanded=False):

        def _toy_style_incomplete(df_s, ref_df):
            styles = pd.DataFrame("", index=df_s.index, columns=df_s.columns)
            for r in df_s.index:
                for c in df_s.columns:
                    if pd.isna(ref_df.at[r, c]):
                        styles.at[r, c] = "background-color:rgba(248,113,113,0.18);color:#F87171;font-weight:700"
            return styles

        def _toy_style_imputed(df_s, mask_df):
            styles = pd.DataFrame("", index=df_s.index, columns=df_s.columns)
            for r in df_s.index:
                for c in df_s.columns:
                    if mask_df.at[r, c]:
                        styles.at[r, c] = "background-color:#c8dff0;color:#1a3a5c;font-weight:700"
            return styles

        def _make_mask(incomplete, complete):
            mask = pd.DataFrame(False, index=complete.index, columns=complete.columns)
            for r in complete.index:
                for c in complete.columns:
                    if r in incomplete.index and pd.isna(incomplete.at[r, c]):
                        mask.at[r, c] = True
            return mask

        # ── r0 / r0* ─────────────────────────────────────────────────
     #   st.markdown('<div class="SENTI-card-label" style="margin-bottom:6px;"></div>', unsafe_allow_html=True)
        r0_inc = pd.DataFrame([
            {"TupleID":"t₁","CargoID":"Charlie-3","Status":"Active", "Period":"Spring", "Country":"France","Hub":"Paris"},
            {"TupleID":"t₂","CargoID":"Alpha-7",  "Status":"Delayed","Period":"Summer", "Country":"USA",   "Hub":None},
            {"TupleID":"t₃","CargoID":"Beta-7",   "Status":"Pending","Period":"Winter", "Country":None,    "Hub":"New York"},
            {"TupleID":"t₄","CargoID":"Alpha-1",  "Status":None,     "Period":"Jul-Aug","Country":"USA",   "Hub":"NYC"},
        ]).set_index("TupleID")
        r0_imp = pd.DataFrame([
            {"TupleID":"t₁","CargoID":"Charlie-3","Status":"Active", "Period":"Spring", "Country":"France","Hub":"Paris"},
            {"TupleID":"t₂","CargoID":"Alpha-7",  "Status":"Delayed","Period":"Summer", "Country":"USA",   "Hub":"NYC"},
            {"TupleID":"t₃","CargoID":"Beta-7",   "Status":"Pending","Period":"Winter", "Country":"USA",   "Hub":"New York"},
            {"TupleID":"t₄","CargoID":"Alpha-1",  "Status":"Delayed","Period":"Jul-Aug","Country":"USA",   "Hub":"NYC"},
        ]).set_index("TupleID")
        r0_mask = _make_mask(r0_inc, r0_imp)

        cl, cr = st.columns(2)
        with cl:
            st.markdown('<div class="SENTI-card-label">r₀ — Incomplete Relation</div>', unsafe_allow_html=True)
            st.dataframe(
                r0_inc.style.apply(lambda s: _toy_style_incomplete(s, r0_inc), axis=None).format(na_rep="⊥"),
                use_container_width=True)
        with cr:
            st.markdown('<div class="SENTI-card-label">r₀* — After SENTI Imputation</div>', unsafe_allow_html=True)
            st.dataframe(
                r0_imp.style.apply(lambda s: _toy_style_imputed(s, r0_mask), axis=None),
                use_container_width=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── u1 / u1* ─────────────────────────────────────────────────
      #  st.markdown('<div class="SENTI-card-label" style="margin-bottom:6px;">BATCH u₁ — Incremental Append</div>', unsafe_allow_html=True)
        u1_inc = pd.DataFrame([
            {"TupleID":"t₅","CargoID":"Delta-9","Status":"Ready",  "Period":"Oct-Nov","Country":"Deutschland","Hub":"Berlin"},
            {"TupleID":"t₆","CargoID":"Beta-7", "Status":None,     "Period":"Winter", "Country":"US",         "Hub":None},
        ]).set_index("TupleID")
        u1_imp = pd.DataFrame([
            {"TupleID":"t₅","CargoID":"Delta-9","Status":"Ready",  "Period":"Oct-Nov","Country":"Deutschland","Hub":"Berlin"},
            {"TupleID":"t₆","CargoID":"Beta-7", "Status":"Pending","Period":"Winter", "Country":"US",         "Hub":"New York"},
        ]).set_index("TupleID")
        u1_mask = _make_mask(u1_inc, u1_imp)

        cl2, cr2 = st.columns(2)
        with cl2:
            st.markdown('<div class="SENTI-card-label">u₁ — Incomplete UPDATE</div>', unsafe_allow_html=True)
            st.dataframe(
                u1_inc.style.apply(lambda s: _toy_style_incomplete(s, u1_inc), axis=None).format(na_rep="⊥"),
                use_container_width=True)
        with cr2:
            st.markdown('<div class="SENTI-card-label">u₁* — After SENTI Imputation</div>', unsafe_allow_html=True)
            st.dataframe(
                u1_imp.style.apply(lambda s: _toy_style_imputed(s, u1_mask), axis=None),
                use_container_width=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── u2 / u2* ─────────────────────────────────────────────────
        #st.markdown('<div class="SENTI-card-label" style="margin-bottom:6px;">BATCH u₂ — Single-Tuple Append</div>', unsafe_allow_html=True)
        u2_inc = pd.DataFrame([
            {"TupleID":"t₇","CargoID":None,    "Status":None, "Period":"Spring","Country":"Germany","Hub":"Berlin"},
        ]).set_index("TupleID")
        u2_imp = pd.DataFrame([
            {"TupleID":"t₇","CargoID":"Delta-9","Status":"Ready","Period":"Spring","Country":"Germany","Hub":"Berlin"},
        ]).set_index("TupleID")
        u2_mask = _make_mask(u2_inc, u2_imp)

        cl3, cr3 = st.columns(2)
        with cl3:
            st.markdown('<div class="SENTI-card-label">u₂ — Incomplete Update</div>', unsafe_allow_html=True)
            st.dataframe(
                u2_inc.style.apply(lambda s: _toy_style_incomplete(s, u2_inc), axis=None).format(na_rep="⊥"),
                use_container_width=True)
        with cr3:
            st.markdown('<div class="SENTI-card-label">u₂* — After SENTI Imputation</div>', unsafe_allow_html=True)
            st.dataframe(
                u2_imp.style.apply(lambda s: _toy_style_imputed(s, u2_mask), axis=None),
                use_container_width=True)

       # st.info(
         #  "**How SENTI works:** Each missing cell (⊥) is filled by semantic retrieval. "
           # "SENTI encodes every row as a vector embedding via a sentence transformer (e.g. LaBSE), "
           # "stores them in a FAISS index, then for each incomplete tuple queries: "
          #  "'which complete row is most semantically similar?' — "
            #"e.g. t₂ missing *Hub* → nearest neighbour is t₁ (Paris) or t₄ (NYC) → **Hub(t₂) = NYC**. "
          #  "Batches u₁ and u₂ demonstrate the incremental mode: new tuples are appended and imputed "
          #  "against the growing FAISS index without reprocessing r₀."
       # )

    # ── Imputer selector (buttons) + optional Transformer select ──
    ss.mode = "Incremental Imputation"  # always incremental

    _strat_options = [
        ("SENTI",   "SENTI",  "Semantic retrieval via sentence transformers"),
        ("Mean",    "mean",   "Column mean (numeric cols)"),
        ("Median",  "median", "Column median (numeric cols)"),
        ("Mode",    "mode",   "Most frequent value"),
        ("KNN",     "knn",    "k-NN imputer (k=5, numeric)"),
        ("F-fill",  "ffill",  "Forward-fill → backward-fill"),
        ("Custom",  "custom", "GitHub repo adapter"),
    ]
    _current_strat = ss.get("strategy") or "SENTI"

    st.markdown('<div class="SENTI-card-label">Imputer</div>', unsafe_allow_html=True)

    _strat_keys   = [k   for (_, k, _) in _strat_options]
    _strat_labels = [lbl for (lbl, _, _) in _strat_options]

    if st.session_state.get('strategy') not in _strat_keys:
        st.session_state['strategy'] = 'SENTI'
    _current_strat = st.session_state.get('strategy', 'SENTI')
    _active_idx    = _strat_keys.index(_current_strat)

    # Style the native radio to look like pills:
    # active = red, inactive = light blue, pill shape, bold text
    st.markdown("""
    <style>
    /* ── Imputer pill radio ── */
    div[data-testid="stRadio"]#imputer-radio [role="radiogroup"] {
        display: flex !important;
        flex-direction: row !important;
        gap: 6px !important;
        flex-wrap: wrap !important;
    }
    div[data-testid="stRadio"]#imputer-radio label {
        padding: 8px 20px !important;
        border-radius: 22px !important;
        border: 1.5px solid #8fb8d4 !important;
        background: #b8d4e8 !important;
        color: #1a2c3d !important;
        font-size: 1.05rem !important;
        font-weight: 700 !important;
        cursor: pointer !important;
        transition: all 0.15s ease !important;
        white-space: nowrap !important;
    }
    div[data-testid="stRadio"]#imputer-radio label:hover {
        background: #9ac4de !important;
        transform: translateY(-1px) !important;
        box-shadow: 0 2px 8px rgba(74,127,165,0.25) !important;
    }
    div[data-testid="stRadio"]#imputer-radio label:has(input:checked) {
        background: #c0392b !important;
        color: #ffffff !important;
        border-color: #a93226 !important;
        box-shadow: 0 2px 8px rgba(192,57,43,0.35) !important;
    }
    div[data-testid="stRadio"]#imputer-radio label span { display: none !important; }
    div[data-testid="stRadio"]#imputer-radio label input[type="radio"] { display: none !important; }
    div[data-testid="stRadio"]#imputer-radio label p {
        margin: 0 !important;
        font-size: 1.05rem !important;
        font-weight: 700 !important;
    }
    div[data-testid="stRadio"]#imputer-radio [data-testid="stWidgetLabel"] {
        display: none !important;
    }
    </style>
    <div id="imputer-radio-anchor"></div>
    """, unsafe_allow_html=True)

    # Attach the id to the radio widget via JS after render
    _sel = st.radio(
        label='imputer',
        options=_strat_labels,
        index=_active_idx,
        horizontal=True,
        key='_imputer_radio',
        label_visibility='collapsed',
    )
    # Map label back to key
    _sel_key = _strat_keys[_strat_labels.index(_sel)] if _sel in _strat_labels else 'SENTI'
    if _sel_key != _current_strat:
        st.session_state['strategy'] = _sel_key
        ss.strategy      = _sel_key
        ss.other_strategy = _sel_key
        st.rerun()

    # JS: assign id to the imputer radio for CSS scoping
    st.markdown("""
    <script>
    (function(){
      var _tries = 0;
      function tag(){
        var doc = window.parent.document;
        // Find all stRadio divs in main content (not sidebar)
        var main = doc.querySelector('section.main') || doc.querySelector('[data-testid="stMain"]') || doc.body;
        var radios = main.querySelectorAll('div[data-testid="stRadio"]');
        for (var i=0; i<radios.length; i++){
          var labels = radios[i].querySelectorAll('label');
          for (var j=0; j<labels.length; j++){
            var txt = labels[j].innerText ? labels[j].innerText.trim() : '';
            if (txt === 'SENTI' || txt === 'Mean' || txt === 'Median'){
              radios[i].id = 'imputer-radio';
              return;
            }
          }
        }
        if(++_tries < 30) setTimeout(tag, 100);
      }
      setTimeout(tag, 50);
    })();
    </script>
    """, unsafe_allow_html=True)

    _current_strat = _sel_key
    ss.strategy = _current_strat
    ss.other_strategy = _current_strat

    # Show transformer selector or desc below buttons
    if _current_strat == "SENTI":
        _tc1, _tc2 = st.columns([2, 5])
        with _tc1:
            st.markdown('<div class="SENTI-card-label" style="margin-top:0.6rem;">Sentence Transformer</div>', unsafe_allow_html=True)
            ss.transformer_choice = st.selectbox(
                "Transformer",
                options=["LaBSE", "all-MiniLM-L6-v2"],
                key="transformer_choice_inpanel",
                index=(["LaBSE", "all-MiniLM-L6-v2"].index(ss.get("transformer_choice", "LaBSE"))
                       if ss.get("transformer_choice") in ["LaBSE", "all-MiniLM-L6-v2"] else 0),
                label_visibility="collapsed",
            )
    elif _current_strat == "custom":
        st.markdown(
            '<div style="font-size:1.075rem;color:var(--text3);line-height:1.55;margin-top:0.4rem;">'
            'Paste any imputation repo URL → SENTI clones it, generates an adapter, '
            'and runs it in an isolated subprocess.'
            '</div>', unsafe_allow_html=True)

    # ── Custom GitHub imputer panel ────────────────────────────────
    if ss.strategy == "custom":
        _custom_imputer_panel()

    st.markdown("<hr>", unsafe_allow_html=True)
    data_loader_ui()

    # Column selection
    if ss.working_df is not None:
        _section("Columns to Impute", badge="SELECT")

        all_cols     = list(ss.working_df.columns)
        missing_cols = [c for c in all_cols if ss.working_df[c].isna().any()]
        num_c  = sum(pd.api.types.is_numeric_dtype(ss.working_df[c]) for c in all_cols)
        cat_c  = len(all_cols) - num_c
        miss_c = len(missing_cols)
        st.markdown(f"""
        <div style="display:flex; gap:10px; margin-bottom:0.6rem; flex-wrap:wrap; font-size:1.152rem; color:var(--text2);">
            <span><span class="col-type-N">N</span> Numeric — {num_c} col(s)</span>
            <span><span class="col-type-C">C</span> Categorical — {cat_c} col(s)</span>
            <span style="color:var(--gold);">⚠ {miss_c} col(s) have missing values</span>
        </div>
        """, unsafe_allow_html=True)

        # Use a dataset fingerprint to detect when the dataset changes (not just reruns)
        _df_sig = (tuple(ss.working_df.columns), len(ss.working_df))
        if ss.get("_col_sel_df_sig") != _df_sig:
            ss._col_sel_df_sig = _df_sig
            # Reset radio to "All" when dataset changes
            st.session_state["col_mode_radio"] = "Select all"
            ss.selected_cols = missing_cols[:]

        # Column-mode radio — two options only
        _mode_labels = ["Select all", "Manual"]
        _mode_keys   = ["all", "manual"]

        if "col_mode_radio" not in st.session_state:
            st.session_state["col_mode_radio"] = "Select all"

        _sel_mode = st.radio(
            "Column selection",
            options=_mode_labels,
            horizontal=True,
            key="col_mode_radio",
            label_visibility="collapsed",
        )
        _sel_key = _mode_keys[_mode_labels.index(_sel_mode)]

        if _sel_key == "all":
            ss.selected_cols = missing_cols[:]
        # "manual" handled below via multiselect

        if _sel_key == "manual":
            _manual_default = [c for c in (ss.selected_cols or []) if c in missing_cols] or missing_cols[:]
            if "cols_to_impute_multiselect" not in st.session_state or \
                    not isinstance(st.session_state["cols_to_impute_multiselect"], list):
                st.session_state["cols_to_impute_multiselect"] = _manual_default
            selected = st.multiselect(
                "Columns",
                options=missing_cols,
                key="cols_to_impute_multiselect",
                label_visibility="collapsed",
                placeholder="Pick columns to impute…",
            )
            ss.selected_cols = selected
        else:
            sel_label = ", ".join(ss.selected_cols) if ss.selected_cols else "none"
            st.markdown(
                f'<div style="font-size:1.098rem;color:var(--text2);padding:4px 0;">'
                f'Selected: <code>{sel_label}</code></div>',
                unsafe_allow_html=True,
            )



        st.markdown("<hr>", unsafe_allow_html=True)

    if ss.working_df is not None:
        # ── Incremental Imputation ───────────────────────────────────

        if ss.source_snapshot is not None and ss.last_imputed_iter == ss.iter_k:
            render_source_preview(ss.source_snapshot, ss.iter_k)
        elif ss.flow_state == "append_phase" and ss.pre_append_snapshot is not None:
            render_source_preview(ss.pre_append_snapshot, ss.iter_k)
        else:
            render_source_preview(ss.working_df, ss.iter_k)

        st.markdown('<div style="margin-top:1rem"></div>', unsafe_allow_html=True)

        _run_c1, _run_c2, _run_c3 = st.columns([1, 2, 1])
        with _run_c2:
            _do_run = st.button("▶  Run Imputation", type="primary",
                                key=f"btn_run_impute_dyn_initial_{ss.iter_k}",
                                use_container_width=True)
        if _do_run:
            ss.source_snapshot = ss.working_df.copy()
            with st.spinner("Running imputation…"):
                imputed, mask = run_imputation(ss.working_df, context_df=None)
            if imputed is not None:
                ss.last_imputed_iter = ss.iter_k
                ss.imputed_df = imputed
                ss.imputed_mask = mask
                # Iteration 1: whole source is the "new" batch
                nm_batch    = int(ss.source_snapshot.isna().sum().sum())
                n_imp_batch = int(mask.sum().sum())
                if not isinstance(ss.get("inc_log"), list):
                    ss.inc_log = []
                ss.inc_log.append({
                    "batch":      ss.iter_k,
                    "new_rows":   len(ss.source_snapshot),
                    "missing_in": nm_batch,
                    "imputed":    n_imp_batch,
                    "method":     ss.strategy,
                })
                ss.flow_state = "post_impute_prompt"
                st.rerun()

        # Persistent display — runs on every rerun after imputation
        if ss.imputed_df is not None and ss.flow_state in ("post_impute_prompt", "append_phase") and ss.last_imputed_iter == ss.iter_k:
            show_imputed(ss.imputed_df, ss.imputed_mask, title=f"Imputed Dataset — Iteration {ss.last_imputed_iter}")

            # ── Batch Log (latest only) + JSON download ────────
            inc_log = ss.get("inc_log") or []
            if inc_log:
                last_entry = inc_log[-1]
                imp_rate   = f"{100*last_entry['imputed']/max(last_entry['missing_in'],1):.0f}%" if last_entry["missing_in"] > 0 else "N/A"
                prev_note  = (
                    f'<span style="font-size:1.106rem;color:var(--text3);margin-left:8px">'
                    f'(+ {len(inc_log)-1} previous batch(es) saved to JSON)</span>'
                    if len(inc_log) > 1 else ""
                )
                last_row_html = (
                    f'<tr>'
                    f'<td style="color:var(--accent2);font-weight:700">#{last_entry["batch"]}</td>'
                    f'<td style="color:var(--text)">{last_entry.get("new_rows", last_entry.get("rows","—"))}</td>'
                    f'<td style="color:var(--red)">{last_entry["missing_in"]}</td>'
                    f'<td style="color:var(--green)">{last_entry["imputed"]} ({imp_rate})</td>'
                    f'<td style="color:var(--text3);font-size:1.152rem">{last_entry["method"]}</td>'
                    f'</tr>'
                )
                st.markdown(f"""
                <div class="SENTI-card" style="margin-top:1rem">
                  <div class="SENTI-card-label">Incremental Batch Log{prev_note}</div>
                  <div style="overflow-x:auto"><table style="width:100%;border-collapse:collapse;font-size:1.26rem">
                    <thead><tr style="color:var(--text3);font-size:1.075rem;letter-spacing:0.08em">
                      <th style="padding:5px 8px;text-align:left">BATCH</th>
                      <th style="padding:5px 8px;text-align:left">NEW ROWS</th>
                      <th style="padding:5px 8px;text-align:left">MISSING (new)</th>
                      <th style="padding:5px 8px;text-align:left">IMPUTED</th>
                      <th style="padding:5px 8px;text-align:left">METHOD</th>
                    </tr></thead>
                    <tbody>{last_row_html}</tbody>
                  </table></div>
                </div>
                """, unsafe_allow_html=True)

            # ── FAISS Index Monitor (latest batch only) ────────
            _render_faiss_panel(latest_only=True)

            # ── Combined JSON download for full history ─────────
            if inc_log:
                import json as _json
                from datetime import datetime as _dt
                try:
                    from zoneinfo import ZoneInfo as _ZI
                    _ts = _dt.now(_ZI("Europe/Rome")).strftime("%Y-%m-%d_%H-%M-%S")
                except Exception:
                    _ts = _dt.now().strftime("%Y-%m-%d_%H-%M-%S")

                faiss_log   = ss.get("faiss_log") or []
                full_record = {
                    "generated_at": _ts,
                    "total_batches": len(inc_log),
                    "batch_log": inc_log,
                    "faiss_log": faiss_log,
                }
                _json_bytes = _json.dumps(full_record, indent=2, default=str).encode("utf-8")
                st.download_button(
                    "⬇  Download Full Log (JSON)",
                    data=_json_bytes,
                    file_name=f"SENTI_log_{_ts}.json",
                    mime="application/json",
                    key=f"dl_full_log_{ss.iter_k}",
                )

            # ── Distribution Drift Monitor ─────────────────────
            if len(inc_log) >= 2 and ss.get("source_snapshot") is not None:
                try:
                    first_df  = ss.get("raw_df") or ss.source_snapshot
                    latest_df = ss.imputed_df
                    num_drift_cols = [c for c in latest_df.columns if pd.api.types.is_numeric_dtype(latest_df[c])]
                    if num_drift_cols:
                        drift_items = ""
                        for col in num_drift_cols[:6]:
                            f_val = first_df[col].dropna().mean() if col in first_df.columns else None
                            l_val = latest_df[col].dropna().mean()
                            if f_val is None or f_val == 0:
                                continue
                            drift_pct = abs((l_val - f_val) / max(abs(f_val), 1e-9)) * 100
                            d_color = "var(--red)" if drift_pct > 20 else ("var(--gold)" if drift_pct > 5 else "var(--green)")
                            arrow   = "↑" if l_val > f_val else ("↓" if l_val < f_val else "→")
                            drift_items += (
                                f'<div style="display:flex;justify-content:space-between;padding:5px 0;'
                                f'border-bottom:1px solid var(--border)">'
                                f'<span style="font-size:1.26rem;color:var(--text)">{col}</span>'
                                f'<span style="font-size:1.26rem;color:{d_color};font-weight:700">'
                                f'{arrow} {drift_pct:.1f}% drift</span>'
                                f'<span style="font-size:1.152rem;color:var(--text3)">'
                                f'{f_val:.2g} → {l_val:.2g}</span></div>'
                            )
                        if drift_items:
                            st.markdown(f"""
                            <div class="SENTI-card" style="margin-top:0.8rem">
                              <div class="SENTI-card-label">Distribution Drift Monitor</div>
                              <p style="font-size:1.198rem;color:var(--text2);margin:0 0 8px;line-height:1.5">
                                Mean shift from initial → current batch.
                                <span style="color:var(--red)">▪ &gt;20% high</span> ·
                                <span style="color:var(--gold)">▪ 5–20% moderate</span> ·
                                <span style="color:var(--green)">▪ &lt;5% stable</span>
                              </p>
                              {drift_items}
                            </div>
                            """, unsafe_allow_html=True)
                except Exception:
                    pass

            if ss.flow_state == "post_impute_prompt":
                Incremental_append_ui()
            else:
                append_panel()

        elif ss.flow_state == "finished":
            if ss.imputed_df is not None:
                show_imputed(ss.imputed_df, ss.imputed_mask, title=f"Final Imputed Dataset — Iteration {ss.last_imputed_iter}")


def page_docs():
    if st.button("← Back", key="btn_back_page_docs"):
        ss.page = ss.get("_last_real_page") or "Imputation"
        ss._doc = False
        st.rerun()

    _section("Guide", badge="GUIDE")

    # CSS for doc elements
    st.markdown("""
    <style>
    .doc-tab-content { padding: 0.5rem 0; }
    .doc-h2 {
        font-size: 1.26rem; font-weight: 700; color: var(--text);
        margin: 1.4rem 0 0.4rem; letter-spacing: 0.02em;
        border-bottom: 1px solid var(--border); padding-bottom: 6px;
    }
    .doc-h3 {
        font-size: 1.08rem; font-weight: 600; color: var(--accent2);
        margin: 1.1rem 0 0.3rem;
    }
    .doc-p {
        font-size: 0.996rem; color: var(--text2); line-height: 1.75;
        margin: 0.3rem 0 0.6rem;
    }
    .doc-note {
        background: rgba(96,165,250,0.10); border-left: 3px solid var(--accent2);
        border-radius: 0 6px 6px 0; padding: 8px 14px; margin: 0.6rem 0;
        font-size: 0.96rem; color: var(--text2); line-height: 1.6;
    }
    .doc-warn {
        background: rgba(251,191,36,0.10); border-left: 3px solid var(--gold);
        border-radius: 0 6px 6px 0; padding: 8px 14px; margin: 0.6rem 0;
        font-size: 0.96rem; color: var(--text2); line-height: 1.6;
    }
    .doc-step-row {
        display: flex; gap: 14px; align-items: flex-start;
        padding: 10px 0; border-bottom: 1px solid var(--border);
    }
    .doc-step-num {
        min-width: 28px; height: 28px; border-radius: 50%;
        background: var(--accent2); color: #fff;
        font-size: 0.9rem; font-weight: 700;
        display: flex; align-items: center; justify-content: center;
        flex-shrink: 0; margin-top: 2px;
    }
    .doc-step-body h4 { font-size: 1.044rem; color: var(--text); margin: 0 0 3px; }
    .doc-step-body p  { font-size: 0.96rem; color: var(--text2); margin: 0; line-height: 1.6; }
    .doc-phase {
        display: flex; gap: 12px; margin: 8px 0; align-items: flex-start;
    }
    .doc-phase-badge {
        min-width: 72px; padding: 3px 8px; border-radius: 12px;
        font-size: 0.84rem; font-weight: 700; text-align: center;
        flex-shrink: 0; margin-top: 2px;
    }
    .doc-phase-text { font-size: 0.96rem; color: var(--text2); line-height: 1.65; }
    .doc-kv { display: flex; gap: 10px; margin: 4px 0; font-size: 0.96rem; }
    .doc-kv-key { color: var(--accent3); min-width: 140px; font-weight: 600; flex-shrink: 0; }
    .doc-kv-val { color: var(--text2); line-height: 1.55; }
    .doc-metric-row {
        display: flex; gap: 10px; flex-wrap: wrap; margin: 0.5rem 0;
    }
    .doc-metric-chip {
        padding: 5px 14px; border-radius: 20px; font-size: 0.924rem;
        font-weight: 600; background: rgba(96,165,250,0.12);
        color: var(--accent2); border: 1px solid rgba(96,165,250,0.25);
    }
    .doc-fk-arrow {
        display: inline-flex; align-items: center; gap: 6px;
        font-size: 0.96rem; padding: 4px 10px; border-radius: 16px;
        background: rgba(52,211,153,0.10); border: 1px solid rgba(52,211,153,0.2);
        color: var(--green); margin: 3px 4px 3px 0;
    }
    </style>
    """, unsafe_allow_html=True)

    tabs = st.tabs([
        "🏠  Overview",
        "🧩  SENTI Algorithm",
        "📈  Incremental Mode",
        "📊  Evaluation",
        "⚙️  Settings & Tips",
    ])

    # ══════════════════════════════════════════════════════════════════
    # TAB 0 — OVERVIEW
    # ══════════════════════════════════════════════════════════════════
    with tabs[0]:
        st.markdown("""
        <div class="doc-p" style="font-size:1.351rem">
        <strong style="color:var(--text)">SENTI</strong> (Zero-shot Imputation Incremental Data Imputation)
        is a research prototype developed at the <strong>DIMES Laboratory, University of Calabria</strong>.
        It imputes missing values in tabular datasets using sentence-transformer embeddings and
        FAISS-based nearest-neighbor retrieval — no model training required.
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="doc-h2">Pages at a glance</div>', unsafe_allow_html=True)
        pages = [
            ("Inject Nulls",   "var(--accent3)", "Upload a complete dataset and inject controlled missing values (MCAR) to create a realistic incomplete dataset for benchmarking."),
            ("Imputation",     "var(--accent2)", "Core imputation page. Supports Static (one-shot) and Incremental (append-then-impute) modes with SENTI or classical baselines."),
            ("Evaluation",     "#A78BFA",        "Upload Incomplete + Imputed + Ground Truth CSVs to compute Semantic Similarity, Exact Match Rate, and RMSE at imputed positions only."),
            ("Guide",  "var(--text3)",   "This page."),
        ]
        for name, color, desc in pages:
            st.markdown(f"""
            <div class="doc-step-row">
              <div class="doc-step-num" style="background:{color}">→</div>
              <div class="doc-step-body">
                <h4>{name}</h4>
                <p>{desc}</p>
              </div>
            </div>""", unsafe_allow_html=True)

        st.markdown('<div class="doc-h2">Supported file formats</div>', unsafe_allow_html=True)
        fmts = [("CSV / TSV","Comma- or tab-separated text files"),
                ("JSON / JSONL","Standard JSON and JSON-Lines"),
                ("Excel (.xlsx/.xls)","Microsoft Excel workbooks"),
                ("Parquet","Apache Parquet columnar format"),
                ]
        for fmt, desc in fmts:
            st.markdown(f'<div class="doc-kv"><span class="doc-kv-key">{fmt}</span><span class="doc-kv-val">{desc}</span></div>', unsafe_allow_html=True)

        st.markdown('<div class="doc-h2">Available imputers</div>', unsafe_allow_html=True)
        imputers = [
            ("SENTI","Embedding-based, Zero-shot Imputation. Uses sentence transformers + FAISS. Works on both categorical and numeric columns simultaneously."),
            ("Mean","Fills numeric missing values with the column mean. Ignores categorical columns."),
            ("Median","Fills numeric missing values with the column median. More robust to outliers than mean."),
            ("Mode / MostFreq","Fills any column with its most frequent value. Works on both types."),
            ("KNN","k-Nearest Neighbors imputation on numeric columns using sklearn."),
            ("ffill","Forward-fill then backward-fill. Useful for time-series ordered data."),
        ]
        for name, desc in imputers:
            st.markdown(f'<div class="doc-kv"><span class="doc-kv-key">{name}</span><span class="doc-kv-val">{desc}</span></div>', unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════
    # TAB 1 — SENTI ALGORITHM
    # ══════════════════════════════════════════════════════════════════
    with tabs[1]:
        st.markdown('<div class="doc-h2">Core idea</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="doc-p">
        SENTI treats each table row as a <em>natural language sentence</em> by concatenating all
        its column values into a single string. A pre-trained sentence transformer encodes this
        string into a dense vector. Missing values are imputed by finding the most similar
        complete (or partially complete) rows in the embedding space — their values are then
        transferred or averaged to fill the gap.
        <br><br>
        Crucially, <strong>no fine-tuning or task-specific training</strong> is needed.
        The transformer is used purely as an encoder; its weights are frozen.
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="doc-h2">The 4-phase pipeline</div>', unsafe_allow_html=True)
        st.markdown('<div class="doc-p">Each call to the SENTI imputer runs four sequential phases:</div>', unsafe_allow_html=True)

        phases = [
            ("#60A5FA", "Phase 1 — Categorical",
             "All rows are embedded <em>as-is</em> (missing cells become empty string). "
             "A FAISS <code>IndexFlatIP</code> (inner-product / cosine) index is built over these embeddings. "
             "For each row with a missing <strong>categorical</strong> column, the k=25 nearest neighbors "
             "are retrieved. If any neighbor has similarity ≥ 0.97 it is copied directly "
             "(<em>direct match</em>); otherwise a <strong>weighted vote</strong> is taken over all "
             "neighbors with similarity ≥ 0.65, with cosine similarity as the weight. "
             "Output: rows with categorical columns filled in."),
            ("#A78BFA", "Phase 2 — Numeric",
             "Rows are re-embedded using the Phase 1 output — now that categorical columns are "
             "filled, the embedding is richer and more discriminative. A new FAISS index is built. "
             "For each row with a missing <strong>numeric</strong> column, a "
             "<strong>weighted average</strong> of neighbor values is computed "
             "(weighted by cosine similarity, threshold ≥ 0.65). "
             "Result is rounded to the nearest integer for integer-typed columns. "
             "Output: rows with numeric columns filled in."),
            ("#36D399", "Phase 3 — Local Fallback",
             "A third index is built on Phase 2 embeddings. Any cell <em>still</em> missing "
             "after phases 1–2 is filled using the <strong>mode</strong> (categorical) or "
             "<strong>median</strong> (numeric) of its k nearest neighbors. "
             "This catches edge cases where weighted voting found no qualifying neighbors."),
            ("var(--gold)", "Phase 4 — Global Fallback",
             "If any cell remains missing after Phase 3 (extremely rare), a "
             "<strong>global column-level fallback</strong> is applied: "
             "column mode for categorical, column median for numeric. "
             "This guarantees a fully complete output with no remaining NaNs."),
        ]
        for color, title, desc in phases:
            st.markdown(f"""
            <div class="doc-phase">
              <div class="doc-phase-badge" style="background:rgba(96,165,250,0.12);color:{color};border:1px solid {color}33">{title.split("—")[0].strip()}</div>
              <div class="doc-phase-text"><strong style="color:{color}">{title}</strong><br>{desc}</div>
            </div>""", unsafe_allow_html=True)

        st.markdown('<div class="doc-h2">FAISS index details</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="doc-p">
        SENTI uses <strong>IndexFlatIP</strong> — an exact (brute-force) inner-product index.
        Because sentence-transformer embeddings are L2-normalised (‖v‖=1), inner product
        equals cosine similarity. No quantisation or approximation is used, so results
        are exact. This is appropriate for dataset sizes up to ~100k rows; for larger datasets
        an approximate index (IVFFlat, HNSW) would be needed.
        </div>
        """, unsafe_allow_html=True)
        kv = [
            ("Index type",       "IndexFlatIP (exact cosine via inner product on L2-normalised vectors)"),
            ("k neighbors",      "25 (MEDIAN_K constant)"),
            ("Direct-copy threshold",  "Cosine similarity ≥ 0.97 — neighbor value copied directly"),
            ("Voting threshold",  "Cosine similarity ≥ 0.65 (SIM_THRESHOLD) — neighbor included in weighted vote/average"),
            ("Embedding models", "LaBSE (768-dim, multilingual) · all-MiniLM-L6-v2 (384-dim, English)"),
            ("Batch size",       "64 sentences per encode call"),
        ]
        for k, v in kv:
            st.markdown(f'<div class="doc-kv"><span class="doc-kv-key">{k}</span><span class="doc-kv-val">{v}</span></div>', unsafe_allow_html=True)

        st.markdown('<div class="doc-h2">Sentence transformer selection</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="doc-p">
        <strong>LaBSE</strong> (Language-agnostic BERT Sentence Embedding) is the default.
        It produces 768-dimensional embeddings and is trained on 109 languages —
        making it robust to datasets with non-English or mixed-language text values
        (e.g. Italian city names, product descriptions). Recommended for most datasets.<br><br>
        <strong>all-MiniLM-L6-v2</strong> produces 384-dimensional embeddings and is
        significantly faster (fewer parameters), but is English-only and may be less
        accurate on multilingual or domain-specific data. Useful when speed matters more than accuracy.
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="doc-note">💡 The model is cached after the first load — subsequent iterations in Incremental mode reuse the same in-memory model, adding no reload overhead.</div>', unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════
    # TAB 2 — MULTI-TABLE
    # ══════════════════════════════════════════════════════════════════
    # ══════════════════════════════════════════════════════════════════
    # TAB 2 — INCREMENTAL MODE
    # ══════════════════════════════════════════════════════════════════
    with tabs[2]:
        st.markdown('<div class="doc-h2">What is Incremental mode?</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="doc-p">
        In many real-world scenarios data arrives in <em>batches</em> — new rows are appended
        to an existing table over time. Re-imputing the entire table from scratch each time is
        wasteful. SENTI's Incremental mode imputes <strong>only the new rows</strong> in each
        batch, while using all previously imputed rows as FAISS context to improve accuracy.
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="doc-h2">How the FAISS index grows</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="doc-p">
        At each iteration <em>k</em>:
        </div>
        <div class="doc-phase">
          <div class="doc-phase-badge" style="background:rgba(96,165,250,0.12);color:var(--text2);border:1px solid #ffffff22">BEFORE</div>
          <div class="doc-phase-text">The FAISS index contains all vectors from batches 1…k-1 (previously imputed rows). These are the <strong>context vectors</strong>.</div>
        </div>
        <div class="doc-phase">
          <div class="doc-phase-badge" style="background:rgba(251,146,60,0.12);color:var(--accent3);border:1px solid var(--accent3)33">NEW</div>
          <div class="doc-phase-text">The new batch k rows are embedded and added to the index.</div>
        </div>
        <div class="doc-phase">
          <div class="doc-phase-badge" style="background:rgba(52,211,153,0.12);color:var(--green);border:1px solid var(--green)33">AFTER</div>
          <div class="doc-phase-text">The index now contains context + new batch = cumulative total. Only new-batch rows are imputed; context rows are used as neighbors but not re-processed.</div>
        </div>
        <div class="doc-p" style="margin-top:0.8rem">
        The <strong>FAISS Index Monitor</strong> panel shows this per-batch, per-phase breakdown
        with a Δ CHECK column confirming the index grows by exactly the new-batch count.
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="doc-h2">Workflow</div>', unsafe_allow_html=True)
        inc_steps = [
            ("Load dataset",    "Upload or select a demo. The dataset is the first batch."),
            ("Run Imputation",  "Click ▶ Run. The full dataset is imputed (iteration 1)."),
            ("Continue?",       "SENTI asks: Add More Tuples or Finish?"),
            ("Append",          "Upload a CSV or enter rows manually. They are appended to the END of the current table."),
            ("Re-impute",       "Click Run Imputation again. Only the NEW rows are imputed; all previous imputed rows act as rich neighbors via the FAISS context."),
            ("Repeat",          "Steps 3–5 repeat for as many iterations as needed."),
            ("Finish",          "Click Finish. The final cumulative imputed dataset is shown and available for download."),
        ]
        for i, (t, d) in enumerate(inc_steps, 1):
            st.markdown(f"""
            <div class="doc-step-row">
              <div class="doc-step-num">{i}</div>
              <div class="doc-step-body"><h4>{t}</h4><p>{d}</p></div>
            </div>""", unsafe_allow_html=True)

        st.markdown('<div class="doc-h2">Batch log & JSON export</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="doc-p">
        After each iteration, the <strong>Incremental Batch Log</strong> shows the latest batch's
        statistics (new rows, missing cells, cells imputed, method). All previous batches are saved
        to a downloadable JSON file that includes both the batch log and the full FAISS index history.
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="doc-note">💡 Later batches benefit from a richer FAISS context and are generally imputed with higher accuracy than batch 1, because more complete neighbor rows are available.</div>', unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════
    # TAB 4 — EVALUATION
    # ══════════════════════════════════════════════════════════════════
    with tabs[3]:
        st.markdown('<div class="doc-h2">How evaluation works</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="doc-p">
        Evaluation compares imputed values <strong>only at positions that were originally missing</strong>
        in the incomplete dataset. Cells that were already present are ignored — this avoids
        rewarding imputers that simply copy non-missing values unchanged.
        <br><br>
        Three files are required, all with identical columns and row counts:
        </div>
        """, unsafe_allow_html=True)

        files = [
            ("Incomplete CSV",    "The dataset <em>after</em> null injection — contains NaN at imputed positions."),
            ("Imputed CSV",       "The output of SENTI (or any imputer). All cells should be filled."),
            ("Ground Truth CSV",  "The original complete dataset before null injection. Used as the reference."),
        ]
        for name, desc in files:
            st.markdown(f'<div class="doc-kv"><span class="doc-kv-key">{name}</span><span class="doc-kv-val">{desc}</span></div>', unsafe_allow_html=True)

        st.markdown('<div class="doc-h2">Metrics</div>', unsafe_allow_html=True)

        metrics = [
            ("Semantic Similarity", "var(--accent2)",
             "Uses a sentence transformer to embed each imputed cell value and its corresponding "
             "ground-truth value as strings, then computes cosine similarity. Scaled from [-1,1] to [0,1]. "
             "Color-coded: <span style='color:#DC3545'>red 0.0–0.7</span> (poor), "
             "<span style='color:var(--accent3)'>orange 0.7–0.95</span> (acceptable), "
             "<span style='color:var(--green)'>green 0.95–1.0</span> (excellent). "
             "Works on both categorical and numeric columns."),
            ("Exact Match Rate",    "#A78BFA",
             "The fraction of imputed cells whose value exactly matches the ground truth. "
             "Reported overall and per-column. A strict metric — near-correct numeric values "
             "may score 0 here but high on semantic similarity."),
            ("RMSE",               "var(--accent3)",
             "Root Mean Squared Error on numeric columns only. Optional normalization "
             "(Min-Max, Z-score, Robust IQR, Divide by constant, User min/max) can be applied "
             "to make RMSE comparable across columns with different scales."),
        ]
        for name, color, desc in metrics:
            st.markdown(f"""
            <div class="doc-phase" style="margin:10px 0">
              <div class="doc-phase-badge" style="background:rgba(96,165,250,0.10);color:{color};border:1px solid {color}44">{name}</div>
              <div class="doc-phase-text">{desc}</div>
            </div>""", unsafe_allow_html=True)

        st.markdown('<div class="doc-h2">Column selection modes</div>', unsafe_allow_html=True)
        col_modes = [
            ("Select all",        "Computes all three metrics (Semantic Similarity, Exact Match, RMSE)."),
            ("Categorical only",  "Computes Semantic Similarity and Exact Match only (RMSE hidden — not meaningful for text)."),
            ("Numerical only",    "Computes Exact Match and RMSE only (Semantic Similarity hidden — redundant for numbers)."),
            ("Manual selection",  "User picks specific columns. All three metrics shown."),
        ]
        for mode, desc in col_modes:
            st.markdown(f'<div class="doc-kv"><span class="doc-kv-key">{mode}</span><span class="doc-kv-val">{desc}</span></div>', unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════
    # TAB 5 — SETTINGS & TIPS
    # ══════════════════════════════════════════════════════════════════
    with tabs[4]:
        st.markdown('<div class="doc-h2">Recommended workflow for benchmarking</div>', unsafe_allow_html=True)
        workflow = [
            ("Start with a complete dataset",  "Use the Inject Nulls page to introduce controlled MCAR missingness. Save the original (ground truth) and the incomplete version separately."),
            ("Impute with SENTI",              "Go to the Imputation page, upload the incomplete dataset, select SENTI + LaBSE, run. Download the imputed CSV."),
            ("Evaluate",                       "Go to Evaluation, upload all three files. Review semantic similarity, exact match, and RMSE."),
            ("Compare baselines",              "Repeat imputation with Mean/Median/Mode. Compare metrics across imputers."),
        ]
        for i, (t, d) in enumerate(workflow, 1):
            st.markdown(f"""
            <div class="doc-step-row">
              <div class="doc-step-num">{i}</div>
              <div class="doc-step-body"><h4>{t}</h4><p>{d}</p></div>
            </div>""", unsafe_allow_html=True)

        st.markdown('<div class="doc-h2">Performance tips</div>', unsafe_allow_html=True)
        tips = [
            ("Use LaBSE for multilingual or mixed data", "LaBSE handles Italian, English, numbers, and codes robustly. Use all-MiniLM-L6-v2 only for speed-sensitive pure-English datasets."),
            ("Select only columns that need imputation", "Excluding already-complete columns from the multiselect keeps the pipeline focused and faster."),
            ("Inject nulls at moderate fractions", "10–30% null fraction per column gives realistic benchmarking. Very high fractions (>50%) leave too few neighbors for reliable imputation."),
            ("Use Incremental mode for streaming data", "If data arrives in periodic batches, Incremental mode avoids re-embedding all historical rows on each update."),
            ("Reset session between experiments", "Use the ↺ Reset Session button in the sidebar to clear all state between benchmark runs."),
        ]
        for tip, desc in tips:
            st.markdown(f'<div class="doc-kv"><span class="doc-kv-key">{tip}</span><span class="doc-kv-val">{desc}</span></div>', unsafe_allow_html=True)

        st.markdown('<div class="doc-h2">Known limitations</div>', unsafe_allow_html=True)
        limits = [
            ("Single-table scope (single-table mode)", "The Imputation page operates on one flat table at a time."),
            ("In-memory only", "All data and FAISS indexes are held in session memory. Very large datasets (>500k rows) may exceed available RAM."),
            ("Exact FAISS index", "IndexFlatIP is exact and O(n) per query. For datasets >100k rows, approximate indexes (IVFFlat, HNSW) would be faster."),
            
            ("MCAR null injection only", "The Inject Nulls page simulates Missing Completely At Random. MAR and MNAR patterns are not currently supported."),
        ]
        for lim, desc in limits:
            st.markdown(f'<div class="doc-kv"><span class="doc-kv-key">{lim}</span><span class="doc-kv-val">{desc}</span></div>', unsafe_allow_html=True)

        st.markdown('<div class="doc-h2">Contact & citation</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="doc-p">
        Developed at the <strong>Department of Informatics, Modeling, Electronics and System
        Engineering (DIMES)</strong>, University of Calabria.<br>
        Contact: <a href="mailto:mahmood.tariq@dimes.unical.it" style="color:var(--accent2)">mahmood.tariq@dimes.unical.it</a><br>
        Profile: <a href="https://dottorato.dimes.unical.it/students/mahmood-tariq" style="color:var(--accent2)" target="_blank">dottorato.dimes.unical.it</a>
        </div>
        """, unsafe_allow_html=True)


def page_eval():
    #_section("Evaluation", badge="")

   # st.markdown("""
   # <div class="SENTI-card">
     #   <p>Upload three aligned CSVs (same columns, same row count) to compute imputation quality metrics 
     #   only at originally missing cell positions.</p>
    #</div>
   # """, unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown('<div class="SENTI-card-label">Incomplete CSV</div>', unsafe_allow_html=True)
        inc_file = st.file_uploader("Incomplete", type=["csv","tsv","json","jsonl","xlsx","xls","parquet"], key="upl_incomplete_eval", label_visibility="collapsed")
    with c2:
        st.markdown('<div class="SENTI-card-label">Imputed CSV</div>', unsafe_allow_html=True)
        imp_file = st.file_uploader("Imputed", type=["csv","tsv","json","jsonl","xlsx","xls","parquet"], key="upl_imputed_eval", label_visibility="collapsed")
    with c3:
        st.markdown('<div class="SENTI-card-label">Ground Truth CSV</div>', unsafe_allow_html=True)
        gt_file = st.file_uploader("Ground Truth", type=["csv","tsv","json","jsonl","xlsx","xls","parquet"], key="upl_gt_eval", label_visibility="collapsed")

    if not (inc_file and imp_file and gt_file):
        _empty_state("📊", "Three files needed",
            "Upload the Incomplete, Imputed, and Ground Truth CSVs above — all must have matching columns and row counts.")
        return

    try:
        inc_df = _read_uploaded_file(inc_file)
        imp_df = _read_uploaded_file(imp_file)
        gt_df  = _read_uploaded_file(gt_file)
    except Exception as e:
        st.error(f"Failed to read one of the files: {e}")
        return

    if not (list(inc_df.columns) == list(imp_df.columns) == list(gt_df.columns)):
        st.error("Column order/names differ across the three CSVs.")
        return
    if not (len(inc_df) == len(imp_df) == len(gt_df)):
        st.error("Row counts differ across the three CSVs.")
        return

    missing_mask = inc_df.isna()
    import pandas.api.types as ptypes
    num_cols = [c for c in gt_df.columns if ptypes.is_numeric_dtype(gt_df[c])]
    cat_cols = [c for c in gt_df.columns if c not in num_cols]
    all_cols = list(gt_df.columns)

    _section("Column Selection", badge="FILTER")

    sel_mode = st.radio(
        "",
        ["Select all", "Categorical only", "Numerical only", "Manual selection"],
        horizontal=True,
        key="eval_col_mode",
    )
    if sel_mode == "Select all":
        selected_cols = all_cols
    elif sel_mode == "Categorical only":
        selected_cols = cat_cols
    elif sel_mode == "Numerical only":
        selected_cols = num_cols
    else:
        selected_cols = st.multiselect("Pick columns", all_cols, default=all_cols, key="eval_cols_manual")

    if not selected_cols:
        st.warning("No columns selected.")
        return

    imp_use = imp_df.copy()
    gt_use = gt_df.copy()
    missing_mask = inc_df[selected_cols].isna()
    missing_mask = missing_mask.reindex_like(imp_use).fillna(False)

    import modules.eval_metrics as evalm
    import modules.highlight as hl

    _section("Metric Settings", badge="CONFIG")

    c_model_col, c_norm_col = st.columns(2)
    with c_model_col:
        st.markdown('<div class="SENTI-card-label">Embedding Model</div>', unsafe_allow_html=True)
        model_options = [
            "sentence-transformers/LaBSE",
            "sentence-transformers/all-MiniLM-L6-v2",
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        ]
        model_name = st.selectbox("Model", model_options, index=0, key="eval_sem_model", label_visibility="collapsed")

    with c_norm_col:
        st.markdown('<div class="SENTI-card-label">RMSE Normalization</div>', unsafe_allow_html=True)
        norm_method = st.selectbox(
            "Normalization",
            ["None", "Min-Max [0,1]", "Z-score (mean/std)", "Robust (median/IQR)", "Divide by constant", "User min/max"],
            key="rmse_norm_method_top",
            label_visibility="collapsed",
        )

    rmse_params = {}
    if norm_method == "Divide by constant":
        rmse_params["divide_by"] = st.number_input("Divide by constant", min_value=0.0, value=1.0, step=0.1, key="rmse_divide_top")
    elif norm_method == "User min/max":
        c1_, c2_ = st.columns(2)
        rmse_params["user_min"] = c1_.number_input("User min", value=0.0, step=0.1, key="rmse_user_min_top")
        rmse_params["user_max"] = c2_.number_input("User max", value=1.0, step=0.1, key="rmse_user_max_top")

    # Determine which buttons to show
    _show_semantic = sel_mode in ("Select all", "Manual selection", "Categorical only")
    _show_exact = True
    _show_rmse = sel_mode in ("Select all", "Manual selection", "Numerical only")

    _section("Compute Metrics", badge="RUN")

    do_sem = do_exact = do_rmse = False
    _btn_defs = []
    if _show_semantic:
        _btn_defs.append(("◈  Semantic Similarity", "btn_compute_sem_one", "sem"))
    if _show_exact:
        _btn_defs.append(("◉  Exact Matches", "btn_compute_exact_one", "exact"))
    if _show_rmse:
        _btn_defs.append(("△  RMSE", "btn_compute_rmse_one", "rmse"))

    if _btn_defs:
        # Build column spec — only add a spacer column when there is room,
        # and ensure every width value is strictly positive (0 is invalid).
        _spacer = 3 - len(_btn_defs)
        _col_spec = [1] * len(_btn_defs) + ([_spacer] if _spacer > 0 else [])
        _cols = st.columns(_col_spec)
        for (_label, _key, _tag), _col in zip(_btn_defs, _cols):
            with _col:
                if _tag == "sem":
                    if st.button(_label, type="primary", key=_key):
                        do_sem = True
                elif _tag == "exact":
                    if st.button(_label, type="primary", key=_key):
                        do_exact = True
                elif _tag == "rmse":
                    if st.button(_label, type="primary", key=_key):
                        do_rmse = True

    if do_sem:
        with st.spinner("Computing semantic similarity…"):
            try:
                out = evalm.semantic_similarity_at_positions(
                    imputed_df=imp_use, ground_df=gt_use, missing_mask=missing_mask, model_name=model_name,
                )
                mean_sim = out['overall']['mean']
                cell_scores = out["cell_scores"]
                cell_scores = ((cell_scores.astype(float) + 1.0) / 2.0).clip(lower=0.0, upper=1.0)

                _section("Semantic Similarity Results", badge="OUTPUT")
                st.markdown(f"""
                <div class="SENTI-stat-row">
                  <div class="SENTI-stat">
                    <span class="stat-label">Average Similarity</span>
                    <span class="stat-val" style="color:var(--green)">{mean_sim:.4f}</span>
                    <span class="stat-sub">across {out['overall']['count']} imputed cells</span>
                  </div>
                </div>
                """, unsafe_allow_html=True)

                st.markdown("""
                <div class="compare-head">
                    <div class="compare-head-item imputed">◈ Imputed (color-coded)</div>
                    <div class="compare-head-item ground">◉ Ground Truth</div>
                </div>
                <div class="sim-legend" style="margin-bottom:0.6rem;">
                    <span><div class="sim-dot" style="background:rgba(220,53,69,0.6)"></div>0.0–0.7 low</span>
                    <span><div class="sim-dot" style="background:rgba(255,159,67,0.6)"></div>0.7–0.95 medium</span>
                    <span><div class="sim-dot" style="background:rgba(46,204,113,0.6)"></div>0.95–1.0 high</span>
                </div>
                """, unsafe_allow_html=True)
                colL, colR = st.columns(2)
                with colL:
                    try:
                        styler = hl.style_similarity_bins(imp_use, cell_scores, missing_mask)
                        st.dataframe(styler, use_container_width=True)
                    except Exception:
                        st.dataframe(imp_use, use_container_width=True)
                with colR:
                    st.dataframe(gt_use, use_container_width=True)
            except Exception as e:
                st.error(f"Semantic similarity failed: {e}")

    if do_exact:
        with st.spinner("Computing exact matches…"):
            try:
                out = evalm.exact_match_at_positions(imp_use, gt_use, missing_mask)
                rate = out["overall"]["rate"]
                _section("Exact Match Results", badge="OUTPUT")
                st.markdown(f"""
                <div class="SENTI-stat-row">
                  <div class="SENTI-stat">
                    <span class="stat-label">Overall Exact Match Rate</span>
                    <span class="stat-val" style="color:var(--accent2)">{rate*100:.2f}%</span>
                    <span class="stat-sub">{out['overall']['true']} / {out['overall']['total']} cells</span>
                  </div>
                </div>
                """, unsafe_allow_html=True)
                st.dataframe(out["per_column"], use_container_width=True)
            except Exception as e:
                st.error(f"Exact match failed: {e}")

    if do_rmse:
        num_c = [c for c in imp_use.columns if ptypes.is_numeric_dtype(gt_use[c])]
        if not num_c:
            st.warning("No numeric columns among the selected columns.")
        else:
            with st.spinner("Computing RMSE…"):
                try:
                    gt_n, imp_n = _normalize_pair(gt_use, imp_use, num_c, norm_method, rmse_params)
                    table, overall_rmse, _ = _rmse_mae_report(gt_n, imp_n, num_c)
                    _section("RMSE Results", badge="OUTPUT")
                    st.markdown(f"""
                    <div class="SENTI-stat-row">
                      <div class="SENTI-stat">
                        <span class="stat-label">Overall RMSE</span>
                        <span class="stat-val" style="color:var(--gold)">{overall_rmse:.6f}</span>
                        <span class="stat-sub">mean across numeric columns</span>
                      </div>
                    </div>
                    """, unsafe_allow_html=True)
                    st.dataframe(table["RMSE"].to_frame().style.format({"RMSE": "{:.6f}"}), use_container_width=True)
                except Exception as e:
                    st.error(f"RMSE computation failed: {e}")

    st.caption(
        "Only originally missing cells are evaluated. Similarity uses cosine distance scaled to [0,1] via sentence-transformers."
    )


def main():
    init_state()
    _inject_master_css()

    page = getattr(ss, "page", "Imputation")
    active = page if not getattr(ss, "_doc", False) else "Imputation"
    _render_topbar(active_page=active)

    try:
        render_sidebar()
    except Exception:
        pass

    if getattr(ss, "_doc", False) or str(page) == "Guide":
        page_docs()
    elif str(page) == "Evaluation":
        page_eval()
    elif str(page) == "Inject nulls":
        page_null_injection()
    else:
        page_senti()

    _render_footer()


if __name__ == "__main__":
    main()


