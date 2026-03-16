# =============================================================================
# main.py  —  Regime Portfolio v2  Entry Point
# Run: streamlit run main.py
# =============================================================================
import warnings
warnings.filterwarnings("ignore")

import streamlit as st
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent))

st.set_page_config(
    page_title="Regime Portfolio v2",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif !important; }

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d0d1a 0%, #12122a 100%);
    border-right: 1px solid #1e1e3a;
}
[data-testid="stSidebar"] * { color: #c8c8e8 !important; }

/* Main bg */
.stApp { background: #0a0a16; }

/* Metric cards */
[data-testid="stMetric"] {
    background: linear-gradient(135deg, #13132a, #1a1a35);
    border: 1px solid #2a2a50;
    border-radius: 10px;
    padding: 14px 18px !important;
}
[data-testid="stMetricValue"] { font-size: 1.4rem !important; color: #e8e8ff !important; }
[data-testid="stMetricLabel"] { color: #8888aa !important; font-size: .8rem !important; }
[data-testid="stMetricDelta"] { font-size: .85rem !important; }

/* Tabs */
button[data-baseweb="tab"] {
    font-family: Inter !important;
    font-weight: 500;
    font-size: .9rem;
    color: #8888cc;
    border-bottom: 2px solid transparent;
    padding-bottom: 8px;
}
button[data-baseweb="tab"][aria-selected="true"] {
    color: #a78bfa !important;
    border-bottom: 2px solid #a78bfa;
}

/* Headers */
h1,h2,h3 { color: #e0e0ff !important; font-family: Inter !important; }
h1 { font-size: 2rem !important; font-weight: 700 !important; }

/* Dataframes */
[data-testid="stDataFrame"] { border: 1px solid #1e1e3a; border-radius: 8px; }

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #667eea, #764ba2) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    font-size: 1rem !important;
    padding: 0.6rem 1.5rem !important;
    transition: opacity .2s;
}
.stButton > button:hover { opacity: 0.85; }

/* Regime tags */
.reg-bull { color:#00C853; font-weight:700; font-size:1.2rem; }
.reg-sideways { color:#FFD600; font-weight:700; font-size:1.2rem; }
.reg-bear { color:#FF1744; font-weight:700; font-size:1.2rem; }

/* Divider */
hr { border-color: #1e1e3a !important; }
</style>
""", unsafe_allow_html=True)

# ── Session state defaults ──────────────────────────────────────────────────────
defaults = {
    "results":      None,
    "regime_probs": None,
    "features":     None,
    "price_df":     None,
    "bench_series": None,
    "port_metrics": None,
    "mc_result":    None,
    "nlp_scores":   None,
    "detector":     None,
    "ran":          False,
}
for k,v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ── Home page ─────────────────────────────────────────────────────────────────
from ui.sidebar import render_sidebar
from ui.home    import render_home
from ui.runner  import run_pipeline

page = render_sidebar()

if page == "home":
    render_home()
elif page == "run":
    run_pipeline()
