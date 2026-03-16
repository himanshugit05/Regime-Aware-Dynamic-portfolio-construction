# =============================================================================
# core/config.py  —  Single Source of Truth for Regime Portfolio v2
# =============================================================================

from pathlib import Path

# ── Project Paths ──────────────────────────────────────────────────────────────
ROOT_DIR  = Path(__file__).parent.parent
CACHE_DIR = ROOT_DIR / ".cache"
CACHE_DIR.mkdir(exist_ok=True)

# ── Sector Universe ────────────────────────────────────────────────────────────
SECTORS = {
    "bank":     {"ticker": "^NSEBANK",   "label": "Nifty Bank",        "color": "#1565C0", "yf_query": "HDFC Bank Nifty Bank India"},
    "it":       {"ticker": "^CNXIT",     "label": "Nifty IT",          "color": "#00897B", "yf_query": "Nifty IT technology India Infosys TCS"},
    "auto":     {"ticker": "^CNXAUTO",   "label": "Nifty Auto",        "color": "#F57F17", "yf_query": "Nifty Auto automobile India Maruti"},
    "pharma":   {"ticker": "^CNXPHARMA", "label": "Nifty Pharma",      "color": "#AD1457", "yf_query": "Nifty Pharma pharmaceutical India Sun Pharma"},
    "fmcg":     {"ticker": "^CNXFMCG",   "label": "Nifty FMCG",        "color": "#558B2F", "yf_query": "Nifty FMCG consumer India HUL"},
    "energy":   {"ticker": "^CNXENERGY", "label": "Nifty Energy",      "color": "#E65100", "yf_query": "Nifty Energy oil gas India Reliance"},
    "infra":    {"ticker": "^CNXINFRA",  "label": "Nifty Infra",       "color": "#4527A0", "yf_query": "Nifty Infrastructure India LT construction"},
    "metal":    {"ticker": "^CNXMETAL",  "label": "Nifty Metal",       "color": "#546E7A", "yf_query": "Nifty Metal steel India Tata Steel"},
    "realty":   {"ticker": "^CNXREALTY", "label": "Nifty Realty",      "color": "#6D4C41", "yf_query": "Nifty Realty real estate India DLF"},
    "consumer": {"ticker": "^CNXCONSUM", "label": "Nifty Consumer",    "color": "#00838F", "yf_query": "Nifty Consumer durables India Titan"},
}

# ── Macro Tickers ──────────────────────────────────────────────────────────────
MACRO = {
    "nifty50":    "^NSEI",
    "nifty_mid":  "^NSEMDCP50",
    "nifty_small":"^CNXSC250",
    "india_vix":  "^INDIAVIX",
    "us_vix":     "^VIX",
    "dxy":        "DX-Y.NYB",
    "crude":      "CL=F",
    "gold":       "GOLDBEES.NS",
    "silver":     "SILVERBEES.NS",
}

MACRO_FALLBACKS = {
    "gold":        ["GC=F"],
    "silver":      ["SI=F"],
    "nifty_mid":   ["MIDCAP150BEES.NS", "MID150BEES.NS"],
    "nifty_small": ["NIFTYSML250.NS",   "SETFNN50.NS"],
}

SECTOR_FALLBACKS = {
    "bank":     ["BANKBEES.NS",   "HDFCBANK.NS"],
    "it":       ["ITBEES.NS",     "INFY.NS"],
    "pharma":   ["PHARMABEES.NS", "SUNPHARMA.NS"],
    "fmcg":     ["FMCGIETF.NS",   "HINDUNILVR.NS"],
    "auto":     ["AUTOBEES.NS",   "MARUTI.NS"],
    "energy":   ["ENERGIETF.NS",  "RELIANCE.NS"],
    "infra":    ["INFRABEES.NS",  "LT.NS"],
    "metal":    ["METALBEES.NS",  "TATASTEEL.NS"],
    "realty":   ["NIFTYREALTY.NS","DLF.NS"],
    "consumer": ["CONSUMPTION.NS","TITAN.NS"],
}

# ── Regime Definitions ─────────────────────────────────────────────────────────
REGIMES     = ["Bull", "Sideways", "Bear"]
N_REGIMES   = 3

REGIME_COLORS = {
    "Bull":     "#00C853",
    "Sideways": "#FFD600",
    "Bear":     "#FF1744",
}
REGIME_BG = {
    "Bull":     "rgba(0,200,83,0.10)",
    "Sideways": "rgba(255,214,0,0.10)",
    "Bear":     "rgba(255,23,68,0.10)",
}

# ── Regime → Bucket Weights ────────────────────────────────────────────────────
REGIME_BUCKETS = {
    "Bull":     {"sectors": 0.72, "nifty_floor": 0.08, "gold": 0.07, "silver": 0.03, "cash": 0.10},
    "Sideways": {"sectors": 0.45, "nifty_floor": 0.05, "gold": 0.17, "silver": 0.05, "cash": 0.28},
    "Bear":     {"sectors": 0.15, "nifty_floor": 0.05, "gold": 0.28, "silver": 0.10, "cash": 0.42},
}

# ── Sector Regime Priors ───────────────────────────────────────────────────────
SECTOR_PRIORS = {
    "Bull": {
        "bank":0.25,"it":0.18,"auto":0.18,"metal":0.14,
        "realty":0.10,"infra":0.08,"energy":0.07,
        "consumer":0.00,"pharma":0.00,"fmcg":0.00,
    },
    "Sideways": {
        "it":0.25,"fmcg":0.20,"pharma":0.18,"consumer":0.12,
        "energy":0.12,"bank":0.08,"infra":0.05,
        "auto":0.00,"metal":0.00,"realty":0.00,
    },
    "Bear": {
        "pharma":0.38,"fmcg":0.35,"it":0.15,"energy":0.12,
        "consumer":0.00,"bank":0.00,"auto":0.00,
        "metal":0.00,"realty":0.00,"infra":0.00,
    },
}

# ── Optimiser Constraints ──────────────────────────────────────────────────────
SECTOR_MAX_WEIGHT = 0.45
PRIOR_BLEND       = 0.40     # α: how much regime prior vs historical data

# ── Sentiment ──────────────────────────────────────────────────────────────────
SENTIMENT_WEIGHTS = {"india_vix": 0.30, "us_vix": 0.20, "dxy": 0.15, "crude": 0.15, "nlp": 0.20}
SENTIMENT_LAMBDA  = 0.35
PERCENTILE_WINDOW = 252

# ── NLP Sentiment ──────────────────────────────────────────────────────────────
NLP_NEWS_LIMIT         = 20      # headlines per sector per fetch
NLP_CACHE_HOURS        = 6       # re-fetch after 6 hours
NLP_MARKET_QUERY       = "India stock market Nifty economy"

# ── Model Hyperparameters ──────────────────────────────────────────────────────
HMM_N_RESTARTS   = 10
HMM_N_ITER       = 500
GMM_N_INIT       = 20
TRAIN_RATIO      = 0.80
MIN_TRAIN_DAYS   = 504

# ── Backtest ───────────────────────────────────────────────────────────────────
REBAL_WINDOWS = {"3 Months":63, "4 Months":84, "6 Months":126, "12 Months":252}
CASH_RATE     = 0.065
RISK_FREE     = 0.065

# ── Monte Carlo ────────────────────────────────────────────────────────────────
MC_SIMULATIONS  = 1000
MC_HORIZON_DAYS = 252      # 1 year forward

# ── Data Quality ──────────────────────────────────────────────────────────────
MAX_MOVE_INDEX = 0.15
MAX_MOVE_ETF   = 0.10

# ── Chart Theme ───────────────────────────────────────────────────────────────
CHART_TEMPLATE  = "plotly_dark"
CHART_FONT      = "Inter"
CHART_BG        = "rgba(0,0,0,0)"
CHART_PAPER_BG  = "rgba(17,17,17,1)"
PRIMARY_COLOR   = "#667eea"
ACCENT_COLOR    = "#764ba2"
POSITIVE_COLOR  = "#00C853"
NEGATIVE_COLOR  = "#FF1744"
NEUTRAL_COLOR   = "#FFD600"
