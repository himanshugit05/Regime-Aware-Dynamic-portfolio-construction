# =============================================================================
# core/models/features.py  —  Feature Engineering for Regime Detection
#
# Feature vector X_t ∈ ℝ¹⁶:
#  Returns    : r_5d, r_21d, r_63d
#  Volatility : vol_21d, vol_63d, vol_ratio (spike detector)
#  Momentum   : momentum_12_1 (Jegadeesh-Titman)
#  India VIX  : percentile rank, 5d change
#  US VIX     : percentile rank, 5d change
#  DXY        : percentile rank, 21d return
#  Crude      : vol percentile rank, 21d return
#  Sentiment  : composite Ct ∈ [0,1]  (VIX + DXY + Crude + NLP)
# =============================================================================

import numpy as np
import pandas as pd
from scipy import stats
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from core.config import PERCENTILE_WINDOW, SENTIMENT_WEIGHTS


# ── Percentile rank (rolling) ─────────────────────────────────────────────────

def _pct_rank(s: pd.Series, window: int = PERCENTILE_WINDOW) -> pd.Series:
    def _last_rank(arr):
        return stats.percentileofscore(arr, arr[-1], kind="rank") / 100.0
    return s.rolling(window, min_periods=window // 4).apply(_last_rank, raw=True)


# ── Align helper ──────────────────────────────────────────────────────────────

def _align(series, index):
    if series is None:
        return None
    return series.reindex(index, method="ffill", limit=5)


# ── Core feature builder ───────────────────────────────────────────────────────

def build_features(price_df: pd.DataFrame, nlp_market_score: float = 0.5) -> pd.DataFrame:
    """
    Build the full feature DataFrame from price_df.

    Parameters
    ----------
    price_df          : wide DataFrame with columns including 'nifty50',
                        'india_vix', 'us_vix', 'dxy', 'crude'
    nlp_market_score  : market-wide NLP fear score [0,1]

    Returns
    -------
    features : pd.DataFrame with index = trading dates
    """
    if "nifty50" not in price_df.columns:
        raise ValueError("price_df must contain 'nifty50' column.")

    nifty = price_df["nifty50"].dropna()
    daily = nifty.pct_change()
    feat  = pd.DataFrame(index=nifty.index)

    # ── Returns ───────────────────────────────────────────────────────────────
    feat["r_5d"]  = nifty.pct_change(5)
    feat["r_21d"] = nifty.pct_change(21)
    feat["r_63d"] = nifty.pct_change(63)

    # ── Realised Volatility ────────────────────────────────────────────────────
    feat["vol_21d"]  = daily.rolling(21).std()  * np.sqrt(252)
    feat["vol_63d"]  = daily.rolling(63).std()  * np.sqrt(252)
    feat["vol_ratio"]= feat["vol_21d"] / (feat["vol_63d"] + 1e-8)

    # ── Momentum (skip 1 month to avoid reversal) ────────────────────────────
    feat["momentum"] = nifty.pct_change(252) - nifty.pct_change(21)

    # ── India VIX ──────────────────────────────────────────────────────────────
    vix_in = _align(price_df.get("india_vix"), nifty.index)
    if vix_in is not None and not vix_in.isna().all():
        feat["vix_in_pct"] = _pct_rank(vix_in)
        feat["vix_in_chg"] = vix_in.pct_change(5)
    else:
        approx = feat["vol_21d"] * 100
        feat["vix_in_pct"] = _pct_rank(approx)
        feat["vix_in_chg"] = approx.pct_change(5)
        vix_in = approx

    # ── US VIX ─────────────────────────────────────────────────────────────────
    vix_us = _align(price_df.get("us_vix"), nifty.index)
    if vix_us is not None and not vix_us.isna().all():
        feat["vix_us_pct"] = _pct_rank(vix_us)
        feat["vix_us_chg"] = vix_us.pct_change(5)
    else:
        feat["vix_us_pct"] = feat["vix_in_pct"]
        feat["vix_us_chg"] = feat["vix_in_chg"]

    # ── DXY ─────────────────────────────────────────────────────────────────────
    dxy = _align(price_df.get("dxy"), nifty.index)
    if dxy is not None and not dxy.isna().all():
        feat["dxy_pct"] = _pct_rank(dxy)
        feat["dxy_ret"] = dxy.pct_change(21)
    else:
        feat["dxy_pct"] = pd.Series(0.5, index=nifty.index)
        feat["dxy_ret"] = pd.Series(0.0, index=nifty.index)

    # ── Crude Oil ──────────────────────────────────────────────────────────────
    crude = _align(price_df.get("crude"), nifty.index)
    if crude is not None and not crude.isna().all():
        cvol = crude.pct_change().rolling(21).std() * np.sqrt(252)
        feat["crude_vol_pct"] = _pct_rank(cvol)
        feat["crude_ret"]     = crude.pct_change(21)
    else:
        feat["crude_vol_pct"] = feat["vol_21d"] / (feat["vol_21d"].rolling(252).max() + 1e-8)
        feat["crude_ret"]     = pd.Series(0.0, index=nifty.index)

    # ── Composite Sentiment Score ──────────────────────────────────────────────
    # Ct = w_vin·VIXin + w_vus·VIXus + w_dxy·DXY + w_cv·CrudeVol + w_nlp·NLP
    feat["sentiment"] = _composite_sentiment(feat, nlp_market_score)

    # Store raw VIX for display
    feat["india_vix_raw"] = _align(price_df.get("india_vix"), nifty.index)

    return feat.dropna(subset=["r_21d", "vol_21d", "momentum", "vix_in_pct", "sentiment"])


def _composite_sentiment(feat: pd.DataFrame, nlp_score: float) -> pd.Series:
    w    = SENTIMENT_WEIGHTS
    cols = {
        "india_vix": "vix_in_pct",
        "us_vix":    "vix_us_pct",
        "dxy":       "dxy_pct",
        "crude":     "crude_vol_pct",
    }
    sent   = pd.Series(0.0, index=feat.index)
    total  = 0.0

    for key, col in cols.items():
        wt = w.get(key, 0)
        if col in feat.columns:
            sent  += wt * feat[col].fillna(0.5)
            total += wt

    # Add static NLP score
    nlp_wt = w.get("nlp", 0)
    sent   += nlp_wt * nlp_score
    total  += nlp_wt

    return (sent / max(total, 1e-8)).clip(0.0, 1.0)


# ── Model feature columns ──────────────────────────────────────────────────────

MODEL_COLS = [
    "r_5d","r_21d","r_63d",
    "vol_21d","vol_63d","vol_ratio",
    "momentum",
    "vix_in_pct","vix_in_chg",
    "vix_us_pct","vix_us_chg",
    "dxy_pct","dxy_ret",
    "crude_vol_pct","crude_ret",
    "sentiment",
]


def get_model_X(feat: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in MODEL_COLS if c in feat.columns]
    return feat[cols].dropna()
