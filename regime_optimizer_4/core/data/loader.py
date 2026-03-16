# =============================================================================
# core/data/loader.py  —  Download, Clean, Cache (Parquet)
# =============================================================================

import pandas as pd
import numpy as np
import yfinance as yf
from pathlib import Path
from datetime import timedelta
import hashlib, warnings
warnings.filterwarnings("ignore")

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from core.config import (
    SECTORS, MACRO, MACRO_FALLBACKS, SECTOR_FALLBACKS,
    MAX_MOVE_INDEX, MAX_MOVE_ETF, CACHE_DIR
)


# ══════════════════════════════════════════════════════════════════════════════
# Cache helpers
# ══════════════════════════════════════════════════════════════════════════════

def _cache_key(start: str, end: str) -> str:
    return hashlib.md5(f"{start}{end}".encode()).hexdigest()[:10]

def _cache_path(key: str) -> Path:
    return CACHE_DIR / f"prices_{key}.parquet"

def _load_cache(start: str, end: str) -> "pd.DataFrame | None":
    p = _cache_path(_cache_key(start, end))
    if p.exists():
        try:
            return pd.read_parquet(p)
        except Exception:
            p.unlink(missing_ok=True)
    return None

def _save_cache(df: pd.DataFrame, start: str, end: str) -> None:
    try:
        df.to_parquet(_cache_path(_cache_key(start, end)))
    except Exception:
        pass

def clear_cache() -> None:
    for f in CACHE_DIR.glob("prices_*.parquet"):
        f.unlink(missing_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
# Core download helpers
# ══════════════════════════════════════════════════════════════════════════════

def _download(ticker: str, start: str, end: str) -> "pd.Series | None":
    try:
        raw = yf.download(ticker, start=start, end=end,
                          auto_adjust=True, progress=False, threads=False)
        if raw.empty:
            return None
        if isinstance(raw.columns, pd.MultiIndex):
            raw = raw["Close"].iloc[:, 0]
        else:
            raw = raw["Close"]
        raw = raw.dropna().astype(float)
        return raw if len(raw) > 20 else None
    except Exception:
        return None


def _clean(series: pd.Series, max_move: float) -> pd.Series:
    """Remove bad ticks, negative prices, extreme outliers."""
    s = series.copy().astype(float)
    s[s <= 0] = np.nan
    med = s.rolling(63, min_periods=5).median()
    s[s < med * 0.01] = np.nan
    s[s.pct_change().abs() > max_move] = np.nan
    return s.ffill(limit=5).dropna()


def _try_tickers(tickers: list, start: str, end: str,
                 cap: float) -> "pd.Series | None":
    for t in tickers:
        raw = _download(t, start, end)
        if raw is not None:
            return _clean(raw, cap)
    return None


# ══════════════════════════════════════════════════════════════════════════════
# Main loader
# ══════════════════════════════════════════════════════════════════════════════

def load_all(start_date: str, end_date: str,
             use_cache: bool = True,
             progress_fn=None) -> pd.DataFrame:
    """
    Download all price series, clean, and return as a wide DataFrame.
    Uses parquet cache keyed by (start, end) — avoids re-downloading.

    Columns: sector keys + macro keys
    Index  : trading dates (buffer extends 500 days before start_date)
    """
    buffer = (pd.to_datetime(start_date) - timedelta(days=500)).strftime("%Y-%m-%d")

    if use_cache:
        cached = _load_cache(buffer, end_date)
        if cached is not None:
            if progress_fn:
                progress_fn(1.0, "Loaded from cache ✓")
            return cached

    prices: dict[str, pd.Series] = {}
    all_jobs = []

    # Build download job list
    for key, meta in SECTORS.items():
        all_jobs.append((key, [meta["ticker"]] + SECTOR_FALLBACKS.get(key, []),
                         MAX_MOVE_INDEX))
    for key, ticker in MACRO.items():
        cap = MAX_MOVE_ETF if key in {"gold","silver"} else MAX_MOVE_INDEX
        fallbacks = MACRO_FALLBACKS.get(key, [])
        all_jobs.append((key, [ticker] + fallbacks, cap))

    total = len(all_jobs)
    for i, (key, tickers, cap) in enumerate(all_jobs):
        if progress_fn:
            progress_fn(i / total, f"Downloading {key}…")
        result = _try_tickers(tickers, buffer, end_date, cap)
        if result is not None:
            prices[key] = result

    if not prices:
        raise RuntimeError("No price data downloaded. Check internet connection.")

    # Align to common index (forward-fill up to 5 days)
    df = pd.DataFrame(prices)
    df = df.ffill(limit=5)

    if progress_fn:
        progress_fn(1.0, "Saving cache…")
    _save_cache(df, buffer, end_date)
    return df


# ══════════════════════════════════════════════════════════════════════════════
# Slicers
# ══════════════════════════════════════════════════════════════════════════════

def get_sector_prices(df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    cols = [k for k in SECTORS if k in df.columns]
    return df.loc[start:end, cols].copy()

def get_portfolio_assets(df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    """Sectors + Nifty50 floor + gold + silver for backtester."""
    cols = [k for k in SECTORS if k in df.columns]
    for extra in ["nifty50", "gold", "silver"]:
        if extra in df.columns:
            cols.append(extra)
    return df.loc[start:end, cols].copy()

def get_benchmark(df: pd.DataFrame, key: str, start: str, end: str,
                  capital: float) -> "pd.Series | None":
    if key not in df.columns:
        return None
    s = df.loc[start:end, key].dropna()
    if len(s) < 2:
        return None
    return s / s.iloc[0] * capital

def get_nifty(df: pd.DataFrame) -> "pd.Series | None":
    return df["nifty50"].dropna() if "nifty50" in df.columns else None
