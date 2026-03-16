# =============================================================================
# core/data/sentiment_nlp.py  —  NLP Sector Sentiment via VADER + yfinance news
#
# Architecture
# ────────────
#  1. Fetch recent news headlines for each sector via yfinance Ticker.news
#  2. Run VADER (Valence Aware Dictionary for Sentiment Reasoning) on each headline
#  3. Aggregate to sector-level sentiment score ∈ [-1, 1]
#  4. Normalise to [0, 1] where 0=bullish, 1=bearish (matches fear convention)
#  5. Cache results for NLP_CACHE_HOURS to avoid rate-limiting
#
# VADER is rule-based (no API needed, no LLM) — fast and works offline
# after the nltk data is downloaded once.
# =============================================================================

import json
import time
import hashlib
import warnings
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from core.config import (
    SECTORS, NLP_NEWS_LIMIT, NLP_CACHE_HOURS,
    NLP_MARKET_QUERY, CACHE_DIR
)

NLP_CACHE_FILE = CACHE_DIR / "nlp_sentiment.json"

# ── VADER setup ───────────────────────────────────────────────────────────────

def _load_vader():
    """Load VADER — download nltk data only if missing."""
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        return SentimentIntensityAnalyzer()
    except ImportError:
        pass
    try:
        import nltk
        from nltk.sentiment.vader import SentimentIntensityAnalyzer
        try:
            return SentimentIntensityAnalyzer()
        except LookupError:
            nltk.download("vader_lexicon", quiet=True)
            return SentimentIntensityAnalyzer()
    except Exception:
        return None


# ── News fetcher ──────────────────────────────────────────────────────────────

def _fetch_headlines(query: str, limit: int = NLP_NEWS_LIMIT) -> list[str]:
    """
    Fetch news headlines using yfinance search.
    Returns list of headline strings.
    """
    headlines = []
    try:
        import yfinance as yf
        # yfinance search for news
        ticker_obj = yf.Search(query, news_count=limit)
        news = ticker_obj.news if hasattr(ticker_obj, "news") else []
        for item in news[:limit]:
            title = item.get("title", "") or item.get("headline", "")
            if title:
                headlines.append(str(title))
    except Exception:
        pass

    # Fallback: try direct ticker news
    if not headlines:
        try:
            import yfinance as yf
            # Use Nifty 50 news as broad market proxy
            t = yf.Ticker("^NSEI")
            news = getattr(t, "news", []) or []
            for item in news[:limit]:
                title = item.get("title", "")
                if title:
                    headlines.append(str(title))
        except Exception:
            pass

    return headlines


# ── Sector score ──────────────────────────────────────────────────────────────

def _score_headlines(headlines: list[str], vader) -> float:
    """
    Compute mean compound VADER score across headlines.
    Returns float ∈ [-1, +1]. Positive = bullish, Negative = bearish.
    """
    if not headlines or vader is None:
        return 0.0
    scores = [vader.polarity_scores(h)["compound"] for h in headlines]
    return float(np.mean(scores)) if scores else 0.0


def _normalize_to_fear(score: float) -> float:
    """
    Convert VADER compound [-1,+1] → fear score [0,1].
    score=-1 (very negative) → fear=1.0
    score=+1 (very positive) → fear=0.0
    """
    return float(np.clip((1.0 - score) / 2.0, 0.0, 1.0))


# ── Cache ─────────────────────────────────────────────────────────────────────

def _load_nlp_cache() -> dict:
    try:
        if NLP_CACHE_FILE.exists():
            data = json.loads(NLP_CACHE_FILE.read_text())
            age_hours = (time.time() - data.get("timestamp", 0)) / 3600
            if age_hours < NLP_CACHE_HOURS:
                return data
    except Exception:
        pass
    return {}


def _save_nlp_cache(data: dict) -> None:
    try:
        data["timestamp"] = time.time()
        NLP_CACHE_FILE.write_text(json.dumps(data, indent=2))
    except Exception:
        pass


# ── Public API ────────────────────────────────────────────────────────────────

def get_sector_sentiment(progress_fn=None) -> dict[str, float]:
    """
    Returns dict: {sector_key: fear_score ∈ [0,1]}
    0 = very bullish sentiment, 1 = very bearish / crisis sentiment.

    Uses cache if fresh (< NLP_CACHE_HOURS old).
    Falls back to neutral (0.5) if news unavailable.
    """
    cached = _load_nlp_cache()
    if cached.get("scores"):
        return cached["scores"]

    vader = _load_vader()
    scores: dict[str, float] = {}
    n = len(SECTORS)

    for i, (sector_key, meta) in enumerate(SECTORS.items()):
        if progress_fn:
            progress_fn(i / n, f"NLP: analysing {meta['label']}…")
        try:
            headlines = _fetch_headlines(meta["yf_query"], NLP_NEWS_LIMIT)
            raw_score = _score_headlines(headlines, vader)
            scores[sector_key] = _normalize_to_fear(raw_score)
        except Exception:
            scores[sector_key] = 0.5    # neutral fallback

    # Market-wide sentiment
    try:
        mkt_headlines = _fetch_headlines(NLP_MARKET_QUERY, NLP_NEWS_LIMIT)
        mkt_score = _score_headlines(mkt_headlines, vader)
        scores["market"] = _normalize_to_fear(mkt_score)
    except Exception:
        scores["market"] = 0.5

    if progress_fn:
        progress_fn(1.0, "NLP sentiment complete ✓")

    _save_nlp_cache({"scores": scores})
    return scores


def get_market_nlp_score() -> float:
    """Return single market-wide NLP fear score [0,1]."""
    scores = get_sector_sentiment()
    return scores.get("market", 0.5)


def get_sentiment_summary(scores: dict[str, float]) -> pd.DataFrame:
    """Return a formatted DataFrame of NLP sentiment scores per sector."""
    rows = []
    for key, score in scores.items():
        if key == "market":
            continue
        label = SECTORS.get(key, {}).get("label", key)
        sentiment = "🟢 Bullish" if score < 0.35 else ("🔴 Bearish" if score > 0.65 else "🟡 Neutral")
        rows.append({"Sector": label, "Fear Score": round(score, 3), "Sentiment": sentiment})
    return pd.DataFrame(rows).sort_values("Fear Score", ascending=True)
