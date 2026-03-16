# =============================================================================
# core/backtest/engine.py  —  Sector Optimizer + Walk-Forward Backtester
# =============================================================================

import numpy as np
import pandas as pd
from scipy.optimize import minimize
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from core.config import (
    SECTORS, REGIME_BUCKETS, SECTOR_PRIORS, SECTOR_MAX_WEIGHT,
    PRIOR_BLEND, SENTIMENT_LAMBDA, CASH_RATE, RISK_FREE,
    REBAL_WINDOWS, MIN_TRAIN_DAYS,
)

ALL_SECTORS = list(SECTORS.keys())


# ══════════════════════════════════════════════════════════════════════════════
# Sector Optimizer
# ══════════════════════════════════════════════════════════════════════════════

class SectorOptimizer:
    """
    3-step allocation:
      1. Blend regime bucket weights by posterior probabilities
      2. Max-Sharpe sector optimization with regime priors (BL-style)
      3. Sentiment overlay on defensives
    """

    def __init__(self, returns: pd.DataFrame,
                 sentiment_lambda: float = SENTIMENT_LAMBDA,
                 prior_blend: float = PRIOR_BLEND):
        ETF = {"gold", "silver"}
        self.ret = returns.copy()
        for c in self.ret.columns:
            cap = 0.10 if c in ETF else 0.15
            self.ret[c] = self.ret[c].clip(-cap, cap)
        self.lam        = sentiment_lambda
        self.prior_blend= prior_blend

    def allocate(self, regime_probs: dict, sentiment: float,
                 current_date: str, nlp_scores: "dict|None" = None) -> dict:
        # Step 1 — bucket weights
        buckets = self._blend_buckets(regime_probs)

        # Step 2 — sector weights within equity bucket
        sector_w = self._sector_max_sharpe(regime_probs, current_date, nlp_scores)

        # Step 3 — sentiment overlay
        buckets = self._sentiment_overlay(buckets, float(np.clip(sentiment, 0, 1)))

        # Assemble
        w = {s: sector_w.get(s, 0.0) * buckets["sectors"] for s in ALL_SECTORS}
        w["nifty50"] = buckets["nifty_floor"]
        w["gold"]    = buckets["gold"]
        w["silver"]  = buckets["silver"]
        w["cash"]    = buckets["cash"]

        total = sum(w.values())
        return {k: v/total for k,v in w.items()} if total > 0 else w

    # ── Step 1 ─────────────────────────────────────────────────────────────────

    def _blend_buckets(self, probs: dict) -> dict:
        keys = list(REGIME_BUCKETS["Bull"].keys())
        b = {k: 0.0 for k in keys}
        for r, p in probs.items():
            if r in REGIME_BUCKETS:
                for k in keys:
                    b[k] += p * REGIME_BUCKETS[r][k]
        return b

    # ── Step 2 ─────────────────────────────────────────────────────────────────

    def _sector_max_sharpe(self, regime_probs: dict, current_date: str,
                           nlp_scores: "dict|None") -> dict:
        top = max(regime_probs, key=regime_probs.get)
        active = [s for s in ALL_SECTORS if SECTOR_PRIORS[top].get(s, 0) > 0]
        if not active:
            active = ALL_SECTORS

        avail = [s for s in active if s in self.ret.columns]
        if len(avail) < 2:
            all_av = [s for s in ALL_SECTORS if s in self.ret.columns]
            n = len(all_av)
            return {s: 1/n for s in all_av} if n > 0 else {}

        hist = self.ret.loc[:current_date, avail].dropna()
        if len(hist) < 63:
            return self._prior_weights(regime_probs, avail)

        mu_hist = hist.mean().values * 252
        cov     = hist.cov().values  * 252 + np.eye(len(avail)) * 1e-6

        # Regime prior signal
        mu_prior = self._prior_signal(regime_probs, avail, mu_hist, nlp_scores)

        # BL-style blend
        alpha = self.prior_blend
        mu_bl = (1 - alpha) * mu_hist + alpha * mu_prior

        n   = len(avail)
        x0  = np.array([SECTOR_PRIORS[top].get(s, 1/n) for s in avail])
        x0  = x0 / max(x0.sum(), 1e-8)

        def neg_sharpe(x):
            r = x @ mu_bl - RISK_FREE
            v = np.sqrt(x @ cov @ x + 1e-10)
            return -(r / v)

        try:
            res = minimize(neg_sharpe, x0,
                           method="SLSQP",
                           bounds=[(0.0, SECTOR_MAX_WEIGHT)] * n,
                           constraints={"type":"eq","fun":lambda x: x.sum()-1},
                           options={"maxiter":500,"ftol":1e-9})
            opt = res.x if res.success else x0
        except Exception:
            opt = x0

        opt = np.clip(opt, 0, None)
        if opt.sum() > 0:
            opt /= opt.sum()

        return {avail[i]: float(opt[i]) for i in range(n)}

    def _prior_weights(self, regime_probs: dict, sectors: list) -> dict:
        b = {s: 0.0 for s in sectors}
        for r, p in regime_probs.items():
            if r in SECTOR_PRIORS:
                for s in sectors:
                    b[s] += p * SECTOR_PRIORS[r].get(s, 0)
        t = sum(b.values())
        return {k: v/t for k,v in b.items()} if t > 0 else \
               {s: 1/len(sectors) for s in sectors}

    def _prior_signal(self, regime_probs: dict, sectors: list,
                      mu_hist: np.ndarray, nlp_scores: "dict|None") -> np.ndarray:
        mean_mu = mu_hist.mean()
        signal  = np.zeros(len(sectors))
        for r, p in regime_probs.items():
            if r in SECTOR_PRIORS:
                for i, s in enumerate(sectors):
                    pw = SECTOR_PRIORS[r].get(s, 0)
                    boost = 0.25 if pw > 0.15 else (0.05 if pw > 0 else -0.20)
                    # NLP adjustment: if sector has bad NLP, reduce boost
                    if nlp_scores and s in nlp_scores:
                        nlp_adj = (0.5 - nlp_scores[s]) * 0.10
                        boost  += nlp_adj
                    signal[i] += p * mean_mu * (1 + boost)
        return signal

    # ── Step 3 ─────────────────────────────────────────────────────────────────

    def _sentiment_overlay(self, b: dict, C: float) -> dict:
        b = dict(b)
        lam = self.lam
        b["gold"]   = min(b["gold"]   + lam     * C * (1-b["gold"]),   0.45)
        b["silver"] = min(b["silver"] + 0.5*lam * C * (1-b["silver"]), 0.20)
        b["cash"]   = min(b["cash"]   + lam     * C * (1-b["cash"]),   0.60)
        defensive   = b["gold"] + b["silver"] + b["cash"] + b["nifty_floor"]
        b["sectors"]= max(1.0 - defensive, 0.05)
        return b


# ══════════════════════════════════════════════════════════════════════════════
# Walk-Forward Backtester
# ══════════════════════════════════════════════════════════════════════════════

class Backtester:

    def __init__(self, prices: pd.DataFrame, features: pd.DataFrame,
                 detector, optimizer: SectorOptimizer,
                 capital: float = 1_000_000):
        self.prices    = prices
        self.features  = features
        self.detector  = detector
        self.optimizer = optimizer
        self.capital   = capital
        self.cash_d    = CASH_RATE / 252

        ETF = {"gold","silver"}
        self.ret = prices.pct_change()
        for c in self.ret.columns:
            cap = 0.10 if c in ETF else 0.15
            self.ret[c] = self.ret[c].clip(-cap, cap)

    def run(self, start: str, end: str, window: str = "3 Months",
            nlp_scores: "dict|None" = None, progress_fn=None) -> dict:

        n_days      = REBAL_WINDOWS[window]
        all_dates   = self.ret.loc[start:end].index
        if not len(all_dates):
            raise ValueError("No trading days in range.")

        rebal = list(all_dates[::n_days])
        if all_dates[-1] not in rebal:
            rebal.append(all_dates[-1])

        capital       = float(self.capital)
        pv: dict      = {}
        wh: list      = []
        rh: list      = []
        turnovers: list = []
        prev_w        = self._equal_w()
        n_rebal       = len(rebal) - 1

        from core.models.features import get_model_X

        for i in range(n_rebal):
            t0, t1 = rebal[i], rebal[i+1]
            if progress_fn:
                progress_fn(i / max(n_rebal,1), f"Backtesting {t0.date()}…")

            feat_slice = self.features.loc[:t0]
            if len(feat_slice) < MIN_TRAIN_DAYS:
                curr_w = self._equal_w()
            else:
                try:
                    X_slice  = get_model_X(feat_slice)
                    probs_df = self.detector.predict_proba(X_slice)
                    latest   = probs_df[["Bull","Sideways","Bear"]].iloc[-1].to_dict()
                    sent     = float(feat_slice["sentiment"].iloc[-1])

                    curr_w = self.optimizer.allocate(
                        regime_probs=latest, sentiment=sent,
                        current_date=str(t0.date()), nlp_scores=nlp_scores,
                    )
                    rh.append({"date":t0, **latest,
                               "sentiment":sent,
                               "regime":max(latest,key=latest.get)})
                except Exception:
                    curr_w = self._equal_w()

            # Turnover
            all_k = set(list(curr_w) + list(prev_w))
            turn  = sum(abs(curr_w.get(k,0)-prev_w.get(k,0)) for k in all_k)/2
            turnovers.append(turn)
            prev_w = dict(curr_w)
            wh.append({"date":t0, **curr_w})

            # Simulate period
            period = all_dates[(all_dates >= t0) & (all_dates < t1)]
            for day in period:
                capital *= (1.0 + self._day_ret(curr_w, day))
                pv[day]  = capital

        if all_dates[-1] not in pv and pv:
            pv[all_dates[-1]] = capital

        return {
            "portfolio_values": pd.Series(pv).sort_index(),
            "weights_history":  pd.DataFrame(wh).set_index("date") if wh else pd.DataFrame(),
            "regime_history":   pd.DataFrame(rh).set_index("date") if rh else pd.DataFrame(),
            "avg_turnover":     float(np.mean(turnovers)) if turnovers else 0.0,
        }

    def _day_ret(self, w: dict, day: pd.Timestamp) -> float:
        ETF = {"gold","silver"}
        total = 0.0
        for a, wt in w.items():
            if a == "cash":
                total += wt * self.cash_d
            elif a in self.ret.columns and day in self.ret.index:
                r = float(self.ret.at[day, a])
                if not (np.isnan(r) or np.isinf(r)):
                    cap = 0.10 if a in ETF else 0.15
                    total += wt * max(min(r, cap), -cap)
        return max(min(total, 0.12), -0.12)

    def _equal_w(self) -> dict:
        assets = [s for s in ALL_SECTORS if s in self.ret.columns]
        assets += ["gold","silver","cash","nifty50"]
        n = len(assets)
        return {a: 1/n for a in assets}
