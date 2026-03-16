# =============================================================================
# core/backtest/metrics.py  —  Risk Metrics + Monte Carlo Simulation
# =============================================================================

import numpy as np
import pandas as pd
from scipy import stats
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from core.config import RISK_FREE, MC_SIMULATIONS, MC_HORIZON_DAYS


# ══════════════════════════════════════════════════════════════════════════════
# Performance Metrics
# ══════════════════════════════════════════════════════════════════════════════

def compute_metrics(portfolio: pd.Series, benchmark: pd.Series) -> dict:
    """Full risk-adjusted metrics for portfolio vs benchmark."""
    rf_d   = RISK_FREE / 252
    common = portfolio.index.intersection(benchmark.index)
    pv     = portfolio.reindex(common).ffill().dropna()
    bv     = benchmark.reindex(common).ffill().dropna()

    pr     = pv.pct_change().dropna().clip(-0.15, 0.15)
    br     = bv.pct_change().dropna().clip(-0.15, 0.15)
    cidx   = pr.index.intersection(br.index)
    pr, br = pr.reindex(cidx), br.reindex(cidx)

    # Rebuild clean NAV from clipped returns for consistent CAGR
    pv2 = (1 + pr).cumprod() * pv.iloc[0]
    bv2 = (1 + br).cumprod() * bv.iloc[0]

    n_days  = len(pr)
    n_years = n_days / 252
    if n_years < 0.1:
        return {}

    def _cagr(v):     return (v.iloc[-1]/v.iloc[0])**(1/n_years) - 1
    def _vol(r):      return r.std() * np.sqrt(252)
    def _maxdd(r):
        c = (1+r).cumprod(); return float(((c - c.cummax())/c.cummax()).min())
    def _sharpe(r, c): return (_cagr(c)-RISK_FREE) / (_vol(r)+1e-10)
    def _sortino(r, c):
        d = r[r < rf_d]; dv = (d.std()*np.sqrt(252)) if len(d)>2 else _vol(r)
        return (_cagr(c)-RISK_FREE) / (dv+1e-10)

    p_cagr, b_cagr   = _cagr(pv2), _cagr(bv2)
    p_vol,  b_vol    = _vol(pr),   _vol(br)
    p_shar, b_shar   = _sharpe(pr,pv2), _sharpe(br,bv2)
    p_sort, b_sort   = _sortino(pr,pv2), _sortino(br,bv2)
    p_dd,   b_dd     = _maxdd(pr), _maxdd(br)
    p_calm, b_calm   = p_cagr/(abs(p_dd)+1e-10), b_cagr/(abs(b_dd)+1e-10)

    cov_m  = np.cov(pr.values, br.values)
    beta   = cov_m[0,1] / (cov_m[1,1]+1e-10)
    alpha  = p_cagr - (RISK_FREE + beta*(b_cagr-RISK_FREE))
    active = pr - br
    ir     = (active.mean()*252) / (active.std()*np.sqrt(252)+1e-10)
    te     = active.std() * np.sqrt(252)

    monthly   = pr.resample("ME").apply(lambda x:(1+x).prod()-1)
    win_rate  = float((monthly > 0).mean())
    var95     = float(np.percentile(pr, 5))
    cvar95    = float(pr[pr<=var95].mean()) if (pr<=var95).any() else var95

    dd_series  = ((1+pr).cumprod() / (1+pr).cumprod().cummax() - 1)
    bdd_series = ((1+br).cumprod() / (1+br).cumprod().cummax() - 1)

    roll_p = pr.rolling(63)
    roll_b = br.rolling(63)
    rs_p   = (roll_p.mean()*252 - RISK_FREE) / (roll_p.std()*np.sqrt(252)+1e-10)
    rs_b   = (roll_b.mean()*252 - RISK_FREE) / (roll_b.std()*np.sqrt(252)+1e-10)

    return {
        # Portfolio
        "cagr":p_cagr, "vol":p_vol, "sharpe":p_shar, "sortino":p_sort,
        "max_dd":p_dd, "calmar":p_calm, "win_rate":win_rate,
        "var95":var95, "cvar95":cvar95,
        # Relative
        "alpha":alpha, "beta":beta, "ir":ir, "te":te,
        # Benchmark
        "b_cagr":b_cagr, "b_vol":b_vol, "b_sharpe":b_shar, "b_sortino":b_sort,
        "b_dd":b_dd, "b_calmar":b_calm,
        # Series
        "dd_series":dd_series, "b_dd_series":bdd_series,
        "port_ret":pr, "bench_ret":br,
        "roll_sharpe_p":rs_p, "roll_sharpe_b":rs_b,
        "total_return": pv2.iloc[-1]/pv2.iloc[0] - 1,
        "b_total_return": bv2.iloc[-1]/bv2.iloc[0] - 1,
        "n_years": n_years,
    }


def metrics_table(m: dict) -> pd.DataFrame:
    rows = [
        ("CAGR",                 f"{m['cagr']:.2%}",     f"{m['b_cagr']:.2%}"),
        ("Total Return",         f"{m['total_return']:.2%}", f"{m['b_total_return']:.2%}"),
        ("Ann. Volatility",      f"{m['vol']:.2%}",      f"{m['b_vol']:.2%}"),
        ("Sharpe Ratio",         f"{m['sharpe']:.3f}",   f"{m['b_sharpe']:.3f}"),
        ("Sortino Ratio",        f"{m['sortino']:.3f}",  f"{m['b_sortino']:.3f}"),
        ("Calmar Ratio",         f"{m['calmar']:.3f}",   f"{m['b_calmar']:.3f}"),
        ("Max Drawdown",         f"{m['max_dd']:.2%}",   f"{m['b_dd']:.2%}"),
        ("Jensen's Alpha",       f"{m['alpha']:.2%}",    "—"),
        ("Beta",                 f"{m['beta']:.3f}",     "1.000"),
        ("Information Ratio",    f"{m['ir']:.3f}",       "—"),
        ("Tracking Error",       f"{m['te']:.2%}",       "—"),
        ("Monthly Win Rate",     f"{m['win_rate']:.1%}", "—"),
        ("VaR 95% (daily)",      f"{m['var95']:.2%}",    "—"),
        ("CVaR 95% (daily)",     f"{m['cvar95']:.2%}",   "—"),
    ]
    return pd.DataFrame(rows, columns=["Metric", "Our Strategy", "Nifty 50"])


# ══════════════════════════════════════════════════════════════════════════════
# Monte Carlo Simulation
# ══════════════════════════════════════════════════════════════════════════════

def monte_carlo(port_ret: pd.Series,
                initial_capital: float,
                n_sims: int = MC_SIMULATIONS,
                horizon_days: int = MC_HORIZON_DAYS,
                seed: int = 42) -> dict:
    """
    GBM Monte Carlo with fat-tail correction (Student-t).

    Model: r_t ~ Student-t(ν, μ, σ)
    Parameters estimated from historical returns.

    Returns
    -------
    dict with simulation paths and statistics:
      paths       : (n_sims, horizon_days) array of portfolio values
      percentiles : dict {5: p5, 25: p25, 50: median, 75: p75, 95: p95}
      prob_profit : P(final value > initial_capital)
      prob_loss20 : P(final value < 0.8 * initial_capital)
      expected    : mean final value
    """
    rng = np.random.default_rng(seed)
    r   = port_ret.dropna().clip(-0.15, 0.15).values

    # Fit Student-t for fat tails
    df_t, loc_t, scale_t = stats.t.fit(r)
    df_t   = max(df_t, 2.1)   # ensure finite variance

    # Simulate
    paths = np.ones((n_sims, horizon_days + 1)) * initial_capital
    daily = stats.t.rvs(df=df_t, loc=loc_t, scale=scale_t,
                        size=(n_sims, horizon_days), random_state=rng)
    daily = np.clip(daily, -0.15, 0.15)

    for t in range(horizon_days):
        paths[:, t+1] = paths[:, t] * (1 + daily[:, t])

    final = paths[:, -1]
    pct   = {p: float(np.percentile(final, p)) for p in [5,10,25,50,75,90,95]}

    # Max drawdown across all paths
    running_max = np.maximum.accumulate(paths, axis=1)
    dd_paths    = (paths - running_max) / running_max
    worst_dd    = float(dd_paths.min(axis=1).mean())

    return {
        "paths":          paths,
        "final":          final,
        "percentiles":    pct,
        "prob_profit":    float((final > initial_capital).mean()),
        "prob_loss_10":   float((final < initial_capital * 0.90).mean()),
        "prob_loss_20":   float((final < initial_capital * 0.80).mean()),
        "expected":       float(final.mean()),
        "t_df":           float(df_t),
        "t_loc":          float(loc_t),
        "t_scale":        float(scale_t),
        "horizon_days":   horizon_days,
        "n_sims":         n_sims,
        "avg_max_dd":     worst_dd,
        "initial_capital":initial_capital,
    }
