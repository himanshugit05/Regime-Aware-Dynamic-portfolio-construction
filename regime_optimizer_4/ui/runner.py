# =============================================================================
# ui/runner.py  —  Orchestrates full pipeline & renders tabbed results
# =============================================================================
import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import traceback

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.config import SECTORS, SECTOR_PRIORS, REGIME_COLORS, MC_SIMULATIONS, MC_HORIZON_DAYS
from core.charts import (
    regime_probability_stack, nifty_with_regimes, sentiment_gauge,
    regime_donut, transition_heatmap,
    portfolio_growth, drawdown_chart, monthly_heatmap, yearly_bar,
    rolling_sharpe, return_distribution,
    sector_allocation_history, sector_weight_heatmap, sector_current_vs_prior,
    nlp_sentiment_bars,
    monte_carlo_fan, monte_carlo_distribution,
    SECTOR_LABELS,
)
from core.backtest.metrics import metrics_table


def run_pipeline():
    p = st.session_state.get("params", {})
    if not p:
        st.warning("Set parameters in the sidebar first."); return

    # ── Run computation only if button was just pressed ────────────────────────
    if not st.session_state.get("ran"):
        _compute(p)

    if not st.session_state.get("ran"):
        return

    _render_results(p)


# ══════════════════════════════════════════════════════════════════════════════
# COMPUTATION
# ══════════════════════════════════════════════════════════════════════════════

def _compute(p: dict):
    prog = st.progress(0, "Starting…")

    try:
        # ── Data ──────────────────────────────────────────────────────────────
        from core.data.loader import load_all, get_portfolio_assets, get_nifty, get_benchmark
        prog.progress(5, "Loading market data…")
        price_df = load_all(p["start"], p["end"],
                            use_cache=p["use_cache"],
                            progress_fn=lambda pct,msg: prog.progress(int(5+pct*20), msg))

        st.session_state["price_df"] = price_df
        nifty = get_nifty(price_df)
        if nifty is None:
            st.error("Nifty 50 data unavailable."); return

        # ── NLP ───────────────────────────────────────────────────────────────
        nlp_scores = None
        if p.get("use_nlp"):
            prog.progress(26, "Running NLP sentiment…")
            try:
                from core.data.sentiment_nlp import get_sector_sentiment
                nlp_scores = get_sector_sentiment(
                    progress_fn=lambda pct,msg: prog.progress(int(26+pct*10), msg))
            except Exception as e:
                st.warning(f"NLP sentiment unavailable: {e}. Using VIX-only sentiment.")
        st.session_state["nlp_scores"] = nlp_scores
        nlp_market = nlp_scores.get("market", 0.5) if nlp_scores else 0.5

        # ── Features ──────────────────────────────────────────────────────────
        prog.progress(37, "Computing features…")
        from core.models.features import build_features, get_model_X
        features = build_features(price_df, nlp_market_score=nlp_market)
        features = features.loc[p["start"]:p["end"]]
        if len(features) < 300:
            st.error("Insufficient data. Extend the start date by ≥2 years."); return
        st.session_state["features"] = features

        # ── Regime Model ──────────────────────────────────────────────────────
        prog.progress(42, "Training HMM + GMM ensemble…")
        from core.models.regime import RegimeDetector
        X = get_model_X(features)
        detector = RegimeDetector(seed=42)
        detector.fit(X, features["r_21d"])
        regime_probs = detector.predict_proba(X)
        st.session_state["detector"]     = detector
        st.session_state["regime_probs"] = regime_probs

        # ── Portfolio ─────────────────────────────────────────────────────────
        prog.progress(55, "Setting up optimizer…")
        import core.config as cfg
        cfg.PRIOR_BLEND = p["alpha"]

        port_prices = get_portfolio_assets(price_df, p["start"], p["end"])
        from core.backtest.engine import SectorOptimizer, Backtester
        optimizer  = SectorOptimizer(port_prices.pct_change().dropna(),
                                     sentiment_lambda=p["lam"],
                                     prior_blend=p["alpha"])
        backtester = Backtester(port_prices, features, detector, optimizer,
                                capital=p["capital"])

        results = backtester.run(
            p["start"], p["end"], window=p["window"],
            nlp_scores=nlp_scores,
            progress_fn=lambda pct,msg: prog.progress(int(55+pct*30), msg),
        )
        st.session_state["results"] = results

        # ── Benchmarks ────────────────────────────────────────────────────────
        prog.progress(87, "Building benchmarks…")
        bench_key_map = {
            "Nifty 50 (Large Cap)": "nifty50",
            "Nifty Midcap":         "nifty_mid",
            "Nifty Smallcap":       "nifty_small",
        }
        bench_series = {}
        for name in p.get("benches",[]):
            key = bench_key_map.get(name)
            if key:
                s = get_benchmark(price_df, key, p["start"], p["end"], p["capital"])
                if s is not None:
                    bench_series[name] = s
        st.session_state["bench_series"] = bench_series

        # ── Metrics ───────────────────────────────────────────────────────────
        prog.progress(90, "Computing metrics…")
        from core.backtest.metrics import compute_metrics
        nifty_bench = bench_series.get("Nifty 50 (Large Cap)")
        port_metrics = {}
        if nifty_bench is not None:
            port_metrics = compute_metrics(results["portfolio_values"], nifty_bench)
        st.session_state["port_metrics"] = port_metrics

        # ── Monte Carlo ───────────────────────────────────────────────────────
        if p.get("use_mc") and port_metrics.get("port_ret") is not None:
            prog.progress(93, f"Monte Carlo ({MC_SIMULATIONS:,} simulations)…")
            from core.backtest.metrics import monte_carlo
            mc = monte_carlo(port_metrics["port_ret"], p["capital"])
            st.session_state["mc_result"] = mc

        prog.progress(100, "Done ✓")
        prog.empty()
        st.session_state["ran"] = True
        st.rerun()

    except Exception as e:
        prog.empty()
        st.error(f"Pipeline failed: {e}")
        st.code(traceback.format_exc())


# ══════════════════════════════════════════════════════════════════════════════
# RENDER
# ══════════════════════════════════════════════════════════════════════════════

def _render_results(p: dict):
    regime_probs = st.session_state["regime_probs"]
    results      = st.session_state["results"]
    features     = st.session_state["features"]
    bench_series = st.session_state["bench_series"]
    port_metrics = st.session_state["port_metrics"]
    mc           = st.session_state.get("mc_result")
    nlp_scores   = st.session_state.get("nlp_scores")
    detector     = st.session_state["detector"]
    price_df     = st.session_state["price_df"]

    port_values  = results["portfolio_values"]
    wh           = results["weights_history"]
    rh           = results["regime_history"]

    # ── KPI Bar ───────────────────────────────────────────────────────────────
    regime_now = regime_probs["regime"].iloc[-1]
    reg_class  = f"reg-{regime_now.lower()}"

    k1,k2,k3,k4,k5,k6 = st.columns(6)
    k1.markdown(f"<div style='margin-top:.3rem'><div style='font-size:.75rem;color:#666'>Current Regime</div>"
                f"<div class='{reg_class}'>{regime_now}</div></div>", unsafe_allow_html=True)

    if port_metrics:
        k2.metric("CAGR", f"{port_metrics['cagr']:.2%}",
                  delta=f"{port_metrics['cagr']-port_metrics['b_cagr']:+.2%} vs Nifty")
        k3.metric("Sharpe", f"{port_metrics['sharpe']:.3f}",
                  delta=f"{port_metrics['sharpe']-port_metrics['b_sharpe']:+.3f}")
        k4.metric("Max Drawdown", f"{port_metrics['max_dd']:.2%}",
                  delta=f"{port_metrics['b_dd']-port_metrics['max_dd']:+.2%} better")
        k5.metric("Alpha", f"{port_metrics['alpha']:.2%}")
        k6.metric("Calmar", f"{port_metrics['calmar']:.3f}",
                  delta=f"{port_metrics['calmar']-port_metrics['b_calmar']:+.3f}")

    st.divider()

    # ── TABS ──────────────────────────────────────────────────────────────────
    tabs = st.tabs([
        "🧠 Regime", "🏭 Sectors", "📈 Performance",
        "🎲 Monte Carlo", "🔬 Risk Metrics",
    ])
    if nlp_scores:
        tabs = st.tabs([
            "🧠 Regime", "🏭 Sectors", "📈 Performance",
            "🗞️ NLP Sentiment", "🎲 Monte Carlo", "🔬 Risk Metrics",
        ])

    tab_idx = {"regime":0,"sectors":1,"perf":2,"nlp":3 if nlp_scores else None,
               "mc": 4 if nlp_scores else 3, "risk": 5 if nlp_scores else 4}

    # ── TAB: REGIME ───────────────────────────────────────────────────────────
    with tabs[tab_idx["regime"]]:
        st.markdown("### Regime Detection")
        info = detector.info()
        c1,c2,c3 = st.columns([1,1,3])
        c1.metric("HMM Weight (BMA)", f"{info['w_hmm']:.1%}")
        c2.metric("GMM Weight (BMA)", f"{info['w_gmm']:.1%}")
        c3.caption("Weights from out-of-sample log-likelihood on 20% held-out validation set.")

        st.plotly_chart(regime_probability_stack(regime_probs), width="stretch")

        nifty_full = price_df["nifty50"].dropna().loc[p["start"]:p["end"]]
        st.plotly_chart(nifty_with_regimes(nifty_full, regime_probs), width="stretch")

        if "sentiment" in features.columns:
            st.plotly_chart(sentiment_gauge(features["sentiment"]), width="stretch")

        c1,c2 = st.columns(2)
        with c1: st.plotly_chart(regime_donut(regime_probs), width="stretch")
        with c2: st.plotly_chart(transition_heatmap(rh), width="stretch")

        if "transmat" in info:
            with st.expander("🔍 Learned HMM Transition Matrix"):
                import numpy as np
                tmat = np.array(info["transmat"])
                state_names = list(detector.state_map.values())
                tm_df = pd.DataFrame(tmat,
                    index=[f"From {s}" for s in state_names],
                    columns=[f"To {s}" for s in state_names])
                st.dataframe(tm_df.style.format("{:.4f}").background_gradient(cmap="RdYlGn"),
                             use_container_width=True)
                st.caption("Aᵢⱼ = P(Sₜ=j | Sₜ₋₁=i). High diagonal = persistent regimes.")

    # ── TAB: SECTORS ──────────────────────────────────────────────────────────
    with tabs[tab_idx["sectors"]]:
        st.markdown("### Sector Rotation")
        if len(wh) > 0:
            st.plotly_chart(sector_allocation_history(wh), width="stretch")
            if len(wh) > 2:
                st.plotly_chart(sector_weight_heatmap(wh), width="stretch")

            latest = wh.iloc[-1]
            st.plotly_chart(sector_current_vs_prior(latest, regime_now), width="stretch")

            # Bucket summary
            sector_keys = [s for s in SECTORS if s in wh.columns]
            eq_w   = sum(latest.get(s,0) for s in sector_keys)
            fl_w   = latest.get("nifty50",0)
            def_w  = sum(latest.get(k,0) for k in ["gold","silver","cash"])
            c1,c2,c3,c4 = st.columns(4)
            c1.metric("Active Sectors", f"{eq_w:.1%}")
            c2.metric("Nifty Floor",    f"{fl_w:.1%}")
            c3.metric("Defensives",     f"{def_w:.1%}")
            c4.metric("Avg Turnover",   f"{results['avg_turnover']:.1%}")
        else:
            st.warning("No allocation history.")

    # ── TAB: PERFORMANCE ──────────────────────────────────────────────────────
    with tabs[tab_idx["perf"]]:
        st.markdown(f"### Portfolio Performance  ·  {p['window']} Rebalancing")
        st.plotly_chart(portfolio_growth(port_values, bench_series, p["capital"]),
                        width="stretch")
        if port_metrics:
            st.plotly_chart(drawdown_chart(port_metrics["dd_series"],
                                           port_metrics["b_dd_series"]), width="stretch")
            st.plotly_chart(monthly_heatmap(port_metrics["port_ret"]), width="stretch")
            st.plotly_chart(yearly_bar(port_metrics["port_ret"],
                                       port_metrics["bench_ret"]), width="stretch")

    # ── TAB: NLP ─────────────────────────────────────────────────────────────
    if nlp_scores and tab_idx["nlp"] is not None:
        with tabs[tab_idx["nlp"]]:
            st.markdown("### 🗞️ NLP Sector Sentiment")
            mkt = nlp_scores.get("market",0.5)
            col_a, col_b, col_c = st.columns(3)
            col_a.metric("Market Sentiment", f"{mkt:.2f}",
                         delta="Bullish" if mkt<0.35 else ("Bearish" if mkt>0.65 else "Neutral"))
            col_b.metric("Sectors Bullish",
                         str(sum(1 for k,v in nlp_scores.items() if k!="market" and v<0.35)))
            col_c.metric("Sectors Bearish",
                         str(sum(1 for k,v in nlp_scores.items() if k!="market" and v>0.65)))

            st.plotly_chart(nlp_sentiment_bars(nlp_scores, SECTOR_LABELS), width="stretch")

            from core.data.sentiment_nlp import get_sentiment_summary
            df_nlp = get_sentiment_summary(nlp_scores)
            st.dataframe(df_nlp, use_container_width=True, hide_index=True)
            st.caption("VADER sentiment on latest news headlines. Score: 0=Bullish, 1=Bearish. "
                       "Refreshes every 6 hours.")

    # ── TAB: MONTE CARLO ─────────────────────────────────────────────────────
    with tabs[tab_idx["mc"]]:
        st.markdown("### 🎲 Monte Carlo Simulation")
        if mc:
            cap = mc["initial_capital"]
            c1,c2,c3,c4 = st.columns(4)
            c1.metric("Prob. of Profit",  f"{mc['prob_profit']:.1%}")
            c2.metric("Prob. Loss >10%",  f"{mc['prob_loss_10']:.1%}")
            c3.metric("Median Outcome",   f"₹{mc['percentiles'][50]/1e5:.1f}L")
            c4.metric("95th Percentile",  f"₹{mc['percentiles'][95]/1e5:.1f}L")

            st.plotly_chart(monte_carlo_fan(mc), width="stretch")
            st.plotly_chart(monte_carlo_distribution(mc), width="stretch")

            # Percentile table
            pct_df = pd.DataFrame([
                {"Percentile": f"{k}th", "Portfolio Value": f"₹{v:,.0f}",
                 "Return": f"{v/cap-1:+.1%}"}
                for k,v in sorted(mc["percentiles"].items())
            ])
            st.dataframe(pct_df, hide_index=True, use_container_width=True)
            st.caption(f"Student-t Monte Carlo: ν={mc['t_df']:.1f}, "
                       f"{mc['n_sims']:,} simulations, "
                       f"{mc['horizon_days']//21}M horizon. "
                       f"Fat-tail correction applied.")
        else:
            st.info("Monte Carlo not run. Enable the toggle in the sidebar.")

    # ── TAB: RISK METRICS ────────────────────────────────────────────────────
    with tabs[tab_idx["risk"]]:
        st.markdown("### 🔬 Risk-Adjusted Metrics")
        if port_metrics:
            tbl = metrics_table(port_metrics)
            def _color(row):
                try:
                    mv = float(row["Our Strategy"].strip("%,").replace("—","0"))
                    bv = float(row["Nifty 50"].strip("%,").replace("—","0"))
                    better = mv > bv
                    if any(x in row["Metric"] for x in ["Drawdown","VaR","CVaR","Beta"]):
                        better = mv > bv
                    return [""]*2 + ["background-color:#0d2b0d" if better else "background-color:#2b0d0d"]
                except Exception:
                    return [""]*3
            st.dataframe(tbl.style.apply(_color,axis=1),
                         hide_index=True, use_container_width=True)

            c1,c2 = st.columns(2)
            with c1:
                st.plotly_chart(rolling_sharpe(port_metrics["roll_sharpe_p"],
                                               port_metrics["roll_sharpe_b"]), width="stretch")
            with c2:
                st.plotly_chart(return_distribution(port_metrics["port_ret"],
                                                    port_metrics["bench_ret"]), width="stretch")
        else:
            st.warning("Nifty 50 benchmark unavailable — metrics not computed.")

    st.divider()
    st.markdown(
        "<small style='color:#444'>Research only. Not investment advice. "
        "Data: Yahoo Finance · VADER NLP · yfinance</small>",
        unsafe_allow_html=True
    )
