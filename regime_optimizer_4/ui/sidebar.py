# =============================================================================
# ui/sidebar.py  —  Sidebar Navigation + Parameters
# =============================================================================

import streamlit as st
from datetime import date, timedelta
from core.config import REBAL_WINDOWS


def render_sidebar() -> str:
    """Renders sidebar, returns page state: 'home' or 'run'."""
    with st.sidebar:
        st.markdown("""
        <div style='text-align:center; padding: 1rem 0 0.5rem'>
          <div style='font-size:2rem'>📊</div>
          <div style='font-weight:700; font-size:1.1rem; color:#a78bfa;'>Regime Portfolio</div>
          <div style='font-size:.75rem; color:#666; margin-top:2px'>v2 · Sector Rotation</div>
        </div>
        """, unsafe_allow_html=True)
        st.divider()

        # ── Date Range ────────────────────────────────────────────────────────
        st.markdown("#### 📅 Backtest Period")
        c1, c2 = st.columns(2)
        with c1:
            start = st.date_input("Start", value=date(2015, 1, 1),
                min_value=date(2008, 1, 1),
                max_value=date.today() - timedelta(days=400),
                label_visibility="visible")
        with c2:
            end = st.date_input("End", value=date.today(),
                min_value=date(2010, 1, 1),
                max_value=date.today(),
                label_visibility="visible")

        # ── Capital ───────────────────────────────────────────────────────────
        st.markdown("#### 💰 Capital")
        capital = st.number_input("Initial Capital (₹)",
            min_value=100_000, max_value=500_000_000,
            value=1_000_000, step=100_000, format="%d",
            label_visibility="collapsed")

        # ── Strategy ──────────────────────────────────────────────────────────
        st.markdown("#### ⚙️ Strategy")
        window = st.selectbox("Rebalancing Window",
            list(REBAL_WINDOWS.keys()), index=0)

        lam = st.slider("Sentiment λ", 0.10, 0.60, 0.35, 0.05,
            help="How aggressively to shift to gold/cash on fear spike.")

        alpha = st.slider("Regime Prior α", 0.10, 0.70, 0.40, 0.05,
            help="α=0.1 → data-driven Max-Sharpe  |  α=0.7 → pure regime priors")

        # ── Options ───────────────────────────────────────────────────────────
        st.markdown("#### 🔧 Options")
        use_nlp  = st.toggle("🗞️ NLP Sector Sentiment", value=True,
            help="Fetch latest news headlines and run VADER sentiment analysis per sector.")
        use_mc   = st.toggle("🎲 Monte Carlo Projection", value=True,
            help="Run 1000-path Student-t Monte Carlo after backtest.")
        use_cache= st.toggle("⚡ Use Data Cache", value=True,
            help="Skip re-downloading if cached data exists.")

        # ── Benchmarks ────────────────────────────────────────────────────────
        st.markdown("#### 📈 Benchmarks")
        benches = st.multiselect("Compare against",
            ["Nifty 50 (Large Cap)", "Nifty Midcap", "Nifty Smallcap"],
            default=["Nifty 50 (Large Cap)"])

        st.divider()

        run = st.button("🚀  Run Analysis", type="primary",
                        use_container_width=True)

        # Save to session state
        st.session_state["params"] = {
            "start":     str(start),
            "end":       str(end),
            "capital":   capital,
            "window":    window,
            "lam":       lam,
            "alpha":     alpha,
            "use_nlp":   use_nlp,
            "use_mc":    use_mc,
            "use_cache": use_cache,
            "benches":   benches,
        }

        if run:
            return "run"

    return "run" if st.session_state.get("ran") else "home"
