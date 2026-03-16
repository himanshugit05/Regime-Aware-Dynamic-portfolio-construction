# ui/home.py — Landing page shown before first run

import streamlit as st
from core.config import SECTOR_PRIORS, SECTORS, REGIME_BUCKETS

def render_home():
    st.markdown("""
    <div style='margin-bottom:2rem'>
      <div style='font-size:2.2rem;font-weight:700;
        background:linear-gradient(135deg,#667eea,#a78bfa);
        -webkit-background-clip:text;-webkit-text-fill-color:transparent;'>
        Regime-Aware Sector Portfolio
      </div>
      <div style='color:#666;font-size:.9rem;margin-top:4px'>
        HMM + GMM Ensemble  ·  10-Sector Rotation  ·  NLP Sentiment  ·
        Monte Carlo  ·  Walk-Forward Backtest
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Cards
    c1,c2,c3,c4 = st.columns(4)
    for col,(ico,title,body) in zip([c1,c2,c3,c4],[
        ("🧠","Regime Engine",
         "HMM + GMM ensemble with BMA weighting. Detects Bull / Sideways / Bear with posterior probabilities."),
        ("🏭","Sector Rotation",
         "Max-Sharpe optimizer with regime-specific Bayesian priors across 10 NSE sectors."),
        ("📡","NLP Sentiment",
         "VADER analysis of live news headlines per sector. Integrated into composite fear score Cₜ."),
        ("🎲","Monte Carlo",
         "Student-t 1000-path forward simulation. Probability of profit, loss, and VaR fan chart."),
    ]):
        col.markdown(f"""
        <div style='background:linear-gradient(135deg,#13132a,#1a1a35);
          border:1px solid #2a2a50;border-radius:12px;padding:1.2rem;height:150px;'>
          <div style='font-size:1.6rem'>{ico}</div>
          <div style='font-weight:600;color:#c8c8ff;margin:.4rem 0 .3rem'>{title}</div>
          <div style='font-size:.8rem;color:#7878a0;line-height:1.5'>{body}</div>
        </div>""", unsafe_allow_html=True)

    st.divider()

    # Math
    st.markdown("### 🔢 Mathematical Framework")
    c1,c2 = st.columns(2)
    with c1:
        st.markdown(r"""
**Regime Detection (BMA Ensemble)**

$$P(S_t{=}k) = w_{HMM} \cdot P_{HMM}(S_t{=}k \mid X_{1:t}) + w_{GMM} \cdot P_{GMM}(S_t{=}k \mid X_t)$$

**BMA Weights** *(out-of-sample log-likelihood)*:

$$w_m \propto \exp\!\left(\overline{LL}_m(X_{val})\right)$$

**Sector Allocation** *(BL-style blend)*:

$$\mu_{BL} = (1-\alpha)\,\mu_{hist} + \alpha\,\mu_{prior}$$
$$\max_w \frac{w'\mu_{BL} - r_f}{\sqrt{w'\Sigma w}} \quad \text{s.t.} \sum w_i=1$$
""")
    with c2:
        st.markdown(r"""
**Composite Sentiment Score**:

$$C_t = 0.30\,VIX^{IN} + 0.20\,VIX^{US} + 0.15\,DXY + 0.15\,CV_t + 0.20\,NLP_t$$

**Sentiment Overlay**:

$$w^*_{gold} = w_{gold} + \lambda C_t(1-w_{gold})$$

**Monte Carlo** *(Student-t fat tails)*:

$$r_t \sim t(\nu, \mu, \sigma), \quad P_T = P_0 \prod_{t=1}^{T}(1+r_t)$$

$$\text{Prob(Profit)} = \mathbb{P}(P_T > P_0)$$
""")

    # Sector Priors Table
    st.markdown("###  Sector Regime Priors")
    import pandas as pd
    prior_df = pd.DataFrame(SECTOR_PRIORS).T
    prior_df.columns = [SECTORS[c]["label"] for c in prior_df.columns]
    prior_df = prior_df.map(lambda x: f"{x:.0%}" if x > 0 else "—")
    st.dataframe(prior_df, use_container_width=True)

    # Bucket weights
    st.markdown("###  Regime Bucket Weights")
    bucket_df = pd.DataFrame(REGIME_BUCKETS).T
    bucket_df.columns = ["Active Sectors","Nifty Floor","Gold","Silver","Cash"]
    bucket_df = bucket_df.map(lambda x: f"{x:.0%}")
    st.dataframe(bucket_df, use_container_width=True)

    st.markdown("""
    <div style='text-align:center;margin-top:2rem;padding:1.5rem;
      background:linear-gradient(135deg,#13132a,#1a1a35);
      border:1px solid #2a2a50;border-radius:12px;'>
      <div style='color:#a78bfa;font-weight:600;font-size:1.1rem'>
        ← Set parameters in the sidebar and click 🚀 Run Analysis
      </div>
    </div>
    """, unsafe_allow_html=True)
