# =============================================================================
# core/charts.py  —  Premium Chart Factory
# Every chart uses a consistent dark theme, proper typography, annotations.
# =============================================================================

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from core.config import (
    REGIME_COLORS, REGIME_BG, SECTORS,
    PRIMARY_COLOR, POSITIVE_COLOR, NEGATIVE_COLOR, NEUTRAL_COLOR,
    CHART_TEMPLATE,
)

SECTOR_COLORS = {k: v["color"] for k,v in SECTORS.items()}
SECTOR_LABELS = {k: v["label"] for k,v in SECTORS.items()}

# ── Base layout ───────────────────────────────────────────────────────────────

def _base(title: str, height: int = 400,
          xtitle: str = "", ytitle: str = "") -> dict:
    return dict(
        title=dict(text=title, font=dict(size=15, color="#e0e0e0", family="Inter")),
        template=CHART_TEMPLATE,
        paper_bgcolor="rgba(20,20,30,1)",
        plot_bgcolor ="rgba(28,28,40,1)",
        font=dict(family="Inter", color="#c0c0c0", size=12),
        height=height,
        margin=dict(t=55, b=40, l=50, r=30),
        xaxis=dict(title=xtitle, gridcolor="#2a2a3e", showgrid=True,
                   zeroline=False, color="#888"),
        yaxis=dict(title=ytitle, gridcolor="#2a2a3e", showgrid=True,
                   zeroline=False, color="#888"),
        hovermode="x unified",
        legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="#333",
                    font=dict(size=11)),
    )


# ══════════════════════════════════════════════════════════════════════════════
# Regime Charts
# ══════════════════════════════════════════════════════════════════════════════

def regime_probability_stack(regime_probs: pd.DataFrame) -> go.Figure:
    """Stacked area chart of posterior regime probabilities."""
    fig = go.Figure()
    fills = {"Bear":"rgba(255,23,68,.50)", "Sideways":"rgba(255,214,0,.50)",
             "Bull":"rgba(0,200,83,.50)"}
    for r in ["Bear","Sideways","Bull"]:
        if r not in regime_probs.columns:
            continue
        fig.add_trace(go.Scatter(
            x=regime_probs.index, y=regime_probs[r],
            name=r, stackgroup="one",
            fillcolor=fills[r],
            line=dict(width=0.8, color=REGIME_COLORS[r]),
            hovertemplate=f"<b>{r}</b>: %{{y:.1%}}<extra></extra>",
        ))
    fig.update_layout(**_base(
        "📊 Ensemble Posterior P(Sₜ = k | data)", 300,
        ytitle="Probability"
    ))
    fig.update_yaxes(tickformat=".0%", range=[0,1])
    return fig


def nifty_with_regimes(nifty: pd.Series, regime_probs: pd.DataFrame) -> go.Figure:
    """Nifty 50 price with regime-colored background bands."""
    fig = go.Figure()

    # Add regime backdrop bands
    prev_r = prev_dt = None
    for dt, row in regime_probs.iterrows():
        cr = row["regime"]
        if cr != prev_r:
            if prev_r and prev_dt:
                fig.add_vrect(x0=prev_dt, x1=dt,
                              fillcolor=REGIME_BG[prev_r],
                              line_width=0, layer="below")
            prev_r, prev_dt = cr, dt
    if prev_r and prev_dt:
        fig.add_vrect(x0=prev_dt, x1=regime_probs.index[-1],
                      fillcolor=REGIME_BG[prev_r], line_width=0, layer="below")

    # Nifty line
    fig.add_trace(go.Scatter(
        x=nifty.index, y=nifty,
        name="Nifty 50",
        line=dict(color=PRIMARY_COLOR, width=2),
        hovertemplate="<b>Nifty 50</b>: %{y:,.0f}<extra></extra>",
    ))

    # Regime annotation at latest point
    latest_regime = regime_probs["regime"].iloc[-1]
    fig.add_annotation(
        x=regime_probs.index[-1], y=nifty.iloc[-1],
        text=f"  {latest_regime}",
        font=dict(color=REGIME_COLORS[latest_regime], size=13, family="Inter"),
        showarrow=False, xanchor="left",
    )

    fig.update_layout(**_base("📈 Nifty 50  ·  Regime Backdrop", 380, ytitle="Index Level"))
    return fig


def sentiment_gauge(sentiment_series: pd.Series) -> go.Figure:
    """Sentiment score over time with fear zones."""
    fig = go.Figure()
    fig.add_hrect(y0=0, y1=0.35, fillcolor="rgba(0,200,83,.07)", line_width=0)
    fig.add_hrect(y0=0.35, y1=0.65, fillcolor="rgba(255,214,0,.07)", line_width=0)
    fig.add_hrect(y0=0.65, y1=1.0,  fillcolor="rgba(255,23,68,.07)", line_width=0)

    fig.add_trace(go.Scatter(
        x=sentiment_series.index, y=sentiment_series,
        fill="tozeroy", fillcolor="rgba(102,126,234,.20)",
        line=dict(color=PRIMARY_COLOR, width=2),
        name="Sentiment Cₜ",
        hovertemplate="<b>Sentiment</b>: %{y:.3f}<extra></extra>",
    ))

    for y, txt, col in [(0.35,"▲ Bullish Zone","#00C853"),
                         (0.65,"▲ Fear Zone",   "#FF1744")]:
        fig.add_hline(y=y, line_dash="dash", line_color=col, line_width=1,
                      annotation_text=txt, annotation_position="top right",
                      annotation_font_color=col)

    fig.update_layout(**_base("📡 Composite Sentiment Score  Cₜ", 250, ytitle="Score"))
    fig.update_yaxes(range=[0,1])
    return fig


def regime_donut(regime_probs: pd.DataFrame) -> go.Figure:
    counts = regime_probs["regime"].value_counts()
    fig = go.Figure(go.Pie(
        labels=counts.index, values=counts.values,
        hole=0.55,
        marker=dict(colors=[REGIME_COLORS[r] for r in counts.index],
                    line=dict(color="#1a1a2e", width=2)),
        textfont=dict(size=12, family="Inter"),
        hovertemplate="<b>%{label}</b><br>%{percent}<extra></extra>",
    ))
    fig.add_annotation(text="Regime<br>Mix", x=0.5, y=0.5,
                       font=dict(size=13, color="#aaa"), showarrow=False)
    fig.update_layout(**_base("🕐 Historical Regime Distribution", 340))
    fig.update_layout(margin=dict(t=55,b=30,l=30,r=30))
    return fig


def transition_heatmap(regime_history: pd.DataFrame) -> go.Figure:
    if len(regime_history) < 3:
        return go.Figure()
    rl = regime_history["regime"].tolist()
    labels = ["Bull","Sideways","Bear"]
    tm = pd.DataFrame(0, index=labels, columns=labels)
    for a,b in zip(rl[:-1], rl[1:]):
        if a in tm.index and b in tm.columns:
            tm.loc[a,b] += 1
    tm_n = tm.div(tm.sum(axis=1).replace(0,1), axis=0)
    fig = px.imshow(tm_n, text_auto=".1%", color_continuous_scale="RdYlGn",
                    zmin=0, zmax=1, aspect="auto")
    fig.update_layout(**_base("🔄 Regime Transition Matrix", 320))
    fig.update_layout(coloraxis_showscale=False)
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# Portfolio Performance Charts
# ══════════════════════════════════════════════════════════════════════════════

def portfolio_growth(port_values: pd.Series, benchmarks: dict[str,pd.Series],
                     initial_capital: float) -> go.Figure:
    fig = go.Figure()
    BENCH_COLS = ["#FF6B6B","#FFA500","#00BCD4"]
    for i,(name,bv) in enumerate(benchmarks.items()):
        b = bv.reindex(port_values.index, method="ffill")
        fig.add_trace(go.Scatter(
            x=b.index, y=b,
            name=name,
            line=dict(color=BENCH_COLS[i%3], width=1.8, dash="dot"),
            hovertemplate=f"<b>{name}</b>: ₹%{{y:,.0f}}<extra></extra>",
            opacity=0.8,
        ))
    fig.add_trace(go.Scatter(
        x=port_values.index, y=port_values,
        name="🎯 Our Strategy",
        line=dict(color=PRIMARY_COLOR, width=3),
        hovertemplate="<b>Strategy</b>: ₹%{y:,.0f}<extra></extra>",
    ))

    # Annotate final values
    fig.add_annotation(
        x=port_values.index[-1], y=port_values.iloc[-1],
        text=f"  ₹{port_values.iloc[-1]/1e5:.1f}L",
        font=dict(color=PRIMARY_COLOR, size=12), showarrow=False, xanchor="left",
    )
    fig.update_layout(**_base(
        f"💰 Growth of ₹{initial_capital/1e5:.0f}L Capital", 440,
        ytitle="Portfolio Value (₹)"
    ))
    fig.update_yaxes(tickformat=",.0f")
    return fig


def drawdown_chart(dd_series: pd.Series, bench_dd: pd.Series) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=bench_dd.index, y=bench_dd*100,
        name="Nifty 50", fill=None,
        line=dict(color="#FF6B6B", width=1.5, dash="dot"),
        hovertemplate="<b>Nifty DD</b>: %{y:.1f}%<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=dd_series.index, y=dd_series*100,
        name="Our Strategy", fill="tozeroy",
        fillcolor="rgba(102,126,234,0.25)",
        line=dict(color=PRIMARY_COLOR, width=2),
        hovertemplate="<b>Strategy DD</b>: %{y:.1f}%<extra></extra>",
    ))
    fig.update_layout(**_base("📉 Drawdown (Underwater) Chart", 280, ytitle="Drawdown (%)"))
    return fig


def monthly_heatmap(port_ret: pd.Series) -> go.Figure:
    MON = {1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",
           7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec"}
    monthly = port_ret.resample("ME").apply(lambda x:(1+x).prod()-1)*100
    df = monthly.to_frame("ret")
    df["yr"] = df.index.year; df["mon"] = df.index.month.map(MON)
    pivot = df.pivot_table(values="ret",index="yr",columns="mon")
    pivot = pivot[[m for m in MON.values() if m in pivot.columns]]

    fig = px.imshow(pivot, text_auto=".1f",
                    color_continuous_scale="RdYlGn",
                    color_continuous_midpoint=0, aspect="auto")
    fig.update_traces(textfont_size=10)
    fig.update_layout(**_base("📅 Monthly Returns Heatmap (%)", max(220, 35*len(pivot))))
    fig.update_layout(coloraxis_showscale=False)
    return fig


def yearly_bar(port_ret: pd.Series, bench_ret: pd.Series) -> go.Figure:
    yr_p = port_ret.resample("YE").apply(lambda x:(1+x).prod()-1)*100
    yr_b = bench_ret.resample("YE").apply(lambda x:(1+x).prod()-1)*100
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=yr_p.index.year, y=yr_p,
        name="Our Strategy",
        marker_color=[POSITIVE_COLOR if v>=0 else NEGATIVE_COLOR for v in yr_p],
        hovertemplate="<b>%{x}</b>: %{y:.1f}%<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=yr_b.index.year, y=yr_b,
        name="Nifty 50",
        line=dict(color="#FF6B6B", width=2),
        mode="lines+markers",
        marker=dict(size=7),
        hovertemplate="<b>Nifty %{x}</b>: %{y:.1f}%<extra></extra>",
    ))
    fig.update_layout(**_base("📊 Annual Returns Comparison", 320, ytitle="Return (%)"))
    return fig


def rolling_sharpe(rs_p: pd.Series, rs_b: pd.Series) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=rs_b.index, y=rs_b, name="Nifty 50",
        line=dict(color="#FF6B6B", width=1.5, dash="dot")))
    fig.add_trace(go.Scatter(x=rs_p.index, y=rs_p, name="Our Strategy",
        line=dict(color=PRIMARY_COLOR, width=2)))
    fig.add_hline(y=0, line_dash="dash", line_color="#555", line_width=1)
    fig.update_layout(**_base("📈 Rolling 63-Day Sharpe Ratio", 300, ytitle="Sharpe"))
    return fig


def return_distribution(port_ret: pd.Series, bench_ret: pd.Series) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=bench_ret*100, name="Nifty 50",
        opacity=0.65, marker_color="#FF6B6B", nbinsx=100))
    fig.add_trace(go.Histogram(x=port_ret*100,  name="Our Strategy",
        opacity=0.75, marker_color=PRIMARY_COLOR, nbinsx=100))
    fig.update_layout(**_base("📊 Daily Return Distribution", 300,
                               xtitle="Daily Return (%)", ytitle="Count"))
    fig.update_layout(barmode="overlay")
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# Sector Charts
# ══════════════════════════════════════════════════════════════════════════════

def sector_allocation_history(wh: pd.DataFrame) -> go.Figure:
    sector_cols = [c for c in SECTORS if c in wh.columns and wh[c].sum() > 0.005]
    fig = go.Figure()
    for s in sector_cols:
        fig.add_trace(go.Bar(
            x=wh.index, y=wh[s]*100,
            name=SECTOR_LABELS.get(s,s),
            marker_color=SECTOR_COLORS.get(s,"#aaa"),
            hovertemplate=f"<b>{SECTOR_LABELS.get(s,s)}</b>: %{{y:.1f}}%<extra></extra>",
        ))
    fig.update_layout(**_base("🏭 Sector Weights at Each Rebalancing", 400, ytitle="Weight (%)"))
    fig.update_layout(barmode="stack", legend=dict(orientation="h",y=-0.35),
                      margin=dict(b=120))
    return fig


def sector_weight_heatmap(wh: pd.DataFrame) -> go.Figure:
    sector_cols = [c for c in SECTORS if c in wh.columns]
    heat = wh[sector_cols].T * 100
    heat.index = [SECTOR_LABELS.get(s,s) for s in sector_cols]
    fig = px.imshow(heat, text_auto=".0f", color_continuous_scale="Blues",
                    aspect="auto", zmin=0, zmax=45)
    fig.update_layout(**_base("🌡️ Sector Weight Heatmap (%) Over Time", 380))
    fig.update_layout(coloraxis_showscale=True)
    return fig


def sector_current_vs_prior(latest_w: pd.Series, regime: str) -> go.Figure:
    from core.config import SECTOR_PRIORS
    sector_cols = [c for c in SECTORS if c in latest_w.index]
    labels = [SECTOR_LABELS[s] for s in sector_cols]
    actual = [latest_w.get(s,0)*100 for s in sector_cols]
    prior  = [SECTOR_PRIORS.get(regime,{}).get(s,0)*100 for s in sector_cols]

    fig = go.Figure()
    fig.add_trace(go.Bar(name="Regime Prior", x=labels, y=prior,
        marker_color="#FFD600", opacity=0.7,
        hovertemplate="<b>%{x}</b> prior: %{y:.1f}%<extra></extra>"))
    fig.add_trace(go.Bar(name="Actual Allocation", x=labels, y=actual,
        marker_color=PRIMARY_COLOR, opacity=0.9,
        hovertemplate="<b>%{x}</b> actual: %{y:.1f}%<extra></extra>"))
    fig.update_layout(**_base(f"🎯 Current Allocation vs {regime} Prior", 360, ytitle="Weight (%)"))
    fig.update_layout(barmode="group")
    return fig


def nlp_sentiment_bars(scores: dict, sector_labels: dict) -> go.Figure:
    items = [(k,v) for k,v in scores.items() if k != "market"]
    items.sort(key=lambda x: x[1])
    labels = [sector_labels.get(k,k) for k,_ in items]
    values = [v for _,v in items]
    colors = [POSITIVE_COLOR if v < 0.35 else (NEGATIVE_COLOR if v > 0.65 else NEUTRAL_COLOR)
              for v in values]
    fig = go.Figure(go.Bar(
        x=values, y=labels, orientation="h",
        marker_color=colors, text=[f"{v:.2f}" for v in values],
        textposition="outside",
        hovertemplate="<b>%{y}</b>: %{x:.3f}<extra></extra>",
    ))
    fig.add_vline(x=0.35, line_dash="dash", line_color=POSITIVE_COLOR, line_width=1)
    fig.add_vline(x=0.65, line_dash="dash", line_color=NEGATIVE_COLOR, line_width=1)
    fig.update_layout(**_base("🗞️ NLP Sector Sentiment  (0=Bullish · 1=Bearish)", 380,
                               xtitle="Fear Score"))
    fig.update_xaxes(range=[0,1])
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# Monte Carlo Charts
# ══════════════════════════════════════════════════════════════════════════════

def monte_carlo_fan(mc: dict) -> go.Figure:
    paths  = mc["paths"]
    cap    = mc["initial_capital"]
    n_days = mc["horizon_days"]
    x      = list(range(n_days + 1))

    pcts = mc["percentiles"]
    fig  = go.Figure()

    # Fan bands
    bands = [
        (5,  95, "rgba(102,126,234,0.10)", "5%–95%"),
        (10, 90, "rgba(102,126,234,0.15)", "10%–90%"),
        (25, 75, "rgba(102,126,234,0.22)", "25%–75%"),
    ]
    for lo, hi, color, name in bands:
        p_lo = np.percentile(paths, lo, axis=0)
        p_hi = np.percentile(paths, hi, axis=0)
        fig.add_trace(go.Scatter(
            x=x+x[::-1], y=list(p_hi)+list(p_lo[::-1]),
            fill="toself", fillcolor=color, line=dict(width=0),
            name=name, hoverinfo="skip",
        ))

    # Median
    p50 = np.percentile(paths, 50, axis=0)
    fig.add_trace(go.Scatter(
        x=x, y=p50, name="Median",
        line=dict(color=PRIMARY_COLOR, width=2.5),
        hovertemplate="<b>Median</b>: ₹%{y:,.0f}<extra></extra>",
    ))

    # Capital line
    fig.add_hline(y=cap, line_dash="dash", line_color="#555",
                  annotation_text="Initial Capital", annotation_font_color="#888")

    # Sample paths (faint)
    rng   = np.random.default_rng(42)
    samp  = rng.choice(len(paths), size=min(100, len(paths)), replace=False)
    for idx in samp:
        fig.add_trace(go.Scatter(
            x=x, y=paths[idx],
            line=dict(color="rgba(150,150,200,0.06)", width=1),
            showlegend=False, hoverinfo="skip",
        ))

    fig.update_layout(**_base(
        f"🎲 Monte Carlo Simulation  ·  {mc['n_sims']:,} paths  ·  "
        f"{mc['horizon_days']//21}M horizon", 480, ytitle="Portfolio Value (₹)"
    ))
    fig.update_yaxes(tickformat=",.0f")
    return fig


def monte_carlo_distribution(mc: dict) -> go.Figure:
    final = mc["final"]
    cap   = mc["initial_capital"]
    fig   = go.Figure()

    fig.add_trace(go.Histogram(
        x=final, nbinsx=80,
        marker_color=PRIMARY_COLOR, opacity=0.75,
        name="Final Values",
        hovertemplate="₹%{x:,.0f}: %{y} paths<extra></extra>",
    ))

    for p, col, lbl in [
        (mc["percentiles"][5],  NEGATIVE_COLOR, "5th pct"),
        (mc["percentiles"][50], NEUTRAL_COLOR,  "Median"),
        (mc["percentiles"][95], POSITIVE_COLOR, "95th pct"),
    ]:
        fig.add_vline(x=p, line_dash="dash", line_color=col,
                      annotation_text=f"  {lbl}: ₹{p/1e5:.1f}L",
                      annotation_font_color=col)

    fig.add_vline(x=cap, line_dash="solid", line_color="#555", line_width=2,
                  annotation_text=f"  Capital: ₹{cap/1e5:.1f}L",
                  annotation_font_color="#888")

    fig.update_layout(**_base("📊 Final Value Distribution After 1 Year",
                               300, xtitle="Portfolio Value (₹)", ytitle="Paths"))
    fig.update_xaxes(tickformat=",.0f")
    return fig
