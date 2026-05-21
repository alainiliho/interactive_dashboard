
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy import stats

st.set_page_config(page_title="The Interactive Dashboard", layout="wide")

@st.cache_data
def load():
    return pd.read_parquet("usa_indicators_1990_2022.parquet")

df = load()
years = df.index.year

UNITS = {
    "GDP_per_capita":"USD","GDP_growth":"%","Inflation":"%",
    "Unemployment":"%","Life_expectancy":"years","Infant_mortality":"‰",
    "Gross_capital":"% GDP","Trade":"% GDP","Current_account":"% GDP",
    "Market_cap":"% GDP",
}

# ── Sidebar ──────────────────────────────────────────────────
st.sidebar.title("⚙️  Controls")
variable  = st.sidebar.selectbox("Variable", df.columns.tolist())
year_rng  = st.sidebar.slider("Year range", int(years.min()),
                               int(years.max()),
                              (int(years.min()), int(years.max())))
win       = st.sidebar.slider("SMA / EMA / Volatility window", 2, 10, 5)
show_raw  = st.sidebar.checkbox("Show raw data",    value=True)
show_sma  = st.sidebar.checkbox("Show SMA",         value=True)
show_ema  = st.sidebar.checkbox("Show EMA",         value=True)
show_vol  = st.sidebar.checkbox("Show Volatility",  value=True)
show_out  = st.sidebar.checkbox("Show Outliers",    value=True)

# ── Filter ───────────────────────────────────────────────────
mask = (years >= year_rng[0]) & (years <= year_rng[1])
s    = df.loc[mask, variable]

sma  = s.rolling(win, center=True).mean()
ema  = s.ewm(span=win, adjust=False).mean()
vol  = s.rolling(win, center=True).std()
z    = np.abs(stats.zscore(s.dropna()))
out_idx = s.index[np.abs(stats.zscore(s.fillna(s.mean()))) > 2.5]

# ── Dynamic stats ─────────────────────────────────────────────
st.title(f"📊  The Interactive Dashboard — {variable.replace('_',' ')} [{UNITS.get(variable,'')}]")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Mean",     f"{s.mean():.3f}")
c2.metric("Variance", f"{s.var():.3f}")
c3.metric("Kurtosis", f"{s.kurtosis():.3f}")
c4.metric("Skewness", f"{s.skew():.3f}")

# ── Plotly figure ─────────────────────────────────────────────
fig = go.Figure()

if show_vol:
    fig.add_trace(go.Scatter(
        x=pd.concat([sma.index.to_series(), sma.index.to_series()[::-1]]),
        y=pd.concat([sma+vol, (sma-vol)[::-1]]),
        fill="toself", fillcolor="rgba(100,149,237,0.15)",
        line=dict(color="rgba(255,255,255,0)"),
        name=f"Volatility ±1σ (w={win})"
    ))

if show_raw:
    fig.add_trace(go.Scatter(x=s.index, y=s.values,
        mode="lines+markers", name="Raw",
        line=dict(color="royalblue", width=1.8),
        marker=dict(size=5)))

if show_sma:
    fig.add_trace(go.Scatter(x=sma.index, y=sma.values,
        mode="lines", name=f"SMA({win})",
        line=dict(color="black", dash="dash", width=2)))

if show_ema:
    fig.add_trace(go.Scatter(x=ema.index, y=ema.values,
        mode="lines", name=f"EMA({win})",
        line=dict(color="crimson", dash="dot", width=2)))

if show_out and len(out_idx) > 0:
    fig.add_trace(go.Scatter(x=out_idx, y=s[out_idx],
        mode="markers", name="Outliers (|Z|>2.5)",
        marker=dict(color="orange", size=10, symbol="x",
                    line=dict(width=2, color="darkorange"))))

fig.update_layout(
    xaxis_title="Year", yaxis_title=UNITS.get(variable,""),
    hovermode="x unified", template="plotly_white",
    legend=dict(orientation="h", yanchor="bottom", y=1.02),
    height=480
)
st.plotly_chart(fig, width="stretch")

with st.expander("📄  Show data table"):
    st.dataframe(
        pd.DataFrame({
            "Year"      : s.index.year,
            variable    : s.values,
            f"SMA({win})": sma.values,
            f"EMA({win})": ema.values,
            "Volatility": vol.values,
        }).set_index("Year").style.format("{:.3f}", na_rep="—")
    )
