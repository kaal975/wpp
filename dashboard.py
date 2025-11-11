# Wind Pattern Dashboard — Streamlit App
# Save this as: dashboard.py
# Run with: streamlit run dashboard.py

import os
import io
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

# -------------------------------
# Helpers
# -------------------------------

@st.cache_data(show_spinner=False)
def load_data():
    """Loads processed data produced by your notebook. If not found, tries to 
    generate a tiny synthetic sample so the app still loads."""
    features_path = os.path.join("outputs", "wind_features.csv")
    regime_path = os.path.join("outputs", "regime_summary.csv")

    if os.path.exists(features_path):
        df = pd.read_csv(features_path, parse_dates=["timestamp"])  
    else:
        # Minimal fallback demo
        periods = 6 * 24 * 3  # 3 days @ 10-min sampling
        start = pd.Timestamp.utcnow().floor('H') - pd.Timedelta(hours=periods//6*1)
        ts = pd.date_range(start, periods=periods, freq="10min")
        minutes = np.arange(periods) * 10
        daily = 3 + 2.5 * np.sin(2 * np.pi * minutes / (24*60))
        speed = np.clip(daily + np.random.normal(scale=0.8, size=periods), 0, None)
        direction = (180 + 30 * np.sin(2 * np.pi * minutes / (3*24*60)) + np.random.vonmises(0, 4, periods) * 180/np.pi) % 360
        rpm = np.clip(10 * (speed - 3), 0, 200)
        rho, area, cp = 1.225, 10.0, 0.35
        power = 0.5 * rho * area * cp * np.power(speed, 3) * 0.001 + np.random.normal(scale=0.05, size=periods)
        power = np.clip(power, 0, None)
        df = pd.DataFrame({
            'timestamp': ts,
            'speed_mps': speed,
            'speed_smooth': pd.Series(speed).rolling(11, min_periods=1, center=True).mean(),
            'direction_deg': direction,
            'direction_circ_mean': np.nan,  # filled later
            'rpm': rpm,
            'power_kw': power,
        })
        df['is_gust'] = df['speed_mps'] > (df['speed_smooth'] + 3.0)
        # simple 3-bin regime by speed for fallback
        df['regime'] = pd.qcut(df['speed_smooth'], q=3, labels=False)

    if os.path.exists(regime_path):
        summary = pd.read_csv(regime_path)
    else:
        summary = (
            df.groupby('regime')
              .agg(mean_speed=('speed_mps', 'mean'),
                   gust_count=('is_gust', 'sum'),
                   samples=('timestamp', 'count'))
              .reset_index()
        )
        summary['dominant_direction'] = 0.0

    # Ensure correct dtypes
    if df['timestamp'].dtype == 'O':
        df['timestamp'] = pd.to_datetime(df['timestamp'])

    # Fill direction_circ_mean if missing
    if df['direction_circ_mean'].isna().all():
        df['direction_circ_mean'] = rolling_circular_mean(df['direction_deg'].values, window=12)

    return df, summary


def circular_mean_deg(angles_deg: np.ndarray) -> float:
    a = np.deg2rad(angles_deg)
    s, c = np.sin(a).mean(), np.cos(a).mean()
    mean = np.rad2deg(np.arctan2(s, c))
    return mean if mean >= 0 else mean + 360


def rolling_circular_mean(angles_deg: np.ndarray, window: int = 12) -> np.ndarray:
    out = np.empty_like(angles_deg, dtype=float)
    for i in range(len(angles_deg)):
        start = max(0, i - window + 1)
        out[i] = circular_mean_deg(angles_deg[start:i+1])
    return out


def make_wind_rose(df: pd.DataFrame, bins: int = 16, weight: str = 'speed_mps') -> go.Figure:
    """Polar bar (wind rose). Weight by 'speed_mps' (default) or by counts."""
    angles = np.linspace(0, 360, bins + 1)
    counts = np.zeros(bins)

    dirs = df['direction_deg'].values % 360
    weights = df[weight].values if weight in df.columns else np.ones_like(dirs)

    for i in range(bins):
        lo, hi = angles[i], angles[i+1]
        mask = (dirs >= lo) & (dirs < hi)
        counts[i] = weights[mask].sum()

    theta = angles[:-1] + (angles[1]-angles[0])/2
    fig = go.Figure(data=go.Barpolar(
        r=counts,
        theta=theta,
        width=np.repeat((360/bins), bins),
        marker_line_color="black",
        marker_line_width=1,
        opacity=0.9,
    ))
    fig.update_layout(
        polar=dict(angularaxis=dict(direction='clockwise', rotation=90)),
        margin=dict(l=20, r=20, t=40, b=20),
        title="Wind Rose (speed-weighted)",
    )
    return fig


def kpi_block(df: pd.DataFrame):
    # Overall KPIs on filtered data
    mean_speed = df['speed_mps'].mean()
    circ_dir = circular_mean_deg(df['direction_deg'].values)
    gust_freq = df['is_gust'].mean() * 100 if 'is_gust' in df.columns else 0.0
    avg_power = df['power_kw'].mean()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Mean wind speed", f"{mean_speed:.2f} m/s")
    c2.metric("Dominant direction", f"{circ_dir:.0f}°")
    c3.metric("Gust frequency", f"{gust_freq:.1f}%")
    c4.metric("Avg. power", f"{avg_power:.2f} kW")


def download_button(df: pd.DataFrame, label: str, filename: str):
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button(label=label, data=csv, file_name=filename, mime='text/csv')


# -------------------------------
# UI — Sidebar
# -------------------------------

st.set_page_config(page_title="Wind Pattern Dashboard", layout="wide")
st.title("🌬 Wind Pattern Dashboard")
st.caption("Interactive view of wind regimes, gusts, directions, and turbine output")

with st.sidebar:
    st.header("Filters")
    df, summary = load_data()

    # Date range selector
    min_date, max_date = df['timestamp'].min(), df['timestamp'].max()
    start, end = st.date_input(
        "Date range",
        value=(min_date.date(), max_date.date()),
        min_value=min_date.date(),
        max_value=max_date.date(),
    )
    # Convert to timestamps
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)

    # Regime multiselect
    regimes = sorted(df['regime'].dropna().unique().tolist())
    selected_regimes = st.multiselect("Regimes", regimes, default=regimes, help="Wind behavior clusters (0..2 in the demo)")

    # Gust highlighting
    highlight_gusts = st.checkbox("Highlight gusts", value=True)

    # Wind-rose weighting
    rose_weight = st.radio("Wind rose weighting", ["speed_mps", "count"], index=0, horizontal=True,
                           help="Weight sectors by speed or just count occurrences")

# Apply filters
mask = (df['timestamp'] >= start_ts) & (df['timestamp'] <= end_ts)
if selected_regimes:
    mask &= df['regime'].isin(selected_regimes)

fdf = df.loc[mask].copy()
if rose_weight == 'count':
    fdf['count'] = 1.0

# Map regime -> color (stable across charts)
palette = px.colors.qualitative.Set2
reg_list = sorted(df['regime'].dropna().unique())
color_map = {int(r): palette[i % len(palette)] for i, r in enumerate(reg_list)}

# -------------------------------
# KPIs
# -------------------------------

kpi_block(fdf)

# -------------------------------
# Tabs with Charts
# -------------------------------

tabs = st.tabs(["Time Series", "Wind Rose", "Power vs Speed", "Regime Summary", "Data Preview"])

with tabs[0]:
    st.subheader("Wind Speed over Time")
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=fdf['timestamp'], y=fdf['speed_mps'], name='Speed (m/s)', mode='lines'))
    if 'speed_smooth' in fdf.columns:
        fig1.add_trace(go.Scatter(x=fdf['timestamp'], y=fdf['speed_smooth'], name='Smoothed', mode='lines'))
    if highlight_gusts and 'is_gust' in fdf.columns and fdf['is_gust'].any():
        gust_pts = fdf.loc[fdf['is_gust']]
        fig1.add_trace(go.Scatter(
            x=gust_pts['timestamp'], y=gust_pts['speed_mps'], name='Gust', mode='markers', marker=dict(size=7, symbol='triangle-up')
        ))
    fig1.update_layout(margin=dict(l=20, r=20, t=30, b=20), legend=dict(orientation='h'))
    st.plotly_chart(fig1, use_container_width=True)

    st.subheader("Power Output over Time")
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=fdf['timestamp'], y=fdf['power_kw'], name='Power (kW)', mode='lines'))
    fig2.update_layout(margin=dict(l=20, r=20, t=30, b=20), legend=dict(orientation='h'))
    st.plotly_chart(fig2, use_container_width=True)

with tabs[1]:
    st.subheader("Wind Rose")
    fig_rose = make_wind_rose(fdf, bins=16, weight='speed_mps' if rose_weight == 'speed_mps' else 'count')
    st.plotly_chart(fig_rose, use_container_width=True)

with tabs[2]:
    st.subheader("Power vs Speed — colored by Regime")
    # color by regime for consistent palette
    fig_scatter = go.Figure()
    for r in selected_regimes or reg_list:
        sub = fdf[fdf['regime'] == r]
        if sub.empty:
            continue
        fig_scatter.add_trace(go.Scatter(
            x=sub['speed_mps'], y=sub['power_kw'], mode='markers', name=f"Regime {int(r)}",
            marker=dict(color=color_map.get(int(r), '#666'), opacity=0.7, size=6)
        ))
    fig_scatter.update_layout(xaxis_title='Wind Speed (m/s)', yaxis_title='Power (kW)', margin=dict(l=20, r=20, t=30, b=20))
    st.plotly_chart(fig_scatter, use_container_width=True)

with tabs[3]:
    st.subheader("Regime Summary")
    st.dataframe(summary, use_container_width=True)
    st.caption("If you ran the full notebook, this shows K-Means regime stats. In fallback mode it's derived on the fly.")

with tabs[4]:
    st.subheader("Filtered Data Preview")
    st.dataframe(fdf.head(1000), use_container_width=True)
    c1, c2 = st.columns(2)
    with c1:
        download_button(fdf, "Download filtered CSV", "filtered_wind_features.csv")
    with c2:
        download_button(summary, "Download regime summary CSV", "regime_summary.csv")

# -------------------------------
# Explanations
# -------------------------------

with st.expander("What am I looking at?"):
    st.markdown(
        """
        - *Time Series*: Raw vs smoothed wind speed, plus gust markers if enabled. Power time-series shows turbine output.
        - *Wind Rose*: Polar bars show which directions dominate. Weight by speed for intensity or by counts for frequency.
        - *Power vs Speed*: Scatter colored by regime (wind behavior clusters). Helps compare turbine performance in different regimes.
        - *Regime Summary*: Aggregated stats per regime from your analysis notebook.
        """
    )
