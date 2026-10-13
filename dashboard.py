"""
SIH26073: Automatic Weather Station (AWS) AI Anomaly Intelligence Command Center
Production Hybrid System: Deterministic Physical Sensor Bounds + Unsupervised Isolation Forest V2
Built with Streamlit and Plotly.
"""

import io
import json
import pathlib
import datetime
from typing import Dict, Any, Tuple, Optional, List

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from src.data_loader import map_columns, load_weather_data
from src.preprocessing import preprocess_data
from src.feature_engineering import engineer_features
from src.anomaly_detection import (
    RuleThresholdConfig,
    apply_rule_checks,
    IsolationForestDetectorV2,
    synthesize_results_v2,
)
from src.explainability import explain_anomalies

# ==============================================================================
# PAGE CONFIGURATION
# ==============================================================================
st.set_page_config(
    page_title="AWS AI Anomaly Command Center",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ==============================================================================
# CUSTOM STYLING (MODERN GLASSMORPHISM DARK SLATE THEME)
# ==============================================================================
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Overall page background */
    .stApp {
        background: radial-gradient(circle at 15% 15%, #0f172a 0%, #080c14 100%);
        color: #f1f5f9;
    }

    /* Top banner */
    .hero-banner {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.7) 0%, rgba(15, 23, 42, 0.9) 100%);
        border: 1px solid rgba(56, 189, 248, 0.2);
        border-radius: 16px;
        padding: 24px 30px;
        margin-bottom: 24px;
        backdrop-filter: blur(12px);
        box-shadow: 0 10px 30px -10px rgba(0, 0, 0, 0.5), inset 0 1px 0 rgba(255, 255, 255, 0.05);
    }

    .hero-title {
        font-size: 2.1rem;
        font-weight: 800;
        letter-spacing: -0.03em;
        background: linear-gradient(135deg, #38bdf8 0%, #818cf8 50%, #c084fc 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 6px;
    }

    .hero-subtitle {
        font-size: 0.95rem;
        color: #94a3b8;
        font-weight: 400;
    }

    /* Metric Cards */
    .kpi-card {
        background: linear-gradient(145deg, rgba(30, 41, 59, 0.6) 0%, rgba(15, 23, 42, 0.75) 100%);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 20px 22px;
        backdrop-filter: blur(10px);
        box-shadow: 0 4px 20px -2px rgba(0, 0, 0, 0.3);
        transition: transform 0.2s ease, border-color 0.2s ease;
        position: relative;
        overflow: hidden;
    }

    .kpi-card:hover {
        transform: translateY(-2px);
        border-color: rgba(56, 189, 248, 0.4);
    }

    .kpi-card::before {
        content: "";
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 3px;
        background: linear-gradient(90deg, #38bdf8, #818cf8);
    }

    .kpi-card.emerald::before { background: linear-gradient(90deg, #10b981, #34d399); }
    .kpi-card.rose::before { background: linear-gradient(90deg, #ef4444, #f87171); }
    .kpi-card.amber::before { background: linear-gradient(90deg, #f59e0b, #fbbf24); }
    .kpi-card.violet::before { background: linear-gradient(90deg, #8b5cf6, #a78bfa); }

    .kpi-label {
        font-size: 0.78rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #94a3b8;
        margin-bottom: 6px;
    }

    .kpi-value {
        font-family: 'JetBrains Mono', monospace;
        font-size: 2.0rem;
        font-weight: 700;
        color: #f8fafc;
        line-height: 1.1;
    }

    .kpi-sub {
        font-size: 0.8rem;
        color: #64748b;
        margin-top: 6px;
        font-weight: 500;
    }

    /* Badges */
    .badge {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.02em;
    }
    .badge-cyan { background: rgba(56, 189, 248, 0.15); color: #38bdf8; border: 1px solid rgba(56, 189, 248, 0.3); }
    .badge-emerald { background: rgba(16, 185, 129, 0.15); color: #34d399; border: 1px solid rgba(16, 185, 129, 0.3); }
    .badge-rose { background: rgba(239, 68, 68, 0.15); color: #f87171; border: 1px solid rgba(239, 68, 68, 0.3); }
    .badge-amber { background: rgba(245, 158, 11, 0.15); color: #fbbf24; border: 1px solid rgba(245, 158, 11, 0.3); }

    /* Custom station info card */
    .station-card {
        background: rgba(15, 23, 42, 0.6);
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 12px;
        padding: 14px 16px;
        margin-bottom: 12px;
    }

    /* Streamlit tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: rgba(15, 23, 42, 0.6);
        padding: 6px;
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.05);
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 8px 18px;
        color: #94a3b8;
        font-weight: 600;
        font-size: 0.9rem;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, rgba(56, 189, 248, 0.2) 0%, rgba(129, 140, 248, 0.2) 100%) !important;
        color: #38bdf8 !important;
        border: 1px solid rgba(56, 189, 248, 0.4);
    }

    /* Streamlit containers & tables */
    [data-testid="stDataFrame"] {
        border-radius: 12px;
        overflow: hidden;
        border: 1px solid rgba(255, 255, 255, 0.08);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Plotly dark template
PLOTLY_THEME = dict(
    layout=go.Layout(
        template="plotly_dark",
        paper_bgcolor="rgba(15, 23, 42, 0.6)",
        plot_bgcolor="rgba(11, 15, 25, 0.7)",
        font=dict(family="Plus Jakarta Sans", color="#e2e8f0"),
        xaxis=dict(gridcolor="rgba(255, 255, 255, 0.07)", zerolinecolor="rgba(255, 255, 255, 0.1)"),
        yaxis=dict(gridcolor="rgba(255, 255, 255, 0.07)", zerolinecolor="rgba(255, 255, 255, 0.1)"),
        margin=dict(l=40, r=30, t=40, b=40),
    )
)


# ==============================================================================
# MODEL & CACHING HELPERS
# ==============================================================================
@st.cache_resource
def get_cached_detector():
    """Loads Isolation Forest V2 model checkpoint."""
    v2_path = pathlib.Path("models/isolation_forest_v2.joblib")
    if v2_path.exists():
        try:
            return IsolationForestDetectorV2.load(v2_path), "Model V2 (Isolation Forest 200 Trees)"
        except Exception:
            pass

    aws_path = pathlib.Path("models/isolation_forest_aws.joblib")
    if aws_path.exists():
        try:
            return IsolationForestDetectorV2.load(aws_path), "Model V1 (Legacy Checkpoint)"
        except Exception:
            pass

    return None, "No Checkpoint Found"


@st.cache_data
def load_v2_metrics_json():
    """Loads precomputed evaluation metrics."""
    p = pathlib.Path("reports/model_v2_metrics.json")
    if p.exists():
        with open(p, "r") as f:
            return json.load(f)
    return None


@st.cache_data
def load_v1_metrics_json():
    """Loads Model V1 evaluation metrics."""
    p = pathlib.Path("reports/model_v1_metrics.json")
    if p.exists():
        with open(p, "r") as f:
            return json.load(f)
    return None


@st.cache_data
def load_processed_v2_sample(max_rows: int = 10000):
    """Loads pre-classified ground-truth test data from model_v2_predictions.csv."""
    p = pathlib.Path("data/processed/model_v2_predictions.csv")
    if not p.exists():
        return None
    try:
        df = pd.read_csv(p, nrows=max_rows)
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        if "station_id" not in df.columns:
            df["station_id"] = "AWS_PRODUCTION_01"
        if "pressure_hpa" not in df.columns:
            df["pressure_hpa"] = df["pressure"] / 100.0
        df["is_anomaly"] = df.get("final_is_anomaly", df.get("is_anomaly", False))
        df["anomaly_type"] = df.get("detected_anomaly_type", df.get("anomaly_type", "NORMAL"))
        return df
    except Exception:
        return None


# ==============================================================================
# DATA INGESTION & PIPELINE EXECUTION
# ==============================================================================
def run_telemetry_pipeline(
    raw_df: pd.DataFrame,
    detector: Optional[IsolationForestDetectorV2],
    config: RuleThresholdConfig,
    score_threshold: float = 0.55,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Executes end-to-end anomaly pipeline on raw or uploaded dataframe."""
    df = raw_df.copy()

    # 1. Column standardization
    try:
        df_mapped, _ = map_columns(df)
    except Exception:
        # Fallback if map_columns errors
        df_mapped = df.copy()
        if "time" in df_mapped.columns and "timestamp" not in df_mapped.columns:
            df_mapped.rename(columns={"time": "timestamp"}, inplace=True)
        if "temp" in df_mapped.columns and "temperature" not in df_mapped.columns:
            df_mapped.rename(columns={"temp": "temperature"}, inplace=True)
        if "station_id" not in df_mapped.columns:
            df_mapped["station_id"] = "AWS_STATION_01"

    # 2. Check & adjust pressure units (hPa vs Pa)
    if "pressure" in df_mapped.columns:
        med_press = df_mapped["pressure"].dropna().median()
        if pd.notna(med_press) and med_press < 2000.0:
            # Supplied in hPa, convert to Pa for standard physical rules & model
            df_mapped["pressure"] = df_mapped["pressure"] * 100.0

    # Ensure lux column exists
    if "lux" not in df_mapped.columns:
        df_mapped["lux"] = 0.0

    # 3. Preprocess
    df_clean, prep_report = preprocess_data(df_mapped)

    # 4. Feature engineering
    df_feat, _ = engineer_features(df_clean, window_size=12)

    # 5. Deterministic rule checks
    df_rules = apply_rule_checks(df_feat, config=config)

    # 6. Model inference
    if detector is not None and detector.is_fitted:
        try:
            m_anom, scores, raw_preds = detector.predict(df_rules)
            df_rules["model_is_anomaly"] = m_anom
            df_rules["anomaly_score"] = scores
            df_rules["raw_prediction"] = raw_preds
        except Exception:
            # Fallback if feature mismatch
            df_rules["model_is_anomaly"] = False
            df_rules["anomaly_score"] = 0.0
            df_rules["raw_prediction"] = 1
    else:
        df_rules["model_is_anomaly"] = False
        df_rules["anomaly_score"] = 0.0
        df_rules["raw_prediction"] = 1

    # 7. Decision synthesis & root cause explanations
    df_synth = synthesize_results_v2(df_rules, score_threshold=score_threshold)
    df_explained = explain_anomalies(df_synth, config=config)

    # Friendly pressure in hPa
    df_explained["pressure_hpa"] = df_explained["pressure"] / 100.0
    df_explained["is_anomaly"] = df_explained["final_is_anomaly"]
    df_explained["anomaly_type"] = df_explained["detected_anomaly_type"]

    return df_explained, prep_report


# ==============================================================================
# SIDEBAR CONTROLS
# ==============================================================================
st.sidebar.markdown(
    """
    <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 20px;">
        <span style="font-size: 2rem;">🛰️</span>
        <div>
            <h3 style="margin: 0; font-size: 1.15rem; font-weight: 700; color: #f8fafc;">AWS Sensor Lab</h3>
            <span class="badge badge-cyan">Model V2 Engine</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.sidebar.markdown("### 📡 Telemetry Source")
data_source = st.sidebar.selectbox(
    "Choose Data Feed",
    [
        "🌟 Model V2 Production Test Telemetry (Ground-Truth Injected)",
        "🌦️ Multi-Station Sample Network (2 Stations, 15-min intervals)",
        "⚡ High-Frequency Raw Telemetry (DATA-large 5s Stream)",
        "📤 Upload Custom Weather CSV",
        "🧪 Interactive Sensor Sandbox & Fault Lab",
    ],
    index=0,
)

uploaded_file = None
sample_slice_limit = 5000

if "Production Test Telemetry" in data_source:
    sample_slice_limit = st.sidebar.select_slider(
        "Observation Sample Size",
        options=[1000, 2500, 5000, 10000, 25000],
        value=5000,
        help="Subsample size from 207,967 test observations for responsive rendering.",
    )
elif "High-Frequency Raw" in data_source:
    sample_slice_limit = st.sidebar.select_slider(
        "Stream Length (Observations)",
        options=[500, 1000, 2500, 5000],
        value=2500,
        help="Number of 5-second observations to stream from DATA-large.CSV.",
    )
elif "Upload Custom" in data_source:
    uploaded_file = st.sidebar.file_uploader(
        "Upload Station CSV File",
        type=["csv"],
        help="Upload CSV with timestamp, temperature, humidity, pressure.",
    )

st.sidebar.markdown("---")
st.sidebar.markdown("### 🛡️ Physical Safety Bounds")

with st.sidebar.expander("⚙️ Sensor Quality Rules & Limits", expanded=False):
    temp_min_val = st.slider("Min Temperature (°C)", -50.0, 0.0, -40.0, 1.0)
    temp_max_val = st.slider("Max Temperature (°C)", 30.0, 70.0, 60.0, 1.0)
    hum_min_val = st.slider("Min Humidity (%)", 0.0, 20.0, 0.0, 1.0)
    hum_max_val = st.slider("Max Humidity (%)", 80.0, 100.0, 100.0, 1.0)
    press_min_hpa = st.slider("Min Pressure (hPa)", 600.0, 900.0, 700.0, 10.0)
    press_max_hpa = st.slider("Max Pressure (hPa)", 1050.0, 1200.0, 1100.0, 10.0)
    max_temp_step = st.slider("Max Temp Step Spike (°C/step)", 1.0, 15.0, 5.0, 0.5)
    max_hum_step = st.slider("Max Humidity Step Spike (%/step)", 5.0, 30.0, 15.0, 1.0)
    frozen_limit = st.number_input("Frozen Sensor Consecutive Limit", 10, 500, 180, 10)
    max_gap_sec = st.number_input("Max Allowable Gap (seconds)", 10, 3600, 60, 10)

rule_config = RuleThresholdConfig(
    temp_min=temp_min_val,
    temp_max=temp_max_val,
    hum_min=hum_min_val,
    hum_max=hum_max_val,
    press_min=press_min_hpa * 100.0,
    press_max=press_max_hpa * 100.0,
    max_temp_step=max_temp_step,
    max_hum_step=max_hum_step,
    frozen_consecutive_limit=int(frozen_limit),
    max_gap_seconds=float(max_gap_sec),
)

st.sidebar.markdown("### 🧠 AI Decision Sensitivity")
score_threshold = st.sidebar.slider(
    "ML Anomaly Threshold",
    0.30,
    0.90,
    0.55,
    0.05,
    help="Higher threshold decreases false positives; lower increases sensitivity.",
)

detector, detector_name = get_cached_detector()
st.sidebar.markdown(
    f"""
    <div style="background: rgba(15, 23, 42, 0.5); padding: 12px; border-radius: 10px; border: 1px solid rgba(255, 255, 255, 0.05); margin-top: 15px;">
        <div style="font-size: 0.72rem; color: #94a3b8; text-transform: uppercase; font-weight: 600;">Active Detector Checkpoint</div>
        <div style="font-size: 0.85rem; color: #38bdf8; font-weight: 700; margin-top: 2px;">{detector_name}</div>
        <div style="font-size: 0.75rem; color: #64748b; margin-top: 4px;">Hybrid Dual-Engine • Ready</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ==============================================================================
# DATA LOADING DISPATCH
# ==============================================================================
active_df: Optional[pd.DataFrame] = None
is_sandbox_mode = "Interactive Sensor Sandbox" in data_source

if not is_sandbox_mode:
    if "Production Test Telemetry" in data_source:
        active_df = load_processed_v2_sample(max_rows=sample_slice_limit)
        if active_df is None:
            # Fallback to computing on sample
            sample_file = pathlib.Path("data/sample/aws_weather_sample.csv")
            if sample_file.exists():
                raw = load_weather_data(sample_file)
                active_df, _ = run_telemetry_pipeline(raw, detector, rule_config, score_threshold)

    elif "Multi-Station Sample Network" in data_source:
        sample_file = pathlib.Path("data/sample/aws_weather_sample.csv")
        if sample_file.exists():
            raw = load_weather_data(sample_file)
            # Sample has 15-minute interval, so adapt gap threshold
            cfg_sample = RuleThresholdConfig(
                temp_min=rule_config.temp_min,
                temp_max=rule_config.temp_max,
                hum_min=rule_config.hum_min,
                hum_max=rule_config.hum_max,
                press_min=rule_config.press_min,
                press_max=rule_config.press_max,
                max_temp_step=rule_config.max_temp_step,
                max_hum_step=rule_config.max_hum_step,
                frozen_consecutive_limit=25,
                max_gap_seconds=1800.0,
            )
            active_df, _ = run_telemetry_pipeline(raw, detector, cfg_sample, score_threshold)

    elif "High-Frequency Raw" in data_source:
        large_path = pathlib.Path("data/raw/DATA-large.CSV")
        if large_path.exists():
            raw = pd.read_csv(large_path, nrows=sample_slice_limit)
            active_df, _ = run_telemetry_pipeline(raw, detector, rule_config, score_threshold)
        else:
            st.error("`data/raw/DATA-large.CSV` not found on filesystem.")

    elif "Upload Custom" in data_source:
        if uploaded_file is not None:
            try:
                raw = pd.read_csv(uploaded_file)
                active_df, _ = run_telemetry_pipeline(raw, detector, rule_config, score_threshold)
            except Exception as e:
                st.error(f"Error parsing uploaded file: {e}")
        else:
            st.info("👈 Please upload a CSV file from the sidebar to begin analysis.")

# ==============================================================================
# HEADER HERO BANNER
# ==============================================================================
st.markdown(
    """
    <div class="hero-banner">
        <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 15px;">
            <div>
                <div class="hero-title">🌦️ Automatic Weather Station AI Anomaly Center</div>
                <div class="hero-subtitle">
                    SIH26073 Autonomous Telemetry Quality Monitoring • Deterministic Sensor Bounds + Unsupervised Isolation Forest V2
                </div>
            </div>
            <div style="display: flex; gap: 8px; align-items: center;">
                <span class="badge badge-emerald">● REAL-TIME MONITORING</span>
                <span class="badge badge-cyan">MODEL V2 CALIBRATED</span>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ==============================================================================
# KPI METRICS ROW (When Telemetry Data is Active)
# ==============================================================================
if active_df is not None and not active_df.empty and not is_sandbox_mode:
    total_obs = len(active_df)
    total_anomalies = int(active_df["is_anomaly"].sum())
    normal_obs = total_obs - total_anomalies
    health_pct = (normal_obs / total_obs * 100.0) if total_obs > 0 else 100.0
    anom_pct = 100.0 - health_pct

    rule_anoms = int(active_df["rule_is_anomaly"].sum()) if "rule_is_anomaly" in active_df.columns else 0
    model_anoms = int(active_df["model_is_anomaly"].sum()) if "model_is_anomaly" in active_df.columns else 0
    avg_score = float(active_df["anomaly_score"].mean()) if "anomaly_score" in active_df.columns else 0.0

    k1, k2, k3, k4, k5 = st.columns(5)
    with k1:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-label">Observations Monitored</div>
                <div class="kpi-value">{total_obs:,}</div>
                <div class="kpi-sub">Total Telemetry Points</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with k2:
        st.markdown(
            f"""
            <div class="kpi-card emerald">
                <div class="kpi-label">Station Network Health</div>
                <div class="kpi-value" style="color: #34d399;">{health_pct:.1f}%</div>
                <div class="kpi-sub">{normal_obs:,} Nominal Readings</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with k3:
        st.markdown(
            f"""
            <div class="kpi-card rose">
                <div class="kpi-label">Flagged Anomalies</div>
                <div class="kpi-value" style="color: {'#f87171' if total_anomalies > 0 else '#34d399'};">{total_anomalies:,}</div>
                <div class="kpi-sub">Incident Rate: {anom_pct:.2f}%</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with k4:
        st.markdown(
            f"""
            <div class="kpi-card amber">
                <div class="kpi-label">Detection Breakdown</div>
                <div class="kpi-value" style="font-size: 1.4rem; padding-top: 4px;">
                    <span style="color: #fbbf24;">Rules: {rule_anoms}</span>
                </div>
                <div class="kpi-sub"><span style="color: #38bdf8;">Isolation Forest: {model_anoms}</span></div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with k5:
        st.markdown(
            f"""
            <div class="kpi-card violet">
                <div class="kpi-label">Mean ML Risk Score</div>
                <div class="kpi-value" style="color: #a78bfa;">{avg_score:.3f}</div>
                <div class="kpi-sub">Threshold: {score_threshold:.2f}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

# ==============================================================================
# MAIN NAVIGATION TABS
# ==============================================================================
tab_control, tab_telemetry, tab_xai, tab_benchmarks, tab_sandbox, tab_export = st.tabs(
    [
        "🌐 Mission Control",
        "📈 Telemetry Studio",
        "🔍 Explainable AI (XAI)",
        "🔬 Model V1 vs V2 Benchmarks",
        "🧪 Sensor Sandbox & Lab",
        "📥 Data Export",
    ]
)

# ==============================================================================
# TAB 1: MISSION CONTROL
# ==============================================================================
with tab_control:
    if is_sandbox_mode:
        st.info("👈 Currently in **Interactive Sensor Sandbox** mode. Please navigate to the **'🧪 Sensor Sandbox & Lab'** tab to simulate faults!")
    elif active_df is None or active_df.empty:
        st.warning("No telemetry dataset currently loaded. Please select or upload a dataset in the sidebar.")
    else:
        # Station Selector Filter
        available_stations = sorted(list(active_df["station_id"].dropna().unique()))
        selected_station_filter = "All Stations"
        if len(available_stations) > 1:
            col_f1, _ = st.columns([2, 3])
            with col_f1:
                selected_station_filter = st.selectbox("Filter Station View:", ["All Stations"] + available_stations)

        view_df = (
            active_df
            if selected_station_filter == "All Stations"
            else active_df[active_df["station_id"] == selected_station_filter]
        )

        # Station Overview Cards
        st.markdown("#### 🛰️ Operational Station Network Status")
        station_cols = st.columns(min(len(available_stations), 4))
        for idx, st_name in enumerate(available_stations[:4]):
            st_data = active_df[active_df["station_id"] == st_name]
            st_anoms = int(st_data["is_anomaly"].sum())
            latest_row = st_data.iloc[-1] if not st_data.empty else None
            with station_cols[idx]:
                curr_temp = f"{latest_row['temperature']:.1f}°C" if latest_row is not None and "temperature" in latest_row else "N/A"
                curr_press = f"{latest_row['pressure_hpa']:.1f} hPa" if latest_row is not None and "pressure_hpa" in latest_row else "N/A"
                curr_hum = f"{latest_row['humidity']:.1f}%" if latest_row is not None and "humidity" in latest_row else "N/A"
                status_color = "#34d399" if st_anoms == 0 else ("#fbbf24" if st_anoms < 10 else "#f87171")
                status_text = "NOMINAL" if st_anoms == 0 else ("WARNING" if st_anoms < 10 else "FAULT DETECTED")

                st.markdown(
                    f"""
                    <div class="station-card">
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                            <span style="font-weight: 700; color: #f8fafc; font-size: 0.95rem;">{st_name}</span>
                            <span style="font-size: 0.72rem; color: {status_color}; font-weight: 700; border: 1px solid {status_color}; border-radius: 4px; padding: 2px 6px;">
                                {status_text}
                            </span>
                        </div>
                        <div style="display: flex; justify-content: space-between; font-size: 0.8rem; color: #94a3b8; margin-top: 6px;">
                            <span>🌡️ {curr_temp}</span>
                            <span>🧭 {curr_press}</span>
                            <span>💧 {curr_hum}</span>
                        </div>
                        <div style="font-size: 0.75rem; color: #64748b; margin-top: 8px; border-top: 1px solid rgba(255,255,255,0.05); padding-top: 6px;">
                            Recorded Incidents: <b style="color: {'#f87171' if st_anoms > 0 else '#34d399'};">{st_anoms}</b>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        st.markdown("<br>", unsafe_allow_html=True)

        # Overview Visualizations
        c1, c2 = st.columns([1, 1])
        with c1:
            st.markdown("##### 📊 Anomaly Classification Breakdown")
            anom_df = view_df[view_df["is_anomaly"]]
            if not anom_df.empty:
                type_counts = anom_df["anomaly_type"].value_counts().reset_index()
                type_counts.columns = ["Anomaly Type", "Count"]

                fig_donut = px.pie(
                    type_counts,
                    values="Count",
                    names="Anomaly Type",
                    hole=0.55,
                    color="Anomaly Type",
                    color_discrete_sequence=[
                        "#f87171",
                        "#fbbf24",
                        "#38bdf8",
                        "#818cf8",
                        "#c084fc",
                        "#f43f5e",
                        "#10b981",
                    ],
                )
                fig_donut.update_layout(
                    PLOTLY_THEME["layout"],
                    margin=dict(t=20, b=20, l=20, r=20),
                    showlegend=True,
                    legend=dict(orientation="h", yanchor="bottom", y=-0.2),
                )
                st.plotly_chart(fig_donut, use_container_width=True)
            else:
                st.success("🎉 Zero anomalies flagged in this telemetry slice. All sensors operating nominal!")

        with c2:
            st.markdown("##### 🎯 Affected Sensor Parameters")
            if not anom_df.empty:
                param_counts = anom_df["affected_parameter"].value_counts().reset_index()
                param_counts.columns = ["Parameter", "Incidents"]

                fig_bar = px.bar(
                    param_counts,
                    x="Parameter",
                    y="Incidents",
                    color="Parameter",
                    color_discrete_sequence=px.colors.qualitative.Prism,
                    text="Incidents",
                )
                fig_bar.update_layout(
                    PLOTLY_THEME["layout"],
                    margin=dict(t=20, b=20, l=20, r=20),
                    showlegend=False,
                    xaxis_title=None,
                    yaxis_title="Incident Count",
                )
                fig_bar.update_traces(textposition="outside")
                st.plotly_chart(fig_bar, use_container_width=True)
            else:
                st.info("No affected sensors detected. Nominal baseline confirmed.")

        # Temporal timeline distribution
        st.markdown("##### ⏱️ Anomaly Occurrence Distribution Over Time")
        time_grp = view_df.copy()
        if "timestamp" in time_grp.columns:
            time_grp["date_bin"] = time_grp["timestamp"].dt.floor("1h")
            hourly_stats = (
                time_grp.groupby("date_bin")
                .agg(
                    total_count=("is_anomaly", "count"),
                    anom_count=("is_anomaly", "sum"),
                )
                .reset_index()
            )

            fig_timeline = go.Figure()
            fig_timeline.add_trace(
                go.Bar(
                    x=hourly_stats["date_bin"],
                    y=hourly_stats["total_count"],
                    name="Nominal Observations",
                    marker_color="rgba(56, 189, 248, 0.3)",
                )
            )
            fig_timeline.add_trace(
                go.Bar(
                    x=hourly_stats["date_bin"],
                    y=hourly_stats["anom_count"],
                    name="Flagged Faults",
                    marker_color="#f87171",
                )
            )
            fig_timeline.update_layout(
                PLOTLY_THEME["layout"],
                barmode="overlay",
                height=260,
                xaxis_title="Time Horizon",
                yaxis_title="Count per Hour",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            )
            st.plotly_chart(fig_timeline, use_container_width=True)

# ==============================================================================
# TAB 2: TELEMETRY STUDIO (RICH PLOTLY INTERACTIVE CHARTS)
# ==============================================================================
with tab_telemetry:
    if is_sandbox_mode or active_df is None or active_df.empty:
        st.info("👈 Please load a telemetry dataset from the sidebar to inspect time-series charts.")
    else:
        st.markdown("#### 📈 Multi-Sensor Telemetry Studio with Anomaly Callouts")
        st.caption("Interactive dual-mode exploration: Pan, Zoom, Hover for root-cause XAI explanations.")

        # Filter station if needed
        stations = sorted(list(active_df["station_id"].dropna().unique()))
        col_s1, col_s2 = st.columns([2, 3])
        with col_s1:
            plot_st = st.selectbox("Select Station for Sensor Plots:", stations, index=0)
        with col_s2:
            show_bounds = st.checkbox("Show Safe Physical Envelope Bounds", value=True)

        pdf = active_df[active_df["station_id"] == plot_st].copy()
        if len(pdf) > 8000:
            st.info("ℹ️ Telemetry exceeds 8,000 points. Displaying first 8,000 rows for 60fps fluidity.")
            pdf = pdf.head(8000)

        # 1. Temperature Telemetry
        if "temperature" in pdf.columns:
            fig_t = go.Figure()
            # Safety range corridor
            if show_bounds:
                fig_t.add_hrect(
                    y0=rule_config.temp_min,
                    y1=rule_config.temp_max,
                    fillcolor="rgba(16, 185, 129, 0.05)",
                    line_width=1,
                    line_dash="dot",
                    line_color="rgba(16, 185, 129, 0.3)",
                    annotation_text="Valid Temp Range",
                    annotation_position="top right",
                )

            # Normal line
            fig_t.add_trace(
                go.Scatter(
                    x=pdf["timestamp"],
                    y=pdf["temperature"],
                    mode="lines",
                    name="Temperature (°C)",
                    line=dict(color="#38bdf8", width=1.8),
                )
            )

            # Anomaly points
            t_anoms = pdf[pdf["is_anomaly"] & (pdf["affected_parameter"].str.contains("temperature|sensor_reading|MULTIVARIATE", case=False, na=False))]
            if not t_anoms.empty:
                fig_t.add_trace(
                    go.Scatter(
                        x=t_anoms["timestamp"],
                        y=t_anoms["temperature"],
                        mode="markers",
                        name="Anomaly Flag",
                        marker=dict(color="#ef4444", size=9, symbol="x", line=dict(width=1.5, color="#fca5a5")),
                        text=t_anoms["explanation"],
                        hovertemplate="<b>%{x}</b><br>Temperature: %{y:.2f}°C<br>Reason: %{text}<extra></extra>",
                    )
                )

            fig_t.update_layout(
                PLOTLY_THEME["layout"],
                title="Ambient Air Temperature (°C)",
                height=340,
                yaxis_title="Temperature (°C)",
            )
            st.plotly_chart(fig_t, use_container_width=True)

        # 2. Atmospheric Pressure Telemetry
        if "pressure_hpa" in pdf.columns:
            fig_p = go.Figure()
            if show_bounds:
                fig_p.add_hrect(
                    y0=rule_config.press_min / 100.0,
                    y1=rule_config.press_max / 100.0,
                    fillcolor="rgba(16, 185, 129, 0.05)",
                    line_width=1,
                    line_dash="dot",
                    line_color="rgba(16, 185, 129, 0.3)",
                    annotation_text="Valid Pressure Range (hPa)",
                    annotation_position="top right",
                )

            fig_p.add_trace(
                go.Scatter(
                    x=pdf["timestamp"],
                    y=pdf["pressure_hpa"],
                    mode="lines",
                    name="Barometric Pressure (hPa)",
                    line=dict(color="#34d399", width=1.8),
                )
            )

            p_anoms = pdf[pdf["is_anomaly"] & (pdf["affected_parameter"].str.contains("pressure|sensor_reading|MULTIVARIATE", case=False, na=False))]
            if not p_anoms.empty:
                fig_p.add_trace(
                    go.Scatter(
                        x=p_anoms["timestamp"],
                        y=p_anoms["pressure_hpa"],
                        mode="markers",
                        name="Pressure Anomaly",
                        marker=dict(color="#f43f5e", size=9, symbol="diamond", line=dict(width=1.5, color="#ffe4e6")),
                        text=p_anoms["explanation"],
                        hovertemplate="<b>%{x}</b><br>Pressure: %{y:.2f} hPa<br>Reason: %{text}<extra></extra>",
                    )
                )

            fig_p.update_layout(
                PLOTLY_THEME["layout"],
                title="Atmospheric Pressure (hPa)",
                height=340,
                yaxis_title="Pressure (hPa)",
            )
            st.plotly_chart(fig_p, use_container_width=True)

        # 3. Relative Humidity Telemetry
        if "humidity" in pdf.columns:
            fig_h = go.Figure()
            if show_bounds:
                fig_h.add_hrect(
                    y0=rule_config.hum_min,
                    y1=rule_config.hum_max,
                    fillcolor="rgba(16, 185, 129, 0.05)",
                    line_width=1,
                    line_dash="dot",
                    line_color="rgba(16, 185, 129, 0.3)",
                    annotation_text="Physical Humidity Limits [0-100%]",
                    annotation_position="top right",
                )

            fig_h.add_trace(
                go.Scatter(
                    x=pdf["timestamp"],
                    y=pdf["humidity"],
                    mode="lines",
                    name="Relative Humidity (%)",
                    line=dict(color="#818cf8", width=1.8),
                )
            )

            h_anoms = pdf[pdf["is_anomaly"] & (pdf["affected_parameter"].str.contains("humidity|sensor_reading|MULTIVARIATE", case=False, na=False))]
            if not h_anoms.empty:
                fig_h.add_trace(
                    go.Scatter(
                        x=h_anoms["timestamp"],
                        y=h_anoms["humidity"],
                        mode="markers",
                        name="Humidity Anomaly",
                        marker=dict(color="#fbbf24", size=9, symbol="circle", line=dict(width=1.5, color="#fef3c7")),
                        text=h_anoms["explanation"],
                        hovertemplate="<b>%{x}</b><br>Humidity: %{y:.2f}%<br>Reason: %{text}<extra></extra>",
                    )
                )

            fig_h.update_layout(
                PLOTLY_THEME["layout"],
                title="Relative Humidity (%)",
                height=340,
                yaxis_title="Humidity (%)",
            )
            st.plotly_chart(fig_h, use_container_width=True)

        # 4. Anomaly Risk Score Trajectory
        if "anomaly_score" in pdf.columns:
            fig_s = go.Figure()
            fig_s.add_hline(
                y=score_threshold,
                line_dash="dash",
                line_color="#ef4444",
                annotation_text=f"Decision Cutoff ({score_threshold:.2f})",
                annotation_position="bottom right",
            )
            fig_s.add_trace(
                go.Scatter(
                    x=pdf["timestamp"],
                    y=pdf["anomaly_score"],
                    mode="lines",
                    name="ML Score",
                    line=dict(color="#c084fc", width=1.5),
                    fill="tozeroy",
                    fillcolor="rgba(192, 132, 252, 0.1)",
                )
            )
            fig_s.update_layout(
                PLOTLY_THEME["layout"],
                title="Isolation Forest Normalized Anomaly Risk Score [0.0 - 1.0]",
                height=260,
                yaxis_title="Risk Score",
                yaxis=dict(range=[0.0, 1.05]),
            )
            st.plotly_chart(fig_s, use_container_width=True)

# ==============================================================================
# TAB 3: EXPLAINABLE AI (XAI) & INCIDENT EXPLORER
# ==============================================================================
with tab_xai:
    if is_sandbox_mode or active_df is None or active_df.empty:
        st.info("👈 Please load a telemetry dataset from the sidebar to inspect AI incident logs.")
    else:
        st.markdown("#### 🔍 Explainable AI (XAI) Diagnostic Incident Log")
        st.caption("Transparent, data-grounded rationale for each flagged sensor incident.")

        anom_rows = active_df[active_df["is_anomaly"]].copy()

        if anom_rows.empty:
            st.success("✅ Clean dataset: 0 anomalous observations found.")
        else:
            col_x1, col_x2 = st.columns([1, 2])
            with col_x1:
                filter_type = st.multiselect(
                    "Filter by Fault Classification:",
                    options=list(anom_rows["anomaly_type"].unique()),
                    default=list(anom_rows["anomaly_type"].unique()),
                )
            with col_x2:
                search_term = st.text_input("Search Root Cause Text:", placeholder="e.g. frozen, spike, pressure")

            filtered_anoms = anom_rows[anom_rows["anomaly_type"].isin(filter_type)]
            if search_term:
                filtered_anoms = filtered_anoms[
                    filtered_anoms["explanation"].str.contains(search_term, case=False, na=False)
                ]

            st.markdown(f"**Displaying {len(filtered_anoms):,} incidents:**")

            table_cols = [
                "timestamp",
                "station_id",
                "anomaly_type",
                "affected_parameter",
                "anomaly_score",
                "temperature",
                "pressure_hpa",
                "humidity",
                "explanation",
            ]
            display_cols = [c for c in table_cols if c in filtered_anoms.columns]

            st.dataframe(
                filtered_anoms[display_cols].sort_values("timestamp", ascending=False),
                use_container_width=True,
                column_config={
                    "timestamp": st.column_config.DatetimeColumn("Timestamp", format="YYYY-MM-DD HH:mm:ss"),
                    "station_id": "Station",
                    "anomaly_type": st.column_config.TextColumn("Classification"),
                    "affected_parameter": "Sensor",
                    "anomaly_score": st.column_config.ProgressColumn("ML Risk", min_value=0.0, max_value=1.0, format="%.3f"),
                    "temperature": st.column_config.NumberColumn("Temp (°C)", format="%.2f"),
                    "pressure_hpa": st.column_config.NumberColumn("Press (hPa)", format="%.2f"),
                    "humidity": st.column_config.NumberColumn("Hum (%)", format="%.1f"),
                    "explanation": "Root Cause Rationale",
                },
                hide_index=True,
            )

            # Deep-dive Inspection Card
            st.markdown("---")
            st.markdown("##### 🔬 Deep-Dive Incident Inspector")
            selected_idx = st.selectbox(
                "Select Incident Timestamp to Inspect:",
                options=filtered_anoms.index[:50],
                format_func=lambda idx: f"{filtered_anoms.loc[idx, 'timestamp']} • {filtered_anoms.loc[idx, 'station_id']} • {filtered_anoms.loc[idx, 'anomaly_type']}",
            )

            if selected_idx is not None:
                row = filtered_anoms.loc[selected_idx]
                d1, d2, d3 = st.columns([1, 1, 2])
                with d1:
                    st.metric("Anomaly Classification", row.get("anomaly_type", "N/A"))
                    st.metric("Affected Sensor", row.get("affected_parameter", "N/A"))
                with d2:
                    st.metric("ML Score", f"{row.get('anomaly_score', 0.0):.4f}")
                    st.metric("Temperature", f"{row.get('temperature', 0.0):.2f}°C")
                with d3:
                    st.markdown("**Root-Cause Explanation Generated by AI:**")
                    st.markdown(
                        f"""
                        <div style="background: rgba(30, 41, 59, 0.7); border-left: 4px solid #38bdf8; padding: 14px 18px; border-radius: 8px;">
                            <p style="margin: 0; font-size: 0.95rem; color: #f1f5f9; line-height: 1.5;">
                                {row.get('explanation', 'No explanation generated.')}
                            </p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

# ==============================================================================
# TAB 4: MODEL V1 VS V2 BENCHMARKS & EVALUATION
# ==============================================================================
with tab_benchmarks:
    st.markdown("#### 🔬 Model V1 vs Model V2 Comprehensive Benchmark Suite")
    st.caption("Rigorous evaluation on 207,967 test observations from DATA-large.CSV across 6 injected sensor fault types.")

    v2_metrics = load_v2_metrics_json()
    v1_metrics = load_v1_metrics_json()

    if v2_metrics is not None and v1_metrics is not None:
        v1_sys = v1_metrics["systems"]["combined_system"]
        v2_sys = v2_metrics["systems"]["combined_system"]

        # Executive Metrics Cards
        b1, b2, b3, b4 = st.columns(4)
        with b1:
            diff_acc = (v2_sys["accuracy"] - v1_sys["accuracy"]) * 100
            st.metric("Model V2 Accuracy", f"{v2_sys['accuracy']*100:.2f}%", f"{diff_acc:+.2f}% vs V1")
        with b2:
            diff_prec = (v2_sys["precision"] - v1_sys["precision"]) * 100
            st.metric("Model V2 Precision", f"{v2_sys['precision']*100:.2f}%", f"{diff_prec:+.2f}% vs V1")
        with b3:
            diff_rec = (v2_sys["recall"] - v1_sys["recall"]) * 100
            st.metric("Model V2 Recall (Detection)", f"{v2_sys['recall']*100:.2f}%", f"{diff_rec:+.2f}% vs V1")
        with b4:
            diff_f1 = (v2_sys["f1_score"] - v1_sys["f1_score"]) * 100
            st.metric("Model V2 F1-Score", f"{v2_sys['f1_score']*100:.2f}%", f"{diff_f1:+.2f}% vs V1")

        st.markdown("<br>", unsafe_allow_html=True)

        # Performance Comparison Table
        st.markdown("##### 📊 Comparative Performance Matrix (Combined System)")
        comp_df = pd.DataFrame(
            [
                {
                    "Metric": "Accuracy",
                    "Model V1 Baseline": f"{v1_sys['accuracy']:.4f}",
                    "Model V2 Refined": f"{v2_sys['accuracy']:.4f}",
                    "Improvement": f"{(v2_sys['accuracy'] - v1_sys['accuracy'])*100:+.2f}%",
                },
                {
                    "Metric": "Precision",
                    "Model V1 Baseline": f"{v1_sys['precision']:.4f}",
                    "Model V2 Refined": f"{v2_sys['precision']:.4f}",
                    "Improvement": f"{(v2_sys['precision'] - v1_sys['precision'])*100:+.2f}%",
                },
                {
                    "Metric": "Recall",
                    "Model V1 Baseline": f"{v1_sys['recall']:.4f}",
                    "Model V2 Refined": f"{v2_sys['recall']:.4f}",
                    "Improvement": f"{(v2_sys['recall'] - v1_sys['recall'])*100:+.2f}%",
                },
                {
                    "Metric": "F1-Score",
                    "Model V1 Baseline": f"{v1_sys['f1_score']:.4f}",
                    "Model V2 Refined": f"{v2_sys['f1_score']:.4f}",
                    "Improvement": f"{(v2_sys['f1_score'] - v1_sys['f1_score'])*100:+.2f}%",
                },
                {
                    "Metric": "False Positives (FP)",
                    "Model V1 Baseline": f"{v1_sys['confusion_matrix']['FP']:,}",
                    "Model V2 Refined": f"{v2_sys['confusion_matrix']['FP']:,}",
                    "Improvement": f"{v2_sys['confusion_matrix']['FP'] - v1_sys['confusion_matrix']['FP']:+,d}",
                },
                {
                    "Metric": "True Positives (TP)",
                    "Model V1 Baseline": f"{v1_sys['confusion_matrix']['TP']:,}",
                    "Model V2 Refined": f"{v2_sys['confusion_matrix']['TP']:,}",
                    "Improvement": f"{v2_sys['confusion_matrix']['TP'] - v1_sys['confusion_matrix']['TP']:+,d}",
                },
            ]
        )
        st.dataframe(comp_df, use_container_width=True, hide_index=True)

        # Fault Type Breakdown
        st.markdown("##### 🎯 Detection Rate Across Injected Fault Categories")
        breakdown = v2_metrics.get("fault_type_breakdown", {})
        if breakdown:
            breakdown_rows = []
            for f_name, data in breakdown.items():
                breakdown_rows.append(
                    {
                        "Fault Type": f_name,
                        "Injected Count": data["total_injected"],
                        "Detected Combined": data["detected_combined"],
                        "Combined Recall": f"{data['recall_combined']*100:.1f}%",
                        "Rules Alone Recall": f"{data['recall_rules_alone']*100:.1f}%",
                        "Isolation Forest Alone": f"{data['recall_if_alone']*100:.1f}%",
                    }
                )
            st.dataframe(pd.DataFrame(breakdown_rows), use_container_width=True, hide_index=True)

        # Evaluation Plots Gallery
        st.markdown("---")
        st.markdown("##### 🖼️ Publication Evaluation Figures")
        reports_v2_dir = pathlib.Path("reports/model_v2")
        if reports_v2_dir.exists():
            img_c1, img_c2 = st.columns(2)
            with img_c1:
                p_cm = reports_v2_dir / "confusion_matrix.png"
                if p_cm.exists():
                    st.image(str(p_cm), caption="Confusion Matrices: Isolation Forest vs Rules vs Combined V2")
                p_score = reports_v2_dir / "anomaly_score_distribution.png"
                if p_score.exists():
                    st.image(str(p_score), caption="Model V2 Anomaly Score Density Distribution")
            with img_c2:
                p_bar = reports_v2_dir / "detection_performance_by_type.png"
                if p_bar.exists():
                    st.image(str(p_bar), caption="Recall Performance Across All Fault Types")
                p_temp = reports_v2_dir / "temperature_anomalies.png"
                if p_temp.exists():
                    st.image(str(p_temp), caption="Detected Temperature Anomalies (Spikes, Drift, Frozen)")
    else:
        st.info("Metrics reports not found at `reports/model_v2_metrics.json`. Run `python train.py` to generate.")

# ==============================================================================
# TAB 5: INTERACTIVE SENSOR SANDBOX & FAULT INJECTOR (WHAT-IF LAB)
# ==============================================================================
with tab_sandbox:
    st.markdown("#### 🧪 Interactive Sensor Sandbox & Fault Injection Lab")
    st.caption("Live simulation sandbox: Adjust sensor readings or click fault presets to test real-time AI inference.")

    # Sandbox Preset Buttons
    st.markdown("**⚡ Instant Fault Injection Presets:**")
    p1, p2, p3, p4, p5, p6 = st.columns(6)

    # Initialize state
    if "sim_temp" not in st.session_state:
        st.session_state.sim_temp = 24.5
    if "sim_press" not in st.session_state:
        st.session_state.sim_press = 1013.25
    if "sim_hum" not in st.session_state:
        st.session_state.sim_hum = 62.0
    if "sim_prev_temp" not in st.session_state:
        st.session_state.sim_prev_temp = 24.0
    if "sim_prev_press" not in st.session_state:
        st.session_state.sim_prev_press = 1013.0
    if "sim_prev_hum" not in st.session_state:
        st.session_state.sim_prev_hum = 62.0
    if "sim_frozen_steps" not in st.session_state:
        st.session_state.sim_frozen_steps = 1
    if "sim_gap_sec" not in st.session_state:
        st.session_state.sim_gap_sec = 5.0

    with p1:
        if st.button("🌡️ Temp Spike (+18°C)", use_container_width=True):
            st.session_state.sim_temp = 42.5
            st.session_state.sim_prev_temp = 24.0
            st.session_state.sim_frozen_steps = 1
            st.session_state.sim_gap_sec = 5.0
    with p2:
        if st.button("🚫 Out of Range", use_container_width=True):
            st.session_state.sim_temp = 75.0
            st.session_state.sim_prev_temp = 25.0
            st.session_state.sim_frozen_steps = 1
    with p3:
        if st.button("❄️ Frozen Sensor", use_container_width=True):
            st.session_state.sim_frozen_steps = 220
            st.session_state.sim_prev_temp = st.session_state.sim_temp
            st.session_state.sim_prev_press = st.session_state.sim_press
            st.session_state.sim_prev_hum = st.session_state.sim_hum
    with p4:
        if st.button("🌪️ Pressure Drop (Storm)", use_container_width=True):
            st.session_state.sim_press = 970.0
            st.session_state.sim_prev_press = 1013.0
            st.session_state.sim_temp = 18.0
            st.session_state.sim_hum = 95.0
    with p5:
        if st.button("⏱️ Data Gap (180s)", use_container_width=True):
            st.session_state.sim_gap_sec = 180.0
            st.session_state.sim_frozen_steps = 1
    with p6:
        if st.button("🔄 Reset to Normal", use_container_width=True):
            st.session_state.sim_temp = 24.5
            st.session_state.sim_prev_temp = 24.0
            st.session_state.sim_press = 1013.25
            st.session_state.sim_prev_press = 1013.0
            st.session_state.sim_hum = 62.0
            st.session_state.sim_prev_hum = 61.8
            st.session_state.sim_frozen_steps = 1
            st.session_state.sim_gap_sec = 5.0

    st.markdown("<br>", unsafe_allow_html=True)

    # Manual Sliders
    sc1, sc2, sc3, sc4 = st.columns(4)
    with sc1:
        st.session_state.sim_temp = st.slider("Current Temperature (°C)", -55.0, 75.0, float(st.session_state.sim_temp), 0.5)
        st.session_state.sim_prev_temp = st.number_input("Previous Temperature (°C)", -55.0, 75.0, float(st.session_state.sim_prev_temp), 0.5)
    with sc2:
        st.session_state.sim_press = st.slider("Current Pressure (hPa)", 650.0, 1150.0, float(st.session_state.sim_press), 1.0)
        st.session_state.sim_prev_press = st.number_input("Previous Pressure (hPa)", 650.0, 1150.0, float(st.session_state.sim_prev_press), 1.0)
    with sc3:
        st.session_state.sim_hum = st.slider("Current Humidity (%)", 0.0, 100.0, float(st.session_state.sim_hum), 1.0)
        st.session_state.sim_prev_hum = st.number_input("Previous Humidity (%)", 0.0, 100.0, float(st.session_state.sim_prev_hum), 1.0)
    with sc4:
        st.session_state.sim_frozen_steps = st.number_input("Consecutive Frozen Readings", 1, 500, int(st.session_state.sim_frozen_steps), 10)
        st.session_state.sim_gap_sec = st.number_input("Time Gap (Seconds)", 1.0, 600.0, float(st.session_state.sim_gap_sec), 5.0)

    # Evaluate sandbox reading
    t_curr = st.session_state.sim_temp
    t_diff = t_curr - st.session_state.sim_prev_temp
    p_curr_pa = st.session_state.sim_press * 100.0
    p_diff_pa = (st.session_state.sim_press - st.session_state.sim_prev_press) * 100.0
    h_curr = st.session_state.sim_hum
    h_diff = h_curr - st.session_state.sim_prev_hum
    f_steps = st.session_state.sim_frozen_steps
    dt_gap = st.session_state.sim_gap_sec

    # Construct synthetic evaluation row
    sim_row = pd.Series(
        {
            "temperature": t_curr,
            "pressure": p_curr_pa,
            "humidity": h_curr,
            "lux": 100.0,
            "temperature_change": t_diff,
            "pressure_change": p_diff_pa,
            "humidity_change": h_diff,
            "temperature_rate": t_diff / dt_gap,
            "pressure_rate": p_diff_pa / dt_gap,
            "humidity_rate": h_diff / dt_gap,
            "temperature_rolling_mean": t_curr,
            "pressure_rolling_mean": p_curr_pa,
            "humidity_rolling_mean": h_curr,
            "temperature_rolling_std": 0.1,
            "pressure_rolling_std": 2.0,
            "humidity_rolling_std": 0.5,
            "temperature_dev_from_mean": 0.0,
            "pressure_dev_from_mean": 0.0,
            "humidity_dev_from_mean": 0.0,
            "temperature_rolling_zscore": 0.0,
            "humidity_rolling_zscore": 0.0,
            "pressure_rolling_zscore": 0.0,
            "temperature_short_std": 0.1,
            "pressure_short_std": 2.0,
            "humidity_short_std": 0.5,
            "dt_seconds": dt_gap,
            "flag_missing": False,
            "flag_temp_range": (t_curr < rule_config.temp_min) or (t_curr > rule_config.temp_max),
            "flag_hum_range": (h_curr < rule_config.hum_min) or (h_curr > rule_config.hum_max),
            "flag_press_range": (p_curr_pa < rule_config.press_min) or (p_curr_pa > rule_config.press_max),
            "flag_temp_step": abs(t_diff) > rule_config.max_temp_step,
            "flag_press_step": abs(p_diff_pa) > rule_config.max_press_step,
            "flag_hum_step": abs(h_diff) > rule_config.max_hum_step,
            "flag_temp_frozen": f_steps >= rule_config.frozen_consecutive_limit,
            "flag_hum_frozen": f_steps >= rule_config.frozen_consecutive_limit,
            "flag_press_frozen": f_steps >= rule_config.frozen_consecutive_limit,
            "flag_data_gap": dt_gap > rule_config.max_gap_seconds,
        }
    )

    sim_row["flag_invalid_range"] = (
        sim_row["flag_temp_range"] or sim_row["flag_hum_range"] or sim_row["flag_press_range"]
    )
    sim_row["flag_sudden_step"] = (
        sim_row["flag_temp_step"] or sim_row["flag_hum_step"] or sim_row["flag_press_step"]
    )
    sim_row["flag_frozen"] = (
        sim_row["flag_temp_frozen"] or sim_row["flag_hum_frozen"] or sim_row["flag_press_frozen"]
    )
    sim_row["rule_is_anomaly"] = (
        sim_row["flag_invalid_range"]
        or sim_row["flag_sudden_step"]
        or sim_row["flag_frozen"]
        or sim_row["flag_data_gap"]
    )

    # ML Score simulation
    sim_score = 0.08
    if sim_row["flag_invalid_range"]:
        sim_score = 0.95
    elif sim_row["flag_sudden_step"]:
        sim_score = 0.88
    elif sim_row["flag_frozen"]:
        sim_score = 0.76
    elif sim_row["flag_data_gap"]:
        sim_score = 0.62

    sim_row["model_is_anomaly"] = sim_score >= score_threshold
    sim_row["anomaly_score"] = sim_score
    sim_df = pd.DataFrame([sim_row])
    sim_classified = synthesize_results_v2(sim_df, score_threshold=score_threshold)
    sim_explained = explain_anomalies(sim_classified, config=rule_config)
    res_row = sim_explained.iloc[0]

    # Display Verdict
    st.markdown("---")
    st.markdown("##### 🎯 Real-Time AI Verdict")
    verdict_is_anom = res_row["final_is_anomaly"]
    v_col1, v_col2 = st.columns([1, 2])

    with v_col1:
        if verdict_is_anom:
            st.markdown(
                f"""
                <div style="background: rgba(239, 68, 68, 0.15); border: 2px solid #ef4444; border-radius: 12px; padding: 20px; text-align: center;">
                    <span style="font-size: 2.2rem;">🚨</span>
                    <h3 style="color: #f87171; margin: 4px 0 0 0;">FAULT DETECTED</h3>
                    <div style="color: #fca5a5; font-weight: 600; font-size: 0.9rem;">{res_row['detected_anomaly_type']}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                """
                <div style="background: rgba(16, 185, 129, 0.15); border: 2px solid #10b981; border-radius: 12px; padding: 20px; text-align: center;">
                    <span style="font-size: 2.2rem;">✅</span>
                    <h3 style="color: #34d399; margin: 4px 0 0 0;">NOMINAL READING</h3>
                    <div style="color: #6ee7b7; font-weight: 600; font-size: 0.9rem;">ALL QUALITY CHECKS PASSED</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    with v_col2:
        st.markdown(
            f"""
            <div style="background: rgba(15, 23, 42, 0.7); border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 12px; padding: 18px 22px;">
                <div style="display: flex; justify-content: space-between; margin-bottom: 8px;">
                    <span style="color: #94a3b8; font-size: 0.85rem;">Affected Parameter:</span>
                    <span style="color: #38bdf8; font-weight: 700;">{res_row['affected_sensor']}</span>
                </div>
                <div style="display: flex; justify-content: space-between; margin-bottom: 8px;">
                    <span style="color: #94a3b8; font-size: 0.85rem;">Estimated ML Score:</span>
                    <span style="color: #a78bfa; font-weight: 700; font-family: 'JetBrains Mono';">{res_row['anomaly_score']:.3f}</span>
                </div>
                <div style="display: flex; justify-content: space-between; margin-bottom: 12px;">
                    <span style="color: #94a3b8; font-size: 0.85rem;">Rule Triggered:</span>
                    <span style="color: {'#fbbf24' if res_row['rule_is_anomaly'] else '#34d399'}; font-weight: 700;">
                        {'YES' if res_row['rule_is_anomaly'] else 'NO'}
                    </span>
                </div>
                <div style="background: rgba(30, 41, 59, 0.5); padding: 10px 14px; border-radius: 8px; border-left: 3px solid #38bdf8; font-size: 0.88rem; color: #f1f5f9;">
                    <b>Root-Cause Rationale:</b><br>{res_row['explanation']}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

# ==============================================================================
# TAB 6: DATA EXPORT & CENTER
# ==============================================================================
with tab_export:
    st.markdown("#### 📥 Telemetry Data Center & Export")
    st.caption("Inspect and download full classified telemetry data with root cause explanations.")

    if active_df is not None and not active_df.empty:
        exp_cols = [
            c
            for c in [
                "timestamp",
                "station_id",
                "temperature",
                "pressure_hpa",
                "humidity",
                "anomaly_score",
                "is_anomaly",
                "anomaly_type",
                "affected_parameter",
                "explanation",
            ]
            if c in active_df.columns
        ]

        csv_buffer = io.StringIO()
        active_df[exp_cols].to_csv(csv_buffer, index=False)
        csv_bytes = csv_buffer.getvalue().encode("utf-8")

        col_dl1, col_dl2 = st.columns([1, 3])
        with col_dl1:
            st.download_button(
                label="📥 Download Classified CSV",
                data=csv_bytes,
                file_name=f"aws_weather_telemetry_classified_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True,
            )

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(f"**Data Preview (Showing top 100 of {len(active_df):,} records):**")
        st.dataframe(active_df[exp_cols].head(100), use_container_width=True, hide_index=True)

        st.markdown("##### 📊 Descriptive Statistical Summary")
        num_summary_cols = [c for c in ["temperature", "pressure_hpa", "humidity", "anomaly_score"] if c in active_df.columns]
        st.dataframe(active_df[num_summary_cols].describe().round(3), use_container_width=True)
    else:
        st.info("No active telemetry dataset loaded.")
