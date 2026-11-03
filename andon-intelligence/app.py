import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from pathlib import Path
import base64
import json

ASSET_DIR = Path(__file__).resolve().parent / "assets"


def asset_data_uri(filename, media_type):
    encoded = base64.b64encode((ASSET_DIR / filename).read_bytes()).decode("ascii")
    return f"data:{media_type};base64,{encoded}"


brand_image = asset_data_uri("andon-stop.png", "image/png")
page_icons = [
    asset_data_uri(f"icon-page-{index}.svg", "image/svg+xml")
    for index in range(1, 5)
]

# --- page configuration 
st.set_page_config(
    page_title="Andon Stop Intelligence",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- page styling
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    :root {
        --andon-sidebar: #e5f4e3;
        --andon-active: #8ce663;
        --andon-ink: #0d2806;
        --andon-muted: #556254;
    }

    html, body, [class*="st"], .stApp {
        font-family: "Inter", sans-serif;
    }

    .stApp,
    [data-testid="stSidebar"] {
        background-color: var(--andon-sidebar);
    }

    .stApp {
        position: relative;
        min-height: 100vh;
    }

    [data-testid="stHeader"] {
        background: transparent;
    }

    [data-testid="stSidebarCollapseButton"] {
        display: none !important;
    }

    .app-shell {
        padding: 0.35rem 0.2rem 1.5rem;
    }

    .app-shell-compact {
        padding-bottom: 0.25rem;
    }

    .app-shell-compact .page-header {
        margin-bottom: 0.25rem;
    }

    .app-shell-balanced {
        padding-bottom: 0.75rem;
    }

    .app-shell-balanced .page-header {
        margin-bottom: 0.65rem;
    }

    .page-header {
        margin: 0 0 1.25rem;
        padding: 1.15rem 1.3rem 1rem;
        border: 1px solid #dfe9e0;
        border-radius: 1rem;
        background: linear-gradient(135deg, #f5fff2 0%, #edf7ff 100%);
    }

    .page-header h1 {
        margin: 0 0 0.15rem;
        color: #153d2d;
        font-size: clamp(2rem, 2vw, 2.5rem);
        line-height: 1.15;
    }

    .page-header p {
        margin: 0;
        color: #506557;
        font-size: 0.92rem;
        font-weight: 500;
    }

    section[data-testid="stSidebar"] {
        position: sticky;
        top: 0;
        min-width: 260px;
        width: 25%;
        max-width: 360px;
        height: 100vh;
        overflow: hidden;
    }

    section[data-testid="stSidebar"] > div {
        padding: 1.4rem 0 2rem 1.05rem;
    }

    .brand-header {
        margin: 0 0 2rem;
    }

    .brand-header img {
        display: block;
        width: 150px;
        max-width: 100%;
        height: auto;
        margin-left: 0.35rem;
    }

    [data-testid="stSidebar"] [data-testid="stRadio"] > label {
        display: none;
    }

    [data-testid="stSidebar"] [data-testid="stRadio"] [role="radiogroup"] {
        gap: 0.75rem;
    }

    [data-testid="stSidebar"] [data-testid="stRadio"] [role="radiogroup"] label {
        position: relative;
        display: flex;
        align-items: center;
        width: calc(100% + 3.2rem);
        min-width: 220px;
        margin-left: -0.6rem;
        margin-right: -3.2rem;
        padding: 0.875rem 1.25rem 0.875rem 3.2rem;
        border-radius: 1.75rem 0 0 1.75rem;
        background-repeat: no-repeat;
        background-position: 1.25rem center;
        background-size: 1.1rem;
        color: var(--andon-muted);
        white-space: nowrap;
        transition: background-color 0.2s ease, color 0.2s ease;
    }

    [data-testid="stSidebar"] [data-testid="stRadio"] [role="radiogroup"] label span {
        display: inline-block;
        white-space: nowrap;
    }

    [data-testid="stSidebar"] [data-testid="stRadio"] [role="radiogroup"] label::before {
        position: absolute;
        top: 50%;
        left: 1.25rem;
        width: 1.1rem;
        height: 1.1rem;
        content: "";
        background-size: contain;
        background-repeat: no-repeat;
        background-position: center;
        transform: translateY(-50%);
        opacity: 0.9;
    }

    [data-testid="stSidebar"] [data-testid="stRadioOption"] > div > div:first-child {
        display: none;
    }

    [data-testid="stSidebar"] [data-testid="stRadioOption"]:has(input:focus-visible) {
        outline: 2px solid #174a32;
        outline-offset: -2px;
    }

    [data-testid="stSidebar"] [data-testid="stRadio"] [role="radiogroup"] label:hover {
        background-color: rgba(0, 0, 0, 0.04);
        color: #1a1a1a;
    }

    [data-testid="stSidebar"] [data-testid="stRadio"] [role="radiogroup"] label:has(input:checked) {
        background: linear-gradient(90deg, #a0f078 0%, #86da5c 100%);
        color: var(--andon-ink);
        font-weight: 700;
        box-shadow: inset -8px 0 0 rgba(24, 52, 25, 0.08);
        margin-right: -0.25rem;
        border-top-right-radius: 0;
        border-bottom-right-radius: 0;
    }

    [data-testid="stMain"] {
        position: sticky;
        top: 0;
        height: 100vh;
        min-height: 100vh;
        background-color: #fff;
        border-radius: 2.5rem 0 0 2.5rem;
        box-shadow: -18px 0 28px rgba(10, 31, 17, 0.08), 0 22px 32px rgba(18, 39, 20, 0.04);
        overflow: hidden;
    }

    [data-testid="stMain"] > div:first-child {
        height: 100vh;
        overflow-y: auto;
        overflow-x: hidden;
        scrollbar-width: thin;
    }

    @media (max-width: 700px) {
        [data-testid="stMain"] {
            border-radius: 1.5rem 0 0 1.5rem;
        }
    }

    .dashboard-kpis {
        display: grid;
        grid-template-columns: repeat(4, minmax(0, 1fr));
        gap: 1rem;
        margin: 0.5rem 0 1.75rem;
    }

    .dashboard-chart-title {
        margin: 0 0 0.65rem;
        color: #174a32;
        font-size: 1.05rem;
        font-weight: 800;
        line-height: 1.3;
        text-transform: uppercase;
    }

    .st-key-dashboard-trend-card,
    .st-key-dashboard-duration-card {
        height: 100%;
    }

    .st-key-dashboard-trend-card [data-testid="stVerticalBlockBorderWrapper"],
    .st-key-dashboard-duration-card [data-testid="stVerticalBlockBorderWrapper"] {
        height: 100%;
    }

    .stat-card {
        min-width: 0;
        padding: 1.1rem 1.2rem;
        border: 1px solid #dce6dc;
        border-radius: 0.9rem;
        background: linear-gradient(180deg, #ffffff 0%, #f7faf7 100%);
        box-shadow: 0 8px 18px rgba(18, 34, 24, 0.03);
    }

    .dashboard-kpi {
        min-width: 0;
        padding: 1.1rem 1.2rem;
        border: 1px solid #dce6dc;
        border-radius: 0.9rem;
        background: linear-gradient(180deg, #ffffff 0%, #f7faf7 100%);
        box-shadow: 0 8px 18px rgba(18, 34, 24, 0.03);
    }

    .dashboard-kpi-label {
        margin-bottom: 0.65rem;
        color: #59665a;
        font-size: 0.76rem;
        font-weight: 700;
    }

    .dashboard-kpi-value {
        color: #174a32;
        font-size: 1.6rem;
        font-weight: 750;
        line-height: 1.2;
    }

    .dashboard-summary-row {
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 1rem;
        margin-top: 1.25rem;
    }

    .dashboard-summary {
        display: flex;
        min-height: 108px;
        flex-direction: column;
        justify-content: center;
        padding: 1.1rem 1.25rem;
        border-radius: 0.75rem;
        color: #fff;
        text-align: center;
        box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.18), inset 0 -8px 12px rgba(0, 0, 0, 0.09);
    }

    .dashboard-summary-title {
        margin-bottom: 0.4rem;
        font-size: 0.78rem;
        font-weight: 750;
        letter-spacing: 0.04em;
        text-transform: uppercase;
    }

    .dashboard-summary-detail {
        font-size: 1rem;
        font-weight: 650;
    }

    @media (max-width: 900px) {
        .dashboard-kpis {
            grid-template-columns: repeat(2, minmax(0, 1fr));
        }

        .dashboard-summary-row {
            grid-template-columns: 1fr;
        }
    }

    @media (max-width: 480px) {
        .dashboard-kpi-value {
            font-size: 1.3rem;
        }
    }

    .analysis-table-wrap {
        position: relative;
        overflow-x: auto;
        border: 1px solid #dce3df;
        border-radius: 0.5rem;
    }

    .analysis-section-title {
        display: flex;
        align-items: center;
        gap: 0.65rem;
        margin: 0 0 0.65rem;
        color: #174a32;
        font-size: 1.05rem;
        font-weight: 800;
        line-height: 1.3;
        text-transform: uppercase;
    }

    .analysis-title-tag {
        padding: 0.2rem 0.5rem;
        border-radius: 999px;
        background: #e5f3e7;
        color: #28643c;
        font-size: 0.65rem;
        font-weight: 750;
        white-space: nowrap;
    }

    .extreme-events-table {
        width: 100%;
        border-collapse: collapse;
        background: #fff;
        color: #24372c;
        font-size: 0.88rem;
    }

    .extreme-events-table th {
        padding: 0.8rem 1rem;
        border: 1px solid #d5d9d6;
        background: #e5e8e6;
        color: #26352b;
        font-weight: 700;
        text-align: left;
        white-space: nowrap;
    }

    .extreme-events-table td {
        padding: 0.7rem 1rem;
        border: 1px solid #e0e5e1;
        white-space: nowrap;
    }

    .extreme-events-table tbody tr:nth-child(even) {
        background: #f8faf8;
    }

    .evaluation-metrics-grid {
        display: grid;
        grid-template-columns: repeat(2, minmax(0, 1fr));
        gap: 0.8rem;
        margin: 0.35rem 0 1.25rem;
    }

    .evaluation-metric {
        padding: 0.8rem 0.9rem;
        border: 1px solid #dce6dc;
        border-radius: 0.8rem;
        background: linear-gradient(180deg, #ffffff 0%, #f7faf7 100%);
        box-shadow: inset 0 1px 0 rgba(255,255,255,0.6);
    }

    .evaluation-metric-label {
        color: #59665a;
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.04em;
        text-transform: uppercase;
    }

    .evaluation-metric-value {
        margin-top: 0.35rem;
        color: #174a32;
        font-size: 1.35rem;
        font-weight: 800;
    }

    .evaluation-regression-grid {
        display: grid;
        grid-template-columns: repeat(2, minmax(0, 1fr));
        gap: 0.8rem;
        margin: 0.55rem 0.45rem 1.25rem 0;
    }

    .evaluation-section-title {
        margin: 0 0 0.5rem;
        color: #174a32;
        font-size: 1.05rem;
        font-weight: 800;
        line-height: 1.3;
        text-transform: uppercase;
    }

    .evaluation-card {
        display: flex;
        flex-direction: column;
        height: 100%;
    }

    .evaluation-card > div:last-child {
        flex: 1;
    }

    .evaluation-section-stack {
        display: grid;
        gap: 1.1rem;
    }

    .evaluation-note-card {
        margin-top: 0.5rem;
        padding: 1rem 1.1rem;
        border: 1px solid rgba(107, 92, 180, 0.22);
        border-radius: 0.8rem;
        background: linear-gradient(135deg, rgba(104, 103, 230, 0.12), rgba(138, 93, 192, 0.18));
        box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.45);
        color: #24372c;
    }

    .evaluation-note-card strong {
        display: block;
        margin-bottom: 0.35rem;
        font-size: 0.76rem;
        letter-spacing: 0.06em;
        text-transform: uppercase;
    }

    .evaluation-note-card p {
        margin: 0;
        font-size: 0.88rem;
        line-height: 1.5;
    }

    .model-limitation-card {
        margin-top: 1rem;
        padding: 1.2rem 1.35rem;
        border-radius: 0.75rem;
        background: #8250b5;
        color: #fff;
    }

    .model-limitation-card strong {
        display: block;
        margin-bottom: 0.55rem;
        font-size: 0.8rem;
    }

    .model-limitation-card p {
        margin: 0;
        font-size: 0.88rem;
        line-height: 1.5;
    }

    .comparison-table-wrap {
        margin: 0.55rem 0 0.85rem;
        overflow-x: auto;
        border: 1px solid #dce3df;
        border-radius: 0.5rem;
    }

    .evaluation-comparison-table {
        width: 100%;
        table-layout: fixed;
        border-collapse: collapse;
        background: #fff;
        color: #24372c;
        font-size: 0.7rem;
    }

    .evaluation-comparison-table th,
    .evaluation-comparison-table td {
        padding: 0.55rem 0.4rem;
        border: 1px solid #e0e5e1;
        text-align: left;
        overflow-wrap: anywhere;
    }

    .evaluation-comparison-table th {
        background: #e5e8e6;
        color: #26352b;
        font-weight: 700;
    }

    .evaluation-comparison-table th:first-child,
    .evaluation-comparison-table td:first-child {
        width: 38%;
    }

    .evaluation-comparison-table th:not(:first-child),
    .evaluation-comparison-table td:not(:first-child) {
        width: 20.66%;
    }

    [data-testid="stNumberInput"] input {
        border: 1px solid #d6e7da;
        border-radius: 0.6rem;
        background: #eff8f0;
    }

    [data-testid="stForm"] {
        border-color: #79b982;
    }

    [data-testid="stNumberInput"] input:focus {
        border-color: #4a8b56;
        box-shadow: 0 0 0 1px #4a8b56;
    }

    .prediction-result-banner {
        padding: 1.35rem 1.5rem;
        border-radius: 0.75rem;
        background: #174a32;
        color: #fff;
    }

    .prediction-result-eyebrow {
        margin-bottom: 1rem;
        color: #b9e7a5;
        font-size: 0.78rem;
        font-weight: 750;
    }

    .prediction-input-title {
        margin: 0 0 0.75rem;
        color: #174a32;
        font-size: 1.05rem;
        font-weight: 800;
        line-height: 1.3;
    }

    .prediction-result-values {
        display: flex;
        justify-content: space-between;
        gap: 1rem;
        font-size: 0.95rem;
        font-weight: 650;
    }

    .prediction-result-values strong {
        display: block;
        margin-top: 0.3rem;
        font-size: 1.4rem;
    }

    .prediction-gauge {
        height: 1rem;
        margin: 1.2rem 0 0.6rem;
        overflow: hidden;
        border-radius: 999px;
        background: #65786c;
    }

    .prediction-gauge-fill {
        height: 100%;
        border-radius: inherit;
        background: #8ce663;
    }

    .prediction-threshold {
        color: #e0e9e2;
        font-size: 0.78rem;
    }

    .prediction-duration-card {
        margin-top: 1rem;
        padding: 1.25rem 1.5rem;
        border: 1px solid #dce6dc;
        border-radius: 0.75rem;
        background: #fff;
    }

    .prediction-duration-title {
        color: #418052;
        font-size: 0.78rem;
        font-weight: 750;
    }

    .prediction-duration-label {
        margin-top: 0.75rem;
        color: #59665a;
    }

    .prediction-duration-value {
        margin-top: 0.2rem;
        color: #174a32;
        font-size: 1.8rem;
        font-weight: 750;
    }

    .prediction-info-banner {
        margin-top: 1rem;
        padding: 1.1rem 1.25rem;
        border-radius: 0.75rem;
        background: #159cbd;
        color: #fff;
        text-align: left;
    }

    .prediction-info-banner strong {
        display: block;
        font-size: 1rem;
    }

    .prediction-info-banner span {
        display: block;
        margin-top: 0.25rem;
        font-size: 0.8rem;
        font-weight: 650;
        line-height: 1.5;
    }

    [data-testid="stFormSubmitButton"] button {
        min-height: 3.1rem;
        border: 0;
        border-radius: 0.65rem;
        background: #174a32;
        color: #fff;
        font-weight: 750;
    }

    [data-testid="stFormSubmitButton"] button:hover {
        border: 0;
        background: #236442;
        color: #fff;
    }

    </style>
    """,
    unsafe_allow_html=True,
)

icon_styles = "\n".join(
    f'[data-testid="stSidebar"] [data-testid="stRadio"] '
    f'[role="radiogroup"] label:has(input[value="{index - 1}"])::before {{ '
    f'background-image: url("{icon}"); }}'
    for index, icon in enumerate(page_icons, start=1)
)

selected_icon_styles = "\n".join(
    f'[data-testid="stSidebar"] [data-testid="stRadio"] '
    f'[role="radiogroup"] label:has(input[value="{index - 1}"]:checked) {{ '
    f'background: linear-gradient(90deg, #a0f078 0%, #86da5c 100%); color: var(--andon-ink); font-weight: 700; }}'
    for index, icon in enumerate(page_icons, start=1)
)
st.markdown(f"<style>{icon_styles}{selected_icon_styles}</style>", unsafe_allow_html=True)

# --- models
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = (
    BASE_DIR.parent
    / "experiment yeo-johnson"
    / "yeo_johnson_experiment_output"
)

PREDICTION_FEATURES = [
    "minutes_since_previous_STOP",
    "STOP_count_last_1min",
    "STOP_count_last_5min",
    "STOP_count_last_10min",
]

yeo_johnson_transformer = joblib.load(
    MODEL_DIR / "yeo_johnson_transformer.joblib"
)

severity_model = joblib.load(
    MODEL_DIR / "severity_logistic_model.joblib"
)

duration_model = joblib.load(
    MODEL_DIR / "duration_log_linear_model.joblib"
)

SEVERITY_THRESHOLD = 0.30


# --- load data
DATA_PATH = BASE_DIR / "../data/mc09_andon.csv"

df = pd.read_csv(DATA_PATH)
df.columns = df.columns.str.strip()

event_columns = {
    "event_id",
    "start_time",
    "end_time",
    "duration_minutes",
    "minutes_since_previous_STOP",
    "STOP_count_last_1min",
    "STOP_count_last_5min",
    "STOP_count_last_10min",
}
telemetry_columns = {"timestamp", "status", "downtimeCurrent"}

if event_columns.issubset(df.columns):
    df["start_time"] = pd.to_datetime(df["start_time"])
    df["end_time"] = pd.to_datetime(df["end_time"])
elif telemetry_columns.issubset(df.columns):
    telemetry = df[["timestamp", "status", "downtimeCurrent"]].copy()
    telemetry["timestamp"] = pd.to_datetime(
        telemetry["timestamp"], errors="coerce"
    )
    telemetry["status"] = telemetry["status"].astype("string").str.strip().str.upper()
    telemetry["downtimeCurrent"] = pd.to_numeric(
        telemetry["downtimeCurrent"], errors="coerce"
    ).fillna(0)
    telemetry = telemetry.dropna(subset=["timestamp"]).sort_values("timestamp")

    stop_mask = telemetry["status"].eq("STOP")
    stop_rows = telemetry.loc[stop_mask].copy()
    stop_rows["_event_group"] = (~stop_mask).astype("int64").cumsum().loc[stop_mask]
    events = (
        stop_rows.groupby("_event_group", sort=False)
        .agg(
            start_time=("timestamp", "first"),
            duration_minutes=("downtimeCurrent", "max"),
        )
        .reset_index(drop=True)
    )
    events["duration_minutes"] /= 60_000
    events["end_time"] = events["start_time"] + pd.to_timedelta(
        events["duration_minutes"], unit="m"
    )
    events["event_id"] = np.arange(1, len(events) + 1)

    start_times = events["start_time"]
    start_values = start_times.astype("int64").to_numpy()
    events["minutes_since_previous_STOP"] = (
        start_times.diff().dt.total_seconds().div(60).fillna(0)
    )
    for window, column in [
        (1, "STOP_count_last_1min"),
        (5, "STOP_count_last_5min"),
        (10, "STOP_count_last_10min"),
    ]:
        events[column] = [
            index - np.searchsorted(
                start_values[:index],
                start_values[index] - window * 60_000_000_000,
                side="right",
            )
            for index in range(len(start_values))
        ]
    df = events
else:
    missing = sorted(event_columns.difference(df.columns))
    st.error(
        "The CSV is neither a prepared STOP-event dataset nor supported MC09 "
        f"telemetry. Missing event columns: {', '.join(missing)}."
    )
    st.stop()

df["date"] = df["start_time"].dt.date


def render_page_header(title, subtitle, spacing="default"):
    shell_class = "app-shell"
    if spacing != "default":
        shell_class += f" app-shell-{spacing}"
    st.markdown(
        f"""
        <div class="{shell_class}">
            <div class="page-header">
                <h1>{title}</h1>
                <p>{subtitle}</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# --- sidebar navi
st.sidebar.markdown(
    f'<div class="brand-header"><img src="{brand_image}" alt="Andon Stop Intelligence"></div>',
    unsafe_allow_html=True,
)

page = st.sidebar.radio(
    "Navigation",
    [
        "Dashboard",
        "STOP Analysis",
        "Prediction",
        "Model Evaluation",
    ],
    index=0,
    label_visibility="collapsed",
)


# --- page 1 : dashboard
if page == "Dashboard":

    render_page_header(
        "Dashboard",
        "Operational overview of STOP activity and duration distribution.",
        spacing="compact",
    )

    total_events = len(df)
    median_duration = df["duration_minutes"].median()
    max_duration = df["duration_minutes"].max()
    prolonged_events = int((df["duration_minutes"] > 5).sum())
    q95 = df["duration_minutes"].quantile(0.95)

    st.markdown(
        f"""
        <div class="dashboard-kpis">
            <div class="dashboard-kpi">
                <div class="dashboard-kpi-label">TOTAL STOP EVENTS</div>
                <div class="dashboard-kpi-value">{total_events:,}</div>
            </div>
            <div class="dashboard-kpi">
                <div class="dashboard-kpi-label">MEDIAN STOP DURATION</div>
                <div class="dashboard-kpi-value">{median_duration:.2f} min</div>
            </div>
            <div class="dashboard-kpi">
                <div class="dashboard-kpi-label">PROLONGED STOP EVENTS</div>
                <div class="dashboard-kpi-value">{prolonged_events:,}</div>
            </div>
            <div class="dashboard-kpi">
                <div class="dashboard-kpi-label">MAXIMUM STOP DURATION</div>
                <div class="dashboard-kpi-value">{max_duration:.2f} min</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    daily_events = (
        df.groupby("date")
        .size()
        .rename("stop_count")
        .rename_axis("date")
        .reset_index()
    )
    daily_events["date"] = pd.to_datetime(daily_events["date"])

    trend_col, distribution_col = st.columns([1, 1], gap="small")

    with trend_col:
        with st.container(border=True, key="dashboard-trend-card"):
            st.markdown(
                '<div class="dashboard-chart-title">STOP EVENT TREND</div>',
                unsafe_allow_html=True,
            )
            st.caption("Number of STOP events per day")

            trend_fig, trend_ax = plt.subplots(figsize=(8, 4.8))
            trend_start = daily_events["date"].min().normalize()
            trend_end = daily_events["date"].max()
            shade_start = trend_start
            week_index = 0
            while shade_start <= trend_end:
                shade_end = shade_start + pd.Timedelta(days=7)
                if week_index % 2 == 0:
                    trend_ax.axvspan(
                        shade_start,
                        shade_end,
                        color="#e5f5f0",
                        zorder=0,
                    )
                shade_start = shade_end
                week_index += 1

            trend_ax.plot(
                daily_events["date"],
                daily_events["stop_count"],
                color="#159c91",
                linewidth=2.2,
                marker="o",
                markersize=3,
                zorder=2,
            )
            trend_ax.set_ylim(0, 160)
            trend_ax.set_ylabel("Number of STOP events")
            trend_ax.set_xlabel("Date")
            trend_ax.grid(axis="y", color="#dfe7e2", linewidth=0.8)
            trend_ax.set_axisbelow(True)
            trend_fig.autofmt_xdate()
            trend_fig.tight_layout()
            st.pyplot(trend_fig, use_container_width=True)
            plt.close(trend_fig)
            st.caption("Daily event counts show whether STOP frequency is stable or whether disruptions cluster in specific periods.")

    with distribution_col:
        with st.container(border=True, key="dashboard-duration-card"):
            st.markdown(
                '<div class="dashboard-chart-title">STOP DURATION DISTRIBUTION</div>',
                unsafe_allow_html=True,
            )

            durations = df["duration_minutes"].dropna().clip(lower=0)
            duration_p95 = max(float(durations.quantile(0.95)), 0.01)
            duration_p99 = max(float(durations.quantile(0.99)), 0.01)
            short_limit = max(min(float(durations.max()), 60), 0.01)
            positive_durations = durations[durations > 0]

            distribution_fig, distribution_axes = plt.subplots(
                2, 2, figsize=(8, 4.8)
            )
            linear_views = [
                (durations[durations <= short_limit], short_limit, "Normal scale · 0–60 min"),
                (durations[durations <= duration_p95], duration_p95, "Normal scale · through P95"),
            ]
            for axis, (values, upper, title) in zip(
                distribution_axes[0], linear_views
            ):
                axis.hist(
                    values,
                    bins=24,
                    range=(0, upper),
                    color="#8ce663",
                    edgecolor="#ffffff",
                    linewidth=0.5,
                )
                axis.set_title(title, fontsize=9)
                axis.set_ylabel("Count")
                axis.set_xlabel("Minutes")
                axis.grid(axis="y", color="#e5e9e6", linewidth=0.6)
                axis.set_axisbelow(True)

            if not positive_durations.empty:
                log_min = float(positive_durations.min())
                log_max = float(positive_durations.max())
                if log_min == log_max:
                    log_max = log_min * 1.01
                log_bins = np.geomspace(log_min, log_max, 25)
                capped_durations = positive_durations[
                    positive_durations <= duration_p99
                ]
                capped_max = max(float(capped_durations.max()), log_min * 1.01)
                capped_bins = np.geomspace(log_min, capped_max, 25)

                log_views = [
                    (positive_durations, log_bins, "Log scale · all positive durations"),
                    (capped_durations, capped_bins, "Log scale · capped at P99"),
                ]
                for axis, (values, bins, title) in zip(
                    distribution_axes[1], log_views
                ):
                    axis.hist(
                        values,
                        bins=bins,
                        color="#3b91d1",
                        edgecolor="#ffffff",
                        linewidth=0.5,
                    )
                    axis.set_xscale("log")
                    axis.set_title(title, fontsize=9)
                    axis.set_ylabel("Count")
                    axis.set_xlabel("Minutes · log")
                    axis.grid(axis="y", color="#e5e9e6", linewidth=0.6)
                    axis.set_axisbelow(True)

            distribution_fig.tight_layout()
            st.pyplot(distribution_fig, use_container_width=True)
            plt.close(distribution_fig)
            st.caption("The long-tail shape highlights that most disruptions are short, but a smaller set of very long STOPs drive the higher duration risk.")

    st.markdown(
        f"""
        <div class="dashboard-summary-row">
            <div class="dashboard-summary" style="background:#00B6CE; box-shadow: inset 0 1px 0 rgba(255,255,255,0.18), inset 0 -8px 12px rgba(0,0,0,0.09);">
                <div class="dashboard-summary-title">TYPICAL STOP</div>
                <div class="dashboard-summary-detail">Median duration: {median_duration:.2f} min</div>
            </div>
            <div class="dashboard-summary" style="background:#AB12A6; box-shadow: inset 0 1px 0 rgba(255,255,255,0.18), inset 0 -8px 12px rgba(0,0,0,0.09);">
                <div class="dashboard-summary-title">LONG-TAIL CHECK</div>
                <div class="dashboard-summary-detail">95% last {q95:.2f} min or less</div>
            </div>
            <div class="dashboard-summary" style="background:#D98600; box-shadow: inset 0 1px 0 rgba(255,255,255,0.18), inset 0 -8px 12px rgba(0,0,0,0.09);">
                <div class="dashboard-summary-title">LONGEST STOP</div>
                <div class="dashboard-summary-detail">{max_duration:.2f} min · {prolonged_events:,} prolonged events</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# --- page 2 : stop analysis
elif page == "STOP Analysis":

    render_page_header(
        "STOP Analysis",
        "Temporal patterns and the most extreme STOP events in the dataset.",
        spacing="balanced",
    )

    temporal_col, frequency_col = st.columns([1, 1])

    gap = pd.to_numeric(df["minutes_since_previous_STOP"], errors="coerce")
    duration = pd.to_numeric(df["duration_minutes"], errors="coerce")
    temporal_data = pd.DataFrame({"gap": gap, "duration": duration}).dropna()
    gap_threshold = temporal_data["gap"].median()
    duration_threshold = temporal_data["duration"].median()
    duration_upper_threshold = temporal_data["duration"].quantile(0.90)
    gap_display_max = max(float(temporal_data["gap"].quantile(0.95)), 1.0)
    duration_display_max = max(
        float(temporal_data["duration"].quantile(0.95)), 1.0
    )

    with temporal_col:
        with st.container(border=True):
            st.markdown(
                '<div class="analysis-section-title">TEMPORAL STOP PATTERN</div>',
                unsafe_allow_html=True,
            )
            st.caption("Previous STOP gap vs current event duration")

            rapid_long = (
                (temporal_data["duration"] > duration_threshold)
                & (temporal_data["gap"] <= gap_threshold)
            )
            recovered_long = (
                (temporal_data["duration"] > duration_threshold)
                & (temporal_data["gap"] > gap_threshold)
            )
            short_stop = temporal_data["duration"] <= duration_threshold

            temporal_fig, temporal_ax = plt.subplots(figsize=(8, 4.6))
            temporal_ax.scatter(
                temporal_data.loc[rapid_long, "gap"].clip(upper=gap_display_max),
                temporal_data.loc[rapid_long, "duration"].clip(
                    upper=duration_display_max
                ),
                color="#d63384",
                marker="s",
                alpha=0.72,
                s=36,
                label="Short gap · longer STOP",
            )
            temporal_ax.scatter(
                temporal_data.loc[recovered_long, "gap"].clip(
                    upper=gap_display_max
                ),
                temporal_data.loc[recovered_long, "duration"].clip(
                    upper=duration_display_max
                ),
                color="#243b70",
                marker="D",
                alpha=0.72,
                s=34,
                label="Long gap · longer STOP",
            )
            temporal_ax.scatter(
                temporal_data.loc[short_stop, "gap"].clip(upper=gap_display_max),
                temporal_data.loc[short_stop, "duration"].clip(
                    upper=duration_display_max
                ),
                color="#78c936",
                marker="^",
                alpha=0.72,
                s=38,
                label="Short STOP",
            )
            temporal_ax.axvline(
                gap_threshold, color="#34443a", linewidth=1.1, linestyle="--"
            )
            temporal_ax.axhline(
                duration_threshold,
                color="#34443a",
                linewidth=1.1,
                linestyle="--",
            )
            temporal_ax.axhline(
                duration_upper_threshold,
                color="#8c9690",
                linewidth=1,
                linestyle=":",
            )
            temporal_ax.text(
                0.02,
                0.96,
                "V7",
                transform=temporal_ax.transAxes,
                color="#26352b",
                fontweight="bold",
                va="top",
            )
            temporal_ax.set_xlim(0, gap_display_max * 1.04)
            temporal_ax.set_ylim(0, duration_display_max * 1.08)
            temporal_ax.set_xlabel("Minutes since previous STOP")
            temporal_ax.set_ylabel("STOP duration (minutes)")
            temporal_ax.grid(color="#e3e8e4", linewidth=0.7)
            temporal_ax.set_axisbelow(True)
            temporal_ax.legend(loc="upper right", fontsize=8, frameon=False)
            temporal_fig.tight_layout()
            st.pyplot(temporal_fig, use_container_width=True)
            plt.close(temporal_fig)
            st.caption(
                "Dashed lines show median thresholds; the dotted line marks P90. "
                "Axes are capped at P95 to keep the main event pattern readable."
            )

    with frequency_col:
        with st.container(border=True):
            st.markdown(
                '<div class="analysis-section-title">STOP FREQUENCY BY TIME WINDOW</div>',
                unsafe_allow_html=True,
            )
            st.caption("Average STOP events per recorded day")

            time_window_labels = ["00–05", "05–10", "10–15", "15–20", "20–24"]
            windowed_events = df[["date", "start_time"]].copy()
            windowed_events["time_window"] = pd.cut(
                windowed_events["start_time"].dt.hour,
                bins=[0, 5, 10, 15, 20, 24],
                labels=time_window_labels,
                right=False,
                include_lowest=True,
            )
            daily_window_counts = pd.crosstab(
                windowed_events["date"], windowed_events["time_window"]
            ).reindex(columns=time_window_labels, fill_value=0)
            average_by_window = daily_window_counts.mean(axis=0)
            bar_colors = ["#e34b4b", "#28b9d2", "#668f75", "#ead45b", "#e982ae"]

            frequency_fig, frequency_ax = plt.subplots(figsize=(7, 4.6))
            bars = frequency_ax.bar(
                time_window_labels,
                average_by_window,
                color=bar_colors,
                width=0.66,
                edgecolor="#ffffff",
                linewidth=1,
            )
            frequency_ax.bar_label(bars, fmt="%.1f", padding=3, fontsize=8)
            frequency_ax.set_ylim(0, 10)
            frequency_ax.set_ylabel("Average STOP events per day")
            frequency_ax.set_xlabel("Time of day")
            frequency_ax.grid(axis="y", color="#e3e8e4", linewidth=0.7)
            frequency_ax.set_axisbelow(True)
            frequency_fig.tight_layout()
            st.pyplot(frequency_fig, use_container_width=True)
            plt.close(frequency_fig)

    st.markdown(
        '<div class="analysis-section-title">EXTREME STOP EVENTS '
        '<span class="analysis-title-tag">TOP 10</span></div>',
        unsafe_allow_html=True,
    )

    longest_events = (
        df[["event_id", "start_time", "end_time", "duration_minutes"]]
        .sort_values("duration_minutes", ascending=False)
        .head(10)
        .rename(
            columns={
                "event_id": "Event ID",
                "start_time": "Start time",
                "end_time": "End time",
                "duration_minutes": "Duration (min)",
            }
        )
        .copy()
    )
    longest_events["Start time"] = longest_events["Start time"].dt.strftime(
        "%Y-%m-%d %H:%M"
    )
    longest_events["End time"] = longest_events["End time"].dt.strftime(
        "%Y-%m-%d %H:%M"
    )
    longest_events["Duration (min)"] = longest_events["Duration (min)"].map(
        lambda value: f"{value:.2f}"
    )
    table_html = longest_events.to_html(
        index=False,
        classes="extreme-events-table",
        border=0,
        escape=True,
    )
    st.markdown(
        f"""
        <div class="analysis-table-wrap">
            {table_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


# --- page 3 : prediction
elif page == "Prediction":

    render_page_header(
        "Prediction",
        "Estimate whether the next STOP is likely to be prolonged and how long it may last.",
        spacing="balanced",
    )

    input_col, results_col = st.columns([0.9, 1.1], gap="medium")

    with input_col:
        with st.container(border=True):
            st.markdown(
                '<div class="prediction-input-title">STOP EVENT INFORMATION</div>',
                unsafe_allow_html=True,
            )

            with st.form("stop_prediction_form"):
                minutes_since_previous_STOP = st.number_input(
                    "Minutes since previous STOP",
                    min_value=0.0,
                    value=10.0,
                    step=0.1,
                )
                STOP_count_last_1min = st.number_input(
                    "STOP count — last 1 minute",
                    min_value=0,
                    value=1,
                    step=1,
                )
                STOP_count_last_5min = st.number_input(
                    "STOP count — last 5 minutes",
                    min_value=0,
                    value=2,
                    step=1,
                )
                STOP_count_last_10min = st.number_input(
                    "STOP count — last 10 minutes",
                    min_value=0,
                    value=3,
                    step=1,
                )
                predict_button = st.form_submit_button(
                    "PREDICT ↗",
                    use_container_width=True,
                )

    if predict_button:
        features = pd.DataFrame([{
            "minutes_since_previous_STOP": minutes_since_previous_STOP,
            "STOP_count_last_1min": STOP_count_last_1min,
            "STOP_count_last_5min": STOP_count_last_5min,
            "STOP_count_last_10min": STOP_count_last_10min,
        }], columns=PREDICTION_FEATURES)
        transformed_features = yeo_johnson_transformer.transform(features)

        severity_probability = float(
            severity_model.predict_proba(transformed_features)[0, 1]
        )
        severity_label = (
            "Prolonged STOP"
            if severity_probability >= SEVERITY_THRESHOLD
            else "Normal STOP"
        )
        duration_log_prediction = duration_model.predict(transformed_features)
        predicted_duration = max(float(np.expm1(duration_log_prediction)[0]), 0)

        st.session_state["prediction_result"] = {
            "severity_label": severity_label,
            "severity_probability": severity_probability,
            "predicted_duration": predicted_duration,
        }

    result = st.session_state.get("prediction_result")
    if result:
        severity_label = result["severity_label"]
        severity_probability = result["severity_probability"]
        predicted_duration = result["predicted_duration"]
        probability_text = f"{severity_probability * 100:.1f}%"
        gauge_width = f"{severity_probability * 100:.1f}%"
        duration_text = f"{predicted_duration:.2f} minutes"
    else:
        severity_label = "Awaiting prediction"
        probability_text = "--.-%"
        gauge_width = "0%"
        duration_text = "-- minutes"

    with results_col:
        st.markdown(
            f"""
            <div class="prediction-result-banner">
                <div class="prediction-result-eyebrow">PREDICTION RESULT</div>
                <div class="prediction-result-values">
                    <div>Severity<strong>{severity_label}</strong></div>
                    <div>Severity Probability<strong>{probability_text}</strong></div>
                </div>
                <div class="prediction-gauge">
                    <div class="prediction-gauge-fill" style="width:{gauge_width}"></div>
                </div>
                <div class="prediction-threshold">
                    Classification threshold: {SEVERITY_THRESHOLD:.2f}
                </div>
            </div>
            <div class="prediction-duration-card">
                <div class="prediction-duration-title">PREDICTION RESULT</div>
                <div class="prediction-duration-label">Estimated STOP Duration</div>
                <div class="prediction-duration-value">{duration_text}</div>
            </div>
            <div class="prediction-info-banner">
                <strong>MODEL NOTE</strong>
                <span>Duration and severity are estimates based on historical STOP patterns. Rare events or unusual operating conditions may reduce accuracy; use these results as decision support, not as a substitute for operator judgment.</span>
            </div>
            """,
            unsafe_allow_html=True,
        )


# --- page 4 : model evaluation
elif page == "Model Evaluation":

    render_page_header(
        "Model Evaluation",
        "Performance summary for the classification and duration models.",
        spacing="balanced",
    )

    evaluation_metrics = pd.read_csv(MODEL_DIR / "metrics.csv")
    classification_test = evaluation_metrics.loc[
        evaluation_metrics["task"].eq("classification")
        & evaluation_metrics["split"].eq("test")
    ].iloc[0]
    regression_validation = evaluation_metrics.loc[
        evaluation_metrics["task"].eq("regression")
        & evaluation_metrics["split"].eq("validation")
    ].iloc[0]
    regression_test = evaluation_metrics.loc[
        evaluation_metrics["task"].eq("regression")
        & evaluation_metrics["split"].eq("test")
    ].iloc[0]

    classification_col, regression_col = st.columns([1, 1], gap="medium")

    with classification_col:
        with st.container(border=True):
            st.markdown(
                '<div class="evaluation-section-title">CLASSIFICATION</div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                f"""
                <div class="evaluation-metrics-grid">
                    <div class="evaluation-metric"><div class="evaluation-metric-label">Test Accuracy</div><div class="evaluation-metric-value">{classification_test['accuracy']:.2%}</div></div>
                    <div class="evaluation-metric"><div class="evaluation-metric-label">Test Precision</div><div class="evaluation-metric-value">{classification_test['precision']:.2%}</div></div>
                    <div class="evaluation-metric"><div class="evaluation-metric-label">Test Recall</div><div class="evaluation-metric-value">{classification_test['recall']:.2%}</div></div>
                    <div class="evaluation-metric"><div class="evaluation-metric-label">Test F1-score</div><div class="evaluation-metric-value">{classification_test['f1']:.2%}</div></div>
                    <div class="evaluation-metric"><div class="evaluation-metric-label">Test ROC-AUC</div><div class="evaluation-metric-value">{classification_test['roc_auc']:.2%}</div></div>
                    <div class="evaluation-metric"><div class="evaluation-metric-label">Test PR-AUC</div><div class="evaluation-metric-value">{classification_test['pr_auc']:.2%}</div></div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with st.container(border=True):
            st.markdown(
                '<div class="evaluation-section-title">CONFUSION MATRIX</div>',
                unsafe_allow_html=True,
            )
            confusion_values = np.array(
                json.loads(classification_test["confusion_matrix"])
            )
            confusion_fig, confusion_ax = plt.subplots(figsize=(5.8, 4.2))
            heatmap = confusion_ax.imshow(
                confusion_values,
                cmap="Greens",
                vmin=2,
                vmax=int(confusion_values.max()),
            )
            confusion_ax.set_xticks([0, 1], labels=["Normal", "Prolonged"])
            confusion_ax.set_yticks([0, 1], labels=["Normal", "Prolonged"])
            confusion_ax.set_xlabel("Predicted Label")
            confusion_ax.set_ylabel("Actual Label")
            confusion_ax.xaxis.set_label_position("top")
            confusion_ax.xaxis.tick_top()
            for row in range(confusion_values.shape[0]):
                for column in range(confusion_values.shape[1]):
                    text_color = (
                        "white"
                        if confusion_values[row, column] > confusion_values.max() * 0.55
                        else "#26352b"
                    )
                    confusion_ax.text(
                        column,
                        row,
                        str(confusion_values[row, column]),
                        ha="center",
                        va="center",
                        color=text_color,
                        fontsize=12,
                        fontweight="bold",
                    )
            confusion_fig.colorbar(heatmap, ax=confusion_ax, label="Events")
            confusion_fig.tight_layout()
            st.pyplot(confusion_fig, use_container_width=True)
            plt.close(confusion_fig)

        st.markdown(
            """
            <div class="evaluation-note-card">
                <strong>Model Limitation</strong>
                <p>The duration model performs better on typical short STOPs than on rare extremes. Current temporal features do not provide enough information to identify those extreme events reliably.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with regression_col:
        with st.container(border=True):
            st.markdown(
                '<div class="evaluation-section-title">REGRESSION</div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                f"""
                <div class="evaluation-regression-grid">
                    <div class="evaluation-metric"><div class="evaluation-metric-label">Validation MAE</div><div class="evaluation-metric-value">{regression_validation['MAE_minutes']:.2f} min</div></div>
                    <div class="evaluation-metric"><div class="evaluation-metric-label">Test MAE</div><div class="evaluation-metric-value">{regression_test['MAE_minutes']:.2f} min</div></div>
                    <div class="evaluation-metric"><div class="evaluation-metric-label">Test RMSE</div><div class="evaluation-metric-value">{regression_test['RMSE_minutes']:.2f} min</div></div>
                    <div class="evaluation-metric"><div class="evaluation-metric-label">Test MAPE</div><div class="evaluation-metric-value">{regression_test['MAPE_percent']:.2f}%</div></div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with st.container(border=True):
            st.markdown(
                '<div class="evaluation-section-title">REGRESSION MODEL COMPARISON · YEO-JOHNSON TEST</div>',
                unsafe_allow_html=True,
            )
            model_comparison = pd.read_csv(
                MODEL_DIR / "regression_model_comparison.csv"
            )
            comparison = model_comparison.loc[
                model_comparison["Split"].eq("test"),
                ["Model", "MAE_minutes", "RMSE_minutes", "MAPE_percent"],
            ].rename(
                columns={
                    "Model": "Model",
                    "MAE_minutes": "MAE (min)",
                    "RMSE_minutes": "RMSE (min)",
                    "MAPE_percent": "MAPE",
                }
            )
            comparison["MAE (min)"] = comparison["MAE (min)"].map(
                lambda value: f"{value:.2f}"
            )
            comparison["RMSE (min)"] = comparison["RMSE (min)"].map(
                lambda value: f"{value:.2f}"
            )
            comparison["MAPE"] = comparison["MAPE"].map(
                lambda value: f"{value:.2f}%"
            )
            comparison_html = comparison.to_html(
                index=False,
                classes="evaluation-comparison-table",
                border=0,
                escape=True,
            )
            st.markdown(
                f'<div class="comparison-table-wrap">{comparison_html}</div>',
                unsafe_allow_html=True,
            )