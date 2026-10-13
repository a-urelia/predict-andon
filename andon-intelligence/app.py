import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from pathlib import Path


# ============================================================
# CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Andon Stop Intelligence",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    :root {
        --andon-sidebar: #e5f4e3;
        --andon-active: #8ce663;
        --andon-ink: #0d2806;
        --andon-muted: #556254;
    }

    .stApp,
    [data-testid="stSidebar"] {
        background-color: var(--andon-sidebar);
    }

    [data-testid="stHeader"] {
        background: transparent;
    }

    section[data-testid="stSidebar"] {
        min-width: 280px;
        width: 280px;
    }

    section[data-testid="stSidebar"] > div {
        padding: 2rem 0 2rem 2rem;
    }

    .brand-header {
        margin: 0 0 2.5rem;
        color: #000;
        font-size: 1.25rem;
        font-weight: 900;
        line-height: 1.1;
    }

    [data-testid="stSidebar"] [data-testid="stRadio"] > label {
        display: none;
    }

    [data-testid="stSidebar"] [data-testid="stRadio"] [role="radiogroup"] {
        gap: 0.75rem;
    }

    [data-testid="stSidebar"] [data-testid="stRadio"] [role="radiogroup"] label {
        margin-right: -2rem;
        padding: 0.875rem 1.25rem;
        border-radius: 1.75rem 0 0 1.75rem;
        color: var(--andon-muted);
        transition: background-color 0.2s ease, color 0.2s ease;
    }

    [data-testid="stSidebar"] [data-testid="stRadio"] [role="radiogroup"] label:hover {
        background-color: rgba(0, 0, 0, 0.04);
        color: #1a1a1a;
    }

    [data-testid="stSidebar"] [data-testid="stRadio"] [role="radiogroup"] label:has(input:checked) {
        background-color: var(--andon-active);
        color: var(--andon-ink);
        font-weight: 600;
    }

    [data-testid="stMain"] {
        min-height: 100vh;
        background-color: #fff;
        border-radius: 2.5rem 0 0 2.5rem;
        box-shadow: -10px 0 30px rgba(0, 0, 0, 0.03);
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
        margin: 1.5rem 0 1.75rem;
    }

    .dashboard-kpi {
        min-width: 0;
        padding: 1.1rem 1.2rem;
        border: 1px solid #dce6dc;
        border-radius: 0.75rem;
        background: #fff;
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
    }

    .dashboard-summary-title {
        margin-bottom: 0.4rem;
        font-size: 0.78rem;
        font-weight: 750;
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
        gap: 0.75rem;
        margin: 1rem 0 1.3rem;
    }

    .evaluation-metric {
        padding: 0.85rem 1rem;
        border: 1px solid #dce6dc;
        border-radius: 0.6rem;
        background: #fff;
    }

    .evaluation-metric-label {
        color: #59665a;
        font-size: 0.76rem;
        font-weight: 650;
    }

    .evaluation-metric-value {
        margin-top: 0.35rem;
        color: #174a32;
        font-size: 1.45rem;
        font-weight: 750;
    }

    .evaluation-regression-grid {
        display: grid;
        grid-template-columns: repeat(2, minmax(0, 1fr));
        gap: 0.75rem;
        margin-top: 0.8rem;
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

    .prediction-result-values {
        display: flex;
        justify-content: space-between;
        gap: 1rem;
        font-size: 1rem;
        font-weight: 650;
    }

    .prediction-result-values strong {
        display: block;
        margin-top: 0.3rem;
        font-size: 1.15rem;
    }

    .prediction-gauge {
        height: 0.6rem;
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
        padding: 1.25rem;
        border-radius: 0.75rem;
        background: #159cbd;
        color: #fff;
        text-align: center;
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


# ============================================================
# LOAD MODELS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR.parent / "models"

severity_model = joblib.load(
    MODEL_DIR / "stop_severity_logistic_model.joblib"
)

duration_model = joblib.load(
    MODEL_DIR / "stop_duration_log_linear_model.joblib"
)

SEVERITY_THRESHOLD = 0.30


# ============================================================
# LOAD DATA
# ============================================================

# Load either the prepared event dataset or raw MC09 telemetry.
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


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.markdown(
    '<div class="brand-header">ANDON STOP<br>INTELLIGENCE</div>',
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
    label_visibility="collapsed",
)


# ============================================================
# PAGE 1 — DASHBOARD
# ============================================================

if page == "Dashboard":

    st.title("Dashboard")

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

    trend_col, distribution_col = st.columns([1.15, 1])

    with trend_col:
        with st.container(border=True):
            st.markdown("#### STOP EVENT TREND")
            st.caption("Number of STOP events per day")

            trend_fig, trend_ax = plt.subplots(figsize=(8, 4.1))
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

    with distribution_col:
        with st.container(border=True):
            st.markdown("#### STOP DURATION DISTRIBUTION")

            durations = df["duration_minutes"].dropna().clip(lower=0)
            duration_p95 = max(float(durations.quantile(0.95)), 0.01)
            duration_p99 = max(float(durations.quantile(0.99)), 0.01)
            short_limit = max(min(float(durations.max()), 60), 0.01)
            positive_durations = durations[durations > 0]

            distribution_fig, distribution_axes = plt.subplots(
                2, 2, figsize=(8, 5.8)
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

    st.markdown(
        f"""
        <div class="dashboard-summary-row">
            <div class="dashboard-summary" style="background:#159cbd">
                <div class="dashboard-summary-title">TYPICAL STOP</div>
                <div class="dashboard-summary-detail">Median duration: {median_duration:.2f} min</div>
            </div>
            <div class="dashboard-summary" style="background:#8250b5">
                <div class="dashboard-summary-title">LONG-TAIL CHECK</div>
                <div class="dashboard-summary-detail">95% last {q95:.2f} min or less</div>
            </div>
            <div class="dashboard-summary" style="background:#9b6500">
                <div class="dashboard-summary-title">LONGEST STOP</div>
                <div class="dashboard-summary-detail">{max_duration:.2f} min · {prolonged_events:,} prolonged events</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# PAGE 2 — STOP ANALYSIS
# ============================================================

elif page == "STOP Analysis":

    st.title("STOP Analysis")

    temporal_col, frequency_col = st.columns([1.15, 1])

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
            st.markdown("#### TEMPORAL STOP PATTERN")
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
            st.markdown("#### STOP FREQUENCY BY TIME WINDOW")
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

    st.markdown("#### EXTREME STOP EVENTS")

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


# ============================================================
# PAGE 3 — PREDICTION
# ============================================================

elif page == "Prediction":

    st.title("Prediction")

    input_col, results_col = st.columns([0.9, 1.1], gap="large")

    with input_col:
        with st.container(border=True):
            st.markdown("#### STOP EVENT INFORMATION")

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
        }])

        severity_probability = float(
            severity_model.predict_proba(features)[0, 1]
        )
        severity_label = (
            "Prolonged STOP"
            if severity_probability >= SEVERITY_THRESHOLD
            else "Normal STOP"
        )
        duration_log_prediction = duration_model.predict(features)
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
                <strong>INFO</strong>
                <span>(ON ESTIMATED RESULT)</span>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ============================================================
# PAGE 4 — MODEL EVALUATION
# ============================================================

elif page == "Model Evaluation":

    st.title("Model Evaluation")

    classification_col, regression_col = st.columns([1, 1], gap="large")

    with classification_col:
        with st.container(border=True):
            st.markdown("#### CLASSIFICATION")
            st.markdown(
                """
                <div class="evaluation-metrics-grid">
                    <div class="evaluation-metric"><div class="evaluation-metric-label">Accuracy</div><div class="evaluation-metric-value">73.49%</div></div>
                    <div class="evaluation-metric"><div class="evaluation-metric-label">Precision</div><div class="evaluation-metric-value">42.86%</div></div>
                    <div class="evaluation-metric"><div class="evaluation-metric-label">Recall</div><div class="evaluation-metric-value">66.67%</div></div>
                    <div class="evaluation-metric"><div class="evaluation-metric-label">F1-score</div><div class="evaluation-metric-value">52.17%</div></div>
                    <div class="evaluation-metric"><div class="evaluation-metric-label">ROC-AUC</div><div class="evaluation-metric-value">74.49%</div></div>
                    <div class="evaluation-metric"><div class="evaluation-metric-label">PR-AUC</div><div class="evaluation-metric-value">41.65%</div></div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.markdown("##### Confusion Matrix")
            confusion_values = np.array([[49, 16], [6, 12]])
            confusion_fig, confusion_ax = plt.subplots(figsize=(5.6, 4.1))
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
            <div class="model-limitation-card">
                <strong>MODEL LIMITATION</strong>
                <p>The duration model performs better on typical short STOPs than on rare extremes. Current temporal features do not provide enough information to identify those extreme events reliably.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with regression_col:
        with st.container(border=True):
            st.markdown("#### REGRESSION")
            st.markdown(
                """
                <div class="evaluation-regression-grid">
                    <div class="evaluation-metric"><div class="evaluation-metric-label">Validation MAE</div><div class="evaluation-metric-value">22.74 min</div></div>
                    <div class="evaluation-metric"><div class="evaluation-metric-label">Test MAE</div><div class="evaluation-metric-value">37.90 min</div></div>
                    <div class="evaluation-metric"><div class="evaluation-metric-label">Test RMSE</div><div class="evaluation-metric-value">223.79 min</div></div>
                    <div class="evaluation-metric"><div class="evaluation-metric-label">Test MAPE</div><div class="evaluation-metric-value">357.49%</div></div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with st.container(border=True):
            st.markdown("#### REGRESSION MODEL COMPARISON")
            comparison = pd.DataFrame({
                "Model": [
                    "Mean Baseline",
                    "Median Baseline",
                    "Linear Regression",
                    "Decision Tree",
                    "Random Forest",
                    "Log-Linear Regression",
                    "Random Forest + Log Target",
                ],
                "MAE (min)": [
                    37.07, 22.80, 25.81, 26.98, 26.74, 22.74, 23.03,
                ],
                "RMSE (min)": [
                    165.69, 167.09, 166.21, 169.59, 167.64, 166.94, 167.09,
                ],
                "MAPE": [
                    3200.94, 232.07, 1097.00, 1316.91, 1056.28, 264.69, 264.95,
                ],
            })
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