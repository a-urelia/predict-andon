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
    layout="wide"
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

st.sidebar.title("⚙️ Andon Stop Intelligence")

st.sidebar.caption("MC09 — Machine STOP Analytics & Prediction")

page = st.sidebar.radio(
    "Navigation",
    [
        "📊 Dashboard",
        "🔎 STOP Analysis",
        "🔮 Prediction",
        "📈 Model Evaluation"
    ]
)


# ============================================================
# PAGE 1 — DASHBOARD
# ============================================================

if page == "📊 Dashboard":

    st.title("MC09 STOP Overview")

    st.caption(
        "Overview of machine STOP events and their duration characteristics."
    )

    st.divider()

    # --------------------------------------------------------
    # KPI
    # --------------------------------------------------------

    total_events = len(df)
    median_duration = df["duration_minutes"].median()
    mean_duration = df["duration_minutes"].mean()
    max_duration = df["duration_minutes"].max()

    prolonged_events = (
        df["duration_minutes"] > 5
    ).sum()

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Total STOP Events",
        f"{total_events:,}"
    )

    col2.metric(
        "Median STOP Duration",
        f"{median_duration:.2f} min"
    )

    col3.metric(
        "Prolonged STOP",
        f"{prolonged_events:,}"
    )

    col4.metric(
        "Maximum STOP Duration",
        f"{max_duration:.2f} min"
    )

    st.divider()

    # --------------------------------------------------------
    # STOP FREQUENCY OVER TIME
    # --------------------------------------------------------

    st.subheader("STOP Event Frequency Over Time")

    daily_events = (
        df.groupby("date")
        .size()
        .reset_index(name="stop_count")
    )

    st.line_chart(
        daily_events.set_index("date")
    )

    st.caption(
        "Number of STOP events recorded for each day."
    )

    st.divider()

    # --------------------------------------------------------
    # DURATION DISTRIBUTION
    # --------------------------------------------------------

    st.subheader("STOP Duration Distribution")

    log_scale = st.checkbox(
        "Use logarithmic scale",
        value=True
    )

    fig, ax = plt.subplots()

    ax.hist(
        df["duration_minutes"],
        bins=40
    )

    ax.set_xlabel("STOP Duration (minutes)")
    ax.set_ylabel("Number of Events")
    ax.set_title("Distribution of STOP Duration")

    if log_scale:
        ax.set_xscale("log")

    st.pyplot(fig)

    st.divider()

    # --------------------------------------------------------
    # DATA INSIGHTS
    # --------------------------------------------------------

    st.subheader("Key Data Insights")

    q95 = df["duration_minutes"].quantile(0.95)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.info(
            f"""
            **Typical STOP Duration**

            50% of STOP events have a duration
            of **{median_duration:.2f} minutes or less**.
            """
        )

    with col2:
        st.info(
            f"""
            **Long-Tail Distribution**

            95% of STOP events have a duration
            of **{q95:.2f} minutes or less**.
            """
        )

    with col3:
        st.warning(
            f"""
            **Extreme Event**

            The longest recorded STOP lasted
            **{max_duration:.2f} minutes**.
            """
        )


# ============================================================
# PAGE 2 — STOP ANALYSIS
# ============================================================

elif page == "🔎 STOP Analysis":

    st.title("STOP Event Analysis")

    st.caption(
        "Analysis of temporal characteristics used as model inputs."
    )

    st.divider()

    # --------------------------------------------------------
    # FEATURE EXPLANATION
    # --------------------------------------------------------

    st.subheader("Temporal Features")

    feature_data = pd.DataFrame({
        "Feature": [
            "minutes_since_previous_STOP",
            "STOP_count_last_1min",
            "STOP_count_last_5min",
            "STOP_count_last_10min"
        ],
        "Meaning": [
            "Time since the previous STOP event",
            "Number of STOP events in the previous 1 minute",
            "Number of STOP events in the previous 5 minutes",
            "Number of STOP events in the previous 10 minutes"
        ]
    })

    st.table(feature_data)

    st.divider()

    # --------------------------------------------------------
    # PREVIOUS STOP GAP VS CURRENT DURATION
    # --------------------------------------------------------

    st.subheader(
        "Previous STOP Gap vs Current STOP Duration"
    )

    fig, ax = plt.subplots()

    ax.scatter(
        df["minutes_since_previous_STOP"],
        df["duration_minutes"],
        alpha=0.5
    )

    ax.set_xlabel(
        "Minutes Since Previous STOP"
    )

    ax.set_ylabel(
        "Current STOP Duration (minutes)"
    )

    ax.set_title(
        "Temporal Relationship Between STOP Events"
    )

    st.pyplot(fig)

    st.caption(
        "Each point represents one STOP event."
    )

    st.divider()

    # --------------------------------------------------------
    # STOP FREQUENCY FEATURES
    # --------------------------------------------------------

    st.subheader("Recent STOP Activity")

    frequency_means = pd.DataFrame({
        "Time Window": [
            "Previous 1 minute",
            "Previous 5 minutes",
            "Previous 10 minutes"
        ],
        "Average STOP Count": [
            df["STOP_count_last_1min"].mean(),
            df["STOP_count_last_5min"].mean(),
            df["STOP_count_last_10min"].mean()
        ]
    })

    st.bar_chart(
        frequency_means.set_index("Time Window")
    )

    st.divider()

    # --------------------------------------------------------
    # LONGEST EVENTS
    # --------------------------------------------------------

    st.subheader("Extreme STOP Events")

    longest_events = (
        df[
            [
                "event_id",
                "start_time",
                "end_time",
                "duration_minutes"
            ]
        ]
        .sort_values(
            "duration_minutes",
            ascending=False
        )
        .head(10)
    )

    st.dataframe(
        longest_events,
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# PAGE 3 — PREDICTION
# ============================================================

elif page == "🔮 Prediction":

    st.title("STOP Event Prediction")

    st.caption(
        "Predict severity and estimated STOP duration from temporal context."
    )

    st.divider()

    st.subheader("STOP Event Context")

    col1, col2 = st.columns(2)

    with col1:

        minutes_since_previous_STOP = st.number_input(
            "Minutes since previous STOP",
            min_value=0.0,
            value=10.0,
            step=0.1
        )

        STOP_count_last_1min = st.number_input(
            "STOP count — previous 1 minute",
            min_value=0,
            value=1,
            step=1
        )

    with col2:

        STOP_count_last_5min = st.number_input(
            "STOP count — previous 5 minutes",
            min_value=0,
            value=2,
            step=1
        )

        STOP_count_last_10min = st.number_input(
            "STOP count — previous 10 minutes",
            min_value=0,
            value=3,
            step=1
        )

    predict_button = st.button(
        "🔮 Predict STOP",
        use_container_width=True
    )

    if predict_button:

        features = pd.DataFrame([{
            "minutes_since_previous_STOP":
                minutes_since_previous_STOP,

            "STOP_count_last_1min":
                STOP_count_last_1min,

            "STOP_count_last_5min":
                STOP_count_last_5min,

            "STOP_count_last_10min":
                STOP_count_last_10min
        }])

        # ----------------------------------------------------
        # CLASSIFICATION
        # ----------------------------------------------------

        severity_probability = (
            severity_model
            .predict_proba(features)[0, 1]
        )

        severity_prediction = (
            severity_probability >= SEVERITY_THRESHOLD
        )

        if severity_prediction:
            severity_label = "Prolonged STOP"
        else:
            severity_label = "Normal STOP"

        # ----------------------------------------------------
        # REGRESSION
        # ----------------------------------------------------

        duration_log_prediction = (
            duration_model.predict(features)
        )

        predicted_duration = np.expm1(
            duration_log_prediction
        )[0]

        predicted_duration = max(
            predicted_duration,
            0
        )

        st.divider()

        st.subheader("Prediction Result")

        col1, col2 = st.columns(2)

        with col1:

            st.metric(
                "Severity",
                severity_label
            )

        with col2:

            st.metric(
                "Severity Probability",
                f"{severity_probability * 100:.2f}%"
            )

        st.progress(
            float(severity_probability)
        )

        st.caption(
            f"Classification threshold: "
            f"{SEVERITY_THRESHOLD * 100:.0f}%"
        )

        st.divider()

        st.metric(
            "Estimated STOP Duration",
            f"{predicted_duration:.2f} minutes"
        )

        st.info(
            f"""
            Based on the provided temporal context, the event
            is classified as **{severity_label}** with an
            estimated STOP duration of
            **{predicted_duration:.2f} minutes**.
            """
        )


# ============================================================
# PAGE 4 — MODEL EVALUATION
# ============================================================

elif page == "📈 Model Evaluation":

    st.title("Model Evaluation")

    st.caption(
        "Performance of the deployed severity and duration models."
    )

    st.divider()

    # --------------------------------------------------------
    # CLASSIFICATION
    # --------------------------------------------------------

    st.subheader("Severity Classification")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Accuracy",
        "73.49%"
    )

    col2.metric(
        "Balanced Accuracy",
        "71.03%"
    )

    col3.metric(
        "Precision",
        "42.86%"
    )

    col4.metric(
        "Recall",
        "66.67%"
    )

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "F1 Score",
        "52.17%"
    )

    col2.metric(
        "ROC-AUC",
        "74.49%"
    )

    col3.metric(
        "PR-AUC",
        "41.65%"
    )

    st.markdown("### Confusion Matrix")

    confusion_matrix = pd.DataFrame(
        [
            [49, 16],
            [6, 12]
        ],
        index=[
            "Actual Normal",
            "Actual Prolonged"
        ],
        columns=[
            "Predicted Normal",
            "Predicted Prolonged"
        ]
    )

    st.dataframe(
        confusion_matrix,
        use_container_width=True
    )

    st.divider()

    # --------------------------------------------------------
    # REGRESSION
    # --------------------------------------------------------

    st.subheader("STOP Duration Prediction")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Validation MAE",
        "22.74 min"
    )

    col2.metric(
        "Test MAE",
        "37.90 min"
    )

    col3.metric(
        "Test RMSE",
        "223.79 min"
    )

    st.metric(
        "Test MAPE",
        "357.49%"
    )

    st.divider()

    st.subheader(
        "Regression Model Comparison"
    )

    comparison = pd.DataFrame({
        "Model": [
            "Mean Baseline",
            "Median Baseline",
            "Linear Regression",
            "Decision Tree",
            "Random Forest",
            "Log-Linear Regression",
            "Random Forest + Log Target"
        ],

        "MAE": [
            37.07,
            22.80,
            25.81,
            26.98,
            26.74,
            22.74,
            23.03
        ],

        "RMSE": [
            165.69,
            167.09,
            166.21,
            169.59,
            167.64,
            166.94,
            167.09
        ],

        "MAPE": [
            3200.94,
            232.07,
            1097.00,
            1316.91,
            1056.28,
            264.69,
            264.95
        ]
    })

    st.dataframe(
        comparison,
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    # --------------------------------------------------------
    # MODEL LIMITATION
    # --------------------------------------------------------

    st.subheader("Model Limitation")

    st.warning(
        """
        The regression model performs better on typical
        short-duration STOP events than on rare
        extreme-duration events.

        The available temporal features describe recent
        STOP activity, but do not contain sufficient
        information to distinguish ordinary STOP events
        from rare extreme-duration events.
        """
    )