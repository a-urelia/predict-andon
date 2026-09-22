import streamlit as st
import joblib
import numpy as np
import pandas as pd


# ============================================================
# LOAD MODELS
# ============================================================

severity_model = joblib.load(
    "models/stop_severity_logistic_model.joblib"
)

duration_model = joblib.load(
    "models/stop_duration_log_linear_model.joblib"
)

SEVERITY_THRESHOLD = 0.30


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Andon Stop Prediction",
    page_icon="⚙️",
    layout="centered"
)


# ============================================================
# TITLE
# ============================================================

st.title("Andon Stop Prediction")
st.caption("MC09 — Machine STOP Severity and Duration Prediction")

st.divider()


# ============================================================
# INPUT FEATURES
# ============================================================

st.subheader("STOP Event Information")

minutes_since_previous_STOP = st.number_input(
    "Minutes since previous STOP",
    min_value=0.0,
    value=10.0,
    step=0.1
)

STOP_count_last_1min = st.number_input(
    "STOP count — last 1 minute",
    min_value=0,
    value=1,
    step=1
)

STOP_count_last_5min = st.number_input(
    "STOP count — last 5 minutes",
    min_value=0,
    value=2,
    step=1
)

STOP_count_last_10min = st.number_input(
    "STOP count — last 10 minutes",
    min_value=0,
    value=3,
    step=1
)


# ============================================================
# PREDICTION
# ============================================================

if st.button("Predict", use_container_width=True):

    # Create DataFrame with the exact feature names
    # used during model training.
    features = pd.DataFrame([{
        "minutes_since_previous_STOP": minutes_since_previous_STOP,
        "STOP_count_last_1min": STOP_count_last_1min,
        "STOP_count_last_5min": STOP_count_last_5min,
        "STOP_count_last_10min": STOP_count_last_10min
    }])

    # --------------------------------------------------------
    # Severity prediction
    # --------------------------------------------------------

    severity_probability = severity_model.predict_proba(
        features
    )[0, 1]

    severity_prediction = (
        1
        if severity_probability >= SEVERITY_THRESHOLD
        else 0
    )

    if severity_prediction == 1:
        severity_label = "Prolonged STOP"
    else:
        severity_label = "Normal STOP"

    # --------------------------------------------------------
    # Duration prediction
    # --------------------------------------------------------

    duration_log_prediction = duration_model.predict(
        features
    )

    predicted_duration = np.expm1(
        duration_log_prediction
    )[0]

    predicted_duration = max(predicted_duration, 0)

    # --------------------------------------------------------
    # Display results
    # --------------------------------------------------------

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

    st.metric(
        "Estimated STOP Duration",
        f"{predicted_duration:.2f} minutes"
    )

    st.caption(
        f"Classification threshold: {SEVERITY_THRESHOLD:.2f}"
    )
