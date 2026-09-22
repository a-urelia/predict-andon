import joblib
import numpy as np
import pandas as pd

# ==========================================
# Load models
# ==========================================

severity_model = joblib.load(
    "models/stop_severity_logistic_model.joblib"
)

duration_model = joblib.load(
    "models/stop_duration_log_linear_model.joblib"
)


# ==========================================
# Model configuration
# ==========================================

SEVERITY_THRESHOLD = 0.30


# ==========================================
# Test input
# ==========================================
# These are temporary test values.
# Later, these will come from the application UI.

minutes_since_previous_STOP = 10.0
STOP_count_last_1min = 1
STOP_count_last_5min = 2
STOP_count_last_10min = 3


# ==========================================
# Prepare features
# ==========================================

features = pd.DataFrame([{
    "minutes_since_previous_STOP": minutes_since_previous_STOP,
    "STOP_count_last_1min": STOP_count_last_1min,
    "STOP_count_last_5min": STOP_count_last_5min,
    "STOP_count_last_10min": STOP_count_last_10min
}])


# ==========================================
# Severity Prediction
# ==========================================

severity_probability = severity_model.predict_proba(features)[0, 1]

severity_prediction = (
    1 if severity_probability >= SEVERITY_THRESHOLD else 0
)


if severity_prediction == 1:
    severity_label = "Prolonged STOP"
else:
    severity_label = "Normal STOP"


# ==========================================
# Duration Prediction
# ==========================================

# The model was trained using log1p(duration)
duration_log_prediction = duration_model.predict(features)

# Convert prediction back to minutes
predicted_duration = np.expm1(duration_log_prediction)[0]

# Prevent negative prediction
predicted_duration = max(predicted_duration, 0)


# ==========================================
# Display Result
# ==========================================

print()
print("=" * 45)
print("        ANDON STOP PREDICTION")
print("=" * 45)

print()
print("Input Features")
print("-" * 45)
print(f"Minutes since previous STOP : {minutes_since_previous_STOP:.2f}")
print(f"STOP count - last 1 min     : {STOP_count_last_1min}")
print(f"STOP count - last 5 min     : {STOP_count_last_5min}")
print(f"STOP count - last 10 min    : {STOP_count_last_10min}")

print()
print("Prediction Result")
print("-" * 45)
print(f"Severity probability       : {severity_probability:.4f}")
print(f"Severity threshold         : {SEVERITY_THRESHOLD:.2f}")
print(f"Severity classification    : {severity_label}")
print(f"Predicted STOP duration    : {predicted_duration:.2f} minutes")

print()
print("=" * 45)