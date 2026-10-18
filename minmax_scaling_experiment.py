"""Run the thesis-controlled Min-Max scaling experiment without altering the baseline artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "mc09_andon.csv"
MODEL_DIR = BASE_DIR / "models"
VIS_DIR = BASE_DIR / "visualizations"
FEATURES = [
    "minutes_since_previous_STOP",
    "STOP_count_last_1min",
    "STOP_count_last_5min",
    "STOP_count_last_10min",
]
THRESHOLD = 0.30
TRAIN_SIZE = 246
VAL_SIZE = 82
TEST_SIZE = 83


def reconstruct_events() -> pd.DataFrame:
    """Build the same STOP-event table used by the thesis workflow and retain the first 411 positive-duration events."""
    telemetry = pd.read_csv(DATA_PATH)
    telemetry = telemetry[["timestamp", "status", "downtimeCurrent"]].copy()
    telemetry["timestamp"] = pd.to_datetime(telemetry["timestamp"], errors="coerce")
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
    events = events[events["duration_minutes"] > 0].sort_values("start_time").reset_index(drop=True)
    events = events.iloc[: 246 + 82 + 83].copy()

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
    return events


def make_split(events: pd.DataFrame):
    X = events[FEATURES].copy()
    y_class = (events["duration_minutes"] > 5).astype(int)
    y_reg = events["duration_minutes"]
    train_end = TRAIN_SIZE
    val_end = TRAIN_SIZE + VAL_SIZE

    X_train = X.iloc[:train_end]
    X_val = X.iloc[train_end:val_end]
    X_test = X.iloc[val_end:]

    y_train_class = y_class.iloc[:train_end]
    y_val_class = y_class.iloc[train_end:val_end]
    y_test_class = y_class.iloc[val_end:]

    y_train_reg = np.log1p(y_reg.iloc[:train_end])
    y_val_reg = np.log1p(y_reg.iloc[train_end:val_end])
    y_test_reg = np.log1p(y_reg.iloc[val_end:])

    return (
        X_train,
        X_val,
        X_test,
        y_train_class,
        y_val_class,
        y_test_class,
        y_train_reg,
        y_val_reg,
        y_test_reg,
        y_reg,
    )


def classification_metrics(y_true, probabilities, threshold=THRESHOLD):
    y_pred = (probabilities >= threshold).astype(int)
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, probabilities),
        "pr_auc": average_precision_score(y_true, probabilities),
        "confusion_matrix": pd.crosstab(y_true, y_pred, rownames=["actual"], colnames=["pred"]).to_numpy().tolist(),
        "threshold": float(threshold),
    }


def regression_metrics(y_true, predictions):
    y_true = np.asarray(y_true, dtype=float)
    predictions = np.asarray(predictions, dtype=float)
    return {
        "MAE_minutes": mean_absolute_error(y_true, predictions),
        "RMSE_minutes": np.sqrt(mean_squared_error(y_true, predictions)),
        "MAPE_percent": np.mean(np.abs((y_true - predictions) / y_true)) * 100.0,
        "actual_mean_minutes": float(y_true.mean()),
        "predicted_mean_minutes": float(predictions.mean()),
        "actual_std_minutes": float(y_true.std()),
        "predicted_std_minutes": float(predictions.std()),
        "actual_max_minutes": float(y_true.max()),
        "predicted_max_minutes": float(predictions.max()),
    }


def build_and_save_models():
    events = reconstruct_events()
    (
        X_train,
        X_val,
        X_test,
        y_train_class,
        y_val_class,
        y_test_class,
        y_train_reg,
        y_val_reg,
        y_test_reg,
        y_reg,
    ) = make_split(events)

    scaler = MinMaxScaler(feature_range=(0, 1))
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    classification_pipeline = Pipeline(
        steps=[
            ("scaler", MinMaxScaler(feature_range=(0, 1))),
            ("classifier", LogisticRegression(max_iter=1000)),
        ]
    )
    classification_pipeline.fit(X_train, y_train_class)
    val_prob = classification_pipeline.predict_proba(X_val)[:, 1]
    test_prob = classification_pipeline.predict_proba(X_test)[:, 1]
    val_metrics = classification_metrics(y_val_class, val_prob)
    test_metrics = classification_metrics(y_test_class, test_prob)

    regression_pipeline = Pipeline(
        steps=[
            ("scaler", MinMaxScaler(feature_range=(0, 1))),
            ("regressor", LinearRegression()),
        ]
    )
    regression_pipeline.fit(X_train, y_train_reg)
    val_pred_log = regression_pipeline.predict(X_val)
    test_pred_log = regression_pipeline.predict(X_test)
    val_pred = np.expm1(val_pred_log)
    test_pred = np.expm1(test_pred_log)

    # Evaluate the actual transform-target model that matches the thesis approach.
    val_reg_metrics = regression_metrics(y_reg.iloc[TRAIN_SIZE:TRAIN_SIZE + VAL_SIZE].to_numpy(), val_pred)
    test_reg_metrics = regression_metrics(y_reg.iloc[TRAIN_SIZE + VAL_SIZE:].to_numpy(), test_pred)

    # Use the fitted scaler separately for explicit artifact persistence per requirement.
    scaler_model = MinMaxScaler(feature_range=(0, 1))
    scaler_model.fit(X_train)
    joblib.dump(classification_pipeline, MODEL_DIR / "stop_severity_logistic_model_minmax.joblib")
    joblib.dump(scaler_model, MODEL_DIR / "stop_severity_logistic_model_minmax_scaler.joblib")
    joblib.dump(regression_pipeline, MODEL_DIR / "stop_duration_log_linear_model_minmax.joblib")
    joblib.dump(scaler_model, MODEL_DIR / "stop_duration_log_linear_model_minmax_scaler.joblib")

    classification_metadata = {
        "model_name": "stop_severity_logistic_model_minmax",
        "model_type": "MinMaxScaler -> LogisticRegression",
        "experiment_name": "MinMax scaling experiment",
        "target": "prolonged_STOP",
        "feature_names": FEATURES,
        "dataset_size": int(len(events)),
        "train_size": int(TRAIN_SIZE),
        "validation_size": int(VAL_SIZE),
        "test_size": int(TEST_SIZE),
        "split_method": "chronological",
        "scaler": "MinMaxScaler",
        "scaler_feature_range": [0, 1],
        "scaler_fitted_on": "training data only",
        "target_transformation": "none",
        "classification_threshold": float(THRESHOLD),
        "validation_metrics": {
            "accuracy": float(val_metrics["accuracy"]),
            "balanced_accuracy": float(val_metrics["balanced_accuracy"]),
            "precision": float(val_metrics["precision"]),
            "recall": float(val_metrics["recall"]),
            "f1": float(val_metrics["f1"]),
            "roc_auc": float(val_metrics["roc_auc"]),
            "pr_auc": float(val_metrics["pr_auc"]),
        },
        "test_metrics": {
            "accuracy": float(test_metrics["accuracy"]),
            "balanced_accuracy": float(test_metrics["balanced_accuracy"]),
            "precision": float(test_metrics["precision"]),
            "recall": float(test_metrics["recall"]),
            "f1": float(test_metrics["f1"]),
            "roc_auc": float(test_metrics["roc_auc"]),
            "pr_auc": float(test_metrics["pr_auc"]),
        },
        "confusion_matrix": test_metrics["confusion_matrix"],
        "feature_metadata": {feature: feature for feature in FEATURES},
    }

    regression_metadata = {
        "model_name": "stop_duration_log_linear_model_minmax",
        "model_type": "MinMaxScaler -> Log-Linear Regression",
        "experiment_name": "MinMax scaling experiment",
        "target": "duration_minutes",
        "target_transformation": "log1p",
        "inverse_target_transformation": "expm1",
        "feature_names": FEATURES,
        "dataset_size": int(len(events)),
        "train_size": int(TRAIN_SIZE),
        "validation_size": int(VAL_SIZE),
        "test_size": int(TEST_SIZE),
        "split_method": "chronological",
        "scaler": "MinMaxScaler",
        "scaler_feature_range": [0, 1],
        "scaler_fitted_on": "training data only",
        "target_unit": "minutes",
        "validation_metrics": {
            "MAE_minutes": float(val_reg_metrics["MAE_minutes"]),
            "RMSE_minutes": float(val_reg_metrics["RMSE_minutes"]),
            "MAPE_percent": float(val_reg_metrics["MAPE_percent"]),
        },
        "test_metrics": {
            "MAE_minutes": float(test_reg_metrics["MAE_minutes"]),
            "RMSE_minutes": float(test_reg_metrics["RMSE_minutes"]),
            "MAPE_percent": float(test_reg_metrics["MAPE_percent"]),
        },
        "feature_metadata": {feature: feature for feature in FEATURES},
    }

    with (MODEL_DIR / "stop_severity_logistic_model_minmax_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(classification_metadata, f, indent=2)
    with (MODEL_DIR / "stop_duration_log_linear_model_minmax_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(regression_metadata, f, indent=2)

    comparison = {
        "classification_baseline": {
            "accuracy": 0.7349,
            "balanced_accuracy": 0.7103,
            "precision": 0.4286,
            "recall": 0.6667,
            "f1": 0.5217,
            "roc_auc": 0.7449,
            "pr_auc": 0.4165,
        },
        "classification_minmax": {
            "accuracy": float(test_metrics["accuracy"]),
            "balanced_accuracy": float(test_metrics["balanced_accuracy"]),
            "precision": float(test_metrics["precision"]),
            "recall": float(test_metrics["recall"]),
            "f1": float(test_metrics["f1"]),
            "roc_auc": float(test_metrics["roc_auc"]),
            "pr_auc": float(test_metrics["pr_auc"]),
        },
        "regression_baseline": {
            "MAE_minutes": 37.9024,
            "RMSE_minutes": 223.7904,
            "MAPE_percent": 357.4918,
        },
        "regression_minmax": {
            "MAE_minutes": float(test_reg_metrics["MAE_minutes"]),
            "RMSE_minutes": float(test_reg_metrics["RMSE_minutes"]),
            "MAPE_percent": float(test_reg_metrics["MAPE_percent"]),
        },
    }
    pd.DataFrame([comparison]).to_csv(VIS_DIR / "minmax_scaling_experiment_summary.csv", index=False)


if __name__ == "__main__":
    MODEL_DIR.mkdir(exist_ok=True, parents=True)
    VIS_DIR.mkdir(exist_ok=True, parents=True)
    build_and_save_models()
    print("Min-Max scaling experiment executed and artifacts were saved in models/ and visualizations/.")
