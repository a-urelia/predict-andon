"""Run a reproducible, paired baseline vs Min-Max preprocessing experiment."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
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

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "mc09_andon.csv"
OUTPUT_DIR = Path(__file__).resolve().parent / "controlled_preprocessing_comparison"
FEATURES = [
    "minutes_since_previous_STOP",
    "STOP_count_last_1min",
    "STOP_count_last_5min",
    "STOP_count_last_10min",
]
THRESHOLD = 0.30
MAX_INTERNAL_GAP_SECONDS = 300
TRAIN_FRACTION = 0.60
VALIDATION_FRACTION = 0.20


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def reconstruct_events() -> pd.DataFrame:
    """Reconstruct STOP segments, then engineer history features before filtering targets."""
    telemetry = pd.read_csv(
        DATA_PATH,
        usecols=["timestamp", "status", "downtimeCurrent"],
    )
    telemetry["timestamp"] = pd.to_datetime(
        telemetry["timestamp"],
        errors="coerce",
    )
    telemetry["status"] = (
        telemetry["status"].astype("string").str.strip().str.upper()
    )
    telemetry["downtimeCurrent"] = pd.to_numeric(
        telemetry["downtimeCurrent"],
        errors="coerce",
    ).fillna(0)
    telemetry = telemetry.dropna(subset=["timestamp"]).sort_values(
        "timestamp",
        kind="quicksort",
    )

    stop_mask = telemetry["status"].eq("STOP")
    telemetry["_source_stop_group"] = (~stop_mask).astype("int64").cumsum()
    stop_rows = telemetry.loc[stop_mask]

    records = []
    for source_group, group in stop_rows.groupby("_source_stop_group", sort=False):
        group = group.sort_values("timestamp", kind="stable")
        segment_ids = (
            group["timestamp"]
            .diff()
            .dt.total_seconds()
            .gt(MAX_INTERNAL_GAP_SECONDS)
            .cumsum()
        )
        for segment_id, segment in group.groupby(segment_ids, sort=False):
            records.append(
                {
                    "source_stop_group": int(source_group),
                    "segment_in_source_group": int(segment_id) + 1,
                    "start_time": segment["timestamp"].iloc[0],
                    "duration_minutes": float(
                        segment["downtimeCurrent"].max() / 60_000
                    ),
                }
            )

    events = pd.DataFrame.from_records(records)
    events = events.sort_values(
        ["start_time", "source_stop_group", "segment_in_source_group"],
        kind="stable",
    ).reset_index(drop=True)
    events["event_id"] = np.arange(1, len(events) + 1, dtype=np.int64)

    start_values = events["start_time"].astype("int64").to_numpy()
    events["minutes_since_previous_STOP"] = (
        events["start_time"].diff().dt.total_seconds().div(60).fillna(0)
    )
    for window, column in [
        (1, "STOP_count_last_1min"),
        (5, "STOP_count_last_5min"),
        (10, "STOP_count_last_10min"),
    ]:
        events[column] = [
            index
            - np.searchsorted(
                start_values[:index],
                start_values[index] - window * 60_000_000_000,
                side="right",
            )
            for index in range(len(start_values))
        ]

    events["prolonged_STOP"] = (events["duration_minutes"] > 5).astype("int64")
    events["split"] = pd.NA
    eligible_indices = events.index[events["duration_minutes"] > 0]
    eligible_count = len(eligible_indices)
    if eligible_count < 3:
        raise ValueError(
            f"Expected at least 3 positive-duration events, found {eligible_count}."
        )

    train_end = int(eligible_count * TRAIN_FRACTION)
    validation_end = int(
        eligible_count * (TRAIN_FRACTION + VALIDATION_FRACTION)
    )
    if train_end == 0 or validation_end <= train_end or validation_end >= eligible_count:
        raise ValueError(
            f"Invalid chronological split boundaries for {eligible_count} events."
        )
    events.loc[eligible_indices[:train_end], "split"] = "train"
    events.loc[eligible_indices[train_end:validation_end], "split"] = "validation"
    events.loc[eligible_indices[validation_end:], "split"] = "test"
    return events


def classification_metrics(
    y_true: pd.Series,
    probabilities: np.ndarray,
) -> dict[str, float | list[list[int]]]:
    y_pred = (probabilities >= THRESHOLD).astype(int)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "pr_auc": float(average_precision_score(y_true, probabilities)),
        "confusion_matrix": pd.crosstab(
            y_true,
            y_pred,
            rownames=["actual"],
            colnames=["predicted"],
        ).reindex(index=[0, 1], columns=[0, 1], fill_value=0).to_numpy().tolist(),
    }


def regression_metrics(
    y_true: pd.Series,
    predictions: np.ndarray,
) -> dict[str, float]:
    actual = y_true.to_numpy(dtype=float)
    return {
        "MAE_minutes": float(mean_absolute_error(actual, predictions)),
        "RMSE_minutes": float(np.sqrt(mean_squared_error(actual, predictions))),
        "MAPE_percent": float(
            np.mean(np.abs((actual - predictions) / actual)) * 100.0
        ),
    }


def make_model(scale_features: bool, task: str) -> Pipeline:
    if task == "classification":
        estimator = LogisticRegression(max_iter=1000)
    elif task == "regression":
        estimator = LinearRegression()
    else:
        raise ValueError(f"Unsupported task: {task}")

    steps = []
    if scale_features:
        steps.append(("scaler", MinMaxScaler(feature_range=(0, 1))))
    steps.append(("model", estimator))
    return Pipeline(steps)


def build_and_save_experiment() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
    events = reconstruct_events()
    eligible = events.loc[events["duration_minutes"] > 0].copy()
    split_assignments = eligible[["event_id", "start_time", "split"]].copy()

    events_path = OUTPUT_DIR / "reconstructed_events.csv"
    split_path = OUTPUT_DIR / "split_assignments.csv"
    events.to_csv(events_path, index=False, date_format="%Y-%m-%dT%H:%M:%S.%f")
    split_assignments.to_csv(
        split_path,
        index=False,
        date_format="%Y-%m-%dT%H:%M:%S.%f",
    )

    train = eligible.loc[eligible["split"] == "train"]
    validation = eligible.loc[eligible["split"] == "validation"]
    test = eligible.loc[eligible["split"] == "test"]
    X_train, X_validation, X_test = (
        frame[FEATURES] for frame in (train, validation, test)
    )
    y_train_class = train["prolonged_STOP"]
    y_validation_class = validation["prolonged_STOP"]
    y_test_class = test["prolonged_STOP"]
    y_train_log = np.log1p(train["duration_minutes"])
    y_validation_duration = validation["duration_minutes"]
    y_test_duration = test["duration_minutes"]

    if y_test_class.nunique() < 2 or y_validation_class.nunique() < 2:
        raise ValueError(
            "Both classes must appear in validation and test to calculate ROC-AUC."
        )

    metrics = {}
    predictions = split_assignments.copy()
    predictions["actual_duration_minutes"] = eligible["duration_minutes"].to_numpy()
    predictions["actual_prolonged_STOP"] = eligible["prolonged_STOP"].to_numpy()

    for variant, scale_features in [("unscaled", False), ("minmax", True)]:
        classifier = make_model(scale_features, "classification")
        classifier.fit(X_train, y_train_class)
        validation_probability = classifier.predict_proba(X_validation)[:, 1]
        test_probability = classifier.predict_proba(X_test)[:, 1]

        regressor = make_model(scale_features, "regression")
        regressor.fit(X_train, y_train_log)
        validation_duration = np.expm1(regressor.predict(X_validation))
        test_duration = np.expm1(regressor.predict(X_test))

        metrics[variant] = {
            "classification": {
                "validation": classification_metrics(
                    y_validation_class,
                    validation_probability,
                ),
                "test": classification_metrics(y_test_class, test_probability),
            },
            "regression": {
                "validation": regression_metrics(
                    y_validation_duration,
                    validation_duration,
                ),
                "test": regression_metrics(y_test_duration, test_duration),
            },
        }
        predictions[f"{variant}_validation_probability"] = np.nan
        predictions[f"{variant}_test_probability"] = np.nan
        predictions[f"{variant}_validation_duration_minutes"] = np.nan
        predictions[f"{variant}_test_duration_minutes"] = np.nan
        predictions.loc[
            predictions["split"] == "validation",
            f"{variant}_validation_probability",
        ] = validation_probability
        predictions.loc[
            predictions["split"] == "test",
            f"{variant}_test_probability",
        ] = test_probability
        predictions.loc[
            predictions["split"] == "validation",
            f"{variant}_validation_duration_minutes",
        ] = validation_duration
        predictions.loc[
            predictions["split"] == "test",
            f"{variant}_test_duration_minutes",
        ] = test_duration

        joblib.dump(classifier, OUTPUT_DIR / f"{variant}_severity_model.joblib")
        joblib.dump(regressor, OUTPUT_DIR / f"{variant}_duration_model.joblib")

    predictions_path = OUTPUT_DIR / "evaluation_predictions.csv"
    predictions.to_csv(
        predictions_path,
        index=False,
        date_format="%Y-%m-%dT%H:%M:%S.%f",
    )

    scaler = make_model(True, "classification").named_steps["scaler"]
    scaler.fit(X_train)
    split_ids = {
        name: frame["event_id"].astype(int).tolist()
        for name, frame in (
            ("train", train),
            ("validation", validation),
            ("test", test),
        )
    }
    metadata = {
        "experiment_name": "Controlled unscaled vs Min-Max preprocessing comparison",
        "status": "new paired rerun; not a reproduction of legacy baseline membership",
        "raw_dataset": str(DATA_PATH.relative_to(BASE_DIR)),
        "raw_dataset_sha256": sha256_file(DATA_PATH),
        "reconstructed_events_sha256": sha256_file(events_path),
        "split_assignments_sha256": sha256_file(split_path),
        "evaluation_predictions_sha256": sha256_file(predictions_path),
        "experiment_script_sha256": sha256_file(Path(__file__).resolve()),
        "event_construction": {
            "rule": "Split consecutive STOP rows when adjacent timestamp gap exceeds the threshold.",
            "max_internal_gap_seconds": MAX_INTERNAL_GAP_SECONDS,
            "duration": "Maximum downtimeCurrent within each segment, converted from milliseconds to minutes.",
            "features_engineered_before_positive_duration_filter": True,
            "features": FEATURES,
        },
        "cohort": {
            "all_reconstructed_stop_segments": int(len(events)),
            "positive_duration_events": int(len(eligible)),
        },
        "split_method": "Chronological over positive-duration events in event start-time order.",
        "split_sizes": {
            name: int(len(frame))
            for name, frame in (
                ("train", train),
                ("validation", validation),
                ("test", test),
            )
        },
        "split_event_ids": split_ids,
        "scaler": {
            "type": "MinMaxScaler",
            "feature_range": [0, 1],
            "fitted_on": "training features only",
            "training_data_min": dict(zip(FEATURES, scaler.data_min_.tolist())),
            "training_data_max": dict(zip(FEATURES, scaler.data_max_.tolist())),
        },
        "classification": {
            "target": "duration_minutes > 5",
            "decision_threshold": THRESHOLD,
            "metrics": {
                variant: values["classification"]
                for variant, values in metrics.items()
            },
        },
        "regression": {
            "target": "duration_minutes",
            "target_transform": "log1p",
            "prediction_inverse_transform": "expm1",
            "metrics": {
                variant: values["regression"]
                for variant, values in metrics.items()
            },
        },
        "software": {
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
        },
    }
    with (OUTPUT_DIR / "experiment_metadata.json").open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(metadata, file, indent=2, allow_nan=False)

    pd.DataFrame(
        [
            {
                "variant": variant,
                "task": task,
                "split": split,
                **scores,
            }
            for variant, tasks in metrics.items()
            for task, splits in tasks.items()
            for split, scores in splits.items()
        ]
    ).to_csv(OUTPUT_DIR / "metrics_comparison.csv", index=False)
    print(f"Paired experiment artifacts saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    build_and_save_experiment()
