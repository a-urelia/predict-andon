"""Run a reproducible RobustScaler experiment on the canonical Andon dataset."""

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
from sklearn.preprocessing import RobustScaler

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "mc09_andon.csv"
OUTPUT_DIR = Path(__file__).resolve().parent / "robust_scaler_experiment_output"
MINMAX_METADATA_PATH = (
    BASE_DIR
    / "scaling min-max"
    / "controlled_preprocessing_comparison"
    / "experiment_metadata.json"
)
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
    """Rebuild event segments and engineer features before positive-duration filtering."""
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


def classification_metrics(y_true: pd.Series, probabilities: np.ndarray) -> dict:
    predictions = (probabilities >= THRESHOLD).astype(int)
    confusion = pd.crosstab(
        y_true,
        predictions,
        rownames=["actual"],
        colnames=["predicted"],
    ).reindex(index=[0, 1], columns=[0, 1], fill_value=0)
    return {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, predictions)),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "pr_auc": float(average_precision_score(y_true, probabilities)),
        "confusion_matrix": confusion.to_numpy().tolist(),
    }


def regression_metrics(y_true: pd.Series, predictions: np.ndarray) -> dict:
    actual = y_true.to_numpy(dtype=float)
    return {
        "MAE_minutes": float(mean_absolute_error(actual, predictions)),
        "RMSE_minutes": float(np.sqrt(mean_squared_error(actual, predictions))),
        "MAPE_percent": float(
            np.mean(np.abs((actual - predictions) / actual)) * 100.0
        ),
    }


def build_and_save_experiment() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
    events = reconstruct_events()
    eligible = events.loc[events["duration_minutes"] > 0].copy()
    train = eligible.loc[eligible["split"] == "train"]
    validation = eligible.loc[eligible["split"] == "validation"]
    test = eligible.loc[eligible["split"] == "test"]

    X_train = train[FEATURES]
    X_validation = validation[FEATURES]
    X_test = test[FEATURES]

    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_validation_scaled = scaler.transform(X_validation)
    X_test_scaled = scaler.transform(X_test)

    y_train_class = train["prolonged_STOP"]
    y_validation_class = validation["prolonged_STOP"]
    y_test_class = test["prolonged_STOP"]
    y_train_log_duration = np.log1p(train["duration_minutes"])
    y_validation_duration = validation["duration_minutes"]
    y_test_duration = test["duration_minutes"]

    if y_validation_class.nunique() < 2 or y_test_class.nunique() < 2:
        raise ValueError(
            "Both classes must appear in validation and test to calculate ROC-AUC."
        )

    classifier = LogisticRegression(max_iter=1000)
    classifier.fit(X_train_scaled, y_train_class)
    validation_probabilities = classifier.predict_proba(X_validation_scaled)[:, 1]
    test_probabilities = classifier.predict_proba(X_test_scaled)[:, 1]

    regressor = LinearRegression()
    regressor.fit(X_train_scaled, y_train_log_duration)
    validation_duration = np.expm1(regressor.predict(X_validation_scaled))
    test_duration = np.expm1(regressor.predict(X_test_scaled))

    metrics = {
        "classification": {
            "validation": classification_metrics(
                y_validation_class,
                validation_probabilities,
            ),
            "test": classification_metrics(y_test_class, test_probabilities),
        },
        "regression": {
            "validation": regression_metrics(
                y_validation_duration,
                validation_duration,
            ),
            "test": regression_metrics(y_test_duration, test_duration),
        },
    }

    events_path = OUTPUT_DIR / "reconstructed_events.csv"
    split_path = OUTPUT_DIR / "split_assignments.csv"
    predictions_path = OUTPUT_DIR / "evaluation_predictions.csv"
    events.to_csv(events_path, index=False, date_format="%Y-%m-%dT%H:%M:%S.%f")
    split_assignments = eligible[["event_id", "start_time", "split"]].copy()
    split_assignments.to_csv(
        split_path,
        index=False,
        date_format="%Y-%m-%dT%H:%M:%S.%f",
    )

    predictions = split_assignments.copy()
    predictions["actual_duration_minutes"] = eligible["duration_minutes"].to_numpy()
    predictions["actual_prolonged_STOP"] = eligible["prolonged_STOP"].to_numpy()
    predictions["validation_probability"] = np.nan
    predictions["test_probability"] = np.nan
    predictions["validation_duration_minutes"] = np.nan
    predictions["test_duration_minutes"] = np.nan
    predictions.loc[
        predictions["split"] == "validation",
        "validation_probability",
    ] = validation_probabilities
    predictions.loc[
        predictions["split"] == "test",
        "test_probability",
    ] = test_probabilities
    predictions.loc[
        predictions["split"] == "validation",
        "validation_duration_minutes",
    ] = validation_duration
    predictions.loc[
        predictions["split"] == "test",
        "test_duration_minutes",
    ] = test_duration
    predictions.to_csv(
        predictions_path,
        index=False,
        date_format="%Y-%m-%dT%H:%M:%S.%f",
    )

    joblib.dump(scaler, OUTPUT_DIR / "robust_scaler.joblib")
    joblib.dump(classifier, OUTPUT_DIR / "severity_logistic_model.joblib")
    joblib.dump(regressor, OUTPUT_DIR / "duration_log_linear_model.joblib")

    metadata = {
        "experiment_name": "RobustScaler preprocessing experiment",
        "status": "RobustScaler fit on training features only",
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
            "train": int(len(train)),
            "validation": int(len(validation)),
            "test": int(len(test)),
        },
        "split_event_ids": {
            name: frame["event_id"].astype(int).tolist()
            for name, frame in (
                ("train", train),
                ("validation", validation),
                ("test", test),
            )
        },
        "scaler": {
            "type": "RobustScaler",
            "quantile_range": [25.0, 75.0],
            "with_centering": True,
            "with_scaling": True,
            "unit_variance": False,
            "fitted_on": "X_train only",
            "n_training_samples": int(len(train)),
            "center_median": dict(zip(FEATURES, scaler.center_.tolist())),
            "scale_iqr": dict(zip(FEATURES, scaler.scale_.tolist())),
        },
        "classification": {
            "model": "LogisticRegression(max_iter=1000)",
            "target": "duration_minutes > 5",
            "decision_threshold": THRESHOLD,
            "metrics": metrics["classification"],
        },
        "regression": {
            "model": "LinearRegression",
            "target": "duration_minutes",
            "target_transform": "log1p",
            "prediction_inverse_transform": "expm1",
            "metrics": metrics["regression"],
        },
        "software": {
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
        },
    }

    reference_metrics = None
    if MINMAX_METADATA_PATH.exists():
        with MINMAX_METADATA_PATH.open(encoding="utf-8") as file:
            minmax_metadata = json.load(file)
        reference_metrics = minmax_metadata
        metadata["same_split_verification"] = {
            "reference_metadata": str(MINMAX_METADATA_PATH.relative_to(BASE_DIR)),
            "raw_dataset_sha256_matches": (
                metadata["raw_dataset_sha256"]
                == minmax_metadata["raw_dataset_sha256"]
            ),
            "reconstructed_events_sha256_matches": (
                metadata["reconstructed_events_sha256"]
                == minmax_metadata["reconstructed_events_sha256"]
            ),
            "split_assignments_sha256_matches": (
                metadata["split_assignments_sha256"]
                == minmax_metadata["split_assignments_sha256"]
            ),
            "split_event_ids_match": (
                metadata["split_event_ids"] == minmax_metadata["split_event_ids"]
            ),
        }
        verification = metadata["same_split_verification"]
        if not all(
            verification[key]
            for key in (
                "raw_dataset_sha256_matches",
                "reconstructed_events_sha256_matches",
                "split_assignments_sha256_matches",
                "split_event_ids_match",
            )
        ):
            raise ValueError(
                "The reconstructed dataset or split does not match the Min-Max experiment."
            )

    with (OUTPUT_DIR / "experiment_metadata.json").open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(metadata, file, indent=2, allow_nan=False)

    metrics_rows = [
        {"task": task, "split": split, **scores}
        for task, split_metrics in metrics.items()
        for split, scores in split_metrics.items()
    ]
    pd.DataFrame(metrics_rows).to_csv(
        OUTPUT_DIR / "metrics.csv",
        index=False,
    )

    comparison_rows = [
        {"variant": "RobustScaler", "task": task, "split": split, **scores}
        for task, split_metrics in metrics.items()
        for split, scores in split_metrics.items()
    ]
    if reference_metrics is not None:
        for variant in ("unscaled", "minmax"):
            for task in ("classification", "regression"):
                for split in ("validation", "test"):
                    comparison_rows.append(
                        {
                            "variant": variant,
                            "task": task,
                            "split": split,
                            **reference_metrics[task]["metrics"][variant][split],
                        }
                    )
    pd.DataFrame(comparison_rows).to_csv(
        OUTPUT_DIR / "preprocessing_comparison.csv",
        index=False,
    )
    print(f"RobustScaler experiment artifacts saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    build_and_save_experiment()