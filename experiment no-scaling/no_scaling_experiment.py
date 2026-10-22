"""Train and evaluate unscaled models on the shared preprocessing cohort."""

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
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
    roc_auc_score,
)

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "mc09_andon.csv"
OUTPUT_DIR = Path(__file__).resolve().parent / "no_scaling_experiment_output"
REFERENCE_METADATA_PATHS = [
    BASE_DIR
    / "experiment min-max"
    / "controlled_preprocessing_comparison"
    / "experiment_metadata.json",
    BASE_DIR
    / "experiment robustscaler"
    / "robust_scaler_experiment_output"
    / "experiment_metadata.json",
    BASE_DIR
    / "experiment yeo-johnson"
    / "yeo_johnson_experiment_output"
    / "experiment_metadata.json",
]
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
    """Build STOP segments and history features using the shared experiment rules."""
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

    start_values = (
        events["start_time"].astype("datetime64[ns]").astype("int64").to_numpy()
    )
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
    train_end = int(eligible_count * TRAIN_FRACTION)
    validation_end = int(
        eligible_count * (TRAIN_FRACTION + VALIDATION_FRACTION)
    )
    if train_end == 0 or validation_end <= train_end or validation_end >= eligible_count:
        raise ValueError(f"Invalid chronological split for {eligible_count} events.")

    events.loc[eligible_indices[:train_end], "split"] = "train"
    events.loc[eligible_indices[train_end:validation_end], "split"] = "validation"
    events.loc[eligible_indices[validation_end:], "split"] = "test"
    return events


def classification_scores(
    actual: pd.Series,
    probability: np.ndarray,
) -> dict[str, object]:
    predicted = (probability >= THRESHOLD).astype("int64")
    return {
        "accuracy": float(accuracy_score(actual, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(actual, predicted)),
        "precision": float(precision_score(actual, predicted, zero_division=0)),
        "recall": float(recall_score(actual, predicted, zero_division=0)),
        "f1": float(f1_score(actual, predicted, zero_division=0)),
        "roc_auc": float(roc_auc_score(actual, probability)),
        "pr_auc": float(average_precision_score(actual, probability)),
        "confusion_matrix": confusion_matrix(
            actual,
            predicted,
            labels=[0, 1],
        ).tolist(),
    }


def regression_scores(
    actual: pd.Series,
    predicted: np.ndarray,
) -> dict[str, float]:
    actual_values = actual.to_numpy(dtype=float)
    return {
        "MAE_minutes": float(mean_absolute_error(actual_values, predicted)),
        "RMSE_minutes": float(
            np.sqrt(mean_squared_error(actual_values, predicted))
        ),
        "MAPE_percent": float(
            np.mean(np.abs((actual_values - predicted) / actual_values)) * 100
        ),
    }


def verify_shared_cohort(
    raw_hash: str,
    events_hash: str,
    split_hash: str,
) -> list[dict[str, object]]:
    checks = []
    for metadata_path in REFERENCE_METADATA_PATHS:
        with metadata_path.open(encoding="utf-8") as file:
            reference = json.load(file)
        matches = {
            "raw_dataset": raw_hash == reference["raw_dataset_sha256"],
            "reconstructed_events": (
                events_hash == reference["reconstructed_events_sha256"]
            ),
            "split_assignments": (
                split_hash == reference["split_assignments_sha256"]
            ),
        }
        if not all(matches.values()):
            raise ValueError(
                f"Shared cohort verification failed for {metadata_path}: {matches}"
            )
        checks.append(
            {
                "reference": str(metadata_path.relative_to(BASE_DIR)),
                "matches": matches,
            }
        )
    return checks


def run_experiment() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    events = reconstruct_events()
    eligible = events.loc[events["duration_minutes"] > 0].copy()
    split_assignments = eligible[["event_id", "start_time", "split"]].copy()

    events_path = OUTPUT_DIR / "reconstructed_events.csv"
    split_path = OUTPUT_DIR / "split_assignments.csv"
    events.to_csv(
        events_path,
        index=False,
        date_format="%Y-%m-%dT%H:%M:%S.%f",
    )
    split_assignments.to_csv(
        split_path,
        index=False,
        date_format="%Y-%m-%dT%H:%M:%S.%f",
    )

    cohort_checks = verify_shared_cohort(
        sha256_file(DATA_PATH),
        sha256_file(events_path),
        sha256_file(split_path),
    )
    splits = {
        name: eligible.loc[eligible["split"].eq(name)].copy()
        for name in ("train", "validation", "test")
    }
    train = splits["train"]
    validation = splits["validation"]
    test = splits["test"]
    if validation["prolonged_STOP"].nunique() < 2 or test["prolonged_STOP"].nunique() < 2:
        raise ValueError("Validation and test must each contain both classes.")

    classifier = LogisticRegression(max_iter=1000)
    classifier.fit(train[FEATURES], train["prolonged_STOP"])
    regressor = LinearRegression()
    regressor.fit(train[FEATURES], np.log1p(train["duration_minutes"]))

    metrics = []
    predictions = split_assignments.copy()
    predictions["actual_duration_minutes"] = eligible["duration_minutes"].to_numpy()
    predictions["actual_prolonged_STOP"] = eligible["prolonged_STOP"].to_numpy()
    predictions["predicted_severity_probability"] = np.nan
    predictions["predicted_prolonged_STOP"] = np.nan
    predictions["predicted_duration_minutes"] = np.nan

    for split_name in ("validation", "test"):
        frame = splits[split_name]
        severity_probability = classifier.predict_proba(frame[FEATURES])[:, 1]
        severity_prediction = (severity_probability >= THRESHOLD).astype("int64")
        duration_prediction = np.expm1(regressor.predict(frame[FEATURES]))
        duration_actual = frame["duration_minutes"]
        severity_actual = frame["prolonged_STOP"]
        classification = classification_scores(
            severity_actual,
            severity_probability,
        )
        regression = regression_scores(duration_actual, duration_prediction)

        metrics.extend(
            [
                {
                    "task": "classification",
                    "split": split_name,
                    **classification,
                },
                {
                    "task": "regression",
                    "split": split_name,
                    **regression,
                },
            ]
        )
        predictions.loc[
            predictions["split"].eq(split_name),
            "predicted_severity_probability",
        ] = severity_probability
        predictions.loc[
            predictions["split"].eq(split_name),
            "predicted_prolonged_STOP",
        ] = severity_prediction
        predictions.loc[
            predictions["split"].eq(split_name),
            "predicted_duration_minutes",
        ] = duration_prediction

    metrics_path = OUTPUT_DIR / "metrics.csv"
    predictions_path = OUTPUT_DIR / "evaluation_predictions.csv"
    pd.DataFrame(metrics).to_csv(metrics_path, index=False)
    predictions.to_csv(
        predictions_path,
        index=False,
        date_format="%Y-%m-%dT%H:%M:%S.%f",
    )
    joblib.dump(classifier, OUTPUT_DIR / "severity_logistic_model_no_scaling.joblib")
    joblib.dump(regressor, OUTPUT_DIR / "duration_log_linear_model_no_scaling.joblib")

    metadata = {
        "experiment_name": "No-scaling preprocessing experiment",
        "status": "Unscaled baseline retrained on the shared 706-event cohort and split.",
        "raw_dataset": str(DATA_PATH.relative_to(BASE_DIR)),
        "raw_dataset_sha256": sha256_file(DATA_PATH),
        "reconstructed_events_sha256": sha256_file(events_path),
        "split_assignments_sha256": sha256_file(split_path),
        "evaluation_predictions_sha256": sha256_file(predictions_path),
        "experiment_script_sha256": sha256_file(Path(__file__).resolve()),
        "shared_cohort_verification": cohort_checks,
        "event_construction": {
            "rule": "Split consecutive STOP rows when adjacent timestamp gap exceeds the threshold.",
            "max_internal_gap_seconds": MAX_INTERNAL_GAP_SECONDS,
            "duration": "Maximum downtimeCurrent within each segment, converted from milliseconds to minutes.",
            "features_engineered_before_positive_duration_filter": True,
        },
        "cohort": {
            "all_reconstructed_stop_segments": int(len(events)),
            "positive_duration_events": int(len(eligible)),
        },
        "split_method": "Chronological, 60% train / 20% validation / 20% test over positive-duration events.",
        "split_sizes": {name: int(len(frame)) for name, frame in splits.items()},
        "split_event_ids": {
            name: frame["event_id"].astype(int).tolist()
            for name, frame in splits.items()
        },
        "preprocessing": {
            "feature_scaling": "none",
            "features": FEATURES,
            "classification_threshold": THRESHOLD,
            "regression_target_transform": "log1p",
            "regression_prediction_inverse_transform": "expm1",
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

    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Reconstructed segments: {len(events)}")
    print(f"Positive-duration events: {len(eligible)}")
    print(f"Split sizes: {metadata['split_sizes']}")
    print(pd.DataFrame(metrics).to_string(index=False))


if __name__ == "__main__":
    run_experiment()
