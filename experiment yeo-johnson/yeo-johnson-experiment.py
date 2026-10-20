"""Run a reproducible Yeo-Johnson preprocessing experiment."""

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
from sklearn.preprocessing import PowerTransformer

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "mc09_andon.csv"
OUTPUT_DIR = Path(__file__).resolve().parent / "yeo_johnson_experiment_output"
MINMAX_DIR = BASE_DIR / "experiment min-max" / "controlled_preprocessing_comparison"
ROBUST_DIR = BASE_DIR / "experiment robustscaler" / "robust_scaler_experiment_output"
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
    """Rebuild STOP segments and engineer history features before filtering."""
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
    positive_indices = events.index[events["duration_minutes"] > 0]
    count = len(positive_indices)
    if count < 3:
        raise ValueError(f"Expected at least 3 positive-duration events, found {count}.")
    train_end = int(count * TRAIN_FRACTION)
    validation_end = int(count * (TRAIN_FRACTION + VALIDATION_FRACTION))
    if train_end == 0 or validation_end <= train_end or validation_end >= count:
        raise ValueError(f"Invalid chronological split boundaries for {count} events.")
    events.loc[positive_indices[:train_end], "split"] = "train"
    events.loc[positive_indices[train_end:validation_end], "split"] = "validation"
    events.loc[positive_indices[validation_end:], "split"] = "test"
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


def load_reference_metadata() -> dict:
    paths = {
        "minmax": MINMAX_DIR / "experiment_metadata.json",
        "robustscaler": ROBUST_DIR / "experiment_metadata.json",
    }
    metadata = {}
    for name, path in paths.items():
        if not path.exists():
            raise FileNotFoundError(
                f"Required {name} comparison metadata is missing: {path}"
            )
        with path.open(encoding="utf-8") as file:
            metadata[name] = json.load(file)
    return metadata


def verify_same_dataset_and_split(
    raw_hash: str,
    events_hash: str,
    split_hash: str,
    split_ids: dict,
    references: dict,
) -> dict:
    checks = {}
    for name, reference in references.items():
        checks[name] = {
            "raw_dataset_sha256_matches": (
                raw_hash == reference["raw_dataset_sha256"]
            ),
            "reconstructed_events_sha256_matches": (
                events_hash == reference["reconstructed_events_sha256"]
            ),
            "split_assignments_sha256_matches": (
                split_hash == reference["split_assignments_sha256"]
            ),
            "split_event_ids_match": split_ids == reference["split_event_ids"],
        }
        if not all(checks[name].values()):
            raise ValueError(
                f"Yeo-Johnson dataset/split differs from {name}: {checks[name]}"
            )
    return checks


def build_and_save_experiment() -> None:
    references = load_reference_metadata()
    OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
    events = reconstruct_events()
    eligible = events.loc[events["duration_minutes"] > 0].copy()
    train = eligible.loc[eligible["split"] == "train"]
    validation = eligible.loc[eligible["split"] == "validation"]
    test = eligible.loc[eligible["split"] == "test"]
    X_train = train[FEATURES]
    X_validation = validation[FEATURES]
    X_test = test[FEATURES]

    transformer = PowerTransformer(method="yeo-johnson", standardize=False)
    X_train_transformed = transformer.fit_transform(X_train)
    X_validation_transformed = transformer.transform(X_validation)
    X_test_transformed = transformer.transform(X_test)

    y_train_class = train["prolonged_STOP"]
    y_validation_class = validation["prolonged_STOP"]
    y_test_class = test["prolonged_STOP"]
    y_train_log_duration = np.log1p(train["duration_minutes"])
    if y_validation_class.nunique() < 2 or y_test_class.nunique() < 2:
        raise ValueError(
            "Both classes must appear in validation and test to calculate ROC-AUC."
        )

    classifier = LogisticRegression(max_iter=1000)
    classifier.fit(X_train_transformed, y_train_class)
    validation_probabilities = classifier.predict_proba(X_validation_transformed)[:, 1]
    test_probabilities = classifier.predict_proba(X_test_transformed)[:, 1]

    regressor = LinearRegression()
    regressor.fit(X_train_transformed, y_train_log_duration)
    validation_duration = np.expm1(regressor.predict(X_validation_transformed))
    test_duration = np.expm1(regressor.predict(X_test_transformed))

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
                validation["duration_minutes"],
                validation_duration,
            ),
            "test": regression_metrics(test["duration_minutes"], test_duration),
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

    split_ids = {
        name: frame["event_id"].astype(int).tolist()
        for name, frame in (
            ("train", train),
            ("validation", validation),
            ("test", test),
        )
    }
    raw_hash = sha256_file(DATA_PATH)
    event_hash = sha256_file(events_path)
    split_hash = sha256_file(split_path)
    split_verification = verify_same_dataset_and_split(
        raw_hash,
        event_hash,
        split_hash,
        split_ids,
        references,
    )

    joblib.dump(transformer, OUTPUT_DIR / "yeo_johnson_transformer.joblib")
    joblib.dump(classifier, OUTPUT_DIR / "severity_logistic_model.joblib")
    joblib.dump(regressor, OUTPUT_DIR / "duration_log_linear_model.joblib")

    metadata = {
        "experiment_name": "Yeo-Johnson preprocessing experiment",
        "status": "PowerTransformer fit on X_train only; standardize=False",
        "raw_dataset": str(DATA_PATH.relative_to(BASE_DIR)),
        "raw_dataset_sha256": raw_hash,
        "reconstructed_events_sha256": event_hash,
        "split_assignments_sha256": split_hash,
        "evaluation_predictions_sha256": sha256_file(predictions_path),
        "experiment_script_sha256": sha256_file(Path(__file__).resolve()),
        "same_dataset_and_split_verification": split_verification,
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
        "split_event_ids": split_ids,
        "transformer": {
            "type": "PowerTransformer",
            "method": "yeo-johnson",
            "standardize": False,
            "fitted_on": "X_train only",
            "n_training_samples": int(len(X_train)),
            "lambdas": dict(zip(FEATURES, transformer.lambdas_.tolist())),
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
    with (OUTPUT_DIR / "experiment_metadata.json").open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(metadata, file, indent=2, allow_nan=False)

    comparison_rows = [
        {"variant": "Yeo-Johnson", "task": task, "split": split, **scores}
        for task, split_metrics in metrics.items()
        for split, scores in split_metrics.items()
    ]
    for reference_name, reference_dir in (
        ("Min-Max", MINMAX_DIR),
        ("RobustScaler", ROBUST_DIR),
    ):
        reference_metrics_path = reference_dir / (
            "metrics_comparison.csv"
            if reference_name == "Min-Max"
            else "preprocessing_comparison.csv"
        )
        if not reference_metrics_path.exists():
            raise FileNotFoundError(
                f"Required {reference_name} comparison metrics are missing: "
                f"{reference_metrics_path}"
            )
        reference_metrics = pd.read_csv(reference_metrics_path)
        if reference_name == "Min-Max":
            reference_metrics = reference_metrics.loc[
                reference_metrics["variant"].isin(["unscaled", "minmax"])
            ]
        elif "variant" in reference_metrics:
            reference_metrics = reference_metrics.loc[
                reference_metrics["variant"].isin(
                    ["unscaled", "minmax", "RobustScaler"]
                )
            ]
        for row in reference_metrics.to_dict(orient="records"):
            variant = row.get("variant", reference_name)
            if variant == "minmax":
                variant = "Min-Max"
            elif variant == "unscaled":
                variant = "No scaling"
            comparison_rows.append(
                {
                    **row,
                    "variant": variant,
                }
            )

    comparison = pd.DataFrame(comparison_rows)
    comparison = comparison.drop_duplicates(
        subset=["variant", "task", "split"],
        keep="last",
    )
    comparison.to_csv(OUTPUT_DIR / "preprocessing_comparison.csv", index=False)
    pd.DataFrame(
        [
            {"task": task, "split": split, **scores}
            for task, split_metrics in metrics.items()
            for split, scores in split_metrics.items()
        ]
    ).to_csv(OUTPUT_DIR / "metrics.csv", index=False)
    print(f"Yeo-Johnson experiment artifacts saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    build_and_save_experiment()