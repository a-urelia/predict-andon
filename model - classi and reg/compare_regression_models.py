"""Compare regression candidates on the saved Yeo-Johnson cohort and split."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.tree import DecisionTreeRegressor

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "yeo_johnson_experiment_output"
FEATURES = [
    "minutes_since_previous_STOP",
    "STOP_count_last_1min",
    "STOP_count_last_5min",
    "STOP_count_last_10min",
]
SPLITS = ("validation", "test")


def regression_metrics(actual: pd.Series, predicted: np.ndarray) -> dict[str, float]:
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


def main() -> None:
    events = pd.read_csv(OUTPUT_DIR / "reconstructed_events.csv")
    eligible = events.loc[events["duration_minutes"] > 0].copy()
    train = eligible.loc[eligible["split"].eq("train")]
    validation = eligible.loc[eligible["split"].eq("validation")]
    test = eligible.loc[eligible["split"].eq("test")]

    split_sizes = {
        "train": len(train),
        "validation": len(validation),
        "test": len(test),
    }
    if split_sizes != {"train": 423, "validation": 141, "test": 142}:
        raise ValueError(f"Unexpected Yeo-Johnson split sizes: {split_sizes}")

    transformer = joblib.load(OUTPUT_DIR / "yeo_johnson_transformer.joblib")
    transformed = {
        name: transformer.transform(frame[FEATURES])
        for name, frame in (
            ("train", train),
            ("validation", validation),
            ("test", test),
        )
    }
    train_target = train["duration_minutes"].to_numpy(dtype=float)

    mean_prediction = float(train_target.mean())
    median_prediction = float(np.median(train_target))
    models = [
        ("Mean Baseline", None, "raw", mean_prediction),
        ("Median Baseline", None, "raw", median_prediction),
        ("Linear Regression", LinearRegression(), "raw", None),
        ("Log-Linear Regression", LinearRegression(), "log1p", None),
        (
            "Decision Tree",
            DecisionTreeRegressor(random_state=42),
            "raw",
            None,
        ),
        (
            "Random Forest",
            RandomForestRegressor(
                n_estimators=100,
                random_state=42,
                n_jobs=-1,
            ),
            "raw",
            None,
        ),
        (
            "Random Forest + Log Target",
            RandomForestRegressor(
                n_estimators=100,
                random_state=42,
                n_jobs=-1,
            ),
            "log1p",
            None,
        ),
    ]

    rows = []
    for model_name, estimator, target_transform, baseline_value in models:
        if estimator is None:
            predictions = {
                split: np.full(len(frame), baseline_value, dtype=float)
                for split, frame in (("validation", validation), ("test", test))
            }
        else:
            fit_target = (
                np.log1p(train_target)
                if target_transform == "log1p"
                else train_target
            )
            estimator.fit(transformed["train"], fit_target)
            predictions = {}
            for split in SPLITS:
                prediction = estimator.predict(transformed[split])
                if target_transform == "log1p":
                    prediction = np.expm1(prediction)
                predictions[split] = prediction

        for split, frame in (("validation", validation), ("test", test)):
            rows.append(
                {
                    "Model": model_name,
                    "Split": split,
                    **regression_metrics(
                        frame["duration_minutes"],
                        predictions[split],
                    ),
                }
            )

    results = pd.DataFrame(rows)
    reference_metrics = pd.read_csv(OUTPUT_DIR / "metrics.csv")
    current_log_linear_test = reference_metrics.loc[
        reference_metrics["task"].eq("regression")
        & reference_metrics["split"].eq("test")
    ].iloc[0]
    comparison_log_linear_test = results.loc[
        results["Model"].eq("Log-Linear Regression")
        & results["Split"].eq("test")
    ].iloc[0]
    for metric, reference_column in (
        ("MAE_minutes", "MAE_minutes"),
        ("RMSE_minutes", "RMSE_minutes"),
        ("MAPE_percent", "MAPE_percent"),
    ):
        if not np.isclose(
            comparison_log_linear_test[metric],
            current_log_linear_test[reference_column],
            rtol=1e-8,
            atol=1e-8,
        ):
            raise ValueError(
                f"Log-Linear Regression {metric} does not match Yeo-Johnson metrics."
            )

    output_path = OUTPUT_DIR / "regression_model_comparison.csv"
    results.to_csv(output_path, index=False, float_format="%.10g")
    print(f"Saved {output_path}")
    print(f"Split sizes: {split_sizes}")
    print(results.loc[results["Split"].eq("test")].to_string(index=False))
    print("Verified: test Log-Linear metrics match the saved Yeo-Johnson experiment.")


if __name__ == "__main__":
    main()
