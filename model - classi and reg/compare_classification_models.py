"""Compare severity classifiers on the saved Yeo-Johnson cohort and split."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.tree import DecisionTreeClassifier

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "yeo_johnson_experiment_output"
FEATURES = [
    "minutes_since_previous_STOP",
    "STOP_count_last_1min",
    "STOP_count_last_5min",
    "STOP_count_last_10min",
]
SPLITS = ("validation", "test")
THRESHOLD = 0.30


def classification_metrics(
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
        "confusion_matrix": json.dumps(
            confusion_matrix(actual, predicted, labels=[0, 1]).tolist()
        ),
    }


def positive_class_probability(model, features: np.ndarray) -> np.ndarray:
    positive_index = list(model.classes_).index(1)
    return model.predict_proba(features)[:, positive_index]


def main() -> None:
    events = pd.read_csv(OUTPUT_DIR / "reconstructed_events.csv")
    eligible = events.loc[events["duration_minutes"] > 0].copy()
    splits = {
        name: eligible.loc[eligible["split"].eq(name)]
        for name in ("train", "validation", "test")
    }
    split_sizes = {name: len(frame) for name, frame in splits.items()}
    if split_sizes != {"train": 423, "validation": 141, "test": 142}:
        raise ValueError(f"Unexpected Yeo-Johnson split sizes: {split_sizes}")

    transformer = joblib.load(OUTPUT_DIR / "yeo_johnson_transformer.joblib")
    transformed = {
        name: transformer.transform(frame[FEATURES])
        for name, frame in splits.items()
    }
    models = [
        ("Majority Class Baseline", DummyClassifier(strategy="most_frequent")),
        ("Logistic Regression", LogisticRegression(max_iter=1000)),
        ("Decision Tree", DecisionTreeClassifier(random_state=42)),
        (
            "Random Forest",
            RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                n_jobs=-1,
            ),
        ),
    ]

    rows = []
    for model_name, model in models:
        model.fit(transformed["train"], splits["train"]["prolonged_STOP"])
        for split in SPLITS:
            probability = positive_class_probability(model, transformed[split])
            rows.append(
                {
                    "Model": model_name,
                    "Split": split,
                    **classification_metrics(
                        splits[split]["prolonged_STOP"],
                        probability,
                    ),
                }
            )

    results = pd.DataFrame(rows)
    reference_metrics = pd.read_csv(OUTPUT_DIR / "metrics.csv")
    for split in SPLITS:
        reference = reference_metrics.loc[
            reference_metrics["task"].eq("classification")
            & reference_metrics["split"].eq(split)
        ].iloc[0]
        logistic = results.loc[
            results["Model"].eq("Logistic Regression")
            & results["Split"].eq(split)
        ].iloc[0]
        for metric in (
            "accuracy",
            "balanced_accuracy",
            "precision",
            "recall",
            "f1",
            "roc_auc",
            "pr_auc",
        ):
            if not np.isclose(
                logistic[metric],
                reference[metric],
                rtol=1e-8,
                atol=1e-8,
            ):
                raise ValueError(
                    f"Logistic Regression {split} {metric} does not match "
                    "the saved Yeo-Johnson metrics."
                )
        if json.loads(logistic["confusion_matrix"]) != json.loads(
            reference["confusion_matrix"]
        ):
            raise ValueError(
                f"Logistic Regression {split} confusion matrix does not match "
                "the saved Yeo-Johnson metrics."
            )

    output_path = OUTPUT_DIR / "classification_model_comparison.csv"
    results.to_csv(output_path, index=False, float_format="%.10g")
    print(f"Saved {output_path}")
    print(f"Split sizes: {split_sizes}")
    print(
        "Class counts by split:\n"
        + eligible.groupby("split")["prolonged_STOP"]
        .value_counts()
        .unstack(fill_value=0)
        .rename(columns={0: "normal", 1: "prolonged"})
        .to_string()
    )
    print(results.to_string(index=False))
    print("Verified: Logistic Regression matches saved Yeo-Johnson metrics.")


if __name__ == "__main__":
    main()
