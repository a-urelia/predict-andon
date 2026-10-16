"""Generate thesis-ready figures and summary tables for the MC09 Andon models."""

import argparse
import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "mc09_andon.csv"
MODEL_DIR = BASE_DIR / "models"
FEATURES = [
    "minutes_since_previous_STOP",
    "STOP_count_last_1min",
    "STOP_count_last_5min",
    "STOP_count_last_10min",
]

INK = "#17352d"
GREEN = "#3c8c62"
MINT = "#dff0e5"
BLUE = "#4d77a8"
PALE_BLUE = "#e7eef7"
ORANGE = "#dd8b32"
PALE_ORANGE = "#fbefd9"
GRAY = "#65756d"
LIGHT_GRAY = "#edf1ee"


def load_events():
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
    telemetry = telemetry.dropna(subset=["timestamp"]).sort_values("timestamp")

    stop_mask = telemetry["status"].eq("STOP")
    stop_rows = telemetry.loc[stop_mask].copy()
    stop_rows["_event_group"] = (
        (~stop_mask).astype("int64").cumsum().loc[stop_mask]
    )
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
        events["duration_minutes"],
        unit="m",
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
            index
            - np.searchsorted(
                start_values[:index],
                start_values[index] - window * 60_000_000_000,
                side="right",
            )
            for index in range(len(start_values))
        ]
    return events


def load_metadata():
    with (MODEL_DIR / "stop_severity_logistic_model_metadata.json").open(
        encoding="utf-8"
    ) as file:
        severity = json.load(file)
    with (MODEL_DIR / "stop_duration_log_linear_model_metadata.json").open(
        encoding="utf-8"
    ) as file:
        duration = json.load(file)
    return severity, duration


def add_box(ax, xy, width, height, text, facecolor, fontsize=11):
    box = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.025,rounding_size=0.025",
        linewidth=1.2,
        edgecolor=GREEN,
        facecolor=facecolor,
    )
    ax.add_patch(box)
    ax.text(
        xy[0] + width / 2,
        xy[1] + height / 2,
        text,
        ha="center",
        va="center",
        color=INK,
        fontsize=fontsize,
        linespacing=1.35,
    )


def add_arrow(ax, start, end, color=GRAY, linewidth=1.5):
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=14,
            linewidth=linewidth,
            color=color,
        )
    )


def save_figure(fig, output_dir, filename):
    fig.savefig(output_dir / filename, dpi=180, bbox_inches="tight")
    plt.close(fig)


def event_construction_figure(output_dir):
    fig, ax = plt.subplots(figsize=(13, 4.3))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title(
        "From MC09 telemetry rows to STOP events",
        loc="left",
        color=INK,
        fontsize=17,
        fontweight="bold",
        pad=15,
    )
    nodes = [
        (0.02, 0.34, 0.19, 0.42, "Raw telemetry\n(timestamp, status,\ndowntimeCurrent)", PALE_BLUE),
        (0.27, 0.34, 0.19, 0.42, "Keep rows where\nstatus = STOP", MINT),
        (0.52, 0.34, 0.19, 0.42, "Group consecutive\nSTOP rows until\nstatus changes", PALE_BLUE),
        (0.77, 0.34, 0.21, 0.42, "Event duration = max\n(downtimeCurrent)\n÷ 60,000 → minutes", PALE_ORANGE),
    ]
    for x, y, width, height, text, color in nodes:
        add_box(ax, (x, y), width, height, text, color)
    for left, right in [(0.21, 0.27), (0.46, 0.52), (0.71, 0.77)]:
        add_arrow(ax, (left + 0.008, 0.55), (right - 0.008, 0.55))
    ax.text(
        0.02,
        0.13,
        "Event start = timestamp of the first STOP row. The plotted reconstruction follows "
        "the current dashboard logic; consecutive STOP runs are not split by elapsed-time gaps.",
        color=GRAY,
        fontsize=9.5,
        va="center",
    )
    save_figure(fig, output_dir, "01_event_construction.png")


def duration_distribution_figure(events, output_dir):
    durations = events.loc[
        events["duration_minutes"] > 0, "duration_minutes"
    ].to_numpy()
    if durations.size == 0:
        raise ValueError("No positive STOP-event durations were reconstructed.")
    q95 = float(np.quantile(durations, 0.95))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    bins = np.geomspace(max(float(durations.min()), 0.001), durations.max(), 40)

    axes[0].hist(durations, bins=bins, color=GREEN, edgecolor="white")
    axes[0].set_xscale("log")
    axes[0].set_title("Full range (log-scaled x-axis)", color=INK, fontweight="bold")
    axes[0].set_xlabel("STOP duration (minutes)")
    axes[0].set_ylabel("Event count")
    axes[0].axvline(q95, color=ORANGE, linestyle="--", linewidth=1.5)
    axes[0].text(
        q95,
        axes[0].get_ylim()[1] * 0.9,
        f"  95th percentile\n  {q95:.1f} min",
        color=INK,
        fontsize=9,
    )

    zoomed = durations[durations <= q95]
    axes[1].hist(
        zoomed,
        bins=30,
        color=BLUE,
        edgecolor="white",
        range=(0, q95),
    )
    axes[1].set_title(f"Zoom: 0–95th percentile ({q95:.1f} min)", color=INK, fontweight="bold")
    axes[1].set_xlabel("STOP duration (minutes)")
    axes[1].set_ylabel("Event count")
    for ax in axes:
        ax.grid(axis="y", color=LIGHT_GRAY, linewidth=0.8)
        ax.set_axisbelow(True)
    fig.suptitle(
        f"Positive STOP-event duration distribution · {len(durations):,} events",
        x=0.04,
        ha="left",
        color=INK,
        fontsize=15,
        fontweight="bold",
    )
    fig.text(
        0.04,
        0.01,
        f"Zero-duration reconstructed groups: {(events['duration_minutes'] <= 0).sum():,} "
        " (shown separately, not included in the histogram).",
        color=GRAY,
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.92))
    save_figure(fig, output_dir, "02_duration_distribution.png")
    return q95


def prediction_problem_figure(output_dir):
    fig, ax = plt.subplots(figsize=(12, 5.4))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title(
        "Prediction problem: shared temporal inputs, two targets",
        loc="left",
        color=INK,
        fontsize=17,
        fontweight="bold",
        pad=15,
    )
    add_box(
        ax,
        (0.03, 0.34),
        0.25,
        0.40,
        "Temporal features\n• Minutes since previous STOP\n• STOP counts in prior 1, 5, 10 min",
        PALE_BLUE,
        fontsize=10,
    )
    add_box(
        ax,
        (0.39, 0.61),
        0.25,
        0.25,
        "Logistic regression\nP(prolonged STOP)",
        MINT,
    )
    add_box(
        ax,
        (0.39, 0.18),
        0.25,
        0.25,
        "Log-linear regression\npredict log1p(duration)",
        PALE_ORANGE,
    )
    add_box(
        ax,
        (0.75, 0.61),
        0.22,
        0.25,
        "Classify prolonged\nif probability ≥ 0.30",
        MINT,
    )
    add_box(
        ax,
        (0.75, 0.18),
        0.22,
        0.25,
        "Estimated duration\nexpm1(prediction) minutes",
        PALE_ORANGE,
    )
    add_arrow(ax, (0.28, 0.59), (0.39, 0.72))
    add_arrow(ax, (0.28, 0.48), (0.39, 0.31))
    add_arrow(ax, (0.64, 0.735), (0.75, 0.735))
    add_arrow(ax, (0.64, 0.305), (0.75, 0.305))
    ax.text(
        0.04,
        0.07,
        "Severity target: duration > 5 minutes. Duration target is continuous; classification and regression are separate model outputs.",
        color=GRAY,
        fontsize=9.5,
    )
    save_figure(fig, output_dir, "03_prediction_problem.png")


def temporal_feature_figure(output_dir):
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title(
        "Temporal feature flow for each STOP event",
        loc="left",
        color=INK,
        fontsize=17,
        fontweight="bold",
        pad=15,
    )
    add_box(ax, (0.03, 0.40), 0.17, 0.27, "Ordered STOP\nevent start times", PALE_BLUE)
    add_box(ax, (0.27, 0.68), 0.28, 0.21, "Gap feature\ncurrent start − previous start\n(minutes)", MINT, fontsize=10)
    add_box(ax, (0.27, 0.38), 0.28, 0.21, "1-minute frequency\nprior STOP starts in\nrolling 1-minute window", PALE_ORANGE, fontsize=10)
    add_box(ax, (0.27, 0.08), 0.28, 0.21, "5- and 10-minute frequency\nprior STOP starts in\nrolling windows", PALE_ORANGE, fontsize=10)
    add_box(ax, (0.68, 0.39), 0.28, 0.28, "Model feature vector\n[gap, count₁, count₅, count₁₀]", PALE_BLUE)
    for end_y in [0.785, 0.485, 0.185]:
        add_arrow(ax, (0.20, 0.535), (0.27, end_y), color=GRAY)
        add_arrow(ax, (0.55, end_y), (0.68, 0.53), color=GRAY)
    ax.text(
        0.03,
        0.015,
        "Rolling counts use earlier STOP event starts only (the current event is not counted as its own history).",
        color=GRAY,
        fontsize=9.5,
    )
    save_figure(fig, output_dir, "04_temporal_feature_flow.png")


def split_timeline_figure(severity_metadata, duration_metadata, output_dir):
    split_names = ["Training", "Validation", "Test"]
    period_keys = ["training_period", "validation_period", "test_period"]
    sizes = [
        duration_metadata["train_size"],
        duration_metadata["validation_size"],
        duration_metadata["test_size"],
    ]
    colors = [GREEN, BLUE, ORANGE]
    periods = [
        (
            pd.Timestamp(severity_metadata[key]["start"]),
            pd.Timestamp(severity_metadata[key]["end"]),
        )
        for key in period_keys
    ]
    origin = min(start for start, _ in periods)
    fig, ax = plt.subplots(figsize=(12, 4.3))
    for row, (name, (start, end), size, color) in enumerate(
        zip(split_names, periods, sizes, colors)
    ):
        left = (start - origin).total_seconds() / 86400
        width = (end - start).total_seconds() / 86400
        ax.barh(row, width, left=left, height=0.52, color=color, alpha=0.9)
        ax.text(
            left + width / 2,
            row,
            f"{name} · n={size}",
            ha="center",
            va="center",
            color="white",
            fontsize=10,
            fontweight="bold",
        )
    ticks = ax.get_xticks()
    ax.set_xticks(ticks)
    ax.set_xticklabels(
        [(origin + pd.to_timedelta(tick, unit="D")).strftime("%b %d") for tick in ticks]
    )
    ax.set_yticks([])
    ax.invert_yaxis()
    ax.set_xlabel("2026 event time")
    ax.set_title(
        "Chronological train / validation / test periods",
        loc="left",
        color=INK,
        fontsize=16,
        fontweight="bold",
    )
    ax.grid(axis="x", color=LIGHT_GRAY)
    ax.set_axisbelow(True)
    fig.text(
        0.05,
        0.02,
        "Periods from severity-model metadata; sample counts are from duration-model metadata.",
        color=GRAY,
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.06, 1, 0.96))
    save_figure(fig, output_dir, "05_train_validation_test_timeline.png")


def confusion_matrix_figure(metadata, output_dir):
    values = np.asarray(metadata["test_confusion_matrix"])
    fig, ax = plt.subplots(figsize=(6, 5))
    image = ax.imshow(values, cmap="Greens", vmin=0)
    ax.set_xticks([0, 1], labels=["Normal", "Prolonged"])
    ax.set_yticks([0, 1], labels=["Normal", "Prolonged"])
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("Actual class")
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position("top")
    for row in range(2):
        for col in range(2):
            ax.text(
                col,
                row,
                f"{values[row, col]}\n{values[row, col] / values.sum():.1%}",
                ha="center",
                va="center",
                color="white" if values[row, col] > values.max() * 0.55 else INK,
                fontsize=13,
                fontweight="bold",
            )
    fig.colorbar(image, ax=ax, label="Test events")
    ax.set_title(
        "Severity classifier test confusion matrix\nThreshold = 0.30",
        color=INK,
        fontsize=14,
        fontweight="bold",
        pad=18,
    )
    fig.tight_layout()
    save_figure(fig, output_dir, "06_confusion_matrix.png")


def model_comparison_figure(output_dir):
    comparison = pd.DataFrame(
        {
            "Model": [
                "Mean Baseline",
                "Median Baseline",
                "Linear Regression",
                "Decision Tree",
                "Random Forest",
                "Log-Linear Regression",
                "Random Forest + Log Target",
            ],
            "MAE_minutes": [37.07, 22.80, 25.81, 26.98, 26.74, 22.74, 23.03],
            "RMSE_minutes": [165.69, 167.09, 166.21, 169.59, 167.64, 166.94, 167.09],
            "MAPE_percent": [3200.94, 232.07, 1097.00, 1316.91, 1056.28, 264.69, 264.95],
        }
    )
    comparison.to_csv(output_dir / "model_comparison.csv", index=False)
    ordered = comparison.sort_values("MAE_minutes", ascending=True)
    fig, ax = plt.subplots(figsize=(10, 5.3))
    bars = ax.barh(ordered["Model"], ordered["MAE_minutes"], color=BLUE)
    best_index = ordered["MAE_minutes"].idxmin()
    bars[list(ordered.index).index(best_index)].set_color(GREEN)
    for bar, value in zip(bars, ordered["MAE_minutes"]):
        ax.text(
            value + 0.35,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.2f}",
            va="center",
            color=INK,
            fontsize=9,
        )
    ax.set_xlim(0, float(ordered["MAE_minutes"].max()) * 1.2)
    ax.set_xlabel("Mean absolute error (minutes) · lower is better")
    ax.set_title(
        "Duration-model comparison on validation data",
        loc="left",
        color=INK,
        fontsize=16,
        fontweight="bold",
    )
    ax.grid(axis="x", color=LIGHT_GRAY)
    ax.set_axisbelow(True)
    fig.text(
        0.05,
        0.015,
        "Comparison values are recorded in the existing Model Evaluation dashboard; they are validation MAE.",
        color=GRAY,
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.96))
    save_figure(fig, output_dir, "07_model_comparison.png")
    return comparison


def predict_durations(events, output_dir):
    features = events[FEATURES]
    model = joblib.load(MODEL_DIR / "stop_duration_log_linear_model.joblib")
    model_features = list(getattr(model, "feature_names_in_", []))
    if model_features and model_features != FEATURES:
        raise ValueError(
            f"Duration model expects {model_features}, but reconstructed features are {FEATURES}."
        )
    events = events.copy()
    events["predicted_duration_minutes"] = np.maximum(
        np.expm1(model.predict(features)),
        0,
    )
    observed = events.loc[events["duration_minutes"] > 0].copy()
    observed["prolonged_STOP"] = observed["duration_minutes"] > 5
    observed.to_csv(output_dir / "duration_predictions.csv", index=False)
    return observed


def actual_predicted_figures(observed, q95, output_dir):
    actual = observed["duration_minutes"].to_numpy()
    predicted = observed["predicted_duration_minutes"].to_numpy()
    count = len(actual)
    if count == 0:
        raise ValueError("No positive-duration events are available for comparison.")

    actual_log = np.log1p(actual)
    predicted_log = np.log1p(predicted)
    upper = float(max(actual_log.max(), predicted_log.max()))
    fig, ax = plt.subplots(figsize=(7.6, 6.4))
    ax.scatter(
        actual_log,
        predicted_log,
        s=22,
        alpha=0.48,
        color=BLUE,
        edgecolors="none",
    )
    ax.plot([0, upper], [0, upper], linestyle="--", color=ORANGE, linewidth=1.5)
    ax.set_xlim(0, upper * 1.03)
    ax.set_ylim(0, upper * 1.03)
    ax.set_xlabel("Actual duration · log1p(minutes)")
    ax.set_ylabel("Predicted duration · log1p(minutes)")
    ax.set_title(
        "Actual vs predicted duration · full positive-duration range",
        loc="left",
        color=INK,
        fontsize=14,
        fontweight="bold",
    )
    ax.grid(color=LIGHT_GRAY)
    ax.set_axisbelow(True)
    fig.text(
        0.06,
        0.015,
        f"{count:,} reconstructed positive-duration events. Descriptive model outputs, not held-out test predictions.",
        color=GRAY,
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    save_figure(fig, output_dir, "08_actual_vs_predicted.png")

    zoom_mask = (actual <= q95) & (predicted <= q95)
    zoom_actual = actual[zoom_mask]
    zoom_predicted = predicted[zoom_mask]
    fig, ax = plt.subplots(figsize=(7.6, 6.4))
    ax.scatter(
        zoom_actual,
        zoom_predicted,
        s=24,
        alpha=0.5,
        color=GREEN,
        edgecolors="none",
    )
    ax.plot([0, q95], [0, q95], linestyle="--", color=ORANGE, linewidth=1.5)
    ax.set_xlim(0, q95)
    ax.set_ylim(0, q95)
    ax.set_xlabel("Actual duration (minutes)")
    ax.set_ylabel("Predicted duration (minutes)")
    ax.set_title(
        f"Actual vs predicted duration · zoom to {q95:.1f} minutes",
        loc="left",
        color=INK,
        fontsize=14,
        fontweight="bold",
    )
    ax.text(
        0.04,
        0.96,
        f"{zoom_mask.sum():,} of {count:,} points shown\n(actual and prediction ≤ actual-duration p95)",
        transform=ax.transAxes,
        va="top",
        color=INK,
        fontsize=9,
    )
    ax.grid(color=LIGHT_GRAY)
    ax.set_axisbelow(True)
    fig.text(
        0.06,
        0.015,
        "Descriptive model outputs, not held-out test predictions.",
        color=GRAY,
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    save_figure(fig, output_dir, "09_actual_vs_predicted_zoom.png")

    fig, ax = plt.subplots(figsize=(9, 5.2))
    bins = np.linspace(0, q95, 32)
    ax.hist(
        actual[actual <= q95],
        bins=bins,
        density=True,
        alpha=0.55,
        color=BLUE,
        label="Actual",
    )
    ax.hist(
        predicted[predicted <= q95],
        bins=bins,
        density=True,
        alpha=0.55,
        color=ORANGE,
        label="Predicted",
    )
    ax.set_xlabel("Duration (minutes), shown through actual-duration p95")
    ax.set_ylabel("Density")
    ax.set_title(
        "Actual and predicted duration distributions",
        loc="left",
        color=INK,
        fontsize=15,
        fontweight="bold",
    )
    ax.legend(frameon=False)
    ax.grid(axis="y", color=LIGHT_GRAY)
    ax.set_axisbelow(True)
    fig.text(
        0.06,
        0.015,
        "Only events with positive observed duration are included; tails beyond p95 are omitted from the display.",
        color=GRAY,
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    save_figure(fig, output_dir, "10_duration_distribution_comparison.png")


def write_summary_tables(events, severity_metadata, duration_metadata, output_dir):
    positive = events.loc[events["duration_minutes"] > 0, "duration_minutes"]
    summary = pd.DataFrame(
        [
            ("Reconstructed STOP groups", len(events)),
            ("Positive-duration groups", int(positive.count())),
            ("Zero-duration groups", int((events["duration_minutes"] <= 0).sum())),
            ("Prolonged groups (>5 min)", int((positive > 5).sum())),
            ("Median positive duration (min)", float(positive.median())),
            ("90th percentile duration (min)", float(positive.quantile(0.90))),
            ("95th percentile duration (min)", float(positive.quantile(0.95))),
            ("Maximum positive duration (min)", float(positive.max())),
        ],
        columns=["Measure", "Value"],
    )
    summary.to_csv(output_dir / "event_summary.csv", index=False)

    confusion = np.asarray(severity_metadata["test_confusion_matrix"])
    model_metrics = pd.DataFrame(
        [
            {
                "Task": "Severity classification",
                "Split": "Validation",
                **severity_metadata["validation_metrics"],
            },
            {
                "Task": "Severity classification",
                "Split": "Test",
                **severity_metadata["final_test_metrics"],
            },
            {
                "Task": "Duration regression",
                "Split": "Validation",
                **duration_metadata["validation_metrics"],
            },
            {
                "Task": "Duration regression",
                "Split": "Test",
                **duration_metadata["test_metrics"],
            },
        ]
    )
    model_metrics.to_csv(output_dir / "saved_model_metrics.csv", index=False)
    pd.DataFrame(
        confusion,
        index=["Actual normal", "Actual prolonged"],
        columns=["Predicted normal", "Predicted prolonged"],
    ).to_csv(output_dir / "severity_test_confusion_matrix.csv")


def main():
    parser = argparse.ArgumentParser(
        description="Generate MC09 STOP-event visualization assets and CSV tables."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=BASE_DIR / "visualizations",
        help="Directory for generated PNG figures and CSV tables.",
    )
    args = parser.parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    severity_metadata, duration_metadata = load_metadata()
    events = load_events()
    model_features = severity_metadata["features"]
    if model_features != FEATURES:
        raise ValueError(
            f"Severity metadata features {model_features} do not match {FEATURES}."
        )
    severity_model = joblib.load(MODEL_DIR / "stop_severity_logistic_model.joblib")
    severity_features = list(getattr(severity_model, "feature_names_in_", []))
    if severity_features and severity_features != FEATURES:
        raise ValueError(
            f"Severity model expects {severity_features}, but reconstructed features are {FEATURES}."
        )

    event_construction_figure(output_dir)
    q95 = duration_distribution_figure(events, output_dir)
    prediction_problem_figure(output_dir)
    temporal_feature_figure(output_dir)
    split_timeline_figure(severity_metadata, duration_metadata, output_dir)
    confusion_matrix_figure(severity_metadata, output_dir)
    model_comparison_figure(output_dir)
    observed = predict_durations(events, output_dir)
    actual_predicted_figures(observed, q95, output_dir)
    write_summary_tables(
        events,
        severity_metadata,
        duration_metadata,
        output_dir,
    )

    print(f"Generated 10 figures and 5 CSV tables in: {output_dir.resolve()}")
    print(
        f"Reconstructed {len(events):,} STOP groups; "
        f"{len(observed):,} have positive duration."
    )


if __name__ == "__main__":
    main()