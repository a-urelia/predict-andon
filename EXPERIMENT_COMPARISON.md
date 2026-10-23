# Preprocessing Experiments: Methods and Comparison

**Prepared:** 2026-10-22  
**Scope:** No scaling, Min-Max, RobustScaler, and Yeo-Johnson experiments on the shared MC09 event cohort.

## Executive Summary

The four current preprocessing runs use the same raw MC09 telemetry, reconstruct the same STOP events, and evaluate on the same chronological split: 706 positive-duration events, with 423 train, 141 validation, and 142 test events. Their raw-data, reconstructed-event, and split-assignment hashes match. This makes the four runs a controlled comparison of feature preprocessing on this split.

Yeo-Johnson is the current app model. At the fixed 0.30 severity threshold, it is the only method that identifies any prolonged STOPs on the test set: 3 of 32, for 9.38% recall. It also has the best test regression MAE and RMSE, but those improvements over the other methods are very small, and its MAPE is substantially worse. The evidence supports Yeo-Johnson as the strongest of these four candidates when detecting prolonged events is important, not as an unqualified winner or a high-recall detector.

The earlier models in `models/` are a separate historical baseline recorded as 411 events. Their exact training-event membership is not persisted, so they are not included in the matched four-method comparison.

## Experiment Inventory

| Method | Script / output | Feature treatment |
|---|---|---|
| No scaling | [Script](experiment%20no-scaling/no_scaling_experiment.py), [outputs](experiment%20no-scaling/no_scaling_experiment_output/) | Original numeric feature values are passed directly to the estimators. |
| Min-Max | [Script](experiment%20min-max/minmax_scaling_experiment.py), [controlled outputs](experiment%20min-max/controlled_preprocessing_comparison/) | `MinMaxScaler` maps each feature to the 0-1 range, fitted on training features only. |
| RobustScaler | [Script](experiment%20robustscaler/robust-scaler-experiment.py), [outputs](experiment%20robustscaler/robust_scaler_experiment_output/) | `RobustScaler` centers by the training median and scales by the training interquartile range. |
| Yeo-Johnson | [Script](experiment%20yeo-johnson/yeo-johnson-experiment.py), [outputs](experiment%20yeo-johnson/yeo_johnson_experiment_output/) | `PowerTransformer(method="yeo-johnson", standardize=False)` is fitted on training features only. |

The application currently loads the Yeo-Johnson transformer and model artifacts from `experiment yeo-johnson/yeo_johnson_experiment_output/` in [andon-intelligence/app.py](andon-intelligence/app.py).

## Shared Data Preparation

All four runs use `data/mc09_andon.csv` as their raw source. The current shared event-construction procedure is:

1. Read `timestamp`, `status`, and `downtimeCurrent`; parse timestamps, normalize status text, convert downtime to numeric, replace missing downtime with zero, drop rows with invalid timestamps, and sort chronologically.
2. Form contiguous STOP blocks, separating a block when a non-STOP status occurs. Within each block, start a new event whenever the gap between adjacent STOP timestamps is greater than 300 seconds.
3. Define each event duration as the maximum `downtimeCurrent` in its segment, converted from milliseconds to minutes. Assign event IDs in chronological order.
4. Before filtering by duration, calculate four historical features: minutes since the previous STOP and STOP counts in the prior 1-, 5-, and 10-minute windows. Engineering features first preserves the preceding event history even when an event itself has zero duration.
5. Keep the 706 events with duration greater than zero for modeling. Define a prolonged STOP as duration greater than 5 minutes.
6. Sort eligible events chronologically and split 60%/20%/20% into train/validation/test: 423/141/142. Model preprocessing is fitted on the training features only; validation and test features are transformed with that fitted preprocessor.

The common cohort contains 1,536 reconstructed segments before the positive-duration filter. The split sizes sum to the modeled cohort: 423 + 141 + 142 = 706.

### Provenance Checks

The recorded shared hashes are:

- Raw CSV: `5d15fa3b6892f29c1765d275d53fb3a910091ad74899c2d1149f69ce0f3919b9`
- Reconstructed event table: `74ccdf25a3d9b69160f57132f39ef1eac4f3c72a009d8cbf55b44be1429451ae`
- Split assignments: `a77afdd8ba3b0c13c782d778a783df9a2c18fe2487d56721b75fb084576538e5`

The no-scaling run independently verifies these hashes against the Min-Max, RobustScaler, and Yeo-Johnson experiment metadata and stops if any differ. The Yeo-Johnson metadata also records successful event and split identity checks against the other two scaled experiments.

## Modeling and Evaluation Protocol

The feature transformation is the experimental variable. Each run uses the same feature names, chronological cohort, split membership, model families, and evaluation threshold:

- **Classification:** `LogisticRegression(max_iter=1000)` predicts whether duration exceeds 5 minutes. Probabilities are classified as prolonged at a fixed threshold of 0.30.
- **Regression:** `LinearRegression` is fitted to `log1p(duration_minutes)`. Predictions are converted back to minutes with `expm1` before calculating error metrics. “No scaling” means no feature scaling; it does not mean the regression target is left untransformed.
- **Classification metrics:** Accuracy, balanced accuracy, precision, recall, F1, ROC-AUC, PR-AUC, and confusion matrix. Confusion matrices below use actual classes as rows and predicted classes as columns, ordered Normal then Prolonged.
- **Regression metrics:** MAE and RMSE in minutes, and MAPE as a percentage. MAPE can become very large when actual durations are small because it divides by the actual value.

Validation metrics are useful for model selection; test metrics are the final held-out comparison. The 0.30 threshold is held fixed here rather than tuned separately for each test result.

## Classification Results

### Validation Split

| Method | Accuracy | Balanced accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC | Confusion matrix |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| No scaling | 80.14% | 50.00% | 0.00% | 0.00% | 0.00% | 66.56% | 35.64% | `[[113, 0], [28, 0]]` |
| Min-Max | 80.14% | 50.00% | 0.00% | 0.00% | 0.00% | 66.25% | 35.55% | `[[113, 0], [28, 0]]` |
| RobustScaler | 80.14% | 50.00% | 0.00% | 0.00% | 0.00% | 66.56% | 35.64% | `[[113, 0], [28, 0]]` |
| Yeo-Johnson | 77.30% | 57.63% | 38.89% | 25.00% | 30.43% | 66.25% | 35.54% | `[[102, 11], [21, 7]]` |

Yeo-Johnson identifies 7 of the 28 prolonged validation events, while the other three methods predict no prolonged events at the fixed threshold. The increased recall comes with 11 false positives in validation.

### Test Split

| Method | Accuracy | Balanced accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC | Confusion matrix |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| No scaling | 76.76% | 49.55% | 0.00% | 0.00% | 0.00% | 66.45% | 36.12% | `[[109, 1], [32, 0]]` |
| Min-Max | 77.46% | 50.00% | 0.00% | 0.00% | 0.00% | 67.10% | 36.31% | `[[110, 0], [32, 0]]` |
| RobustScaler | 76.76% | 49.55% | 0.00% | 0.00% | 0.00% | 66.59% | 36.15% | `[[109, 1], [32, 0]]` |
| Yeo-Johnson | 76.76% | 52.87% | 42.86% | 9.38% | 15.38% | 67.19% | 36.36% | `[[106, 4], [29, 3]]` |

The test set contains 110 Normal and 32 Prolonged events. Min-Max has the highest accuracy because it predicts every test event as Normal, correctly classifying all 110 Normal cases but missing all 32 prolonged cases. Thus, accuracy alone gives a misleading picture of usefulness for an early-warning task. Yeo-Johnson catches 3 prolonged events, misses 29, and raises 4 false alarms.

## Regression Results

| Method | Validation MAE (min) | Validation RMSE (min) | Validation MAPE | Test MAE (min) | Test RMSE (min) | Test MAPE |
|---|---:|---:|---:|---:|---:|---:|
| No scaling | 8.02 | 41.80 | 674.37% | 10.29 | 57.58 | 744.51% |
| Min-Max | 8.02 | 41.80 | 674.37% | 10.29 | 57.58 | 744.51% |
| RobustScaler | 8.02 | 41.80 | 674.37% | 10.29 | 57.58 | 744.51% |
| Yeo-Johnson | 8.05 | 41.64 | 874.02% | 10.27 | 57.45 | 869.07% |

No scaling, Min-Max, and RobustScaler produce effectively identical regression results. This is expected for ordinary least-squares linear regression under affine feature transformations: rescaling inputs generally changes coefficient units, not the predictions. Yeo-Johnson is nonlinear, so it can change the fitted relationships. On the test set its MAE improves by about 0.013 minutes (less than one second) and RMSE by about 0.135 minutes (about eight seconds), while MAPE increases by about 124.55 percentage points compared with the other three.

## Interpretation and Model Choice

### Why the Application Uses Yeo-Johnson

The app currently uses the Yeo-Johnson artifacts. Relative to these four runs, the practical reason to retain that candidate is classification behavior at the app's current 0.30 threshold:

- It is the only method with nonzero test recall, precision, and F1. It detects 3 of 32 prolonged test events; the other methods detect none.
- It has the highest test ROC-AUC (67.19%) and PR-AUC (36.36%), though these gains over Min-Max and the other methods are small.
- Its test MAE and RMSE are numerically the lowest, but only marginally lower than the nearly identical affine-scaling/no-scaling results.

This makes Yeo-Johnson the most defensible current candidate if detecting any prolonged events is prioritized over raw accuracy. It is not evidence that the model is reliable for detecting prolonged events: test recall is only 9.38%, so it misses 90.62% of prolonged cases. MAPE is also high, and its result is worse than the other three methods on that metric. If duration error is the primary objective, this experiment does not establish a strong overall advantage for Yeo-Johnson.

### Recommended Reporting

Describe the choice as a **provisional, task-prioritized selection**: Yeo-Johnson was chosen because it provides the best observed prolonged-event detection at the selected threshold, with slightly lower test MAE/RMSE. Report its low recall and high MAPE alongside that rationale. Do not say it has the best accuracy or that it performs well overall.

Before making a stronger deployment claim, select a classification threshold using validation data according to the cost of missed prolonged events versus false alarms, then evaluate that locked threshold once on the test set. A single chronological holdout provides limited evidence about performance across other periods; repeated time-based evaluation or a later untouched period would give stronger evidence.

## Legacy 411-Event Baseline

The original models in `models/` and the standalone Min-Max artifacts record 411 events with splits of 246/82/83. Those are not the same cohort as the current 706-event comparison. The original training event IDs and exact preprocessing lineage are not available, so their metric differences cannot be attributed to scaling alone.

The Min-Max folder contains both generations of artifacts. Use the 706-event `controlled_preprocessing_comparison/` output for comparisons with RobustScaler, Yeo-Johnson, and the new no-scaling run. Do not mix its results with the older top-level Min-Max model metadata that reports 411 events. The prior provenance review is available at [temuan_investigasi.md](experiment%20min-max/minmax_provenance_investigation_2026-10-19/temuan_investigasi.md).

## Artifact References

- [No-scaling metrics](experiment%20no-scaling/no_scaling_experiment_output/metrics.csv) and [metadata](experiment%20no-scaling/no_scaling_experiment_output/experiment_metadata.json)
- [Min-Max paired metrics](experiment%20min-max/controlled_preprocessing_comparison/metrics_comparison.csv) and [metadata](experiment%20min-max/controlled_preprocessing_comparison/experiment_metadata.json)
- [RobustScaler metrics](experiment%20robustscaler/robust_scaler_experiment_output/metrics.csv), [preprocessing comparison](experiment%20robustscaler/robust_scaler_experiment_output/preprocessing_comparison.csv), and [metadata](experiment%20robustscaler/robust_scaler_experiment_output/experiment_metadata.json)
- [Yeo-Johnson metrics](experiment%20yeo-johnson/yeo_johnson_experiment_output/metrics.csv), [preprocessing comparison](experiment%20yeo-johnson/yeo_johnson_experiment_output/preprocessing_comparison.csv), and [metadata](experiment%20yeo-johnson/yeo_johnson_experiment_output/experiment_metadata.json)
- [Legacy duration-model metadata](models/stop_duration_log_linear_model_metadata.json) and [legacy classifier metadata](models/stop_severity_logistic_model_metadata.json)
