# Regression Model Comparison on Yeo–Johnson Features

**Scope:** Seven regression candidates evaluated using Yeo–Johnson-transformed input features and the experiment's shared chronological split. This report covers duration regression only; it is not a comparison of severity classifiers.

## Summary

The Log-Linear Regression has the lowest test MAE (10.27 minutes) and RMSE (57.45 minutes). The Median Baseline is a close MAE competitor at 10.35 minutes and has a much lower test MAPE (210.29% versus 869.07%). On validation, the Median Baseline has the lowest MAE (8.04 minutes), while Log-Linear Regression has the lowest RMSE (41.64 minutes). The model choice therefore depends on which error measure is prioritized; Log-Linear Regression is not best on every metric.

The app's current final model choice remains Log-Linear Regression. This comparison does not change the app model or its selection; it provides the requested candidate-model results on Yeo–Johnson features.

## Dataset and Split

All methods use the saved Yeo–Johnson experiment cohort:

- 706 positive-duration STOP events.
- 423 training events, 141 validation events, and 142 test events.
- The same event membership and chronological split are used for every method.
- The Yeo–Johnson transformer is loaded from the experiment output. It was fitted on training features only; validation and test features use that same fitted transformer.

The four input features are `minutes_since_previous_STOP`, `STOP_count_last_1min`, `STOP_count_last_5min`, and `STOP_count_last_10min`. The continuous target is `duration_minutes`.

## Methodology

The comparison script loads `reconstructed_events.csv`, filters to events with positive duration, and uses the saved split labels. It transforms train, validation, and test features using the stored Yeo–Johnson transformer, then fits each learned model on the transformed training features only.

- **Mean Baseline:** predicts the mean training duration for every validation and test event.
- **Median Baseline:** predicts the median training duration for every validation and test event.
- **Linear Regression:** ordinary least-squares `LinearRegression` fitted to raw duration.
- **Log-Linear Regression:** ordinary least-squares `LinearRegression` fitted to `log1p(duration)`; predictions are converted back to minutes using `expm1`.
- **Decision Tree:** `DecisionTreeRegressor(random_state=42)` fitted to raw duration.
- **Random Forest:** `RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)` fitted to raw duration.
- **Random Forest + Log Target:** the same Random Forest configuration fitted to `log1p(duration)`; predictions are converted back using `expm1`.

The original seven-row chart stored only metric values and did not preserve the Decision Tree or Random Forest hyperparameters. Therefore, those two methods were recomputed with the explicit deterministic settings above. The comparison script checks that the recomputed Log-Linear Regression test metrics match the saved Yeo–Johnson experiment metrics.

## Metrics

- **MAE:** mean absolute duration error, in minutes. Lower is better.
- **RMSE:** square-root mean squared error, in minutes. Lower is better and large errors have more influence than with MAE.
- **MAPE:** mean absolute percentage error. Lower is better, but it can be very large when actual durations are close to zero because each error is divided by the actual duration.

## Validation Results

| Model | MAE (min) | RMSE (min) | MAPE |
|---|---:|---:|---:|
| Mean Baseline | 43.46 | 55.21 | 17,916.45% |
| Median Baseline | 8.04 | 42.11 | 185.54% |
| Linear Regression | 79.42 | 102.96 | 31,979.27% |
| Log-Linear Regression | 8.05 | 41.64 | 874.02% |
| Decision Tree | 28.63 | 100.48 | 8,263.92% |
| Random Forest | 37.21 | 106.74 | 24,064.31% |
| Random Forest + Log Target | 18.61 | 65.99 | 2,865.48% |

The Median Baseline has the best validation MAE and MAPE. Log-Linear Regression has the best validation RMSE. The difference between the median and log-linear MAE is approximately 0.01 minutes, or 0.59 seconds.

## Test Results

| Model | MAE (min) | RMSE (min) | MAPE |
|---|---:|---:|---:|
| Mean Baseline | 47.33 | 66.43 | 19,898.22% |
| Median Baseline | 10.35 | 57.81 | 210.29% |
| Linear Regression | 76.90 | 102.98 | 30,426.86% |
| Log-Linear Regression | 10.27 | 57.45 | 869.07% |
| Decision Tree | 86.46 | 543.18 | 9,395.17% |
| Random Forest | 80.51 | 349.25 | 13,133.13% |
| Random Forest + Log Target | 20.32 | 77.62 | 1,833.43% |

Log-Linear Regression has the lowest test MAE and RMSE, but its MAE advantage over the Median Baseline is only about 0.072 minutes, or 4.3 seconds. The Median Baseline's test MAPE is substantially lower. The raw-target Linear Regression, Decision Tree, and Random Forest have much larger test errors; the tree and forest also have large validation-to-test error increases, suggesting their fit did not generalize well to this chronological test period.

## Interpretation

The results support keeping Log-Linear Regression as a reasonable duration model when absolute error and squared-error performance are the primary criteria. The test result is slightly better than the Median Baseline on MAE and RMSE, and the validation RMSE is also best among the candidates. However, validation MAE slightly favors the Median Baseline, and the Median Baseline has much lower MAPE on both splits. Thus, the evidence does not show that Log-Linear Regression is uniformly superior.

The high MAPE values across several candidates indicate that percentage error is unstable for this duration distribution, particularly for short events. MAE and RMSE should be reported alongside MAPE, not replaced by it. If relative percentage accuracy is a primary operational goal, the current results do not establish Log-Linear Regression as the best choice.

Model selection should be based on validation metrics and the operational cost of different errors. The test set should remain a final held-out assessment rather than be used to repeatedly tune or choose models. These results represent one chronological split, so they do not establish stability across different time periods.

## Reproducibility and Artifacts

- Comparison implementation: [compare_regression_models.py](compare_regression_models.py)
- Full precision results for both splits: [regression_model_comparison.csv](yeo_johnson_experiment_output/regression_model_comparison.csv)
- Experiment metrics used to validate Log-Linear Regression: [metrics.csv](yeo_johnson_experiment_output/metrics.csv)
- Experiment cohort and split provenance: [experiment_metadata.json](yeo_johnson_experiment_output/experiment_metadata.json)
