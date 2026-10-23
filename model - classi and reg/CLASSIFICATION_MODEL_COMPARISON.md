# Severity Classification Comparison on Yeo–Johnson Features

**Scope:** Four severity-classification candidates evaluated using the saved Yeo–Johnson-transformed features and the experiment's shared chronological split. This report covers classification only; duration regression results are documented separately in [REGRESSION_MODEL_COMPARISON.md](REGRESSION_MODEL_COMPARISON.md).

## Summary

At the app's fixed probability threshold of 0.30, Logistic Regression has the highest test ROC-AUC (67.19%), PR-AUC (36.36%), and precision (42.86%). It identifies 3 of the 32 prolonged STOPs in the test set, for 9.38% recall. The Decision Tree and Random Forest each identify 8 of 32 prolonged events (25.00% recall), with lower precision (26.67% and 25.00%) and weaker probability-ranking metrics. The majority-class baseline identifies no prolonged events.

There is no universally best classifier across all goals. Logistic Regression gives fewer false alarms and stronger test ranking scores; the tree and forest identify more prolonged events at the selected threshold, but generate many more false alarms. The app's final model remains Logistic Regression; this comparison does not change the production model or threshold.

## Dataset and Split

All classifiers use the same saved Yeo–Johnson experiment cohort and split:

| Split | Normal STOP | Prolonged STOP | Total |
|---|---:|---:|---:|
| Train | 358 | 65 | 423 |
| Validation | 113 | 28 | 141 |
| Test | 110 | 32 | 142 |

A prolonged STOP is defined as duration greater than 5 minutes. The feature inputs are minutes since the previous STOP and STOP counts in the preceding 1-, 5-, and 10-minute windows. The PowerTransformer is fitted on training features only; its saved transform is applied to all three splits. Classifiers are fitted on the transformed training features.

## Methodology

All models use the same training labels, features, transformed matrices, chronological validation/test sets, and probability threshold of 0.30:

- **Majority Class Baseline:** `DummyClassifier(strategy="most_frequent")`, trained on the training split. It predicts the most frequent class and has no event-detection ability.
- **Logistic Regression:** `LogisticRegression(max_iter=1000)`, matching the selected Yeo–Johnson severity classifier.
- **Decision Tree:** `DecisionTreeClassifier(random_state=42)` with scikit-learn defaults otherwise.
- **Random Forest:** `RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)` with scikit-learn defaults otherwise.

The original results did not preserve tree/forest hyperparameters. The tree and forest were therefore rerun with the deterministic configurations above. The comparison script verifies that Logistic Regression's validation and test metrics, including its confusion matrices, match the saved Yeo–Johnson experiment metrics.

## Metrics

- **Accuracy:** fraction of all events classified correctly; it can be misleading when normal events are more common.
- **Balanced accuracy:** average of recall for the Normal and Prolonged classes.
- **Precision:** fraction of predicted prolonged events that were actually prolonged.
- **Recall:** fraction of actual prolonged events detected.
- **F1:** harmonic mean of precision and recall.
- **ROC-AUC and PR-AUC:** ranking metrics computed from predicted probabilities, rather than applying the 0.30 threshold. PR-AUC is especially relevant when prolonged cases are less frequent.
- **Confusion matrix:** actual classes are rows and predicted classes are columns, ordered Normal then Prolonged.

## Validation Results

| Model | Accuracy | Balanced accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC | Confusion matrix |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| Majority Class Baseline | 80.14% | 50.00% | 0.00% | 0.00% | 0.00% | 50.00% | 19.86% | `[[113, 0], [28, 0]]` |
| Logistic Regression | 77.30% | 57.63% | 38.89% | 25.00% | 30.43% | 66.25% | 35.54% | `[[102, 11], [21, 7]]` |
| Decision Tree | 64.54% | 48.32% | 17.65% | 21.43% | 19.35% | 48.51% | 19.62% | `[[85, 28], [22, 6]]` |
| Random Forest | 67.38% | 52.78% | 23.53% | 28.57% | 25.81% | 56.89% | 24.50% | `[[87, 26], [20, 8]]` |

On validation, Logistic Regression has the highest F1, ROC-AUC, and PR-AUC. Random Forest has slightly higher recall, detecting 8 of 28 prolonged cases instead of Logistic Regression's 7, but at the cost of more false alarms (26 versus 11).

## Test Results

| Model | Accuracy | Balanced accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC | Confusion matrix |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| Majority Class Baseline | 77.46% | 50.00% | 0.00% | 0.00% | 0.00% | 50.00% | 22.54% | `[[110, 0], [32, 0]]` |
| Logistic Regression | 76.76% | 52.87% | 42.86% | 9.38% | 15.38% | 67.19% | 36.36% | `[[106, 4], [29, 3]]` |
| Decision Tree | 67.61% | 52.50% | 26.67% | 25.00% | 25.81% | 52.73% | 24.04% | `[[88, 22], [24, 8]]` |
| Random Forest | 66.20% | 51.59% | 25.00% | 25.00% | 25.00% | 52.67% | 24.53% | `[[86, 24], [24, 8]]` |

At threshold 0.30, Logistic Regression flags 7 events: 3 are prolonged and 4 are normal. It misses 29 of 32 prolonged events. The Decision Tree flags 30 events, detecting 8 prolonged events and incorrectly flagging 22 normal events. The Random Forest flags 32 events, detecting 8 prolonged events and incorrectly flagging 24 normal events. Thus, tree-based candidates increase recall to 25% but have substantially lower precision and substantially more false alarms.

The majority baseline's 77.46% test accuracy is not useful for detection: it classifies every event as Normal and misses all prolonged cases. Accuracy should not be the sole selection metric for this imbalanced task.

## Interpretation and Model Choice

The comparison supports retaining Logistic Regression as a reasonable, interpretable probability model when avoiding false alarms and maintaining probability ranking quality are priorities. It has the best test ROC-AUC, PR-AUC, and precision of the four tested candidates, and it matches the existing Yeo–Johnson experiment result.

However, the current threshold yields only 9.38% test recall, so the selected classifier misses most prolonged events. If the operational priority is to catch more prolonged STOPs, the Decision Tree and Random Forest detect more events at this threshold, but their false-alarm burden is much higher and their ROC-AUC/PR-AUC are weaker. That is a business tradeoff, not an unambiguous algorithm ranking.

A threshold should be selected on validation data according to the relative cost of missed prolonged events and false alarms, then locked before final test evaluation. These results come from one chronological split and do not demonstrate stability across other periods. No claims about production alert reliability should be made from this comparison alone.

## Reproducibility and Artifacts

- Comparison implementation: [compare_classification_models.py](compare_classification_models.py)
- Full precision validation/test results: [classification_model_comparison.csv](yeo_johnson_experiment_output/classification_model_comparison.csv)
- Existing Yeo–Johnson metrics used to verify Logistic Regression: [metrics.csv](yeo_johnson_experiment_output/metrics.csv)
- Experiment cohort and split provenance: [experiment_metadata.json](yeo_johnson_experiment_output/experiment_metadata.json)
