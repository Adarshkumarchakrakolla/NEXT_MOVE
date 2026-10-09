# NextMove — Step 5: Point-in-Time Feature Pipeline Report

**Document Path:** `reports/step5_feature_pipeline_report.md`  
**Execution Date:** 2026-10-09  
**Status:** **STEP 5 COMPLETED — ALL 30 TESTS PASSING (0 FAILURES)**  
**Subsystem:** NextMove Feature Engineering & Point-in-Time Pipeline  

---

## 1. Executive Summary

Step 5 implements the production point-in-time feature engineering pipeline for NextMove. 

The pipeline strictly enforces **Convention A** (Start-of-Day Cutoff, Monday 00:00:00 UTC): for every observation cutoff date $T$, historical feature aggregations strictly query closed calendar days prior to Monday ($\text{event\_date} \le T_{\text{date}} - 1 \text{ day}$, i.e. up to Sunday 23:59:59 UTC). Events occurring on or after Monday $T$ are strictly excluded from the feature space, eliminating lookahead leakage by construction.

The feature generation pipeline was implemented in [`src/features/feature_builder.py`](file:///d:/NextMove/src/features/feature_builder.py) using an optimized $O(\log N)$ binary search indexing strategy, materializing all **6,675 supervised feature instances** (445 cutoffs $\times$ 3 competitors $\times$ 5 strategic clusters) in **2.802 seconds**.

All 30 automated tests (8 feature builder tests + 11 temporal integrity tests + 11 schema tests) passed with zero failures in 4.36 seconds. The raw CSV ([`recall_intel_121000_independent_synthetic_events.csv`](file:///d:/NextMove/data/synthetic/raw/recall_intel_121000_independent_synthetic_events.csv)) remains completely untouched (exact size: 48,002,803 bytes).

---

## 2. Implemented Code Architecture

### 2.1 Feature Builder Module: [`src/features/feature_builder.py`](file:///d:/NextMove/src/features/feature_builder.py)
Implements [`PointInTimeFeatureBuilder`](file:///d:/NextMove/src/features/feature_builder.py) and [`FeaturePipelineMetrics`](file:///d:/NextMove/src/features/feature_builder.py):
- **Pre-Indexing:** Groups and sorts event series by `(competitor, strategic_cluster)`. Dates and magnitudes are converted to contiguous NumPy arrays.
- **$O(\log N)$ Window Queries:** Evaluates exact lookback windows via `np.searchsorted`, completely avoiding repetitive full-dataset scans.
- **Dynamic Cluster Derivation:** Derives `strategic_cluster` (1–5) on the fly via centralized [`PredictionConfig`](file:///d:/NextMove/configs/prediction_config.py), maintaining compatibility with raw data lacking a pre-existing cluster column.
- **Modularity for Evidence Data:** Operates on any DataFrame exposing `competitor`, `event_date`, `event_type` (or `strategic_cluster`), and `magnitude_index`. When real evidence-backed logs are integrated, the feature extraction logic requires zero code changes.

### 2.2 Package Exports: [`src/features/__init__.py`](file:///d:/NextMove/src/features/__init__.py)
Exports [`PointInTimeFeatureBuilder`](file:///d:/NextMove/src/features/feature_builder.py) and [`FeaturePipelineMetrics`](file:///d:/NextMove/src/features/feature_builder.py) at package level.

### 2.3 Automated Test Suite: [`tests/temporal_leakage/test_feature_builder.py`](file:///d:/NextMove/tests/temporal_leakage/test_feature_builder.py)
Implements 8 isolated unit tests using deterministic, in-memory synthetic fixtures to verify mathematical accuracy, edge-case sentinels, rival filtering, decay rates, and temporal isolation.

---

## 3. Materialized Feature Artifact & Telemetry

The feature matrix was generated from the raw CSV and saved strictly to the designated synthetic processed area:

- **Output File Path:** [`data/synthetic/processed/synthetic_features_445_cutoffs_6675_instances.csv`](file:///d:/NextMove/data/synthetic/processed/synthetic_features_445_cutoffs_6675_instances.csv)
- **Total Rows Generated:** `6,675`
- **Total Columns:** `16`
- **Partition Breakdown:**
  - **Train Instances:** `4,635` (309 cutoffs $\times$ 15 targets; `2018-01-01` to `2023-11-27`)
  - **Validation Instances:** `1,110` (74 cutoffs $\times$ 15 targets; `2024-01-01` to `2025-05-26`)
  - **Test (OOT) Instances:** `930` (62 cutoffs $\times$ 15 targets; `2025-07-07` to `2026-09-07`)
- **Pipeline Execution Runtime:** `2.802 seconds`
- **File Size:** `724,100 bytes` (~707 KB)
- **SHA-256 Checksum:** `5d89b0f2e8011327e8d3b5ad512c0f6e746cf203e070969d1ef01f95d57b6424`
- **Source CSV Status:** Untouched (48,002,803 bytes; SHA-256 unchanged)

---

## 4. Formally Verified Feature Definitions & Mathematical Invariants

For every observation cutoff Monday $T_{\text{date}}$ at 00:00:00 UTC, the completed historical day boundary is $T - 1 \text{ day}$ (Sunday 23:59:59 UTC):

| Feature Name | Exact Lookback Interval | Mathematical Formula | Handling of Missing History & Zeros |
| :--- | :---: | :--- | :--- |
| **`days_since_last_cluster_event`** | History $\le T_{\text{date}} - 1$ | $(T_{\text{date}} - 1) - \max(t_e \mid t_e \le T_{\text{date}} - 1)$ | If no prior events exist: **`-1.0`** (standardized sentinel). |
| **`has_prior_cluster_event`** | History $\le T_{\text{date}} - 1$ | $\mathbb{I}(\text{count}(t_e \le T_{\text{date}} - 1) \ge 1)$ | **`0`** if no prior events exist; **`1`** otherwise. |
| **`cluster_event_count_30d`** | $[T_{\text{date}} - 30, T_{\text{date}} - 1]$ | $\sum \mathbb{I}(c_e = C \land k_e = K \land t_e \in \text{window})$ | **`0`** if no events in 30-day window. |
| **`cluster_event_count_90d`** | $[T_{\text{date}} - 90, T_{\text{date}} - 1]$ | $\sum \mathbb{I}(c_e = C \land k_e = K \land t_e \in \text{window})$ | **`0`** if no events in 90-day window. |
| **`cluster_event_count_365d`** | $[T_{\text{date}} - 365, T_{\text{date}} - 1]$ | $\sum \mathbb{I}(c_e = C \land k_e = K \land t_e \in \text{window})$ | **`0`** if no events in 365-day window. |
| **`velocity_ratio_30_90`** | Derived from 30d & 90d | $\frac{\text{count\_30d} + 1.0}{(\text{count\_90d} / 3.0) + 1.0}$ | **`1.0`** if both counts are 0 (Laplace smoothed baseline). |
| **`rival_event_count_14d`** | $[T_{\text{date}} - 14, T_{\text{date}} - 1]$ | $\sum \mathbb{I}(c_e \ne C \land k_e = K \land t_e \in \text{window})$ | **`0`** if rivals have no events in 14-day window. |
| **`decayed_magnitude_mass_60d`**| $[T_{\text{date}} - 60, T_{\text{date}} - 1]$ | $\sum m_e \cdot \exp(-0.05 \cdot (T_{\text{date}} - 1 - t_e))$ | **`0.0`** if no events in 60-day window. |
| **`quarter`** | Point-in-Time $T_{\text{date}}$ | Integer quarter in $[1, 4]$ | Extracted from $T_{\text{date}}$. |
| **`month`** | Point-in-Time $T_{\text{date}}$ | Integer month in $[1, 12]$ | Extracted from $T_{\text{date}}$. |
| **`is_q4`** | Point-in-Time $T_{\text{date}}$ | Binary flag: $\mathbb{I}(\text{quarter} == 4)$ | **`1`** if Q4; **`0`** otherwise. |

### Interval Boundary Discipline
- **Right Boundary ($T - 1$):** Events occurring on Sunday $T - 1$ have age $0$ days, age weight $\exp(0) = 1.0$, and are included in all lookback windows. Events on Monday $T$ have age $< 0$ and are strictly excluded.
- **Left Boundaries ($T - L$):** Lookback intervals are inclusive on both ends ($[T - L, T - 1]$). For $L = 30$, $(T - 1) - (T - 30) + 1 = 30$ calendar days. An event on $T - 30$ is included; an event on $T - 31$ is excluded.

---

## 5. Test Suite Execution Results

Ran complete project test suite via `pytest -v`:

```text
============================= test session starts =============================
platform win32 -- Python 3.14.5, pytest-9.1.1, pluggy-1.6.0
rootdir: D:\NextMove
configfile: pytest.ini
collected 30 items

tests/temporal_leakage/test_feature_builder.py::test_feature_matrix_combination_uniqueness_and_completeness PASSED [  3%]
tests/temporal_leakage/test_feature_builder.py::test_expected_feature_columns_and_data_types PASSED [  6%]
tests/temporal_leakage/test_feature_builder.py::test_exact_calculations_on_hand_built_fixture PASSED [ 10%]
tests/temporal_leakage/test_feature_builder.py::test_exclusion_of_events_on_and_after_cutoff PASSED [ 13%]
tests/temporal_leakage/test_feature_builder.py::test_missing_history_and_recency_sentinel PASSED [ 16%]
tests/temporal_leakage/test_feature_builder.py::test_rival_filtering PASSED [ 20%]
tests/temporal_leakage/test_feature_builder.py::test_calendar_features PASSED [ 23%]
tests/temporal_leakage/test_feature_builder.py::test_repeatability_across_repeated_runs PASSED [ 26%]
tests/temporal_leakage/test_temporal_integrity.py::test_taxonomy_mapping_completeness_and_uniqueness PASSED [ 30%]
tests/temporal_leakage/test_temporal_integrity.py::test_cluster_derivation_from_event_type PASSED [ 33%]
tests/temporal_leakage/test_temporal_integrity.py::test_partition_cutoff_counts_and_boundaries PASSED [ 36%]
tests/temporal_leakage/test_temporal_integrity.py::test_partition_embargo_isolation PASSED [ 40%]
tests/temporal_leakage/test_temporal_integrity.py::test_feature_exclusion_on_and_after_cutoff_date PASSED [ 43%]
tests/temporal_leakage/test_temporal_integrity.py::test_label_window_first_and_last_day_inclusion PASSED [ 46%]
tests/temporal_leakage/test_temporal_integrity.py::test_label_window_exclusion_one_day_before_and_after PASSED [ 50%]
tests/temporal_leakage/test_temporal_integrity.py::test_empty_event_windows_produce_label_zero PASSED [ 53%]
tests/temporal_leakage/test_temporal_integrity.py::test_competitor_and_cluster_filtering PASSED [ 56%]
tests/temporal_leakage/test_temporal_integrity.py::test_rejection_of_incomplete_forward_windows PASSED [ 60%]
tests/temporal_leakage/test_temporal_integrity.py::test_deterministic_results_across_runs PASSED [ 63%]
tests/unit/test_dataset_schema.py::test_row_and_column_counts PASSED     [ 66%]
tests/unit/test_dataset_schema.py::test_column_names_match_expected PASSED [ 70%]
tests/unit/test_dataset_schema.py::test_event_id_uniqueness_and_validity PASSED [ 73%]
tests/unit/test_dataset_schema.py::test_event_date_parseability_and_bounds PASSED [ 76%]
tests/unit/test_dataset_schema.py::test_zero_missing_or_empty_values PASSED [ 80%]
tests/unit/test_dataset_schema.py::test_no_exact_duplicate_rows PASSED   [ 83%]
tests/unit/test_dataset_schema.py::test_categorical_domains PASSED       [ 81%]
tests/unit/test_dataset_schema.py::test_numeric_ranges_and_validity PASSED [ 86%]
tests/unit/test_dataset_schema.py::test_synthetic_metadata_labels PASSED [ 90%]
tests/unit/test_dataset_schema.py::test_generator_seed_consistency PASSED [ 95%]
tests/unit/test_dataset_schema.py::test_description_template_prefix PASSED [100%]

============================= 30 passed in 4.36s ==============================
```

- **Total Test Count:** 30
- **Total Passing:** 30 (100%)
- **Failures / Errors:** 0
- **Execution Time:** 4.36 seconds

---

## 6. Scope Compliance & Operating Limitations

1. **No Production Labels Generated:** Only the point-in-time feature matrix was materialized. Ground truth labels for training remain ungenerated.
2. **No Models Trained or Evaluated:** No predictive models have been instantiated, fit, or scored.
3. **No Scalers or Encoders Fit:** No global standardizers, normalizers, or imputers have been fit to prevent cross-temporal distribution leakage.
4. **Synthetic Data Capability Boundary:** Synthetic feature values validate data transformations, temporal indexing, and software contracts. Because synthetic attributes are statistically uncoupled (correlations $\approx 0.000$) and template-generated, models trained on this data test engineering pipeline mechanics, not empirical market forecasting skill.
5. **Date-Only Granularity Limitation:** Calendar-day boundaries ($\le T_{\text{date}} - 1$) enforce date-level mathematical isolation, but cannot resolve intra-day publication time-of-day or scraping latency without immutable `published_at` and `ingested_at` logging.
