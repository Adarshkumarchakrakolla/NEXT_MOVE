# Step 6: Binary Label Generation and Modeling Dataset Integrity Report

**NextMove: Production Competitive Intelligence Agent**  
**Component:** Target Label Generation & Modeling Dataset Assembly Pipeline (`src/labels/label_builder.py`)  
**Status:** Complete & Fully Validated  
**Execution Date:** 2026-10-09  

---

## 1. Executive Summary

In Step 6 of the NextMove project, we implemented the target label generation pipeline and assembled the final modeling dataset artifact. The pipeline generates the binary decision target `target_30d` over the approved forward window:

$$\text{cutoff\_date} \le \text{event\_date} \le \text{cutoff\_date} + 29\text{ days (inclusive)}$$

Every instance is identified by the canonical tripartite key `(cutoff_date, competitor, strategic_cluster)`. The label indicates whether at least one qualifying strategic move is executed by that competitor within that strategic cluster during the 30-calendar-day forward window ($1 = \text{true}$, $0 = \text{false}$).

The labels were joined with the 16 point-in-time feature columns materialized in Step 5 (`data/synthetic/processed/synthetic_features_445_cutoffs_6675_instances.csv`), enforcing a strict 1:1 bijection with zero missing keys, zero duplicate keys, zero null values, and complete preservation of all original feature values.

The resulting artifact was serialized to disk at [data/synthetic/processed/synthetic_modeling_dataset_445_cutoffs_6675_instances.csv](file:///d:/NextMove/data/synthetic/processed/synthetic_modeling_dataset_445_cutoffs_6675_instances.csv).

> [!IMPORTANT]
> **Synthetic Dataset Benchmark Notice**  
> Consistent with project directives, **no predictive models were trained, no preprocessing transformers were fitted, no classification thresholds were tuned, and no predictive skill is claimed**. This pipeline verifies mathematical correctness, temporal isolation, and engineering robustness on synthetic data prior to production deployment.

---

## 2. Implemented Architecture and Code Modules

The label pipeline and test suites were implemented across the following dedicated modules:

| Module Path | Type | Purpose |
| :--- | :--- | :--- |
| [src/labels/__init__.py](file:///d:/NextMove/src/labels/__init__.py) | Package Init | Exports `TargetLabelBuilder` and `LabelPipelineMetrics`. |
| [src/labels/label_builder.py](file:///d:/NextMove/src/labels/label_builder.py) | Implementation | Core pipeline class for forward window indexing, binary label derivation, domain validation, and feature joining. |
| [tests/labels/test_label_builder.py](file:///d:/NextMove/tests/labels/test_label_builder.py) | Test Suite | 13 automated tests verifying boundary inclusion, outside exclusion, empty windows, isolation, join integrity, and determinism. |
| [data/synthetic/processed/synthetic_modeling_dataset_445_cutoffs_6675_instances.csv](file:///d:/NextMove/data/synthetic/processed/synthetic_modeling_dataset_445_cutoffs_6675_instances.csv) | Data Artifact | Final assembled modeling dataset (6,675 rows, 17 columns). |

### Key Pipeline Components in `TargetLabelBuilder`

1. **Centralized Taxonomy Mapping:** Uses `assign_strategic_clusters()` and `PredictionConfig.event_type_to_cluster` to derive cluster IDs (1–5) from raw `event_type` strings. No cluster ID is assumed to pre-exist in the raw log.
2. **Fast Binary-Search Temporal Indexing:** Events are indexed by `(competitor, strategic_cluster)` into sorted NumPy date arrays (`datetime64[D]`). Window event counts over $[T, T + 29\text{d}]$ are computed via `np.searchsorted` in $\mathcal{O}(\log N)$, processing all 6,675 combinations in under 0.05 seconds.
3. **Right-Censoring Rejection:** Any cutoff where $T + 29\text{ days} > \text{dataset\_max\_date}$ (`2026-10-09`) is rejected with a `ValueError`.
4. **Bijective Feature-Label Joining:**
   - Enforces key uniqueness on `(cutoff_date, competitor, strategic_cluster)`.
   - Rejects unexpected competitors ($\notin \{\text{Amazon}, \text{Walmart}, \text{eBay}\}$) or invalid cluster IDs ($\notin \{1, 2, 3, 4, 5\}$).
   - Rejects orphan feature instances or orphan labels.
   - Enforces partition consistency across features and labels.
   - Verifies all original feature values remain bitwise unchanged.

---

## 3. Strict Temporal Leakage Prevention (Convention A)

The dataset guarantees absolute temporal isolation between features and targets:

```
Historical Feature Lookback Window            Prediction Cutoff (T)       Forward Label Window
[Sunday T - 365d  ...  Sunday T - 1d]         Monday 00:00:00 UTC         [Monday T  ...  Tuesday T + 29d]
<----------------------------------->                  |                  <------------------------------>
Features query ONLY: t_e <= T - 1                      |                  Labels query ONLY: T <= t_e <= T + 29
(Strictly completed history)                           |                  (Strict 30-day forward window)
                                                       V
                                          Zero Temporal Overlap
```

- **Feature Boundary:** Features are calculated using strictly $t_e \le T - 1\text{ day}$ (completed historical days through Sunday 23:59:59 UTC).
- **Label Boundary:** Labels are calculated using strictly $T \le t_e \le T + 29\text{ days}$ (Monday 00:00:00 UTC through Tuesday 23:59:59 UTC 4 weeks later).
- **Separation:** Intersection between feature events and label events is empty ($\emptyset$).
- **Embargo Protection:** The 35-day embargo between Train and Validation (`2023-11-28` to `2023-12-31`) and 42-day embargo between Validation and Test (`2025-05-27` to `2025-07-06`) ensure that no forward label window extends into the cutoff date of a subsequent partition.

---

## 4. Modeling Dataset Artifact Reconciliation

### Artifact Verification Metadata

- **Output File Path:** `data/synthetic/processed/synthetic_modeling_dataset_445_cutoffs_6675_instances.csv`
- **File Size:** 737,461 bytes (720.18 KB)
- **Total Row Count:** 6,675 rows
- **Total Column Count:** 17 columns
- **SHA-256 Checksum:** `6019116a428ac0f43f7c289cd73512310e8e82045745bdb9b87980e642a89c0d`
- **Pipeline Execution Runtime:** 2.994 seconds

### Schema Order and Types

| Column Index | Column Name | Data Type | Role | Value Constraints / Range |
| :---: | :--- | :--- | :--- | :--- |
| 1 | `cutoff_date` | `string` (`YYYY-MM-DD`) | Key | Weekly Mondays (`2018-01-01` to `2026-09-07`) |
| 2 | `competitor` | `string` | Key | `Amazon`, `Walmart`, `eBay` |
| 3 | `strategic_cluster` | `int64` | Key | 1, 2, 3, 4, 5 |
| 4 | `cluster_name` | `string` | Metadata | Approved human-readable cluster title |
| 5 | `partition` | `string` | Split | `train`, `validation`, `test` |
| 6 | `days_since_last_cluster_event` | `float64` | Feature | $[0.0, \infty)$ or $-1.0$ (sentinel) |
| 7 | `has_prior_cluster_event` | `int64` | Feature | $0$ or $1$ |
| 8 | `cluster_event_count_30d` | `int64` | Feature | $\ge 0$ |
| 9 | `cluster_event_count_90d` | `int64` | Feature | $\ge 0$ |
| 10 | `cluster_event_count_365d` | `int64` | Feature | $\ge 0$ |
| 11 | `velocity_ratio_30_90` | `float64` | Feature | $(0.0, \infty)$ |
| 12 | `rival_event_count_14d` | `int64` | Feature | $\ge 0$ |
| 13 | `decayed_magnitude_mass_60d` | `float64` | Feature | $\ge 0.0$ |
| 14 | `quarter` | `int64` | Feature | 1, 2, 3, 4 |
| 15 | `month` | `int64` | Feature | 1 through 12 |
| 16 | `is_q4` | `int64` | Feature | $0$ or $1$ |
| 17 | `target_30d` | `int64` | **Target** | **$0$ or $1$** |

Zero missing, NaN, infinite, or unparseable values exist anywhere in the artifact.

---

## 5. Label Distribution and Synthetic Benchmark Analysis

### Overall Label Distribution

| Metric | Measured Value | Percentage |
| :--- | :---: | :---: |
| **Total Modeling Instances** | 6,675 | 100.0% |
| **Positive Labels (`target_30d = 1`)** | 6,675 | 100.0% |
| **Negative Labels (`target_30d = 0`)** | 0 | 0.0% |
| **Positive Prevalence** | **1.000000** | **100.0%** |

### Partition Breakdown

| Partition | Cutoffs | Total Instances | Positive Count | Negative Count | Prevalence |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Train** | 309 | 4,635 | 4,635 | 0 | 100.0% |
| **Validation** | 74 | 1,110 | 1,110 | 0 | 100.0% |
| **Test** | 62 | 930 | 930 | 0 | 100.0% |
| **Total** | **445** | **6,675** | **6,675** | **0** | **100.0%** |

### Competitor Breakdown

| Competitor | Total Instances | Positive Count | Negative Count | Prevalence | Mean Window Event Count | Window Count Range [Min, Max] |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Amazon** | 2,225 | 2,225 | 0 | 100.0% | 90.81 | [41, 152] |
| **Walmart** | 2,225 | 2,225 | 0 | 100.0% | 79.49 | [32, 133] |
| **eBay** | 2,225 | 2,225 | 0 | 100.0% | 56.29 | [23, 93] |

### Strategic Cluster Breakdown

| Cluster ID | Cluster Name | Instances | Positive Count | Negative Count | Prevalence | Mean Window Event Count | Window Count Range [Min, Max] |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | Pricing, Promotions & Retail Media | 1,335 | 1,335 | 0 | 100.0% | 73.80 | [31, 113] |
| **2** | Logistics, Fulfillment & Expansion | 1,335 | 1,335 | 0 | 100.0% | 64.12 | [31, 101] |
| **3** | Product, AI & Platform Ecosystem | 1,335 | 1,335 | 0 | 100.0% | 96.97 | [51, 152] |
| **4** | Marketplace & Seller Operations | 1,335 | 1,335 | 0 | 100.0% | 51.24 | [23, 84] |
| **5** | Corporate Strategy, M&A & Workforce | 1,335 | 1,335 | 0 | 100.0% | 91.51 | [43, 143] |

### Partition $\times$ Competitor Cross-Tabulation

| Partition | Competitor | Total Instances | Positive Count | Negative Count | Prevalence | Mean Window Event Count |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Train** | Amazon | 1,545 | 1,545 | 0 | 100.0% | 90.65 |
| **Train** | Walmart | 1,545 | 1,545 | 0 | 100.0% | 79.37 |
| **Train** | eBay | 1,545 | 1,545 | 0 | 100.0% | 56.36 |
| **Validation** | Amazon | 370 | 370 | 0 | 100.0% | 91.07 |
| **Validation** | Walmart | 370 | 370 | 0 | 100.0% | 79.79 |
| **Validation** | eBay | 370 | 370 | 0 | 100.0% | 56.28 |
| **Test** | Amazon | 310 | 310 | 0 | 100.0% | 91.29 |
| **Test** | Walmart | 310 | 310 | 0 | 100.0% | 79.74 |
| **Test** | eBay | 310 | 310 | 0 | 100.0% | 55.97 |

### Explanation of 100% Empirical Prevalence on Synthetic Data

The 100% empirical positive prevalence observed across all 6,675 instances is an expected mathematical consequence of the synthetic event generator:

1. **Uniform High Density:** The raw dataset contains 121,000 events distributed across 3,204 calendar days (an average of 37.7 events per day).
2. **Dense Competitor-Cluster Coverage:** Across 3 competitors and 5 strategic clusters (15 combinations), each combination averages 8,066 events over the dataset period (approximately 2.5 events per day).
3. **Window Count Distribution:** Over any 30-day window, the expected number of events for any competitor-cluster pair is $\approx 75$ events. Across all 6,675 forward windows, the **minimum observed count is 23 events** (observed for eBay in Cluster 4), while the maximum is 152 events (Amazon in Cluster 3).
4. **Implication:** Because the minimum count across all windows is 23 ($> 0$), every single window contains at least one event.
5. **Engineering Verification:** Because the synthetic data has high uniform event frequency, unit tests using small in-memory fixtures are essential to prove that the pipeline correctly assigns negative labels ($0$) when windows are empty or events fall outside boundaries.

---

## 6. Automated Test Suite Results

The automated test suite in [tests/labels/test_label_builder.py](file:///d:/NextMove/tests/labels/test_label_builder.py) was executed alongside all existing project tests.

### Test Execution Command
```bash
.venv/Scripts/python.exe -m pytest tests -v
```

### Complete Test Results (43 / 43 Passed in 6.34s)

```
tests/labels/test_label_builder.py::test_first_and_last_day_of_window_inclusion PASSED [  2%]
tests/labels/test_label_builder.py::test_immediately_outside_window_exclusion PASSED [  4%]
tests/labels/test_label_builder.py::test_empty_windows_produce_label_zero PASSED [  6%]
tests/labels/test_label_builder.py::test_competitor_and_cluster_isolation PASSED [  9%]
tests/labels/test_label_builder.py::test_binary_labels_only_even_with_multiple_events PASSED [ 11%]
tests/labels/test_label_builder.py::test_rejection_of_truncated_windows PASSED [ 13%]
tests/labels/test_label_builder.py::test_join_features_and_labels_exact_match PASSED [ 16%]
tests/labels/test_label_builder.py::test_join_rejects_duplicate_keys PASSED [ 18%]
tests/labels/test_label_builder.py::test_join_rejects_missing_or_orphan_keys PASSED [ 20%]
tests/labels/test_label_builder.py::test_join_rejects_unexpected_competitor_or_invalid_cluster PASSED [ 23%]
tests/labels/test_label_builder.py::test_no_change_to_existing_feature_values_after_joining PASSED [ 25%]
tests/labels/test_label_builder.py::test_deterministic_label_output_across_repeated_runs PASSED [ 27%]
tests/labels/test_label_builder.py::test_all_6675_expected_keys_appear_exactly_once PASSED [ 30%]
tests/temporal_leakage/test_feature_builder.py::test_feature_matrix_combination_uniqueness_and_completeness PASSED [ 32%]
tests/temporal_leakage/test_feature_builder.py::test_expected_feature_columns_and_data_types PASSED [ 34%]
tests/temporal_leakage/test_feature_builder.py::test_exact_calculations_on_hand_built_fixture PASSED [ 37%]
tests/temporal_leakage/test_feature_builder.py::test_exclusion_of_events_on_and_after_cutoff PASSED [ 39%]
tests/temporal_leakage/test_feature_builder.py::test_missing_history_and_recency_sentinel PASSED [ 41%]
tests/temporal_leakage/test_feature_builder.py::test_rival_filtering PASSED [ 44%]
tests/temporal_leakage/test_feature_builder.py::test_calendar_features PASSED [ 46%]
tests/temporal_leakage/test_feature_builder.py::test_repeatability_across_repeated_runs PASSED [ 48%]
tests/temporal_leakage/test_temporal_integrity.py::test_taxonomy_mapping_completeness_and_uniqueness PASSED [ 51%]
tests/temporal_leakage/test_temporal_integrity.py::test_cluster_derivation_from_event_type PASSED [ 53%]
tests/temporal_leakage/test_temporal_integrity.py::test_partition_cutoff_counts_and_boundaries PASSED [ 55%]
tests/temporal_leakage/test_temporal_integrity.py::test_partition_embargo_isolation PASSED [ 58%]
tests/temporal_leakage/test_temporal_integrity.py::test_feature_exclusion_on_and_after_cutoff_date PASSED [ 60%]
tests/temporal_leakage/test_temporal_integrity.py::test_label_window_first_and_last_day_inclusion PASSED [ 62%]
tests/temporal_leakage/test_temporal_integrity.py::test_label_window_exclusion_one_day_before_and_after PASSED [ 65%]
tests/temporal_leakage/test_temporal_integrity.py::test_empty_event_windows_produce_label_zero PASSED [ 67%]
tests/temporal_leakage/test_temporal_integrity.py::test_competitor_and_cluster_filtering PASSED [ 69%]
tests/temporal_leakage/test_temporal_integrity.py::test_rejection_of_incomplete_forward_windows PASSED [ 72%]
tests/temporal_leakage/test_temporal_integrity.py::test_deterministic_results_across_runs PASSED [ 74%]
tests/unit/test_dataset_schema.py::test_row_and_column_counts PASSED     [ 76%]
tests/unit/test_dataset_schema.py::test_column_names_match_expected PASSED [ 79%]
tests/unit/test_dataset_schema.py::test_event_id_uniqueness_and_validity PASSED [ 81%]
tests/unit/test_dataset_schema.py::test_event_date_parseability_and_bounds PASSED [ 83%]
tests/unit/test_dataset_schema.py::test_zero_missing_or_empty_values PASSED [ 86%]
tests/unit/test_dataset_schema.py::test_no_exact_duplicate_rows PASSED   [ 88%]
tests/unit/test_dataset_schema.py::test_categorical_domains PASSED       [ 90%]
tests/unit/test_dataset_schema.py::test_numeric_ranges_and_validity PASSED [ 93%]
tests/unit/test_dataset_schema.py::test_synthetic_metadata_labels PASSED [ 95%]
tests/unit/test_dataset_schema.py::test_generator_seed_consistency PASSED [ 97%]
tests/unit/test_description_template_prefix PASSED [100%]

============================= 43 passed in 6.34s ==============================
```

### Coverage of Step 6 Test Requirements

1. **Events on First ($T$) and Last ($T+29$) Days:** Verified by `test_first_and_last_day_of_window_inclusion` (both trigger $1$).
2. **Events Immediately Outside ($T-1$ and $T+30$):** Verified by `test_immediately_outside_window_exclusion` (both evaluate to $0$).
3. **Empty Windows Producing 0:** Verified by `test_empty_windows_produce_label_zero`.
4. **Competitor and Cluster Isolation:** Verified by `test_competitor_and_cluster_isolation` (an event for Walmart in Cluster 3 does not activate Amazon or any other cluster).
5. **All 6,675 Expected Keys Exactly Once:** Verified by `test_all_6675_expected_keys_appear_exactly_once`.
6. **Binary Labels Only:** Verified by `test_binary_labels_only_even_with_multiple_events` and full grid assertions ($\text{target\_30d} \in \{0, 1\}$).
7. **Rejection of Truncated Windows:** Verified by `test_rejection_of_truncated_windows` (cutoffs after `2026-09-07` raise `ValueError`).
8. **Correct Feature-Label Join Keys:** Verified by `test_join_features_and_labels_exact_match`, `test_join_rejects_duplicate_keys`, `test_join_rejects_missing_or_orphan_keys`, and `test_join_rejects_unexpected_competitor_or_invalid_cluster`.
9. **No Change to Existing Feature Values:** Verified by `test_no_change_to_existing_feature_values_after_joining`.
10. **Deterministic Output Across Repeated Runs:** Verified by `test_deterministic_label_output_across_repeated_runs`.

---

## 7. Next Steps & Gate Verification

Step 6 has satisfied all requirements specified in the project roadmap:
- The target label generation module is fully implemented in [src/labels/label_builder.py](file:///d:/NextMove/src/labels/label_builder.py).
- The modeling dataset artifact is materialized at [data/synthetic/processed/synthetic_modeling_dataset_445_cutoffs_6675_instances.csv](file:///d:/NextMove/data/synthetic/processed/synthetic_modeling_dataset_445_cutoffs_6675_instances.csv).
- All 43 unit, temporal leakage, and label integrity tests pass cleanly.
- Statistical telemetry and distributions are fully documented.
- No model training or evaluation was performed. Execution stops here as instructed.
