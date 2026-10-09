# NextMove — Step 4: Dataset Contract & Temporal Integrity Report

**Document Path:** `reports/step4_temporal_integrity_report.md`  
**Execution Date:** 2026-10-09  
**Status:** **STEP 4 COMPLETED — ALL 22 TESTS PASSING (0 FAILURES)**  
**Auditor / Subsystem:** NextMove Architecture & Temporal Validation  

---

## 1. Executive Summary

Step 4 implements the production dataset contract, centralized configuration, and automated temporal leakage test suite for the NextMove competitive intelligence platform. 

Prior to code implementation, [`reports/prediction_target_spec.md`](file:///d:/NextMove/reports/prediction_target_spec.md) was corrected by removing the unsupported claim that Monday 00:00 UTC guarantees a 24–48-hour settling buffer for weekend announcements, while preserving the approved Convention A rules and formalizing the date-only availability limitations.

All temporal utilities and automated tests were executed via `pytest`. **All 22 automated tests (11 temporal leakage tests + 11 schema tests) passed in 3.14 seconds with 0 failures.** The raw dataset ([`recall_intel_121000_independent_synthetic_events.csv`](file:///d:/NextMove/data/synthetic/raw/recall_intel_121000_independent_synthetic_events.csv)) remains completely untouched (exact size: 48,002,803 bytes).

---

## 2. Specification Correction

In [`reports/prediction_target_spec.md`](file:///d:/NextMove/reports/prediction_target_spec.md) (Section 4.3), the sentence:
> *"Convention A's weekend boundary mitigates this by providing a 24–48h settling buffer for Friday/weekend announcements before Monday 00:00 UTC."*

was removed and replaced with the verified operational reality:
> *"Because disclosures occurring over the weekend or late Friday may experience publication, crawling, or indexing delays in live operations, date-only data does not guarantee that weekend announcements were publicly accessible prior to Monday 00:00 UTC. In production competitive intelligence, real-world forecasting claims require immutable publication timestamps (`published_at`) or crawling ingestion timestamps (`ingested_at`). For this prototype, Convention A enforces strict calendar-day boundaries ($\le T_{\text{date}} - 1$) as a formal date-level contract, acknowledging that actual point-in-time public availability cannot be established from calendar dates alone."*

---

## 3. Implemented Files & Code Architecture

### 3.1 Centralized Configuration: [`configs/prediction_config.py`](file:///d:/NextMove/configs/prediction_config.py)
A single immutable configuration dataclass (`PredictionConfig`) centralizes all task parameters:
- **Raw CSV Path:** `Path("data/synthetic/raw/recall_intel_121000_independent_synthetic_events.csv")`
- **Competitors:** `["Amazon", "Walmart", "eBay"]`
- **Taxonomy Mapping:** Complete bidirectional mappings (`EVENT_TYPE_TO_CLUSTER` and `CLUSTER_TO_EVENT_TYPES`) for all 30 distinct event types across the 5 approved strategic clusters.
- **Convention A Temporal Parameters:** Start-of-day cutoff (Monday 00:00:00 UTC), 30-day inclusive forward label window, and max eligible cutoff `2026-09-07`.
- **Partition Boundaries:** Exact cutoff boundaries and expected counts (Train: 309, Val: 74, Test: 62 = 445 total Mondays; 6,675 supervised target instances).
- **Self-Validation:** `config.validate()` asserts 30 mapped event types, cluster ID range [1, 5], bi-directional mapping identity, and instance count consistency.

### 3.2 Deterministic Temporal Utilities: [`src/splitting/temporal.py`](file:///d:/NextMove/src/splitting/temporal.py)
Provides clean, reusable functions exported via [`src/splitting/__init__.py`](file:///d:/NextMove/src/splitting/__init__.py):
- `normalize_to_date(ts)`: Normalizes timestamps/strings to UTC date level.
- `map_event_type_to_cluster(event_type)`: Deterministic single-event type mapping.
- `assign_strategic_clusters(df)`: Derives `strategic_cluster` column from `event_type` without assuming a pre-existing column in the raw CSV.
- `generate_weekly_monday_cutoffs(start, end, max_dataset_date)`: Generates Mondays and rejects ranges whose forward window exceeds max dataset date.
- `get_partition_cutoffs(partition)`: Retrieves verified Monday cutoffs for "train", "validation", or "test", asserting exact expected counts.
- `get_label_window_bounds(cutoff_date, horizon_days, max_dataset_date)`: Defines inclusive $[T, T + 29 \text{ days}]$ window; raises `ValueError` if exceeding dataset max date.
- `filter_feature_events(df, cutoff_date)`: Enforces Convention A by asserting $t_e < T_{\text{date}}$ (strictly excluding events on or after cutoff).
- `compute_binary_window_label(...)`: Evaluates binary label ($1$ or $0$) over $[T, T + 29]$ independently of event presence (empty sets produce $0$ without error).

### 3.3 Test Runner Configuration: [`pytest.ini`](file:///d:/NextMove/pytest.ini)
Configures `testpaths = tests` and `pythonpath = .` to isolate NextMove tests from third-party vendor directories and ensure clean root-level module imports.

### 3.4 Automated Test Suite: [`tests/temporal_leakage/test_temporal_integrity.py`](file:///d:/NextMove/tests/temporal_leakage/test_temporal_integrity.py)
Implements 11 isolated unit tests using fast in-memory synthetic fixtures to verify all temporal and taxonomy contracts.

---

## 4. Test Suite Execution Results

Ran complete test suite via `pytest -v` across all test directories:

```text
============================= test session starts =============================
platform win32 -- Python 3.14.5, pytest-9.1.1, pluggy-1.6.0
rootdir: D:\NextMove
configfile: pytest.ini
collected 22 items

tests/temporal_leakage/test_temporal_integrity.py::test_taxonomy_mapping_completeness_and_uniqueness PASSED [  4%]
tests/temporal_leakage/test_temporal_integrity.py::test_cluster_derivation_from_event_type PASSED [  9%]
tests/temporal_leakage/test_temporal_integrity.py::test_partition_cutoff_counts_and_boundaries PASSED [ 13%]
tests/temporal_leakage/test_temporal_integrity.py::test_partition_embargo_isolation PASSED [ 18%]
tests/temporal_leakage/test_temporal_integrity.py::test_feature_exclusion_on_and_after_cutoff_date PASSED [ 22%]
tests/temporal_leakage/test_temporal_integrity.py::test_label_window_first_and_last_day_inclusion PASSED [ 27%]
tests/temporal_leakage/test_temporal_integrity.py::test_label_window_exclusion_one_day_before_and_after PASSED [ 31%]
tests/temporal_leakage/test_temporal_integrity.py::test_empty_event_windows_produce_label_zero PASSED [ 36%]
tests/temporal_leakage/test_temporal_integrity.py::test_competitor_and_cluster_filtering PASSED [ 40%]
tests/temporal_leakage/test_temporal_integrity.py::test_rejection_of_incomplete_forward_windows PASSED [ 45%]
tests/temporal_leakage/test_temporal_integrity.py::test_deterministic_results_across_runs PASSED [ 50%]
tests/unit/test_dataset_schema.py::test_row_and_column_counts PASSED     [ 54%]
tests/unit/test_dataset_schema.py::test_column_names_match_expected PASSED [ 59%]
tests/unit/test_dataset_schema.py::test_event_id_uniqueness_and_validity PASSED [ 63%]
tests/unit/test_dataset_schema.py::test_event_date_parseability_and_bounds PASSED [ 68%]
tests/unit/test_dataset_schema.py::test_zero_missing_or_empty_values PASSED [ 72%]
tests/unit/test_dataset_schema.py::test_no_exact_duplicate_rows PASSED   [ 77%]
tests/unit/test_dataset_schema.py::test_categorical_domains PASSED       [ 81%]
tests/unit/test_dataset_schema.py::test_numeric_ranges_and_validity PASSED [ 86%]
tests/unit/test_dataset_schema.py::test_synthetic_metadata_labels PASSED [ 90%]
tests/unit/test_dataset_schema.py::test_generator_seed_consistency PASSED [ 95%]
tests/unit/test_dataset_schema.py::test_description_template_prefix PASSED [100%]

============================= 22 passed in 3.14s ==============================
```

- **Total Tests Collected:** 22
- **Total Tests Passed:** 22 (100%)
- **Failures / Errors:** 0
- **Execution Duration:** 3.14 seconds

---

## 5. Formally Verified Invariants

| Invariant | Test Verification Method | Result |
| :--- | :--- | :---: |
| **Taxonomy Completeness** | Exactly 30 event types map to 1..5; reverse mapping matches; cluster assignment derives `strategic_cluster` dynamically. | **VERIFIED** |
| **Cutoff Counts** | Train: 309 Mondays, Validation: 74 Mondays, Test: 62 Mondays (Total: 445 Mondays $\times$ 15 = 6,675 instances). | **VERIFIED** |
| **Feature Boundary (Convention A)** | `filter_feature_events` excludes events dated on cutoff Monday $T$ and future dates; features query strictly $\le T - 1$. | **VERIFIED** |
| **Label Window Boundary** | Events on day $T$ and day $T + 29$ produce label `1`; events on $T - 1$ and $T + 30$ produce label `0`. | **VERIFIED** |
| **Empty Window Handling** | Windows with zero matching events produce label `0` cleanly without error. | **VERIFIED** |
| **Competitor & Cluster Isolation** | Cross-competitor and cross-cluster events inside window do not trigger false positive labels for target pair. | **VERIFIED** |
| **Right-Censoring Rejection** | Cutoff `2026-09-14` (window ends `2026-10-13` > `2026-10-09`) raises `ValueError`. Final cutoff is strictly `2026-09-07`. | **VERIFIED** |
| **Embargo Isolation** | Train label end (`2023-12-26`) to Val cutoff (`2024-01-01`): **5-day gap**. Val label end (`2025-06-24`) to Test cutoff (`2025-07-07`): **12-day gap**. Zero cross-boundary label overlap. | **VERIFIED** |
| **Deterministic Consistency** | Grid generation, feature masking, and label computation produce identical outputs across repeated runs. | **VERIFIED** |
| **Raw CSV Immutability** | Measured file size remains exactly 48,002,803 bytes. | **VERIFIED** |

---

## 6. Operating Boundaries & Remaining Limitations

1. **Date-Only Record Limitation:** Calendar-day boundaries ($\le T_{\text{date}} - 1$) enforce strict mathematical isolation, but cannot resolve intra-day publication time-of-day or scraping latency. Real-world commercial forecasting claims require immutable `published_at` and `ingested_at` logging.
2. **Synthetic Data Constraint:** The synthetic dataset validates pipeline mechanics, join logic, and temporal isolation contracts. Because synthetic attributes are statistically uncoupled (correlations $\approx 0.000$) and template-generated, models trained on this data test software engineering integrity, not real-world predictive foresight.
3. **Current State:** No models have been trained, no full feature store has been materialized, and no production label tables have been generated. Implementation is halted pending Step 5 authorization.
