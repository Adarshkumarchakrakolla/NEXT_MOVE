# Logical Review and Defect Analysis: Prediction Target Specification

**Document Reviewed:** [`reports/prediction_target_spec.md`](file:///d:/NextMove/reports/prediction_target_spec.md)  
**Review Date:** 2026-10-09  
**Review Type:** Pre-Implementation Verification & Correctness Audit  
**Auditor:** NextMove Architecture & Modeling Subsystem  
**Review Status:** **AMENDMENTS REQUIRED (LOGICAL DEFECTS IDENTIFIED)**  

---

## Executive Summary of Findings

A systematic logical correctness review was conducted on the proposed prediction-target specification for the NextMove competitive intelligence platform. While the core decision to frame the task as a **30-day forward window strategic move probability** is methodologically sound and superior to next-event micro-classification, the review identified **three significant defects** and **two areas requiring formalization** before implementation:

1. **Defect 1 (Taxonomy Incompleteness):** The specification omitted **6 of the 30 distinct event types** (`advertising`, `data_service`, `international_expansion`, `partnership`, `payments`, `retail_media`), which collectively account for **22,429 rows (18.54% of the dataset)**.
2. **Defect 2 (Undefined Edge Handling in Label Construction):** The specification did not formally define right-censored incomplete forward windows (cutoff dates within 30 days of dataset or split boundaries) or multi-event day tie-breaking logic.
3. **Defect 3 (Cutoff-Time Ambiguity & Overstated Embargo Sufficiency):** The proposed observation grid used Monday 00:00 UTC while features included events on cutoff date $T$. Conditioning on Monday's events at Monday 00:00 UTC is lookahead leakage. Furthermore, a 30-day temporal embargo alone does not prevent global transformer leakage or event-triggered sampling leakage.
4. **Formalization Need 1 (Synthetic vs. Real Boundary):** The functional boundary between what the synthetic dataset can prove vs. what requires evidence-backed data was described conceptually but needs concrete capability criteria.
5. **Formalization Need 2 (Exact Baseline Formulations):** The benchmark baselines (Historical Base-Rate and Trailing-Momentum) lacked exact mathematical formulas and formal decision-threshold comparison metrics.

---

## 1. Complete Event Taxonomy Audit

### 1.1 Findings & Measured Defect
In Section 4.1 of `prediction_target_spec.md`, 24 event types were partitioned across 5 clusters. A cross-reference against the actual dataset ([`recall_intel_121000_independent_synthetic_events.csv`](file:///d:/NextMove/data/synthetic/raw/recall_intel_121000_independent_synthetic_events.csv)) reveals **30 distinct event types**:

| Status | Count | Event Types | Total Records Affected |
| :--- | :---: | :--- | :---: |
| **Mapped in Spec** | 24 | `pricing_change`, `promotion`, `membership`, `delivery`, `fulfillment`, `logistics_expansion`, `store_expansion`, `product_launch`, `feature_launch`, `ai_launch`, `app_update`, `live_commerce`, `marketplace_change`, `seller_program`, `trust_safety`, `authentication`, `earnings`, `hiring`, `restructuring`, `leadership_change`, `acquisition`, `investment`, `regulatory`, `sustainability` | 98,571 (81.46%) |
| **Unmapped (Omitted)** | **6** | `advertising`, `data_service`, `international_expansion`, `partnership`, `payments`, `retail_media` | **22,429 (18.54%)** |

Silently discarding 18.54% of the dataset would severely bias competitor activity monitoring, especially in monetization and tech infrastructure where Amazon and Walmart frequently announce ad tech and payment initiatives.

### 1.2 Unmapped Category Breakdown
- `partnership`: 5,261 rows (4.35%)
- `advertising`: 4,675 rows (3.86%)
- `data_service`: 3,789 rows (3.13%)
- `international_expansion`: 3,051 rows (2.52%)
- `payments`: 2,926 rows (2.42%)
- `retail_media`: 2,727 rows (2.25%)

### 1.3 Recommended Taxonomy Correction (100% Complete Mapping)
To achieve complete coverage without expanding cluster count beyond executive interpretability, the 30 event types should be mapped into **5 Expanded Strategic Action Clusters**:

| Cluster ID | Cluster Name | Event Types Included (All 30 Covered) | Record Count | Dataset Share |
| :---: | :--- | :--- | :---: | :---: |
| **Cluster 1** | **Pricing, Promotions & Retail Media** | `pricing_change`, `promotion`, `membership`, `advertising`, `retail_media` | 23,660 | 19.55% |
| **Cluster 2** | **Logistics, Fulfillment & Expansion** | `delivery`, `fulfillment`, `logistics_expansion`, `store_expansion`, `international_expansion` | 20,554 | 16.99% |
| **Cluster 3** | **Product, AI & Platform Ecosystem** | `product_launch`, `feature_launch`, `ai_launch`, `app_update`, `live_commerce`, `data_service`, `payments` | 27,991 | 23.13% |
| **Cluster 4** | **Marketplace & Seller Operations** | `marketplace_change`, `seller_program`, `trust_safety`, `authentication` | 16,438 | 13.59% |
| **Cluster 5** | **Corporate Strategy, M&A & Workforce** | `earnings`, `hiring`, `restructuring`, `leadership_change`, `acquisition`, `investment`, `regulatory`, `sustainability`, `partnership` | 32,357 | 26.74% |
| **Total** | **All 5 Clusters** | **30 of 30 Distinct Event Types** | **121,000** | **100.00%** |

---

## 2. Label Construction & Edge-Case Protocol

### 2.1 Formal Binary Label Definition
Under the recommended **Start-of-Day Cutoff Convention (Convention A)**, for observation cutoff Monday $T_{\text{date}}$ at 00:00:00 UTC, competitor $C \in \{\text{Amazon}, \text{Walmart}, \text{eBay}\}$, and strategic cluster $K \in \{1, 2, 3, 4, 5\}$:

$$Y_{C, K, T} = \begin{cases} 
1 & \text{if } \sum_{e \in \mathcal{E}} \mathbb{I}(c_e = C \land k_e = K \land T_{\text{date}} \le t_e \le T_{\text{date}} + 29) \ge 1 \\
0 & \text{if } \sum_{e \in \mathcal{E}} \mathbb{I}(c_e = C \land k_e = K \land T_{\text{date}} \le t_e \le T_{\text{date}} + 29) = 0
\end{cases}$$

### 2.2 Edge Cases & Formal Handling Rules

#### A. Incomplete Future Windows (Right-Censoring)
- **Problem:** If cutoff date $T$ is within 30 days of the end of the available dataset ($T + 29 > \text{2026-10-09}$), the forward observation window is truncated. Truncated windows artificially depress positive event counts and cause false-negative bias.
- **Rule:** Any cutoff date $T$ where $T + 29 > \text{2026-10-09}$ ($T > \text{2026-09-10}$) is **strictly ineligible for label generation**. The final allowable cutoff Monday is strictly **`2026-09-07`**.

#### B. Incomplete Windows at Partition Boundaries
- **Rule:** The cutoff selection must ensure that the label window of the last training cutoff $[T_{\text{train\_last}}, T_{\text{train\_last}} + 29]$ **does not extend into the validation partition**. 
  - If Validation starts on Monday `2024-01-01`, then $T_{\text{train\_last}} + 29 \le \text{2023-12-31}$, which is satisfied by $T_{\text{train\_last}} = \text{2023-11-27}$ ($T_{\text{train\_last}} + 29 = \text{2023-12-26}$).

#### C. Duplicate and Overlapping Events
- **Intra-Day Event Collisions:** If competitor $C$ logs multiple events in cluster $K$ on the same day within $[T_{\text{date}}, T_{\text{date}} + 29]$, $Y_{C, K, T} = 1$. The target is strictly binary existence, immune to intra-day duplicate count noise.
- **Cross-Sectional Overlapping Windows:** When cutoffs are sampled on a weekly grid ($T_0, T_1 = T_0 + 7, \dots$), the forward window $[T_0, T_0 + 29]$ shares 23 days with $[T_1, T_1 + 29]$. This is standard rolling-window panel structure. It does **not** constitute temporal leakage as long as chronological split boundaries and embargoes are strictly maintained.

---

## 3. Point-in-Time Features & Leakage Defense

### 3.1 Strict Historical Temporal Boundaries (Convention A: Start-of-Day)
Every feature vector $X_{C, K, T}$ computed for observation cutoff Monday $T_{\text{date}}$ at 00:00:00 UTC must satisfy:
$$\mathcal{F}_T = \{ e \in \mathcal{E} \mid t_e \le T_{\text{date}} - 1 \text{ day} \}$$
Under no circumstances may any transformation query $t_e \ge T_{\text{date}}$. Features query strictly completed days up to Sunday ($T_{\text{date}} - 1$).

### 3.2 Specific Lookback Feature Boundaries
1. **Recency Features:**
   $$\Delta t_{\text{last}}(C, K, T) = (T_{\text{date}} - 1) - \max(\{ t_e \mid c_e = C \land k_e = K \land t_e \le T_{\text{date}} - 1 \} \cup \{ T_0 \})$$
   If no prior event exists in the historical log, $\Delta t$ must be assigned a fixed sentinel value ($999.0$ days) along with a binary indicator `is_first_observed_event = 1`.
2. **Rolling Velocity Windows:**
   - 30-Day Velocity: Count of qualifying events in interval $[T_{\text{date}} - 30, T_{\text{date}} - 1]$ (inclusive of Sunday, length = 30 days).
   - 90-Day Velocity: Count of qualifying events in interval $[T_{\text{date}} - 90, T_{\text{date}} - 1]$ (length = 90 days).
   - 365-Day Baseline: Count of qualifying events in interval $[T_{\text{date}} - 365, T_{\text{date}} - 1]$.
   *Crucial Rule:* The forward prediction window starts on Monday $T_{\text{date}}$ and spans $[T_{\text{date}}, T_{\text{date}} + 29]$. Thus, the feature window $[T_{\text{date}} - 30, T_{\text{date}} - 1]$ and label window $[T_{\text{date}}, T_{\text{date}} + 29]$ share **zero** overlapping days.
3. **Cross-Competitor Rivalry Features:**
   - Rival reaction velocity over $[T_{\text{date}} - 14, T_{\text{date}} - 1]$ (length = 14 days) across $c_e \ne C$.

### 3.3 Transformation Leakage Guard
Global transformations fit over the full dataset (e.g. `StandardScaler().fit(X_all)`, target encoding, or imputer medians) leak future distribution parameters into the past.
- **Mandatory Protocol:** All statistical feature scaling and encoders must be fit **exclusively on the training partition cutoffs** ($\le T_{\text{train\_last}}$) and applied out-of-sample to validation and test partitions.

---

## 4. Temporal Split Integrity & Embargo Verification

### 4.1 Multi-Layered Leakage Defense
An embargo between partition cutoffs is necessary, but **not sufficient**. Four additional failure modes are guarded:
1. **Global Preprocessing Leakage:** Scalers and encoders fit strictly on Train.
2. **Event-Triggered Cutoff Sampling:** Cutoffs placed on an exogenous weekly calendar grid (Mondays at 00:00 UTC), not on event arrival dates.
3. **Cross-Cluster Target Encoding:** No out-of-sample target labels used in feature engineering.
4. **Fine-Tuning on Future Text:** No representation learning on text after $T_{\text{train\_last}}$.

### 4.2 Formalized Chronological Partitions (Convention A: 445 Eligible Mondays)
Assuming a weekly observation grid sampled every **Monday at 00:00:00 UTC**:

```
[------------- TRAIN -------------] [EMBARGO 1] [---------- VALIDATION ----------] [EMBARGO 2] [------------- TEST -------------]
2018-01-01              2023-11-27    34 Days    2024-01-01              2025-05-26    41 Days   2025-07-07              2026-09-07
Label Window up to:     2023-12-26               Label Window up to:     2025-06-24              Label Window up to:     2026-10-06
```

| Partition | Start Cutoff $T_{\text{start}}$ | End Cutoff $T_{\text{end}}$ | Max Feature Date | Max Label Date Observed | Eligible Mondays | Supervised Instances (3 comp $\times$ 5 clusters) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Train** | `2018-01-01` | `2023-11-27` | `2023-11-26` | `2023-12-26` | 309 | 4,635 |
| **Embargo 1** | `2023-11-28` | `2023-12-31` | N/A | *Buffer ensures Train labels close by 2023-12-26* | 0 | 0 |
| **Validation** | `2024-01-01` | `2025-05-26` | `2025-05-25` | `2025-06-24` | 74 | 1,110 |
| **Embargo 2** | `2025-05-27` | `2025-07-06` | N/A | *Buffer ensures Val labels close by 2025-06-24* | 0 | 0 |
| **Test (OOT)** | `2025-07-07` | `2026-09-07` | `2026-09-06` | `2026-10-06` | 62 | 930 |
| **Total** | | | | | **445 cutoffs** | **6,675 instances** |

*Verification of Zero Overlap:*
- Train Label End: `2023-12-26` < Validation Cutoff Start: `2024-01-01` (Gap: 5 days).
- Val Label End: `2025-06-24` < Test Cutoff Start: `2025-07-07` (Gap: 12 days).
- Test Label End: `2026-10-06` $\le$ Dataset Max: `2026-10-09` (Zero truncated label windows).

---

## 5. Dataset Suitability: Synthetic vs. Evidence-Backed Reality

```mermaid
graph TD
    subgraph Synthetic Pipeline [Synthetic Experimentation: data/synthetic/]
        S1[Point-in-Time Rolling Feature Extraction]
        S2[Temporal Split & Embargo Validation]
        S3[Leakage Detection Test Suites]
        S4[Probability Calibration Algorithms]
        S5[Pipeline Architecture Contract]
    end

    subgraph Evidence Pipeline [Evidence-Backed Real World: data/evidence/]
        E1[Empirical Competitor Retaliation Lags]
        E2[Authentic Macroeconomic Seasonality & Shocks]
        E3[Semantic Features from Unstructured 8-K/Press Releases]
        E4[Production Strategic Competitive Forecasting]
        E5[True Model Outperformance over Trailing Base-Rates]
    end

    S5 -.->|Contracts & Fixtures Only| Evidence Pipeline
```

| Dimension | Supported by Synthetic Dataset (`data/synthetic/`) | Requires Real Evidence-Backed Data (`data/evidence/`) |
| :--- | :--- | :--- |
| **Data Pipeline Testing** | **YES.** End-to-end rolling-window aggregations, point-in-time joins, and caching. | No, identical interface. |
| **Leakage Defense Verification** | **YES.** `tests/temporal_leakage/` fixtures verify that future timestamps are unreachable. | No, identical interface. |
| **Model Code Execution** | **YES.** XGBoost/scikit-learn training, prediction inference, and logging pipelines run cleanly. | No, identical interface. |
| **Calibration Verification** | **YES.** Testing Platt scaling and Isotonic regression calibration scripts. | No, identical interface. |
| **Competitive Retaliation Signal** | **NO.** Synthetic features have Pearson correlation $\approx 0.000$; rivals do not react to each other. | **YES.** Real pricing wars and delivery expansions exhibit authentic cross-firm causality. |
| **Realistic Event Pacing** | **NO.** Events arrive at a constant ~38/day rate with zero seasonality. | **YES.** Real corporate disclosures show burstiness around earnings calls and holiday quarters. |
| **NLP / Semantic Extraction** | **NO.** 100% of descriptions follow a single rigid slot-filled template. | **YES.** Real press releases and SEC filings contain nuanced strategic intent. |
| **True Model Superiority Claim** | **NO.** Models cannot genuinely outperform a simple Poisson prior on independent random data. | **YES.** Models must beat baseline momentum on real empirical events. |

---

## 6. Baseline Models & Performance Benchmark Specifications

### 6.1 Baseline 1: Historical Base-Rate Predictor (Stationary Prior)
- **Definition:** Outputs the historical empirical positive prevalence observed across the training partition:
  $$\hat{p}_{\text{base}}(C, K) = \frac{1}{|T_{\text{train}}|} \sum_{T \in T_{\text{train}}} Y_{C, K, T}$$
- **Characteristics:** Constant probability output per $(C, K)$ across all evaluation cutoffs $T$. Sets the lower bound for PR-AUC.

### 6.2 Baseline 2: Trailing-Momentum Predictor (Dynamic Empirical Inertia)
- **Definition:** Computes competitor $C$'s event arrival intensity in cluster $K$ over trailing 90 days $[T_{\text{date}} - 90, T_{\text{date}} - 1]$, projecting forward probability via Poisson arrival:
  $$\hat{\lambda}_{C, K, T} = \frac{\sum_{t_e \in [T_{\text{date}}-90, T_{\text{date}}-1]} \mathbb{I}(c_e = C \land k_e = K)}{90}$$
  $$\hat{p}_{\text{momentum}}(C, K, T) = 1 - \exp\left(-30 \cdot \hat{\lambda}_{C, K, T}\right)$$

### 6.3 Evaluation Metrics and Benchmark Comparison Protocol

| Metric | Formulation | Benchmark Criterion | Practical Meaning for Competitive Intel |
| :--- | :--- | :--- | :--- |
| **PR-AUC (Average Precision)** | $\sum_n (R_n - R_{n-1}) P_n$ | **$\text{PR-AUC}_{\text{model}} > \text{PR-AUC}_{\text{momentum}} > \text{PR-AUC}_{\text{base}}$** | Evaluates ranking quality on positive strategic actions; robust to class imbalance. |
| **Brier Score (BS)** | $\frac{1}{N} \sum_{i=1}^N (\hat{p}_i - y_i)^2$ | **$\text{BSS} = 1 - \frac{\text{BS}_{\text{model}}}{\text{BS}_{\text{base}}} > 0$** | Measures overall probabilistic accuracy; penalizes overconfidence. |
| **Expected Calibration Error (ECE)** | $\sum_{m=1}^{10} \frac{|B_m|}{N} |\text{acc}(B_m) - \text{conf}(B_m)|$ | **$\text{ECE}_{\text{model}} < 0.05$ (5%)** | Ensures predicted probabilities match empirical reality. |
| **Precision @ High Confidence ($p \ge 0.70$)** | $\frac{\text{TP}_{p \ge 0.70}}{\text{TP}_{p \ge 0.70} + \text{FP}_{p \ge 0.70}}$ | **$\text{Precision}_{\text{model}} \ge 0.80$ (80%)** | Minimizes false alerts on executive CI dashboards. |

---

## 7. Recommended Amendments to `prediction_target_spec.md` & Automated Leakage Tests

Before commencing Step 4 (Feature Store & Splitting Implementation), the following updates are recommended:

### 7.1 Specification Document Revisions
1. **Adopt Convention A (Start-of-Day Cutoff):** Formally define predictions at Monday 00:00:00 UTC, features strictly $\le T - 1 \text{ day}$, and label window as $[T, T + 29 \text{ days}]$.
2. **Complete 30-Category Taxonomy:** Replace the 24-item taxonomy mapping with the complete 30-item mapping (Section 1.3).
3. **Rigorous Incomplete Window Rule:** Explicitly forbid cutoffs after `2026-09-07`.
4. **Chronological Partition Table:** Update with the verified 445-Monday schedule and embargo boundaries.
5. **Baseline Formulas:** Embed the formal definitions of Base-Rate and Trailing-Momentum predictors.

### 7.2 Updated Automated Temporal Leakage Tests (`tests/temporal_leakage/`)
1. **`test_feature_temporal_boundary_convention_a`:** Asserts $\max(t_e) < T_{\text{date}}$ for every feature calculation.
2. **`test_label_window_temporal_bounds`:** Asserts $\min(t_e) \ge T_{\text{date}}$ and $\max(t_e) \le T_{\text{date}} + 29$ for every label window.
3. **`test_partition_embargo_isolation`:** Asserts $\max(t_{\text{label\_train}}) < \min(T_{\text{val\_cutoff}})$ and $\max(t_{\text{label\_val}}) < \min(T_{\text{test\_cutoff}})$.
4. **`test_no_truncated_test_windows`:** Asserts no cutoff $T$ is sampled after `2026-09-07`.
