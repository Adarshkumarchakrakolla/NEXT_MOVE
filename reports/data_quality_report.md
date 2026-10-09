# NextMove Data-Quality Audit Report

**Dataset:** [`recall_intel_121000_independent_synthetic_events.csv`](file:///d:/NextMove/data/synthetic/raw/recall_intel_121000_independent_synthetic_events.csv)  
**Audit Timestamp:** `2026-10-09T04:06:00+05:30`  
**File Size:** 48,002,803 bytes (45.78 MB)  
**Mode:** Read-Only Audit (No data modified, overwritten, or imputed)  

---

## 1. Executive Summary & Verdict

### Final Verdict: `PASS WITH WARNINGS`

The raw dataset adheres to strict structural and schema standards: exactly 121,000 data rows, all 21 expected columns present with no duplicate or missing columns, zero missing or empty values across the entire dataset, 100% unique and valid `event_id` keys, and 100% parseable dates within the expected historical range. Numeric fields contain zero invalid values and zero infinities.

However, **6 critical warnings** regarding synthetic data generation artifacts must be accounted for before downstream feature engineering or model training:
1. **Pervasive Title Repetition:** Only 54,081 unique titles exist across 121,000 rows (a **55.30% repetition rate**).
2. **Rigid Description Templating:** **100.00%** of descriptions follow a deterministic fixed string format starting with `"Synthetic event for benchmark training: ..."`.
3. **Static Generator Seed:** `generator_seed` is uniform across all rows at `20261009` with zero variance.
4. **Severe Gated Zero-Inflation:** `headcount_delta` is **92.74% zeros**, `price_delta_pct` is **86.57% zeros**, and `growth_delta_pct` is **81.60% zeros**, strictly activated only for discrete `event_type` categories.
5. **Uncoupled Synthetic Dimensions:** Pairwise Pearson correlations among numeric features are statistically ~0.000, revealing independent random generation rather than realistic macroeconomic dynamics.
6. **Nominal Independence Artifact:** `independence_status` is hardcoded to `'independent_record'` for 100% of rows despite evident template reuse and combinatorial synthesis.

---

## 2. Schema Checks & Structural Integrity

| Check | Expected | Measured | Status |
| :--- | :--- | :--- | :--- |
| **Data Row Count** | 121,000 | 121,000 | PASS |
| **Column Count** | 21 | 21 | PASS |
| **Missing Expected Columns** | 0 | 0 | PASS |
| **Unexpected Extra Columns** | 0 | 0 | PASS |
| **Duplicate Column Headers** | 0 | 0 | PASS |
| **`event_id` Uniqueness** | 100% Unique | 121,000 / 121,000 (0 duplicates) | PASS |
| **`event_id` Non-Empty** | 0 null / empty | 0 null / empty | PASS |
| **`event_date` Parseability** | 100% Parseable ISO | 121,000 parseable (0 failures) | PASS |
| **`event_date` Temporal Range** | 2018-01-01 to 2026-10-09 | 2018-01-01 to 2026-10-09 (0 out-of-range) | PASS |

### Columns Identified (in file order):
```
 1. event_id
 2. competitor
 3. event_date
 4. event_type
 5. title
 6. description
 7. strategic_area
 8. market
 9. channel
10. impact_level
11. signal_strength
12. extraction_confidence
13. magnitude_index
14. price_delta_pct
15. growth_delta_pct
16. headcount_delta
17. source_type
18. provenance_status
19. evidence_status
20. independence_status
21. generator_seed
```

---

## 3. Data Hygiene & Missing Value Analysis

All 21 columns were audited for nulls (`NaN`), empty strings `""`, and whitespace-only entries.

| Column | Missing (NaN) | Empty String ("") | Total Invalid / Missing | Completeness |
| :--- | :---: | :---: | :---: | :---: |
| `event_id` | 0 | 0 | 0 | 100.00% |
| `competitor` | 0 | 0 | 0 | 100.00% |
| `event_date` | 0 | 0 | 0 | 100.00% |
| `event_type` | 0 | 0 | 0 | 100.00% |
| `title` | 0 | 0 | 0 | 100.00% |
| `description` | 0 | 0 | 0 | 100.00% |
| `strategic_area` | 0 | 0 | 0 | 100.00% |
| `market` | 0 | 0 | 0 | 100.00% |
| `channel` | 0 | 0 | 0 | 100.00% |
| `impact_level` | 0 | 0 | 0 | 100.00% |
| `signal_strength` | 0 | 0 | 0 | 100.00% |
| `extraction_confidence` | 0 | 0 | 0 | 100.00% |
| `magnitude_index` | 0 | 0 | 0 | 100.00% |
| `price_delta_pct` | 0 | 0 | 0 | 100.00% |
| `growth_delta_pct` | 0 | 0 | 0 | 100.00% |
| `headcount_delta` | 0 | 0 | 0 | 100.00% |
| `source_type` | 0 | 0 | 0 | 100.00% |
| `provenance_status` | 0 | 0 | 0 | 100.00% |
| `evidence_status` | 0 | 0 | 0 | 100.00% |
| `independence_status` | 0 | 0 | 0 | 100.00% |
| `generator_seed` | 0 | 0 | 0 | 100.00% |

- **Exact Duplicate Rows across all 21 columns:** `0` (Zero redundant records).
- **Duplicate `event_id` values:** `0` (Strictly unique primary keys `SYN-000001` to `SYN-121000`).

---

## 4. Categorical Feature Distributions

### A. Competitor Distribution
| Competitor | Record Count | Percentage |
| :--- | :---: | :---: |
| **Amazon** | 48,513 | 40.09% |
| **Walmart** | 42,431 | 35.07% |
| **eBay** | 30,056 | 24.84% |

### B. Impact Level Distribution
| Impact Level | Record Count | Percentage |
| :--- | :---: | :---: |
| **`medium`** | 47,564 | 39.31% |
| **`high`** | 30,455 | 25.17% |
| **`low`** | 26,880 | 22.21% |
| **`very_high`** | 16,101 | 13.31% |

### C. Metadata & Provenance Status
| Metadata Field | Distinct Value | Count | Uniformity |
| :--- | :--- | :---: | :---: |
| `source_type` | `synthetic` | 121,000 | 100.00% |
| `provenance_status` | `synthetic_generated` | 121,000 | 100.00% |
| `evidence_status` | `not_applicable` | 121,000 | 100.00% |
| `independence_status` | `independent_record` | 121,000 | 100.00% |

### D. Event Type Distribution (Top 15 of 30)
| Event Type | Count | Share | Event Type | Count | Share |
| :--- | :---: | :---: | :--- | :---: | :---: |
| `feature_launch` | 6,557 | 5.42% | `membership` | 3,895 | 3.22% |
| `pricing_change` | 6,484 | 5.36% | `data_service` | 3,789 | 3.13% |
| `product_launch` | 5,939 | 4.91% | `app_update` | 3,600 | 2.98% |
| `promotion` | 5,880 | 4.86% | `authentication` | 3,477 | 2.87% |
| `earnings` | 5,458 | 4.51% | `store_expansion` | 3,269 | 2.70% |
| `ai_launch` | 5,281 | 4.36% | `international_expansion` | 3,051 | 2.52% |
| `partnership` | 5,261 | 4.35% | `investment` | 3,015 | 2.49% |
| `logistics_expansion` | 4,823 | 3.99% | `live_commerce` | 3,009 | 2.49% |
| `fulfillment` | 4,792 | 3.96% | `payments` | 2,926 | 2.42% |
| `advertising` | 4,675 | 3.86% | `retail_media` | 2,727 | 2.25% |
| `marketplace_change` | 4,671 | 3.86% | `restructuring` | 2,715 | 2.24% |
| `delivery` | 4,621 | 3.82% | `sustainability` | 2,324 | 1.92% |
| `seller_program` | 4,193 | 3.47% | `acquisition` | 2,294 | 1.90% |
| `trust_safety` | 4,097 | 3.39% | `regulatory` | 2,107 | 1.74% |
| `hiring` | 4,059 | 3.35% | `leadership_change` | 2,011 | 1.66% |

### E. Strategic Area Distribution (20 Categories)
| Strategic Area | Count | Share | Strategic Area | Count | Share |
| :--- | :---: | :---: | :--- | :---: | :---: |
| Logistics & Fulfillment | 9,542 | 7.89% | Recommerce | 5,778 | 4.78% |
| AI & Personalization | 8,781 | 7.26% | Seller Economics | 5,206 | 4.30% |
| Marketplace & Sellers | 8,122 | 6.71% | Retail Operations | 5,125 | 4.24% |
| Product & Platform | 8,030 | 6.64% | Data & Analytics | 5,044 | 4.17% |
| Customer Experience | 8,018 | 6.63% | Workforce | 4,396 | 3.63% |
| Corporate Strategy | 7,374 | 6.09% | International Expansion | 4,381 | 3.62% |
| Pricing & Membership | 7,366 | 6.09% | Payments | 4,379 | 3.62% |
| Advertising & Retail Media | 6,613 | 5.47% | Live Commerce | 3,774 | 3.12% |
| Partnerships | 5,898 | 4.87% | Sustainability | 3,711 | 3.07% |
| Trust & Safety | 5,833 | 4.82% | M&A | 3,629 | 3.00% |

### F. Market and Channel Distribution
Markets (10 balanced categories, ~10% each): Brazil (12,315), UK (12,170), Mexico (12,152), Canada (12,126), US (12,096), Australia (12,091), Global (12,080), France (12,035), Germany (11,995), India (11,940)

Channels (8 balanced categories, ~12.5% each): logistics (15,337), payments (15,210), ads (15,202), mobile (15,198), physical_retail (15,198), web (14,992), marketplace (14,932), seller_platform (14,931)

---

## 5. Numeric Feature Profiles & Range Checks

| Feature | Type | Min | Max | Mean | Std | Median | Zero % | Invalid / Inf |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `signal_strength` | `float64` | 0.3 | 0.99 | 0.7337 | 0.1507 | 0.755 | 0.0% | 0 / 0 |
| `extraction_confidence` | `float64` | 0.6 | 0.995 | 0.8347 | 0.0983 | 0.851 | 0.0% | 0 / 0 |
| `magnitude_index` | `float64` | 0.1 | 9.219 | 1.1645 | 0.6905 | 1.002 | 0.0% | 0 / 0 |
| `price_delta_pct` | `float64` | -29.99 | 25.0 | -0.3357 | 5.8537 | 0.0 | 86.57% | 0 / 0 |
| `growth_delta_pct` | `float64` | -9.98 | 80.0 | 6.4151 | 17.4972 | 0.0 | 81.6% | 0 / 0 |
| `headcount_delta` | `int64` | -1000.0 | 11996.0 | 401.0804 | 1757.254 | 0.0 | 92.74% | 0 / 0 |
| `generator_seed` | `int64` | 20261009.0 | 20261009.0 | 20261009.0 | 0.0 | 20261009.0 | 0.0% | 0 / 0 |

### Distributional Anomalies Observed in Numeric Variables:
- **`price_delta_pct`:** 86.57% exact zeros. Non-zero values range from `-29.99%` to `+25.00%`. Active exclusively for `pricing_change` (6,483 rows), `promotion` (5,877 rows), and `membership` (3,894 rows).
- **`growth_delta_pct`:** 81.60% exact zeros. Non-zero values range from `-9.98%` to `+80.00%`. Active exclusively for `earnings` (5,456 rows), `fulfillment` (4,792 rows), `marketplace_change` (4,670 rows), `delivery` (4,619 rows), and `retail_media` (2,727 rows).
- **`headcount_delta`:** 92.74% exact zeros. Non-zero values range from `-1,000` to `+11,996`. Active exclusively for `hiring` (4,059 rows), `restructuring` (2,715 rows), and `leadership_change` (2,011 rows).
- **`generator_seed`:** 100.00% invariant at `20261009`. Std is `0.0000`.
- **`signal_strength` and `extraction_confidence`:** Continuous bounded distributions ([0.30, 0.99] and [0.60, 0.995]), symmetrically distributed around medians 0.755 and 0.851.
- **`magnitude_index`:** Moderately right-skewed positive distribution with min 0.10, median 1.002, and max 9.219.

---

## 6. Temporal Distribution Analysis

The event dates span from **2018-01-01** to **2026-10-09**. All dates are parseable ISO dates.

| Year | Event Count | Annual Share | Operational Notes |
| :--- | :---: | :---: | :--- |
| 2018 | 13,779 | 11.39% | Full calendar year |
| 2019 | 13,997 | 11.57% | Full calendar year |
| 2020 | 13,792 | 11.40% | Full calendar year |
| 2021 | 13,670 | 11.30% | Full calendar year |
| 2022 | 13,743 | 11.36% | Full calendar year |
| 2023 | 13,716 | 11.34% | Full calendar year |
| 2024 | 13,720 | 11.34% | Full calendar year |
| 2025 | 13,853 | 11.45% | Full calendar year |
| 2026 | 10,730 | 8.87% | Partial year (Jan 1 to Oct 9, 2026) |

> **Temporal Regularity Finding:** Full years (2018–2025) average ~13,780 events/year with under 2.4% variation between any two years. The partial year 2026 has 10,730 events across ~282 days, representing an identical daily rate (~38.0 events/day). This demonstrates uniform temporal sampling without macroeconomic seasonality.

---

## 7. Text Repetition & Template Pattern Analysis

- **Total Rows:** 121,000  
- **Unique Titles:** 54,081 (**55.3% repetition rate**)  
- **Unique Descriptions:** 116,591 (**3.64% repetition rate**)  
- **Descriptions with Prefix `"Synthetic event for benchmark training:"`:** **100.0%** (All 121,000 rows)  

### Top 5 Repeated Titles
| Repeated Title | Occurrences |
| :--- | :---: |
| "Amazon changed its marketplace offering in seller tools in Brazil" | 14 |
| "Amazon expanded a promotion for seller tools in UK" | 14 |
| "Amazon changed pricing for delivery capacity in UK" | 14 |
| "Amazon introduced search in US" | 13 |
| "Amazon introduced a trust-and-safety initiative for seller tools in Germany" | 12 |

### Top Repeated Description Template Format
```text
Count: 4x -> Synthetic event for benchmark training: Amazon announced a partnership around third-party marketplace. Primary market: Germany; channel: payments; impact: medium.
Count: 4x -> Synthetic event for benchmark training: Amazon introduced marketplace sellers. Primary market: Germany; channel: payments; impact: medium.
Count: 4x -> Synthetic event for benchmark training: Amazon expanded into advertising platform. Primary market: Mexico; channel: seller_platform; impact: medium.
```

---

## 8. Representative Sample Inspection (20 Evenly Spaced Records)

Below is the 20-record representative cross-section across indices `0`, `6368`, ..., `120999`:

| Event ID | Date | Competitor | Event Type | Title | Impact | Signal | Confidence | Deltas (Price / Growth / HC) |
| :--- | :---: | :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| `SYN-000001` | 2023-11-24 | **Walmart** | `promotion` | Walmart expanded a promotion for logistics partnership in Canada | `high` | 0.876 | 0.829 | `-0.57% / 0.0% / 0` |
| `SYN-006369` | 2020-01-04 | **eBay** | `store_expansion` | eBay expanded store operations in discounting in Mexico | `medium` | 0.658 | 0.6 | `0.0% / 0.0% / 0` |
| `SYN-012737` | 2024-02-24 | **Amazon** | `live_commerce` | Amazon expanded live-commerce activity in seller tools in UK | `high` | 0.835 | 0.733 | `0.0% / 0.0% / 0` |
| `SYN-019106` | 2021-01-14 | **Amazon** | `product_launch` | Amazon launched pricing in Brazil | `high` | 0.837 | 0.979 | `0.0% / 0.0% / 0` |
| `SYN-025474` | 2021-03-27 | **Amazon** | `logistics_expansion` | Amazon expanded logistics capacity in third-party marketplace in UK | `very_high` | 0.962 | 0.93 | `0.0% / 0.0% / 0` |
| `SYN-031842` | 2025-05-03 | **Walmart** | `seller_program` | Walmart introduced a seller program for supplier media in US | `medium` | 0.735 | 0.855 | `0.0% / 0.0% / 0` |
| `SYN-038211` | 2024-10-31 | **eBay** | `trust_safety` | eBay introduced a trust-and-safety initiative for shopping experience in Germany | `low` | 0.607 | 0.724 | `0.0% / 0.0% / 0` |
| `SYN-044579` | 2023-03-06 | **Amazon** | `product_launch` | Amazon launched merchant payments in Australia | `medium` | 0.778 | 0.886 | `0.0% / 0.0% / 0` |
| `SYN-050947` | 2026-03-22 | **Walmart** | `product_launch` | Walmart launched retail partnership in Australia | `very_high` | 0.928 | 0.918 | `0.0% / 0.0% / 0` |
| `SYN-057316` | 2025-01-28 | **Amazon** | `payments` | Amazon expanded payments capabilities in mobile app in Brazil | `high` | 0.822 | 0.986 | `0.0% / 0.0% / 0` |
| `SYN-063684` | 2018-03-24 | **eBay** | `partnership` | eBay announced a partnership around discovery in Brazil | `high` | 0.817 | 0.666 | `0.0% / 0.0% / 0` |
| `SYN-070053` | 2025-02-23 | **Walmart** | `sustainability` | Walmart advanced a sustainability initiative in marketplace sellers in Global | `high` | 0.881 | 0.955 | `0.0% / 0.0% / 0` |
| `SYN-076421` | 2022-08-06 | **eBay** | `hiring` | eBay expanded hiring in seller tools in US | `high` | 0.844 | 0.7 | `0.0% / 0.0% / 1447` |
| `SYN-082789` | 2021-03-26 | **eBay** | `acquisition` | eBay announced an acquisition involving renewable energy in France | `high` | 0.844 | 0.889 | `0.0% / 0.0% / 0` |
| `SYN-089158` | 2020-12-09 | **Walmart** | `partnership` | Walmart announced a partnership around pre-owned goods in Germany | `low` | 0.619 | 0.906 | `0.0% / 0.0% / 0` |
| `SYN-095526` | 2020-09-05 | **Amazon** | `international_expansion` | Amazon expanded into checkout in Australia | `very_high` | 0.933 | 0.797 | `0.0% / 0.0% / 0` |
| `SYN-101894` | 2019-05-31 | **Walmart** | `feature_launch` | Walmart introduced circular commerce in Canada | `low` | 0.422 | 0.969 | `0.0% / 0.0% / 0` |
| `SYN-108263` | 2026-05-14 | **Amazon** | `fulfillment` | Amazon changed fulfillment operations in returns in UK | `high` | 0.877 | 0.907 | `0.0% / 16.51% / 0` |
| `SYN-114631` | 2020-11-05 | **eBay** | `marketplace_change` | eBay changed its marketplace offering in pricing in Australia | `high` | 0.879 | 0.818 | `0.0% / 54.37% / 0` |
| `SYN-121000` | 2023-12-15 | **Amazon** | `product_launch` | Amazon launched shopping experience in UK | `high` | 0.893 | 0.929 | `0.0% / 0.0% / 0` |

---

## 9. Synthetic Integrity & Correlation Analysis

### A. Correlation Matrix (Numeric Features)
| Feature | signal_strength | extraction_confidence | magnitude_index | price_delta_pct | growth_delta_pct | headcount_delta |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `signal_strength` | +1.0000 | +0.0038 | +0.0009 | -0.0023 | +0.0034 | -0.0024 |
| `extraction_confidence` | +0.0038 | +1.0000 | -0.0016 | +0.0048 | +0.0037 | -0.0008 |
| `magnitude_index` | +0.0009 | -0.0016 | +1.0000 | -0.0032 | -0.0032 | -0.0046 |
| `price_delta_pct` | -0.0023 | +0.0048 | -0.0032 | +1.0000 | +0.0210 | +0.0131 |
| `growth_delta_pct` | +0.0034 | +0.0037 | -0.0032 | +0.0210 | +1.0000 | -0.0837 |
| `headcount_delta` | -0.0024 | -0.0008 | -0.0046 | +0.0131 | -0.0837 | +1.0000 |

> **Finding:** All pairwise correlations between `signal_strength`, `extraction_confidence`, `magnitude_index`, and deltas are `< 0.005` (with slight `-0.0837` correlation between `growth_delta_pct` and `headcount_delta`). In real-world market intelligence, major product or pricing disruptions correlate with extraction confidence, signal strength, and magnitude. The near-zero correlation is a mathematical signature of synthetic independent uncoupled generator distributions.

### B. Synthetic Provenance Status
- `source_type == 'synthetic'`: **100% of records**
- `provenance_status == 'synthetic_generated'`: **100% of records**
- `evidence_status == 'not_applicable'`: **100% of records**
- `independence_status == 'independent_record'`: **100% of records**

> **Critical Caution on Independence:** While each row has a distinct `event_id` and is tagged `independent_record`, the records are not independent real-world business events. They are generated via combinatorial slot-filling (competitor + verb + topic + market + channel). Modeling methods that assume true econometric independence or natural time-series clustering will experience distribution shift when evaluated against empirical market signals.

---

## 10. Audit Summary & Downstream Recommendations

### Passed Checks
- [x] **Row count exactly matches 121,000**
- [x] **Column count exactly matches 21**
- [x] **Zero missing, unexpected, or duplicate column names**
- [x] **event_id values are 100% unique, non-null, and non-empty**
- [x] **event_date is 100% parseable ISO format and strictly within [2018-01-01, 2026-10-09]**
- [x] **Zero missing (NaN) and zero empty string values across all 21 columns**
- [x] **Zero exact duplicate rows across all 21 columns**
- [x] **Numeric columns have zero invalid values, zero infinities, and all lie within mathematically valid bounds**
- [x] **All records are explicitly labeled with source_type='synthetic' and provenance_status='synthetic_generated'**

### Warnings
- [!] **WARNING:** Severe title repetition: 54,081 unique titles across 121,000 rows (55.30% repetition rate due to combinatorial slot filling)
- [!] **WARNING:** Rigid description template: 100.00% of descriptions start with 'Synthetic event for benchmark training:' and follow a fixed string interpolation pattern
- [!] **WARNING:** Invariant generator seed: generator_seed is identically 20261009 across all 121,000 records (zero variance)
- [!] **WARNING:** Conditional zero-inflation in target deltas: headcount_delta (92.74% zeros), price_delta_pct (86.57% zeros), growth_delta_pct (81.60% zeros) are gated strictly by discrete event types
- [!] **WARNING:** Uncorrelated synthetic numeric dimensions: Pairwise Pearson correlations between numeric features are ~0.000, revealing independent uncoupled synthetic sampling rather than organic market relationships
- [!] **WARNING:** Nominal independence claim: independence_status is hardcoded to 'independent_record' for 100% of records, but records reflect synthetic template permutations rather than empirical independent events

### Recommended Downstream Actions:
1. **Feature Engineering on Zero-Inflated Deltas:** Do not treat `price_delta_pct`, `growth_delta_pct`, and `headcount_delta` as standard Gaussian continuous variables. Use two-part (hurdle) models or explicit binary activity flags (`is_price_event`, `is_headcount_event`).
2. **Text Feature Representation:** NLP/LLM embeddings should strip the invariant prefix `"Synthetic event for benchmark training: "` prior to vector encoding to avoid wasting embedding attention on constant tokens.
3. **Split Strategy Validation:** Because titles repeat across years, ensure temporal splitting (`event_date`) is strictly enforced so that models do not inadvertently memorize template combinations across train/validation/test splits.
4. **Synthetic Metadata Columns Exclusion:** Exclude `source_type`, `provenance_status`, `evidence_status`, `independence_status`, and `generator_seed` from predictive model feature sets, as they have zero variance.
