import os

CUTOFF_MD_PATH = r"d:/NextMove/reports/cutoff_convention_review.md"

content = """# Prediction Cutoff-Time Convention Review & Ambiguity Resolution

**Document:** `reports/cutoff_convention_review.md`  
**System:** NextMove Competitive Intelligence Platform  
**Target:** Resolving Temporal Ambiguity in Observation Grid & Lookback Windows  
**Status:** Pre-Implementation Specification Review  

---

## 1. Executive Summary & Core Ambiguity

A critical temporal ambiguity was identified in the interaction between the observation grid and feature calculation rules:
- The raw dataset ([`recall_intel_121000_independent_synthetic_events.csv`](file:///d:/NextMove/data/synthetic/raw/recall_intel_121000_independent_synthetic_events.csv)) contains calendar dates (`event_date`, e.g. `2024-01-08`), but **no intra-day timestamps** (`00:00:00` vs `16:30:00`) and **no source-availability timestamps** (`ingested_at`, `published_at`).
- The draft specification proposed sampling predictions on a weekly grid at **Monday 00:00:00 UTC**, while simultaneously defining lookback features as including events dated on cutoff date $T$.
- If predictions are generated at Monday 00:00:00 UTC, events with `event_date == Monday` have **not yet occurred** in the real world. Conditioning on Monday's events at Monday 00:00:00 UTC is a textbook lookahead leakage bug.

To eliminate this ambiguity, two internally consistent temporal cutoff conventions are evaluated.

---

## 2. Comparative Analysis of Cutoff Conventions

```mermaid
graph TD
    subgraph Convention A: Start-of-Day [Monday 00:00:00 UTC - Recommended]
        A_Feat[Features: Historical Days strictly <= Sunday T-1 day] -->|Cutoff at Monday 00:00 UTC| A_Pred[Prediction Issued Monday Morning]
        A_Pred --> A_Label[Label Window: 30 Days inclusive of Monday [T, T+29 days]]
    end

    subgraph Convention B: End-of-Day [Monday 23:59:59 UTC]
        B_Feat[Features: Historical Days <= Monday T inclusive] -->|Cutoff at Monday 23:59 UTC| B_Pred[Prediction Issued Monday Night]
        B_Pred --> B_Label[Label Window: 30 Days starting Tuesday [T+1, T+30 days]]
    end
```

---

### Convention A: Start-of-Day Cutoff (Monday 00:00:00 UTC) — *RECOMMENDED*

#### 1. Exact Cutoff Timestamp & Eligible Feature Dates
- **Cutoff Timestamp:** Every Monday at `00:00:00 UTC` ($T_{\text{ts}}$).
- **Eligible Historical Events:** Strictly events with $\text{event\_date} \le T_{\text{date}} - 1 \text{ day}$ (Sunday at 23:59:59 UTC).
- **Lookback Windows:**
  - 30-Day Velocity: $[T_{\text{date}} - 30, T_{\text{date}} - 1]$ (30 full calendar days: preceding Saturday through preceding Sunday).
  - 90-Day Velocity: $[T_{\text{date}} - 90, T_{\text{date}} - 1]$ (90 full calendar days).
  - 365-Day Baseline: $[T_{\text{date}} - 365, T_{\text{date}} - 1]$.
- **Recency Clock:** $\Delta t = (T_{\text{date}} - 1) - \max(t_e \mid t_e \le T_{\text{date}} - 1)$.

#### 2. Precise 30-Day Forward Label Window (Zero Overlap)
- **Forward Label Horizon:** Starts at `Monday 00:00:00 UTC` ($T$) and spans exactly **30 full calendar days**, terminating on `Tuesday 23:59:59 UTC` ($T + 29 \text{ days}$).
- **Label Query Date Range:** 
  $$\text{Label Window} = [T_{\text{date}}, T_{\text{date}} + 29 \text{ days}]$$
- **Mathematical Disjointness Proof:**
  - Feature Max Date: $T_{\text{date}} - 1$
  - Label Min Date: $T_{\text{date}}$
  - $\text{Feature Dates} \cap \text{Label Dates} = \emptyset$ (Strictly disjoint; 0 days of overlap).

---

### Convention B: End-of-Day Cutoff (Monday 23:59:59 UTC)

#### 1. Exact Cutoff Timestamp & Eligible Feature Dates
- **Cutoff Timestamp:** Every Monday at `23:59:59 UTC` ($T_{\text{ts}}$), executed after Monday's full batch ingestion closes.
- **Eligible Historical Events:** All events with $\text{event\_date} \le T_{\text{date}}$ (inclusive of Monday).
- **Lookback Windows:**
  - 30-Day Velocity: $[T_{\text{date}} - 29, T_{\text{date}}]$ (30 calendar days: preceding Sunday through Monday).
  - 90-Day Velocity: $[T_{\text{date}} - 89, T_{\text{date}}]$ (90 calendar days).
- **Recency Clock:** $\Delta t = T_{\text{date}} - \max(t_e \mid t_e \le T_{\text{date}})$.

#### 2. Precise 30-Day Forward Label Window (Zero Overlap)
- **Forward Label Horizon:** Starts at `Tuesday 00:00:00 UTC` ($T + 1 \text{ day}$) and terminates on `Wednesday 23:59:59 UTC` ($T + 30 \text{ days}$).
- **Label Query Date Range:**
  $$\text{Label Window} = [T_{\text{date}} + 1 \text{ day}, T_{\text{date}} + 30 \text{ days}]$$
- **Mathematical Disjointness Proof:**
  - Feature Max Date: $T_{\text{date}}$
  - Label Min Date: $T_{\text{date}} + 1$
  - $\text{Feature Dates} \cap \text{Label Dates} = \emptyset$ (Strictly disjoint; 0 days of overlap).

---

## 3. Date-Only Data Limitations on Availability & Ground Truth

In real-world competitive intelligence, date-only records introduce fundamental limitations that must be acknowledged before interpreting results:

| Phenomenon | Impact on Convention A (Start-of-Day) | Impact on Convention B (End-of-Day) | Real-World Operational Reality |
| :--- | :--- | :--- | :--- |
| **Ingestion Pipeline Latency** | **Resilient.** Provides a natural 24–48h weekend buffer. Events from Friday/Saturday have completed scraping, deduplication, and parsing by Monday 00:00 UTC. | **Fragile.** Requires Monday 23:59:59 UTC ingestion to be instantaneous. Late-breaking Monday press releases (e.g. 23:45 UTC) will not be indexed by midnight. | Real CI pipelines experience 15m to 4h pipeline delays. |
| **Timezone & Dateline Discrepancies** | Minor. International events from Tokyo or London on Sunday are already closed by Monday 00:00 UTC. | Moderate. A Seattle (PST) event filed at 5:00 PM PST Monday carries a Tuesday 01:00 UTC timestamp in GMT logs. | Date strings without timezone metadata create 1-day boundary jitter. |
| **Retroactive Backdating / Revisions** | Unaffected by definition. | Vulnerable if historical edits alter Monday records post-cutoff. | SEC 8-K amendments or news retractions alter historical records after issuance. |
| **Operational Business Fit** | **Optimal.** Delivers the "Monday Morning Executive Briefing" before trading opens. | Suboptimal. Delivers forecast late Monday night, reducing decision lead time. | Corporate strategy teams plan their weekly agendas on Monday mornings. |

---

## 4. Recommendation for NextMove Initial Prototype

### Recommended Choice: **Convention A (Start-of-Day Cutoff at Monday 00:00:00 UTC)**

#### Key Justifications:
1. **Executive Operational Alignment:** In commercial competitive intelligence, executive stakeholders want predictions at the start of the business week (Monday morning). Forecasting the upcoming 30 days (starting Monday through 4 weeks later) maps directly to corporate review cadences.
2. **Built-in Weekend Ingestion Buffer:** By drawing the feature boundary at Sunday 23:59:59 UTC ($T - 1$), all work-week announcements from the prior week (Monday–Friday) are guaranteed to be fully ingested and indexed, providing realistic temporal safety without assuming zero-latency scraping.
3. **Purity of Label Window:** Monday itself is part of the future forecast window $[T, T+29]$. If Amazon launches a major discount on Monday afternoon, the model is evaluated on whether it foresaw that action from Sunday's stance, rather than trivially conditioning on Monday morning leaks.
4. **Clean Code Semantics:**
   ```python
   # Convention A Implementation Invariant:
   features_df = events_df[events_df["event_date"] < cutoff_date]
   label_window_df = events_df[
       (events_df["event_date"] >= cutoff_date) & 
       (events_df["event_date"] < cutoff_date + pd.Timedelta(days=30))
   ]
   ```

---

## 5. Formalized Partition Schedule & Leakage Test Suite under Convention A

With Convention A established, the chronological observation grid and automated test invariants are finalized:

### 5.1 Observation Grid Schedule (445 Eligible Mondays)
- **Eligible Grid Days:** Every Monday $T$ where $T \ge \text{2018-01-01}$ and $T + 29 \text{ days} \le \text{2026-10-09}$ (Last eligible Monday: **`2026-09-07`**).

| Partition | Start Cutoff $T_{\text{start}}$ | End Cutoff $T_{\text{end}}$ | Eligible Feature Dates | Forward Label Window Dates | Total Mondays |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Train** | `2018-01-01` | `2023-11-27` | $\le \text{2023-11-26}$ | `[2018-01-01, 2023-12-26]` | 309 |
| **Embargo 1** | `2023-11-28` | `2023-12-31` | *No Cutoffs Sampled* | *Buffer ensures Train labels close by 2023-12-26* | 0 |
| **Validation** | `2024-01-01` | `2025-05-26` | $\le \text{2025-05-25}$ | `[2024-01-01, 2025-06-24]` | 74 |
| **Embargo 2** | `2025-05-27` | `2025-07-06` | *No Cutoffs Sampled* | *Buffer ensures Val labels close by 2025-06-24* | 0 |
| **Test (OOT)** | `2025-07-07` | `2026-09-07` | $\le \text{2026-09-06}$ | `[2025-07-07, 2026-10-06]` | 62 |
| **Total** | | | | | **445 Mondays** |

*Verification of Zero Overlap:*
- Train Label End: `2023-12-26` < Validation Start: `2024-01-01` (Gap: 5 calendar days).
- Val Label End: `2025-06-24` < Test Start: `2025-07-07` (Gap: 12 calendar days).
- Test Label End: `2026-10-06` $\le$ Dataset Max: `2026-10-09` (Zero truncated label windows).

### 5.2 Updated Temporal Leakage Automated Tests (`tests/temporal_leakage/`)
The following 4 test contracts will be implemented in Step 4 to guarantee leakage defense:

1. **`test_feature_temporal_boundary_convention_a`:**
   - Asserts that for every sample point with cutoff $T$, $\max(\text{feature\_event\_date}) < T$. If any feature queries an event where $\text{event\_date} \ge T$, test raises `AssertionError`.
2. **`test_label_window_temporal_bounds`:**
   - Asserts that for every cutoff $T$, $\min(\text{label\_event\_date}) \ge T$ and $\max(\text{label\_event\_date}) \le T + 29 \text{ days}$.
3. **`test_partition_embargo_isolation`:**
   - Asserts that the maximum label date across all training cutoffs is strictly less than the minimum cutoff date of the validation partition, and similarly for validation vs. test.
4. **`test_no_truncated_test_windows`:**
   - Asserts that no observation cutoff is sampled after `2026-09-07`, ensuring 100% of test instances have full 30-day ground-truth visibility.
"""

os.makedirs(os.path.dirname(CUTOFF_MD_PATH), exist_ok=True)
with open(CUTOFF_MD_PATH, 'w', encoding='utf-8') as f:
    f.write(content)

print("Generated cutoff convention review:", CUTOFF_MD_PATH)
