"""Verification script for NextMove Step 6 — Target Label Generation & Integrity Checks.

Executes and verifies:
1. Path anchoring and project root resolution.
2. Taxonomy mapping completeness across all 121,000 raw events (0 unmapped).
3. Window boundaries and right-censoring compliance (window ends 2026-10-06 <= 2026-10-09).
4. Boundary proofs (T, T + 29 inclusion; T - 1, T + 30 exclusion; empty window; isolation).
5. Label generation on complete 6,675 key grid with partition and binary checks.
6. Separate feature join verification on (cutoff_date, competitor, strategic_cluster)
   with 1:1 row validation and bitwise feature value preservation.
7. Materialization of the modeling dataset and printed distributions.
"""

import hashlib
from pathlib import Path
import sys
import time

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from configs.prediction_config import config
from src.splitting.temporal import (
    assign_strategic_clusters,
    get_partition_cutoffs,
    get_label_window_bounds,
    normalize_to_date,
)
from src.labels.label_builder import TargetLabelBuilder


def main():
    start_total_time = time.time()
    print("=" * 70)
    print("NEXTMOVE STEP 6: TARGET LABEL GENERATION VERIFICATION")
    print("=" * 70)

    # -------------------------------------------------------------------------
    # 1. Project Root & File Path Checks
    # -------------------------------------------------------------------------
    print("\n[CHECK 1] Path and Project Root Verification")
    print(f"  Project Root:     {config.project_root}")
    print(f"  Raw CSV Path:     {config.raw_csv_path}")
    assert config.raw_csv_path.exists(), f"Raw CSV not found at {config.raw_csv_path}"
    print(f"  Raw CSV Exists:   True (Size: {config.raw_csv_path.stat().st_size:,} bytes)")
    print("  -> Path check PASSED.")

    # -------------------------------------------------------------------------
    # 2. Window Completeness & Right-Censoring Checks
    # -------------------------------------------------------------------------
    print("\n[CHECK 2] Forward Window Completeness & Right-Censoring")
    test_cutoffs = get_partition_cutoffs("test")
    last_cutoff = test_cutoffs[-1]
    start_w, end_w = get_label_window_bounds(last_cutoff, horizon_days=config.label_window_days)
    print(f"  Latest Test Cutoff:         {last_cutoff.strftime('%Y-%m-%d')} (Monday)")
    print(f"  Forward Label Window:       {start_w.strftime('%Y-%m-%d')} to {end_w.strftime('%Y-%m-%d')} (30 days)")
    print(f"  Dataset Maximum Event Date: {config.dataset_max_date} (Friday)")
    assert end_w <= normalize_to_date(config.dataset_max_date), "Window exceeds maximum dataset date!"
    buffer_days = (normalize_to_date(config.dataset_max_date) - end_w).days
    print(f"  Remaining Buffer:           {buffer_days} calendar days before dataset boundary")
    print("  -> Right-censoring check PASSED.")

    # -------------------------------------------------------------------------
    # 3. Taxonomy Mapping & Date Types Verification
    # -------------------------------------------------------------------------
    print("\n[CHECK 3] Taxonomy Mapping and Date Parsing")
    raw_df = pd.read_csv(config.raw_csv_path, low_memory=False)
    assert len(raw_df) == 121000, f"Expected 121,000 raw events, found {len(raw_df)}"
    mapped_df = assign_strategic_clusters(raw_df)
    unmapped_count = int(mapped_df["strategic_cluster"].isna().sum())
    print(f"  Total Raw Events:           {len(mapped_df):,}")
    print(f"  Unmapped Events Count:      {unmapped_count}")
    assert unmapped_count == 0, f"Found {unmapped_count} unmapped events!"
    print("  Cluster Distribution across raw event log:")
    for c_id in range(1, 6):
        cnt = int((mapped_df["strategic_cluster"] == c_id).sum())
        pct = cnt / len(mapped_df) * 100
        print(f"    Cluster {c_id} ({config.cluster_names[c_id]}): {cnt:,} events ({pct:.2f}%)")
    print("  -> Taxonomy mapping check PASSED.")

    # -------------------------------------------------------------------------
    # 4. Label-Window Boundaries & Edge-Case Proofs
    # -------------------------------------------------------------------------
    print("\n[CHECK 4] Label-Window Boundary & Edge-Case Fixtures")
    T = pd.Timestamp("2024-01-15")  # Monday. Window: [2024-01-15, 2024-02-13]
    t_minus_1 = T - pd.Timedelta(days=1)   # 2024-01-14 (Sunday)
    t_0 = T                                # 2024-01-15 (Monday, first day)
    t_29 = T + pd.Timedelta(days=29)       # 2024-02-13 (Tuesday, 30th day)
    t_30 = T + pd.Timedelta(days=30)       # 2024-02-14 (Wednesday, day after)

    # 4a. First day inclusion (T)
    b_t0 = TargetLabelBuilder(config=config).fit_events(pd.DataFrame([{
        "event_id": "E_FIRST", "competitor": "Amazon", "event_date": t_0.strftime("%Y-%m-%d"), "event_type": "pricing_change"
    }]))
    lbl_t0 = b_t0.compute_labels_for_cutoff(T)
    amz_c1_t0 = [r for r in lbl_t0 if r["competitor"] == "Amazon" and r["strategic_cluster"] == 1][0]["target_30d"]
    assert amz_c1_t0 == 1, "First-day event (T) must produce target_30d = 1"

    # 4b. Last day inclusion (T + 29)
    b_t29 = TargetLabelBuilder(config=config).fit_events(pd.DataFrame([{
        "event_id": "E_LAST", "competitor": "Amazon", "event_date": t_29.strftime("%Y-%m-%d"), "event_type": "pricing_change"
    }]))
    lbl_t29 = b_t29.compute_labels_for_cutoff(T)
    amz_c1_t29 = [r for r in lbl_t29 if r["competitor"] == "Amazon" and r["strategic_cluster"] == 1][0]["target_30d"]
    assert amz_c1_t29 == 1, "Last-day event (T + 29) must produce target_30d = 1"

    # 4c. Outside window exclusion (T - 1 and T + 30)
    b_out = TargetLabelBuilder(config=config).fit_events(pd.DataFrame([
        {"event_id": "E_PREV", "competitor": "Amazon", "event_date": t_minus_1.strftime("%Y-%m-%d"), "event_type": "pricing_change"},
        {"event_id": "E_NEXT", "competitor": "Amazon", "event_date": t_30.strftime("%Y-%m-%d"), "event_type": "pricing_change"},
    ]))
    lbl_out = b_out.compute_labels_for_cutoff(T)
    amz_c1_out = [r for r in lbl_out if r["competitor"] == "Amazon" and r["strategic_cluster"] == 1][0]["target_30d"]
    assert amz_c1_out == 0, "Events outside [T, T + 29] must yield target_30d = 0"

    # 4d. Empty window -> 0
    b_empty = TargetLabelBuilder(config=config).fit_events(pd.DataFrame(columns=["competitor", "event_date", "event_type"]))
    lbl_empty = b_empty.compute_labels_for_cutoff(T)
    assert all(r["target_30d"] == 0 for r in lbl_empty), "Empty window must produce all 0s"

    # 4e. Competitor & cluster isolation
    b_iso = TargetLabelBuilder(config=config).fit_events(pd.DataFrame([{
        "event_id": "E_ISO", "competitor": "Walmart", "event_date": (T + pd.Timedelta(days=5)).strftime("%Y-%m-%d"), "event_type": "payments"  # Cluster 3
    }]))
    lbl_iso = b_iso.compute_labels_for_cutoff(T)
    for r in lbl_iso:
        if r["competitor"] == "Walmart" and r["strategic_cluster"] == 3:
            assert r["target_30d"] == 1
        else:
            assert r["target_30d"] == 0, f"Cell ({r['competitor']}, {r['strategic_cluster']}) leaked label 1!"

    print("  -> Boundary inclusion (T, T+29), exclusion (T-1, T+30), empty window, and isolation PASSED.")

    # -------------------------------------------------------------------------
    # 5. Label Generation & Full Grid Verification
    # -------------------------------------------------------------------------
    print("\n[CHECK 5] Target Label Generation on Complete Observation Grid")
    builder = TargetLabelBuilder(config=config)
    builder.fit_events(mapped_df)
    labels_df = builder.build_label_matrix()

    # Exact expected row formula and partition checks
    expected_rows = (
        len(get_partition_cutoffs("train"))
        + len(get_partition_cutoffs("val"))
        + len(get_partition_cutoffs("test"))
    ) * len(config.competitors) * 5

    assert len(labels_df) == expected_rows, f"Expected {expected_rows} rows, got {len(labels_df)}"
    assert labels_df["target_30d"].isin([0, 1]).all(), "Target contains non-binary values!"
    assert labels_df["target_30d"].notna().all(), "Target contains null values!"
    assert labels_df["partition"].notna().all(), "Partition column contains null values!"

    expected_partitions = {
        "train": 4635,
        "validation": 1110,
        "test": 930,
    }
    assert labels_df.groupby("partition").size().to_dict() == expected_partitions, "Partition row counts mismatch!"
    assert set(labels_df["strategic_cluster"].unique()) == {1, 2, 3, 4, 5}, "Strategic cluster domain mismatch!"
    assert set(labels_df["competitor"].unique()) == set(config.competitors), "Competitor domain mismatch!"

    # Coverage check: every single expected key is present
    all_expected_cutoffs = (
        get_partition_cutoffs("train") + get_partition_cutoffs("val") + get_partition_cutoffs("test")
    )
    expected_key_set = {
        (c.strftime("%Y-%m-%d"), comp, cl)
        for c in all_expected_cutoffs
        for comp in config.competitors
        for cl in range(1, 6)
    }
    actual_key_set = set(zip(labels_df["cutoff_date"], labels_df["competitor"], labels_df["strategic_cluster"]))
    assert actual_key_set == expected_key_set, "Mismatch in expected key set!"
    assert len(labels_df) == len(actual_key_set), "Duplicate join keys detected in labels!"

    print("All key, label, partition, and coverage checks PASSED.")

    # -------------------------------------------------------------------------
    # 6. Feature Join Verification
    # -------------------------------------------------------------------------
    print("\n[CHECK 6] Feature Join Verification on (cutoff_date, competitor, strategic_cluster)")
    features_csv_path = config.project_root / "data" / "synthetic" / "processed" / "synthetic_features_445_cutoffs_6675_instances.csv"
    assert features_csv_path.exists(), f"Feature artifact not found at {features_csv_path}"
    features_df = pd.read_csv(features_csv_path)
    print(f"  Loaded Features Artifact:   {len(features_df):,} rows, {len(features_df.columns)} columns")

    # Perform separate join
    joined_df = builder.join_features_and_labels(features_df, labels_df)
    assert len(joined_df) == len(features_df) == 6675, "Row count changed after join!"
    assert len(joined_df.columns) == len(features_df.columns) + 1, "Column count mismatch after join!"
    assert "target_30d" in joined_df.columns, "'target_30d' missing from joined dataset!"
    assert joined_df.isna().sum().sum() == 0, "Null values found after join!"

    # Verify bitwise feature preservation
    for col in features_df.columns:
        assert np.array_equal(joined_df[col].values, features_df[col].values), (
            f"Feature column '{col}' values altered during join!"
        )
    print("  -> Feature join is strictly 1:1, zero nulls, and all feature values UNCHANGED.")

    # -------------------------------------------------------------------------
    # 7. Materialize Labeled Modeling Dataset Artifact
    # -------------------------------------------------------------------------
    print("\n[CHECK 7] Materializing Labeled Modeling Dataset Artifact")
    output_csv_path = config.project_root / "data" / "synthetic" / "processed" / "synthetic_modeling_dataset_445_cutoffs_6675_instances.csv"
    joined_df.to_csv(output_csv_path, index=False)
    file_size_bytes = output_csv_path.stat().st_size

    # Checksum calculation
    hasher = hashlib.sha256()
    with open(output_csv_path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    checksum = hasher.hexdigest()

    print(f"  Artifact Path:     {output_csv_path}")
    print(f"  File Size:         {file_size_bytes:,} bytes")
    print(f"  SHA-256 Checksum:  {checksum}")
    print(f"  Total Rows:        {len(joined_df):,}")
    print(f"  Total Columns:     {len(joined_df.columns)}")

    # -------------------------------------------------------------------------
    # 8. Printed Distribution Reconciliation
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("PRINTED LABEL DISTRIBUTIONS ACROSS SYNTHETIC MODELING DATASET")
    print("=" * 70)

    print("\n--- DISTRIBUTION BY PARTITION ---")
    p_summary = joined_df.groupby("partition")["target_30d"].agg(
        total="count",
        positives="sum",
        prevalence="mean"
    )
    p_summary["negatives"] = p_summary["total"] - p_summary["positives"]
    print(p_summary[["total", "positives", "negatives", "prevalence"]])

    print("\n--- DISTRIBUTION BY COMPETITOR ---")
    c_summary = joined_df.groupby("competitor")["target_30d"].agg(
        total="count",
        positives="sum",
        prevalence="mean"
    )
    c_summary["negatives"] = c_summary["total"] - c_summary["positives"]
    print(c_summary[["total", "positives", "negatives", "prevalence"]])

    print("\n--- DISTRIBUTION BY STRATEGIC CLUSTER ---")
    s_summary = joined_df.groupby("strategic_cluster")["target_30d"].agg(
        total="count",
        positives="sum",
        prevalence="mean"
    )
    s_summary["negatives"] = s_summary["total"] - s_summary["positives"]
    print(s_summary[["total", "positives", "negatives", "prevalence"]])

    total_time = round(time.time() - start_total_time, 3)
    print(f"\nVerification and artifact generation completed in {total_time}s.")
    print("=" * 70)


if __name__ == "__main__":
    main()
