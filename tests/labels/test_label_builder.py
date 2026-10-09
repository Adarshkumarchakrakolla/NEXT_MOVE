"""Automated test suite for TargetLabelBuilder and Step 6 Target Label Generation.

Verifies:
- In-memory window boundary tests (first day T, last day T + 29 inclusion)
- Immediately outside window exclusion (T - 1, T + 30)
- Empty windows producing 0
- Competitor and strategic cluster isolation
- Multiple events producing binary 1 (never count > 1)
- Rejection of truncated/incomplete forward windows (right-censoring)
- Explicit handling/rejection of unmapped event types and invalid clusters
- Deterministic label output across repeated runs
- Full grid key coverage (all 6,675 expected instances, no duplicates, non-null binary)
- Verification that raw dataset remains completely unaltered
- Artifact serialization with build_and_save_labels
"""

import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from configs.prediction_config import (
    COMPETITORS,
    CLUSTER_NAMES,
    PredictionConfig,
    config,
)
from src.labels.label_builder import TargetLabelBuilder
from src.splitting.temporal import get_partition_cutoffs, normalize_to_date


# ==============================================================================
# Fixtures
# ==============================================================================


@pytest.fixture
def target_cutoff() -> pd.Timestamp:
    """Monday cutoff date for controlled label testing."""
    return pd.Timestamp("2024-01-15")  # Monday. Window: 2024-01-15 to 2024-02-13 inclusive


@pytest.fixture
def label_builder() -> TargetLabelBuilder:
    """TargetLabelBuilder initialized with default project config."""
    return TargetLabelBuilder(config=config)


# ==============================================================================
# 1. Forward Window Boundary Invariants
# ==============================================================================


def test_first_and_last_day_of_window_inclusion(label_builder: TargetLabelBuilder, target_cutoff: pd.Timestamp):
    """Events occurring on the first day (T) and last day (T + 29) must produce label 1."""
    T = target_cutoff
    last_day = T + pd.Timedelta(days=29)  # 2024-02-13

    # Event on the very first day of the window (T)
    events_t0 = pd.DataFrame([{
        "event_id": "E_FIRST",
        "competitor": "Amazon",
        "event_date": T.strftime("%Y-%m-%d"),
        "event_type": "pricing_change",  # Cluster 1
    }])
    builder_t0 = TargetLabelBuilder(config=config).fit_events(events_t0)
    labels_t0 = builder_t0.compute_labels_for_cutoff(T)
    amz_c1_t0 = [r for r in labels_t0 if r["competitor"] == "Amazon" and r["strategic_cluster"] == 1][0]
    assert amz_c1_t0["target_30d"] == 1, "Event on cutoff date T must produce target_30d = 1"

    # Event on the 30th day of the window (T + 29 days)
    events_t29 = pd.DataFrame([{
        "event_id": "E_LAST",
        "competitor": "Amazon",
        "event_date": last_day.strftime("%Y-%m-%d"),
        "event_type": "pricing_change",  # Cluster 1
    }])
    builder_t29 = TargetLabelBuilder(config=config).fit_events(events_t29)
    labels_t29 = builder_t29.compute_labels_for_cutoff(T)
    amz_c1_t29 = [r for r in labels_t29 if r["competitor"] == "Amazon" and r["strategic_cluster"] == 1][0]
    assert amz_c1_t29["target_30d"] == 1, "Event on T + 29 days must produce target_30d = 1"


def test_immediately_outside_window_exclusion(label_builder: TargetLabelBuilder, target_cutoff: pd.Timestamp):
    """Events occurring on T - 1 (Sunday) and T + 30 (Wednesday) must NOT trigger label 1."""
    T = target_cutoff
    day_before = T - pd.Timedelta(days=1)   # 2024-01-14 (Sunday)
    day_after = T + pd.Timedelta(days=30)   # 2024-02-14 (Wednesday)

    events_df = pd.DataFrame([
        {
            "event_id": "E_BEFORE",
            "competitor": "Amazon",
            "event_date": day_before.strftime("%Y-%m-%d"),
            "event_type": "pricing_change",
        },
        {
            "event_id": "E_AFTER",
            "competitor": "Amazon",
            "event_date": day_after.strftime("%Y-%m-%d"),
            "event_type": "pricing_change",
        },
    ])

    builder = TargetLabelBuilder(config=config).fit_events(events_df)
    labels = builder.compute_labels_for_cutoff(T)
    amz_c1 = [r for r in labels if r["competitor"] == "Amazon" and r["strategic_cluster"] == 1][0]

    assert amz_c1["target_30d"] == 0, "Events outside [T, T + 29] must yield target_30d = 0"


def test_empty_windows_produce_label_zero(label_builder: TargetLabelBuilder, target_cutoff: pd.Timestamp):
    """An empty event log or unpopulated window must yield target_30d = 0."""
    empty_events = pd.DataFrame(columns=["competitor", "event_date", "event_type"])
    builder = TargetLabelBuilder(config=config).fit_events(empty_events)
    labels = builder.compute_labels_for_cutoff(target_cutoff)

    assert len(labels) == 15
    for row in labels:
        assert row["target_30d"] == 0, f"Expected 0 for empty window, got {row['target_30d']}"


# ==============================================================================
# 2. Competitor and Cluster Isolation
# ==============================================================================


def test_competitor_and_cluster_isolation(label_builder: TargetLabelBuilder, target_cutoff: pd.Timestamp):
    """An event for a specific competitor and cluster must not leak into any other cell."""
    T = target_cutoff
    in_window_date = T + pd.Timedelta(days=5)

    # Event for Walmart in Cluster 3 (payments -> Cluster 3)
    events_df = pd.DataFrame([{
        "event_id": "W_AI",
        "competitor": "Walmart",
        "event_date": in_window_date.strftime("%Y-%m-%d"),
        "event_type": "payments",  # Maps to Cluster 3
    }])

    builder = TargetLabelBuilder(config=config).fit_events(events_df)
    labels = builder.compute_labels_for_cutoff(T)

    assert len(labels) == 15
    for row in labels:
        if row["competitor"] == "Walmart" and row["strategic_cluster"] == 3:
            assert row["target_30d"] == 1, "Target cell must be 1"
        else:
            assert row["target_30d"] == 0, (
                f"Isolated cell ({row['competitor']}, {row['strategic_cluster']}) leaked to 1!"
            )


def test_binary_labels_only_even_with_multiple_events(label_builder: TargetLabelBuilder, target_cutoff: pd.Timestamp):
    """Multiple matching events must yield target_30d = 1, never a count > 1."""
    T = target_cutoff
    events_df = pd.DataFrame([
        {
            "event_id": f"E_{i}",
            "competitor": "eBay",
            "event_date": (T + pd.Timedelta(days=i)).strftime("%Y-%m-%d"),
            "event_type": "marketplace_change",
        }
        for i in range(10)
    ])

    builder = TargetLabelBuilder(config=config).fit_events(events_df)
    labels = builder.compute_labels_for_cutoff(T)

    ebay_c4 = [r for r in labels if r["competitor"] == "eBay" and r["strategic_cluster"] == 4][0]
    assert ebay_c4["target_30d"] == 1
    assert type(ebay_c4["target_30d"]) in (int, np.int64, np.int32)
    assert ebay_c4["event_count_in_window"] == 10, "Diagnostic count should record exact occurrences"


# ==============================================================================
# 3. Truncated Forward Window Rejection & Domain Validation
# ==============================================================================


def test_rejection_of_truncated_windows(label_builder: TargetLabelBuilder):
    """Cutoffs whose forward window [T, T + 29] extends past dataset max date must be rejected."""
    empty_events = pd.DataFrame(columns=["competitor", "event_date", "event_type"])
    builder = TargetLabelBuilder(config=config).fit_events(empty_events)

    # Max eligible cutoff is 2026-09-07 (window ends 2026-10-06 <= 2026-10-09)
    # Next Monday is 2026-09-14 (window ends 2026-10-13 > 2026-10-09)
    invalid_cutoff = "2026-09-14"

    with pytest.raises(ValueError, match="exceeding max.*dataset date"):
        builder.compute_labels_for_cutoff(invalid_cutoff)


def test_unmapped_event_types_handled_explicitly(label_builder: TargetLabelBuilder):
    """Event log containing unknown or unmapped event types must raise ValueError."""
    invalid_events = pd.DataFrame([{
        "event_id": "INV_1",
        "competitor": "Amazon",
        "event_date": "2024-01-15",
        "event_type": "alien_invasion_sale",  # Not in 30 approved taxonomy types
    }])

    with pytest.raises(ValueError, match="Found unmapped event types"):
        TargetLabelBuilder(config=config).fit_events(invalid_events)


def test_invalid_strategic_cluster_ids_rejected(label_builder: TargetLabelBuilder):
    """Event log containing invalid strategic cluster IDs must raise ValueError."""
    invalid_cluster_events = pd.DataFrame([{
        "event_id": "INV_2",
        "competitor": "Amazon",
        "event_date": "2024-01-15",
        "strategic_cluster": 99,  # Invalid cluster outside 1..5
    }])

    with pytest.raises(ValueError, match="Found invalid strategic cluster IDs"):
        TargetLabelBuilder(config=config).fit_events(invalid_cluster_events)


# ==============================================================================
# 4. Determinism Across Repeated Runs
# ==============================================================================


def test_deterministic_label_output_across_repeated_runs(label_builder: TargetLabelBuilder, target_cutoff: pd.Timestamp):
    """Running label generation twice on identical events yields identical matrices."""
    events_df = pd.DataFrame([
        {"event_id": "E1", "competitor": "Amazon", "event_date": "2024-01-20", "event_type": "pricing_change"},
        {"event_id": "E2", "competitor": "Walmart", "event_date": "2024-01-25", "event_type": "delivery"},
    ])
    cutoffs = [("train", "2024-01-15"), ("train", "2024-01-22")]

    builder1 = TargetLabelBuilder(config=config).fit_events(events_df)
    labels1 = builder1.build_label_matrix(cutoffs=cutoffs)

    builder2 = TargetLabelBuilder(config=config).fit_events(events_df)
    labels2 = builder2.build_label_matrix(cutoffs=cutoffs)

    pd.testing.assert_frame_equal(labels1, labels2)


# ==============================================================================
# 5. Full Grid Key Coverage (All 6,675 Expected Instances)
# ==============================================================================


def test_all_6675_expected_keys_appear_exactly_once(label_builder: TargetLabelBuilder):
    """The default observation schedule produces exactly 6,675 unique keys with non-null binary labels."""
    empty_events = pd.DataFrame(columns=["competitor", "event_date", "event_type"])
    builder = TargetLabelBuilder(config=config).fit_events(empty_events)

    labels_df = builder.build_label_matrix()

    assert len(labels_df) == 6675, f"Expected 6,675 rows, got {len(labels_df)}"

    # Check key uniqueness
    keys = list(zip(labels_df["cutoff_date"], labels_df["competitor"], labels_df["strategic_cluster"]))
    assert len(keys) == len(set(keys)), "Duplicate keys found in full grid!"

    # Check partition counts
    assert (labels_df["partition"] == "train").sum() == 309 * 15 == 4635
    assert (labels_df["partition"] == "validation").sum() == 74 * 15 == 1110
    assert (labels_df["partition"] == "test").sum() == 62 * 15 == 930

    # Check binary invariant and non-null
    assert labels_df["target_30d"].notna().all(), "Null labels found in output!"
    assert set(labels_df["target_30d"].unique()).issubset({0, 1})


# ==============================================================================
# 6. Raw Dataset Immutability Verification
# ==============================================================================


def test_raw_dataset_remains_unchanged(tmp_path: Path):
    """Verifies that fitting and generating labels does not modify the raw dataset file."""
    raw_path = config.raw_csv_path
    assert raw_path.exists(), f"Raw CSV not found at {raw_path}"

    initial_size = raw_path.stat().st_size
    initial_mtime = raw_path.stat().st_mtime_ns

    # Compute fast hash of raw dataset head and tail
    with open(raw_path, "rb") as f:
        head_sample = f.read(10000)
        f.seek(-10000, 2)
        tail_sample = f.read(10000)
    sample_hash_before = hashlib.sha256(head_sample + tail_sample).hexdigest()

    # Instantiate and fit using small sample without modifying original
    small_df = pd.read_csv(raw_path, nrows=50)
    builder = TargetLabelBuilder(config=config).fit_events(small_df)
    _ = builder.compute_labels_for_cutoff("2024-01-15")

    # Verify original file properties remained bitwise identical
    assert raw_path.stat().st_size == initial_size
    assert raw_path.stat().st_mtime_ns == initial_mtime

    with open(raw_path, "rb") as f:
        head_sample_after = f.read(10000)
        f.seek(-10000, 2)
        tail_sample_after = f.read(10000)
    sample_hash_after = hashlib.sha256(head_sample_after + tail_sample_after).hexdigest()

    assert sample_hash_before == sample_hash_after, "Raw dataset file was modified!"


# ==============================================================================
# 7. Label Artifact Serialization
# ==============================================================================


def test_build_and_save_labels_serialization(tmp_path: Path):
    """Verifies build_and_save_labels generates the expected artifact file and metrics."""
    out_file = tmp_path / "test_labels.csv"
    empty_events = pd.DataFrame(columns=["competitor", "event_date", "event_type"])
    builder = TargetLabelBuilder(config=config)

    cutoffs = [("train", "2024-01-08"), ("validation", "2024-01-15")]
    df, metrics = builder.build_and_save_labels(
        output_csv_path=out_file,
        events_df=empty_events,
        cutoffs=cutoffs,
        include_diagnostic_count=True,
    )

    assert out_file.exists()
    assert len(df) == 30
    assert metrics.row_count == 30
    assert metrics.train_rows == 15
    assert metrics.val_rows == 15
    assert "event_count_in_window" in df.columns
    assert "target_30d" in df.columns
    assert (df["target_30d"] == 0).all()
