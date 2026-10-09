"""Automated test suite for TargetLabelBuilder and Modeling Dataset Integrity.

Verifies:
- In-memory window boundary tests (first day, last day, outside window)
- Empty windows producing 0
- Competitor and strategic cluster isolation
- Binary label invariants (strictly {0, 1})
- Rejection of truncated forward windows
- Feature-label join validation (1:1 bijection, duplicate rejection, domain validation)
- Feature value preservation (unaltered feature matrix after join)
- Deterministic output across repeated runs
- Full grid key coverage (all 6,675 expected instances)
"""

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
from src.features.feature_builder import PointInTimeFeatureBuilder
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
        {"event_id": f"E_{i}", "competitor": "eBay", "event_date": (T + pd.Timedelta(days=i)).strftime("%Y-%m-%d"), "event_type": "marketplace_change"}
        for i in range(10)
    ])

    builder = TargetLabelBuilder(config=config).fit_events(events_df)
    labels = builder.compute_labels_for_cutoff(T)

    ebay_c4 = [r for r in labels if r["competitor"] == "eBay" and r["strategic_cluster"] == 4][0]
    assert ebay_c4["target_30d"] == 1
    assert type(ebay_c4["target_30d"]) in (int, np.int64, np.int32)


# ==============================================================================
# 3. Truncated Forward Window Rejection
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


# ==============================================================================
# 4. Feature-Label Join Integrity and Validation
# ==============================================================================


def test_join_features_and_labels_exact_match(label_builder: TargetLabelBuilder, target_cutoff: pd.Timestamp):
    """Assert successful join when features and labels have exact matching keys."""
    cutoffs = [("train", "2024-01-08"), ("train", "2024-01-15")]

    # Mock features dataframe
    feat_rows = []
    for part, c in cutoffs:
        for comp in config.competitors:
            for cluster_id in range(1, 6):
                feat_rows.append({
                    "cutoff_date": c,
                    "competitor": comp,
                    "strategic_cluster": cluster_id,
                    "partition": part,
                    "dummy_feature_1": 42.0,
                    "dummy_feature_2": 99,
                })
    features_df = pd.DataFrame(feat_rows)

    # Empty events -> all labels 0
    builder = TargetLabelBuilder(config=config).fit_events(
        pd.DataFrame(columns=["competitor", "event_date", "event_type"])
    )
    labels_df = builder.build_label_matrix(cutoffs=cutoffs)

    joined_df = builder.join_features_and_labels(features_df, labels_df)

    assert len(joined_df) == len(features_df) == 30
    assert "target_30d" in joined_df.columns
    assert (joined_df["target_30d"] == 0).all()
    assert joined_df["dummy_feature_1"].tolist() == features_df["dummy_feature_1"].tolist()
    assert joined_df["dummy_feature_2"].tolist() == features_df["dummy_feature_2"].tolist()


def test_join_rejects_duplicate_keys(label_builder: TargetLabelBuilder):
    """Join must reject duplicate join keys in features or labels."""
    features_df = pd.DataFrame([
        {"cutoff_date": "2024-01-15", "competitor": "Amazon", "strategic_cluster": 1, "feat": 1.0},
        {"cutoff_date": "2024-01-15", "competitor": "Amazon", "strategic_cluster": 1, "feat": 2.0},  # Duplicate
    ])
    labels_df = pd.DataFrame([
        {"cutoff_date": "2024-01-15", "competitor": "Amazon", "strategic_cluster": 1, "target_30d": 1},
    ])

    with pytest.raises(ValueError, match="Duplicate join keys detected in features"):
        label_builder.join_features_and_labels(features_df, labels_df)

    # Duplicate in labels
    valid_features = pd.DataFrame([
        {"cutoff_date": "2024-01-15", "competitor": "Amazon", "strategic_cluster": 1, "feat": 1.0},
    ])
    dup_labels = pd.DataFrame([
        {"cutoff_date": "2024-01-15", "competitor": "Amazon", "strategic_cluster": 1, "target_30d": 1},
        {"cutoff_date": "2024-01-15", "competitor": "Amazon", "strategic_cluster": 1, "target_30d": 0},
    ])
    with pytest.raises(ValueError, match="Duplicate join keys detected in labels"):
        label_builder.join_features_and_labels(valid_features, dup_labels)


def test_join_rejects_missing_or_orphan_keys(label_builder: TargetLabelBuilder):
    """Join must reject any mismatch between feature instances and label instances."""
    features_df = pd.DataFrame([
        {"cutoff_date": "2024-01-15", "competitor": "Amazon", "strategic_cluster": 1, "feat": 1.0},
        {"cutoff_date": "2024-01-15", "competitor": "Amazon", "strategic_cluster": 2, "feat": 1.0},
    ])
    labels_df = pd.DataFrame([
        {"cutoff_date": "2024-01-15", "competitor": "Amazon", "strategic_cluster": 1, "target_30d": 1},
        # Missing cluster 2 in labels
    ])

    with pytest.raises(ValueError, match="missing target labels"):
        label_builder.join_features_and_labels(features_df, labels_df)


def test_join_rejects_unexpected_competitor_or_invalid_cluster(label_builder: TargetLabelBuilder):
    """Join must reject competitors not in {Amazon, Walmart, eBay} or clusters outside 1..5."""
    features_df = pd.DataFrame([
        {"cutoff_date": "2024-01-15", "competitor": "Target", "strategic_cluster": 1, "feat": 1.0},
    ])
    labels_df = pd.DataFrame([
        {"cutoff_date": "2024-01-15", "competitor": "Target", "strategic_cluster": 1, "target_30d": 1},
    ])
    with pytest.raises(ValueError, match="unexpected competitors"):
        label_builder.join_features_and_labels(features_df, labels_df)

    # Invalid cluster ID 99
    features_invalid_cluster = pd.DataFrame([
        {"cutoff_date": "2024-01-15", "competitor": "Amazon", "strategic_cluster": 99, "feat": 1.0},
    ])
    labels_invalid_cluster = pd.DataFrame([
        {"cutoff_date": "2024-01-15", "competitor": "Amazon", "strategic_cluster": 99, "target_30d": 1},
    ])
    with pytest.raises(ValueError, match="invalid cluster IDs"):
        label_builder.join_features_and_labels(features_invalid_cluster, labels_invalid_cluster)


def test_no_change_to_existing_feature_values_after_joining(label_builder: TargetLabelBuilder):
    """All existing feature columns and their values must be identical before and after join."""
    features_df = pd.DataFrame([
        {
            "cutoff_date": "2024-01-15",
            "competitor": "Amazon",
            "strategic_cluster": 1,
            "days_since_last_cluster_event": 5.25,
            "has_prior_cluster_event": 1,
            "cluster_event_count_30d": 12,
            "velocity_ratio_30_90": 1.333333,
            "quarter": 1,
            "is_q4": 0,
        },
        {
            "cutoff_date": "2024-01-15",
            "competitor": "Walmart",
            "strategic_cluster": 2,
            "days_since_last_cluster_event": -1.0,
            "has_prior_cluster_event": 0,
            "cluster_event_count_30d": 0,
            "velocity_ratio_30_90": 1.0,
            "quarter": 1,
            "is_q4": 0,
        },
    ])
    labels_df = pd.DataFrame([
        {"cutoff_date": "2024-01-15", "competitor": "Amazon", "strategic_cluster": 1, "target_30d": 1},
        {"cutoff_date": "2024-01-15", "competitor": "Walmart", "strategic_cluster": 2, "target_30d": 0},
    ])

    joined_df = label_builder.join_features_and_labels(features_df, labels_df)

    for col in features_df.columns:
        np.testing.assert_array_equal(joined_df[col].values, features_df[col].values)


# ==============================================================================
# 5. Determinism Across Repeated Runs
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
# 6. Full Grid Integration Test (All 6,675 Expected Keys)
# ==============================================================================


def test_all_6675_expected_keys_appear_exactly_once(label_builder: TargetLabelBuilder):
    """The default observation schedule produces exactly 6,675 unique keys with binary labels."""
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

    # Check binary invariant
    assert set(labels_df["target_30d"].unique()).issubset({0, 1})
