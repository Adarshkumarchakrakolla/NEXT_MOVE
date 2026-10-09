"""Automated temporal integrity and leakage test suite for NextMove.

Tests Convention A invariants, cutoff grid generation, embargo isolation,
and binary label generation using fast in-memory synthetic fixtures.
"""

import pandas as pd
import pytest

from configs.prediction_config import (
    CLUSTER_NAMES,
    EVENT_TYPE_TO_CLUSTER,
    config,
)
from src.splitting.temporal import (
    assign_strategic_clusters,
    compute_binary_window_label,
    filter_feature_events,
    generate_weekly_monday_cutoffs,
    get_label_window_bounds,
    get_partition_cutoffs,
    normalize_to_date,
)


# ==============================================================================
# In-Memory Synthetic Fixtures
# ==============================================================================


@pytest.fixture
def sample_cutoff() -> pd.Timestamp:
    """Standard sample cutoff Monday for unit testing."""
    return pd.Timestamp("2024-01-08")  # Monday


@pytest.fixture
def synthetic_events_df(sample_cutoff: pd.Timestamp) -> pd.DataFrame:
    """Create a minimal synthetic dataframe with exact boundary dates relative to sample_cutoff."""
    T = sample_cutoff
    return pd.DataFrame([
        # Historical events (< T)
        {"event_id": "E1", "competitor": "Amazon", "event_date": (T - pd.Timedelta(days=2)).strftime("%Y-%m-%d"), "event_type": "pricing_change"},
        {"event_id": "E2", "competitor": "Amazon", "event_date": (T - pd.Timedelta(days=1)).strftime("%Y-%m-%d"), "event_type": "pricing_change"},
        # On cutoff date (T)
        {"event_id": "E3", "competitor": "Amazon", "event_date": T.strftime("%Y-%m-%d"), "event_type": "pricing_change"},
        # Inside label window [T, T + 29]
        {"event_id": "E4", "competitor": "Walmart", "event_date": (T + pd.Timedelta(days=10)).strftime("%Y-%m-%d"), "event_type": "pricing_change"},
        {"event_id": "E5", "competitor": "Amazon", "event_date": (T + pd.Timedelta(days=29)).strftime("%Y-%m-%d"), "event_type": "pricing_change"},
        # Outside label window (>= T + 30)
        {"event_id": "E6", "competitor": "Amazon", "event_date": (T + pd.Timedelta(days=30)).strftime("%Y-%m-%d"), "event_type": "pricing_change"},
    ])


# ==============================================================================
# 1. Taxonomy & Configuration Invariants
# ==============================================================================


def test_taxonomy_mapping_completeness_and_uniqueness():
    """Verify exactly 30 event types mapped once across 5 clusters with no unmapped types."""
    assert len(EVENT_TYPE_TO_CLUSTER) == 30
    assert set(CLUSTER_NAMES.keys()) == {1, 2, 3, 4, 5}

    mapped_event_types = set(EVENT_TYPE_TO_CLUSTER.keys())
    assert len(mapped_event_types) == 30

    # Ensure every cluster has event types assigned
    clusters_with_events = set(EVENT_TYPE_TO_CLUSTER.values())
    assert clusters_with_events == {1, 2, 3, 4, 5}

    # Configuration self-validation
    assert config.validate() is True


def test_cluster_derivation_from_event_type():
    """Verify strategic cluster assignment is explicitly derived from event_type."""
    df = pd.DataFrame([
        {"event_type": "pricing_change"},
        {"event_type": "delivery"},
        {"event_type": "ai_launch"},
        {"event_type": "seller_program"},
        {"event_type": "earnings"},
    ])

    derived = assign_strategic_clusters(df)
    assert "strategic_cluster" in derived.columns
    assert list(derived["strategic_cluster"]) == [1, 2, 3, 4, 5]

    # Test error on unknown event type
    bad_df = pd.DataFrame([{"event_type": "unknown_event_category"}])
    with pytest.raises(ValueError, match="Found unmapped event types"):
        assign_strategic_clusters(bad_df)


# ==============================================================================
# 2. Cutoff Grid & Partition Count Invariants
# ==============================================================================


def test_partition_cutoff_counts_and_boundaries():
    """Verify expected cutoff counts: train 309, validation 74, test 62 (total 445)."""
    train_cutoffs = get_partition_cutoffs("train")
    val_cutoffs = get_partition_cutoffs("validation")
    test_cutoffs = get_partition_cutoffs("test")

    assert len(train_cutoffs) == 309
    assert len(val_cutoffs) == 74
    assert len(test_cutoffs) == 62
    assert len(train_cutoffs) + len(val_cutoffs) + len(test_cutoffs) == 445

    # Boundary dates
    assert train_cutoffs[0] == pd.Timestamp("2018-01-01")
    assert train_cutoffs[-1] == pd.Timestamp("2023-11-27")
    assert val_cutoffs[0] == pd.Timestamp("2024-01-01")
    assert val_cutoffs[-1] == pd.Timestamp("2025-05-26")
    assert test_cutoffs[0] == pd.Timestamp("2025-07-07")
    assert test_cutoffs[-1] == pd.Timestamp("2026-09-07")

    # All generated cutoffs must be Mondays (dayofweek == 0)
    for c in train_cutoffs + val_cutoffs + test_cutoffs:
        assert c.dayofweek == 0


def test_partition_embargo_isolation():
    """Verify that train and validation label windows strictly close before succeeding partition cutoffs."""
    train_cutoffs = get_partition_cutoffs("train")
    val_cutoffs = get_partition_cutoffs("validation")
    test_cutoffs = get_partition_cutoffs("test")

    # Train last label window: [2023-11-27, 2023-12-26]
    _, train_last_label_end = get_label_window_bounds(train_cutoffs[-1])
    assert train_last_label_end == pd.Timestamp("2023-12-26")
    assert train_last_label_end < val_cutoffs[0]
    gap_train_to_val = (val_cutoffs[0] - train_last_label_end).days - 1
    assert gap_train_to_val == 5  # 5 full calendar days buffer

    # Validation last label window: [2025-05-26, 2025-06-24]
    _, val_last_label_end = get_label_window_bounds(val_cutoffs[-1])
    assert val_last_label_end == pd.Timestamp("2025-06-24")
    assert val_last_label_end < test_cutoffs[0]
    gap_val_to_test = (test_cutoffs[0] - val_last_label_end).days - 1
    assert gap_val_to_test == 12  # 12 full calendar days buffer


# ==============================================================================
# 3. Convention A Feature Exclusion Invariants
# ==============================================================================


def test_feature_exclusion_on_and_after_cutoff_date(synthetic_events_df: pd.DataFrame, sample_cutoff: pd.Timestamp):
    """Enforce Convention A: Features must exclude events on cutoff date T and after it."""
    features_df = filter_feature_events(synthetic_events_df, cutoff_date=sample_cutoff)

    # Must contain E1 (T - 2) and E2 (T - 1)
    assert set(features_df["event_id"]) == {"E1", "E2"}

    # Must strictly exclude E3 (dated on T) and future events (E4, E5, E6)
    assert "E3" not in features_df["event_id"].values
    assert "E4" not in features_df["event_id"].values
    assert "E5" not in features_df["event_id"].values
    assert "E6" not in features_df["event_id"].values

    # Invariant: max feature date < cutoff date
    max_feat_date = pd.to_datetime(features_df["event_date"]).max()
    assert max_feat_date < sample_cutoff


# ==============================================================================
# 4. Label Window Exact Boundary Behavior
# ==============================================================================


def test_label_window_first_and_last_day_inclusion(sample_cutoff: pd.Timestamp):
    """Events on the first day (T) and last day (T + 29) of the 30-day window must produce label 1."""
    T = sample_cutoff

    # Event on first day of window (T)
    first_day_df = pd.DataFrame([{
        "event_id": "FIRST",
        "competitor": "Amazon",
        "event_date": T.strftime("%Y-%m-%d"),
        "event_type": "pricing_change",  # Cluster 1
    }])
    label_first = compute_binary_window_label(
        events_df=first_day_df,
        competitor="Amazon",
        cluster_id=1,
        cutoff_date=T,
    )
    assert label_first == 1

    # Event on last day of window (T + 29 days)
    last_day_df = pd.DataFrame([{
        "event_id": "LAST",
        "competitor": "Amazon",
        "event_date": (T + pd.Timedelta(days=29)).strftime("%Y-%m-%d"),
        "event_type": "pricing_change",  # Cluster 1
    }])
    label_last = compute_binary_window_label(
        events_df=last_day_df,
        competitor="Amazon",
        cluster_id=1,
        cutoff_date=T,
    )
    assert label_last == 1


def test_label_window_exclusion_one_day_before_and_after(sample_cutoff: pd.Timestamp):
    """Events one day before (T - 1) or one day after (T + 30) the window must produce label 0."""
    T = sample_cutoff

    # Event one day before window (T - 1)
    before_df = pd.DataFrame([{
        "event_id": "BEFORE",
        "competitor": "Amazon",
        "event_date": (T - pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
        "event_type": "pricing_change",  # Cluster 1
    }])
    label_before = compute_binary_window_label(
        events_df=before_df,
        competitor="Amazon",
        cluster_id=1,
        cutoff_date=T,
    )
    assert label_before == 0

    # Event one day after window (T + 30)
    after_df = pd.DataFrame([{
        "event_id": "AFTER",
        "competitor": "Amazon",
        "event_date": (T + pd.Timedelta(days=30)).strftime("%Y-%m-%d"),
        "event_type": "pricing_change",  # Cluster 1
    }])
    label_after = compute_binary_window_label(
        events_df=after_df,
        competitor="Amazon",
        cluster_id=1,
        cutoff_date=T,
    )
    assert label_after == 0


def test_empty_event_windows_produce_label_zero(sample_cutoff: pd.Timestamp):
    """An empty event window must correctly produce label 0 without error."""
    empty_df = pd.DataFrame(columns=["event_id", "competitor", "event_date", "event_type"])
    label = compute_binary_window_label(
        events_df=empty_df,
        competitor="Amazon",
        cluster_id=1,
        cutoff_date=sample_cutoff,
    )
    assert label == 0


# ==============================================================================
# 5. Competitor & Cluster Filtering Invariants
# ==============================================================================


def test_competitor_and_cluster_filtering(sample_cutoff: pd.Timestamp):
    """An event for another competitor or another cluster must not trigger a false positive."""
    T = sample_cutoff

    events_df = pd.DataFrame([
        # Walmart event in Cluster 1 inside window
        {"event_id": "W1", "competitor": "Walmart", "event_date": (T + pd.Timedelta(days=5)).strftime("%Y-%m-%d"), "event_type": "pricing_change"},
        # Amazon event in Cluster 2 (delivery) inside window
        {"event_id": "A2", "competitor": "Amazon", "event_date": (T + pd.Timedelta(days=15)).strftime("%Y-%m-%d"), "event_type": "delivery"},
    ])

    # Target: Amazon in Cluster 1 (pricing) -> No Amazon pricing events exist
    assert compute_binary_window_label(events_df, competitor="Amazon", cluster_id=1, cutoff_date=T) == 0

    # Target: Walmart in Cluster 1 (pricing) -> Walmart pricing event exists
    assert compute_binary_window_label(events_df, competitor="Walmart", cluster_id=1, cutoff_date=T) == 1

    # Target: Amazon in Cluster 2 (logistics) -> Amazon delivery event exists
    assert compute_binary_window_label(events_df, competitor="Amazon", cluster_id=2, cutoff_date=T) == 1

    # Target: eBay in any cluster -> 0
    assert compute_binary_window_label(events_df, competitor="eBay", cluster_id=1, cutoff_date=T) == 0


# ==============================================================================
# 6. Right-Censoring & Incomplete Forward Windows
# ==============================================================================


def test_rejection_of_incomplete_forward_windows():
    """Reject cutoffs whose 30-day forward window extends beyond the dataset's max date."""
    max_dataset_date = "2026-10-09"

    # Eligible cutoff: 2026-09-07 (Window: [2026-09-07, 2026-10-06] <= 2026-10-09)
    valid_start, valid_end = get_label_window_bounds("2026-09-07", max_dataset_date=max_dataset_date)
    assert valid_start == pd.Timestamp("2026-09-07")
    assert valid_end == pd.Timestamp("2026-10-06")

    # Incomplete cutoff: 2026-09-14 (Window: [2026-09-14, 2026-10-13] > 2026-10-09)
    with pytest.raises(ValueError, match="exceeding max dataset date"):
        get_label_window_bounds("2026-09-14", max_dataset_date=max_dataset_date)

    # Grid generator must reject ranges that include incomplete cutoffs
    with pytest.raises(ValueError, match="extends beyond max dataset date"):
        generate_weekly_monday_cutoffs(
            start_date="2026-09-01",
            end_date="2026-09-21",
            max_dataset_date=max_dataset_date,
        )


# ==============================================================================
# 7. Determinism Across Runs
# ==============================================================================


def test_deterministic_results_across_runs(synthetic_events_df: pd.DataFrame, sample_cutoff: pd.Timestamp):
    """Verify that temporal grid generation, feature filtering, and label computation are 100% deterministic."""
    train_cutoffs_run1 = get_partition_cutoffs("train")
    train_cutoffs_run2 = get_partition_cutoffs("train")
    assert train_cutoffs_run1 == train_cutoffs_run2

    filtered_run1 = filter_feature_events(synthetic_events_df, sample_cutoff)
    filtered_run2 = filter_feature_events(synthetic_events_df, sample_cutoff)
    pd.testing.assert_frame_equal(filtered_run1, filtered_run2)

    label_run1 = compute_binary_window_label(synthetic_events_df, "Amazon", 1, sample_cutoff)
    label_run2 = compute_binary_window_label(synthetic_events_df, "Amazon", 1, sample_cutoff)
    assert label_run1 == label_run2
