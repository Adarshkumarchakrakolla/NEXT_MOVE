"""Unit test suite for PointInTimeFeatureBuilder.

Validates lookback calculations, Convention A temporal exclusion, missing history sentinels,
rival filtering, decay calculations, and schema invariants using fast in-memory fixtures.
"""

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


# ==============================================================================
# Fixtures
# ==============================================================================


@pytest.fixture
def target_cutoff() -> pd.Timestamp:
    """Monday cutoff date for controlled feature testing."""
    return pd.Timestamp("2024-01-15")  # Monday. T - 1 = 2024-01-14 (Sunday)


@pytest.fixture
def controlled_events_df(target_cutoff: pd.Timestamp) -> pd.DataFrame:
    """Construct a deterministic event log with exact age distances from target_cutoff."""
    T = target_cutoff
    sunday = T - pd.Timedelta(days=1)  # 2024-01-14

    return pd.DataFrame([
        # --- Amazon, Cluster 1 (Pricing) ---
        # Event 1: Sunday T - 1 (age = 0 days), magnitude = 2.0
        {"event_id": "A1", "competitor": "Amazon", "event_date": sunday.strftime("%Y-%m-%d"), "event_type": "pricing_change", "magnitude_index": 2.0},
        # Event 2: T - 11 days (age = 10 days, in 14d, 30d, 60d, 90d, 365d), magnitude = 3.0
        {"event_id": "A2", "competitor": "Amazon", "event_date": (sunday - pd.Timedelta(days=10)).strftime("%Y-%m-%d"), "event_type": "promotion", "magnitude_index": 3.0},
        # Event 3: T - 60 days (age = 59 days, in 60d, 90d, 365d), magnitude = 1.0
        {"event_id": "A3", "competitor": "Amazon", "event_date": (sunday - pd.Timedelta(days=59)).strftime("%Y-%m-%d"), "event_type": "membership", "magnitude_index": 1.0},
        # Event 4: T - 75 days (age = 74 days, in 90d, 365d; outside 60d), magnitude = 4.0
        {"event_id": "A4", "competitor": "Amazon", "event_date": (sunday - pd.Timedelta(days=74)).strftime("%Y-%m-%d"), "event_type": "pricing_change", "magnitude_index": 4.0},
        # Event 5: T - 200 days (age = 199 days, in 365d; outside 90d), magnitude = 5.0
        {"event_id": "A5", "competitor": "Amazon", "event_date": (sunday - pd.Timedelta(days=199)).strftime("%Y-%m-%d"), "event_type": "advertising", "magnitude_index": 5.0},
        # Event 6: On cutoff date T (Monday 2024-01-15) - MUST BE EXCLUDED!
        {"event_id": "A_LEAK1", "competitor": "Amazon", "event_date": T.strftime("%Y-%m-%d"), "event_type": "pricing_change", "magnitude_index": 999.0},
        # Event 7: After cutoff date (Tuesday 2024-01-16) - MUST BE EXCLUDED!
        {"event_id": "A_LEAK2", "competitor": "Amazon", "event_date": (T + pd.Timedelta(days=1)).strftime("%Y-%m-%d"), "event_type": "pricing_change", "magnitude_index": 999.0},

        # --- Rivals in Cluster 1 (Pricing) within 14-day window [T - 14, T - 1] ---
        # Walmart has 3 events in [T - 14, T - 1]
        {"event_id": "W1", "competitor": "Walmart", "event_date": (sunday - pd.Timedelta(days=2)).strftime("%Y-%m-%d"), "event_type": "pricing_change", "magnitude_index": 1.5},
        {"event_id": "W2", "competitor": "Walmart", "event_date": (sunday - pd.Timedelta(days=5)).strftime("%Y-%m-%d"), "event_type": "promotion", "magnitude_index": 2.0},
        {"event_id": "W3", "competitor": "Walmart", "event_date": (sunday - pd.Timedelta(days=12)).strftime("%Y-%m-%d"), "event_type": "retail_media", "magnitude_index": 1.0},
        # eBay has 2 events in [T - 14, T - 1]
        {"event_id": "E1", "competitor": "eBay", "event_date": (sunday - pd.Timedelta(days=3)).strftime("%Y-%m-%d"), "event_type": "pricing_change", "magnitude_index": 1.0},
        {"event_id": "E2", "competitor": "eBay", "event_date": (sunday - pd.Timedelta(days=8)).strftime("%Y-%m-%d"), "event_type": "membership", "magnitude_index": 2.5},

        # --- Amazon in Cluster 2 (Logistics) ---
        # 1 delivery event 5 days ago
        {"event_id": "A_LOG1", "competitor": "Amazon", "event_date": (sunday - pd.Timedelta(days=5)).strftime("%Y-%m-%d"), "event_type": "delivery", "magnitude_index": 3.5},
    ])


# ==============================================================================
# 1. Schema, Keys, and Uniqueness
# ==============================================================================


def test_feature_matrix_combination_uniqueness_and_completeness(controlled_events_df: pd.DataFrame):
    """Assert every competitor-cluster-cutoff combination exists exactly once with zero nulls."""
    builder = PointInTimeFeatureBuilder().fit_events(controlled_events_df)
    cutoffs = [
        ("train", "2024-01-08"),
        ("train", "2024-01-15"),
        ("train", "2024-01-22"),
    ]
    df = builder.build_feature_matrix(cutoffs=cutoffs)

    # 3 cutoffs * 3 competitors * 5 clusters = 45 instances
    assert len(df) == 45

    # Check key uniqueness
    keys = list(zip(df["cutoff_date"], df["competitor"], df["strategic_cluster"]))
    assert len(keys) == len(set(keys)), "Duplicate feature keys found!"

    # Assert zero NaN / null values across entire matrix
    assert df.isna().sum().sum() == 0, "Missing or NaN feature values detected!"


def test_expected_feature_columns_and_data_types(controlled_events_df: pd.DataFrame, target_cutoff: pd.Timestamp):
    """Verify expected feature columns and strictly conforming data types."""
    builder = PointInTimeFeatureBuilder().fit_events(controlled_events_df)
    df = builder.build_feature_matrix(cutoffs=[("train", target_cutoff)])

    expected_columns = [
        "cutoff_date",
        "competitor",
        "strategic_cluster",
        "cluster_name",
        "partition",
        "days_since_last_cluster_event",
        "has_prior_cluster_event",
        "cluster_event_count_30d",
        "cluster_event_count_90d",
        "cluster_event_count_365d",
        "velocity_ratio_30_90",
        "rival_event_count_14d",
        "decayed_magnitude_mass_60d",
        "quarter",
        "month",
        "is_q4",
    ]
    assert list(df.columns) == expected_columns

    # Check types
    assert pd.api.types.is_numeric_dtype(df["strategic_cluster"])
    assert pd.api.types.is_float_dtype(df["days_since_last_cluster_event"])
    assert pd.api.types.is_integer_dtype(df["has_prior_cluster_event"])
    assert pd.api.types.is_integer_dtype(df["cluster_event_count_30d"])
    assert pd.api.types.is_integer_dtype(df["cluster_event_count_90d"])
    assert pd.api.types.is_integer_dtype(df["cluster_event_count_365d"])
    assert pd.api.types.is_float_dtype(df["velocity_ratio_30_90"])
    assert pd.api.types.is_integer_dtype(df["rival_event_count_14d"])
    assert pd.api.types.is_float_dtype(df["decayed_magnitude_mass_60d"])
    assert pd.api.types.is_integer_dtype(df["quarter"])
    assert pd.api.types.is_integer_dtype(df["month"])
    assert pd.api.types.is_integer_dtype(df["is_q4"])


# ==============================================================================
# 2. Hand-Built Mathematical Accuracy
# ==============================================================================


def test_exact_calculations_on_hand_built_fixture(controlled_events_df: pd.DataFrame, target_cutoff: pd.Timestamp):
    """Verify exact numerical calculations against manually derived mathematical formulas."""
    builder = PointInTimeFeatureBuilder().fit_events(controlled_events_df)
    df = builder.build_feature_matrix(cutoffs=[("train", target_cutoff)])

    # Isolate Amazon in Cluster 1 (Pricing)
    amz_p1 = df[(df["competitor"] == "Amazon") & (df["strategic_cluster"] == 1)].iloc[0]

    # 1. Recency: Event A1 occurred on Sunday T - 1 (age = 0.0 days)
    assert amz_p1["has_prior_cluster_event"] == 1
    assert amz_p1["days_since_last_cluster_event"] == 0.0

    # 2. Counts:
    # 30d window: Event A1 (age 0), Event A2 (age 10) = 2 events
    assert amz_p1["cluster_event_count_30d"] == 2

    # 90d window: Event A1 (0), A2 (10), A3 (59), A4 (74) = 4 events
    assert amz_p1["cluster_event_count_90d"] == 4

    # 365d window: Event A1 (0), A2 (10), A3 (59), A4 (74), A5 (199) = 5 events
    assert amz_p1["cluster_event_count_365d"] == 5

    # 3. Smoothed velocity ratio: (2 + 1) / (4/3 + 1) = 3 / (7/3) = 9/7 = 1.285714
    expected_ratio = (2.0 + 1.0) / ((4.0 / 3.0) + 1.0)
    assert amz_p1["velocity_ratio_30_90"] == pytest.approx(expected_ratio, rel=1e-5)

    # 4. Decayed magnitude mass over 60 days [age <= 59 days]:
    # A1: mag 2.0 * exp(-0.05 * 0) = 2.0
    # A2: mag 3.0 * exp(-0.05 * 10) = 3.0 * exp(-0.5) = 1.819591979
    # A3: mag 1.0 * exp(-0.05 * 59) = 1.0 * exp(-2.95) = 0.052339705
    # (A4 has age 74 > 59, so not in 60d window)
    expected_decay = (
        2.0 * np.exp(-0.05 * 0)
        + 3.0 * np.exp(-0.05 * 10)
        + 1.0 * np.exp(-0.05 * 59)
    )
    assert amz_p1["decayed_magnitude_mass_60d"] == pytest.approx(expected_decay, rel=1e-4)


# ==============================================================================
# 3. Temporal Leakage & Exclusion
# ==============================================================================


def test_exclusion_of_events_on_and_after_cutoff(controlled_events_df: pd.DataFrame, target_cutoff: pd.Timestamp):
    """Enforce Convention A: Events on cutoff date T or after it must have zero impact on features."""
    builder = PointInTimeFeatureBuilder().fit_events(controlled_events_df)
    df = builder.build_feature_matrix(cutoffs=[("train", target_cutoff)])

    amz_p1 = df[(df["competitor"] == "Amazon") & (df["strategic_cluster"] == 1)].iloc[0]

    # If A_LEAK1 (mag 999, dated on T) were leaked:
    # decayed mass would be > 1000 and count_30d would be >= 3.
    assert amz_p1["cluster_event_count_30d"] == 2
    assert amz_p1["decayed_magnitude_mass_60d"] < 10.0


# ==============================================================================
# 4. Missing History and Recency Sentinel
# ==============================================================================


def test_missing_history_and_recency_sentinel(controlled_events_df: pd.DataFrame, target_cutoff: pd.Timestamp):
    """Verify that unobserved history sets has_prior_cluster_event=0 and days_since=-1.0."""
    builder = PointInTimeFeatureBuilder().fit_events(controlled_events_df)
    df = builder.build_feature_matrix(cutoffs=[("train", target_cutoff)])

    # Amazon has ZERO events in Cluster 5 (Corporate Strategy)
    amz_p5 = df[(df["competitor"] == "Amazon") & (df["strategic_cluster"] == 5)].iloc[0]

    assert amz_p5["has_prior_cluster_event"] == 0
    assert amz_p5["days_since_last_cluster_event"] == -1.0
    assert amz_p5["cluster_event_count_30d"] == 0
    assert amz_p5["cluster_event_count_90d"] == 0
    assert amz_p5["cluster_event_count_365d"] == 0
    assert amz_p5["velocity_ratio_30_90"] == 1.0  # (0 + 1) / (0 + 1)
    assert amz_p5["decayed_magnitude_mass_60d"] == 0.0


# ==============================================================================
# 5. Rival Filtering
# ==============================================================================


def test_rival_filtering(controlled_events_df: pd.DataFrame, target_cutoff: pd.Timestamp):
    """Verify rival event count over [T - 14, T - 1] sums other competitors in the cluster."""
    builder = PointInTimeFeatureBuilder().fit_events(controlled_events_df)
    df = builder.build_feature_matrix(cutoffs=[("train", target_cutoff)])

    # In Cluster 1 over [T - 14, T - 1]:
    # Amazon has 2 events (A1 at age 0, A2 at age 10)
    # Walmart has 3 events (W1, W2, W3)
    # eBay has 2 events (E1, E2)
    amz_p1 = df[(df["competitor"] == "Amazon") & (df["strategic_cluster"] == 1)].iloc[0]
    wal_p1 = df[(df["competitor"] == "Walmart") & (df["strategic_cluster"] == 1)].iloc[0]
    ebay_p1 = df[(df["competitor"] == "eBay") & (df["strategic_cluster"] == 1)].iloc[0]

    # For Amazon: rivals are Walmart (3) + eBay (2) = 5
    assert amz_p1["rival_event_count_14d"] == 5

    # For Walmart: rivals are Amazon (2) + eBay (2) = 4
    assert wal_p1["rival_event_count_14d"] == 4

    # For eBay: rivals are Amazon (2) + Walmart (3) = 5
    assert ebay_p1["rival_event_count_14d"] == 5


# ==============================================================================
# 6. Calendar Context Features
# ==============================================================================


def test_calendar_features(controlled_events_df: pd.DataFrame):
    """Verify quarter, month, and is_q4 calendar features from cutoff date."""
    builder = PointInTimeFeatureBuilder().fit_events(controlled_events_df)

    # Q1 test
    df_q1 = builder.build_feature_matrix(cutoffs=[("train", "2024-01-15")])
    row_q1 = df_q1.iloc[0]
    assert row_q1["quarter"] == 1
    assert row_q1["month"] == 1
    assert row_q1["is_q4"] == 0

    # Q4 test (e.g. October Monday)
    df_q4 = builder.build_feature_matrix(cutoffs=[("train", "2024-10-14")])
    row_q4 = df_q4.iloc[0]
    assert row_q4["quarter"] == 4
    assert row_q4["month"] == 10
    assert row_q4["is_q4"] == 1


# ==============================================================================
# 7. Determinism and Repeatability
# ==============================================================================


def test_repeatability_across_repeated_runs(controlled_events_df: pd.DataFrame, target_cutoff: pd.Timestamp):
    """Assert repeated executions produce bitwise identical feature tables."""
    builder = PointInTimeFeatureBuilder().fit_events(controlled_events_df)
    cutoffs = [("train", target_cutoff), ("validation", "2024-06-10")]

    df1 = builder.build_feature_matrix(cutoffs=cutoffs)
    df2 = builder.build_feature_matrix(cutoffs=cutoffs)

    pd.testing.assert_frame_equal(df1, df2)
