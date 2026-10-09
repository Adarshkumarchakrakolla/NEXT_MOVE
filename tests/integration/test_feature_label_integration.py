"""Integration test suite for Step 7: Feature-Label Integration.

Verifies:
- Successful one-to-one join on (cutoff_date, competitor, strategic_cluster)
- Rejection of duplicate keys in features or labels
- Rejection of missing labels or orphan keys
- Binary target validation (non-binary labels or nulls rejected)
- Partition consistency between feature matrix and label matrix
- Feature values strictly unchanged after joining (verified by key, not row order)
- No duplicated columns (no _x, _y suffixes) and no diagnostic leakage (event_count_in_window excluded)
- Deterministic output across repeated runs
- Correct row counts by partition (Train: 4,635, Val: 1,110, Test: 930, Total: 6,675)
- Verification against the materialized artifacts on disk
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
from src.labels.label_builder import TargetLabelBuilder, integrate_features_and_labels


# ==============================================================================
# Fixtures
# ==============================================================================


@pytest.fixture
def builder() -> TargetLabelBuilder:
    """TargetLabelBuilder instance initialized with project config."""
    return TargetLabelBuilder(config=config)


@pytest.fixture
def sample_feature_df() -> pd.DataFrame:
    """Controlled 2-cutoff (30 instance) feature dataframe."""
    rows = []
    for c in ["2024-01-08", "2024-01-15"]:
        for comp in config.competitors:
            for cluster_id in range(1, 6):
                rows.append({
                    "cutoff_date": c,
                    "competitor": comp,
                    "strategic_cluster": cluster_id,
                    "cluster_name": config.cluster_names[cluster_id],
                    "partition": "validation",
                    "days_since_last_cluster_event": 5.0,
                    "has_prior_cluster_event": 1,
                    "cluster_event_count_30d": 10,
                    "cluster_event_count_90d": 25,
                    "cluster_event_count_365d": 100,
                    "velocity_ratio_30_90": 1.2,
                    "rival_event_count_14d": 3,
                    "decayed_magnitude_mass_60d": 4.5,
                    "quarter": 1,
                    "month": 1,
                    "is_q4": 0,
                })
    return pd.DataFrame(rows)


@pytest.fixture
def sample_label_df(sample_feature_df: pd.DataFrame) -> pd.DataFrame:
    """Matching label dataframe for the sample feature dataframe."""
    rows = []
    for _, row in sample_feature_df.iterrows():
        rows.append({
            "cutoff_date": row["cutoff_date"],
            "competitor": row["competitor"],
            "strategic_cluster": row["strategic_cluster"],
            "partition": row["partition"],
            "target_30d": 1 if row["strategic_cluster"] % 2 == 1 else 0,
            "event_count_in_window": 15 if row["strategic_cluster"] % 2 == 1 else 0,
        })
    return pd.DataFrame(rows)


# ==============================================================================
# 1. Successful One-to-One Join and Schema Preservation
# ==============================================================================


def test_successful_one_to_one_join(builder: TargetLabelBuilder, sample_feature_df: pd.DataFrame, sample_label_df: pd.DataFrame):
    """Features and labels join 1:1, adding target_30d and keeping all original feature columns."""
    joined = builder.join_features_and_labels(sample_feature_df, sample_label_df)

    assert len(joined) == len(sample_feature_df) == 30
    assert len(joined.columns) == len(sample_feature_df.columns) + 1
    assert "target_30d" in joined.columns
    assert joined.isna().sum().sum() == 0

    # Ensure target values match expected
    for _, row in joined.iterrows():
        expected_target = 1 if row["strategic_cluster"] % 2 == 1 else 0
        assert row["target_30d"] == expected_target


def test_join_on_shuffled_order(builder: TargetLabelBuilder, sample_feature_df: pd.DataFrame, sample_label_df: pd.DataFrame):
    """Join must be robust against shuffled row order and not rely on row position."""
    shuffled_features = sample_feature_df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    shuffled_labels = sample_label_df.sample(frac=1.0, random_state=99).reset_index(drop=True)

    joined = builder.join_features_and_labels(shuffled_features, shuffled_labels)

    assert len(joined) == 30
    # Match by key
    for _, row in joined.iterrows():
        expected_target = 1 if row["strategic_cluster"] % 2 == 1 else 0
        assert row["target_30d"] == expected_target


# ==============================================================================
# 2. Key Uniqueness and Rejection of Duplicates
# ==============================================================================


def test_join_rejects_duplicate_keys_in_features(builder: TargetLabelBuilder, sample_feature_df: pd.DataFrame, sample_label_df: pd.DataFrame):
    """Duplicate keys in features dataframe must raise ValueError."""
    dup_features = pd.concat([sample_feature_df, sample_feature_df.iloc[[0]]], ignore_index=True)

    with pytest.raises(ValueError, match="Duplicate join keys detected in features"):
        builder.join_features_and_labels(dup_features, sample_label_df)


def test_join_rejects_duplicate_keys_in_labels(builder: TargetLabelBuilder, sample_feature_df: pd.DataFrame, sample_label_df: pd.DataFrame):
    """Duplicate keys in labels dataframe must raise ValueError."""
    dup_labels = pd.concat([sample_label_df, sample_label_df.iloc[[0]]], ignore_index=True)

    with pytest.raises(ValueError, match="Duplicate join keys detected in labels"):
        builder.join_features_and_labels(sample_feature_df, dup_labels)


# ==============================================================================
# 3. Rejection of Missing and Orphan Keys
# ==============================================================================


def test_join_rejects_missing_labels(builder: TargetLabelBuilder, sample_feature_df: pd.DataFrame, sample_label_df: pd.DataFrame):
    """Missing label instance must raise ValueError."""
    missing_one_label = sample_label_df.iloc[1:].copy()

    with pytest.raises(ValueError, match="feature instances missing target labels"):
        builder.join_features_and_labels(sample_feature_df, missing_one_label)


def test_join_rejects_extra_orphan_labels(builder: TargetLabelBuilder, sample_feature_df: pd.DataFrame, sample_label_df: pd.DataFrame):
    """Extra label instance with no corresponding feature row must raise ValueError."""
    missing_one_feature = sample_feature_df.iloc[1:].copy()

    with pytest.raises(ValueError, match="target labels with no matching feature row"):
        builder.join_features_and_labels(missing_one_feature, sample_label_df)


# ==============================================================================
# 4. Binary Target Validation
# ==============================================================================


def test_binary_target_validation(builder: TargetLabelBuilder, sample_feature_df: pd.DataFrame, sample_label_df: pd.DataFrame):
    """Non-binary target values (e.g. 2, -1) or NaN must raise ValueError."""
    corrupted_labels = sample_label_df.copy()
    corrupted_labels.loc[0, "target_30d"] = 2

    with pytest.raises(ValueError, match="non-binary values"):
        builder.join_features_and_labels(sample_feature_df, corrupted_labels)

    nan_labels = sample_label_df.copy()
    nan_labels.loc[0, "target_30d"] = np.nan

    with pytest.raises(ValueError, match="NaN / null values"):
        builder.join_features_and_labels(sample_feature_df, nan_labels)


# ==============================================================================
# 5. Partition Consistency
# ==============================================================================


def test_partition_consistency_rejection(builder: TargetLabelBuilder, sample_feature_df: pd.DataFrame, sample_label_df: pd.DataFrame):
    """Mismatched partition values between features and labels must raise ValueError."""
    mismatched_labels = sample_label_df.copy()
    mismatched_labels.loc[0, "partition"] = "train"  # Features have 'validation'

    with pytest.raises(ValueError, match="Partition mismatch between features and labels"):
        builder.join_features_and_labels(sample_feature_df, mismatched_labels)


# ==============================================================================
# 6. Feature Values Unchanged & No Target Leakage
# ==============================================================================


def test_feature_values_unchanged_after_joining(builder: TargetLabelBuilder, sample_feature_df: pd.DataFrame, sample_label_df: pd.DataFrame):
    """All 16 original feature column values must be bitwise identical after join."""
    joined = builder.join_features_and_labels(sample_feature_df, sample_label_df)

    key_cols = ["cutoff_date", "competitor", "strategic_cluster"]
    joined_sorted = joined.sort_values(key_cols).reset_index(drop=True)
    orig_sorted = sample_feature_df.sort_values(key_cols).reset_index(drop=True)

    for col in sample_feature_df.columns:
        np.testing.assert_array_equal(
            joined_sorted[col].values,
            orig_sorted[col].values,
            err_msg=f"Feature column '{col}' modified during join!",
        )


def test_no_duplicated_columns_or_accidental_target_leakage(builder: TargetLabelBuilder, sample_feature_df: pd.DataFrame, sample_label_df: pd.DataFrame):
    """Ensures no duplicated columns (_x, _y) and diagnostic event_count_in_window is NOT in output."""
    joined = builder.join_features_and_labels(sample_feature_df, sample_label_df)

    # Check no duplicate suffixes
    assert not any(col.endswith("_x") or col.endswith("_y") for col in joined.columns)

    # Check diagnostic column exclusion
    assert "event_count_in_window" not in joined.columns, (
        "Diagnostic column 'event_count_in_window' leaked into feature matrix!"
    )


# ==============================================================================
# 7. Determinism Across Repeated Runs
# ==============================================================================


def test_deterministic_output_across_repeated_runs(builder: TargetLabelBuilder, sample_feature_df: pd.DataFrame, sample_label_df: pd.DataFrame):
    """Repeated executions produce identical DataFrames."""
    run1 = builder.join_features_and_labels(sample_feature_df, sample_label_df)
    run2 = builder.join_features_and_labels(sample_feature_df, sample_label_df)

    pd.testing.assert_frame_equal(run1, run2)


# ==============================================================================
# 8. Full Materialized Artifact Validation on Disk
# ==============================================================================


def test_full_integrated_artifact_on_disk(builder: TargetLabelBuilder):
    """End-to-end integration test validating the actual materialized artifacts on disk."""
    feat_path = config.project_root / "data" / "synthetic" / "processed" / "synthetic_features_445_cutoffs_6675_instances.csv"
    label_path = config.project_root / "data" / "synthetic" / "processed" / "synthetic_labels_445_cutoffs_6675_instances.csv"

    assert feat_path.exists(), f"Step 5 feature artifact not found: {feat_path}"
    assert label_path.exists(), f"Step 6 label artifact not found: {label_path}"

    feat_df = pd.read_csv(feat_path)
    label_df = pd.read_csv(label_path)

    # Validate inputs
    assert len(feat_df) == 6675
    assert len(label_df) == 6675
    assert len(feat_df.columns) == 16

    # Execute join
    integrated_df = builder.join_features_and_labels(feat_df, label_df)

    # Assert integrated properties
    assert len(integrated_df) == 6675
    assert len(integrated_df.columns) == 17
    assert "target_30d" in integrated_df.columns
    assert "event_count_in_window" not in integrated_df.columns
    assert integrated_df.isna().sum().sum() == 0

    # Assert partition counts
    assert (integrated_df["partition"] == "train").sum() == 4635
    assert (integrated_df["partition"] == "validation").sum() == 1110
    assert (integrated_df["partition"] == "test").sum() == 930

    # Assert target binary values
    assert set(integrated_df["target_30d"].unique()).issubset({0, 1})
