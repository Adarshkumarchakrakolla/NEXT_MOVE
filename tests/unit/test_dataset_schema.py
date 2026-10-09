"""Automated schema, data quality, and synthetic integrity tests.

Audits: data/synthetic/raw/recall_intel_121000_independent_synthetic_events.csv
Mode: Read-only verification.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

DATASET_PATH = Path("data/synthetic/raw/recall_intel_121000_independent_synthetic_events.csv")

EXPECTED_COLUMNS = [
    "event_id",
    "competitor",
    "event_date",
    "event_type",
    "title",
    "description",
    "strategic_area",
    "market",
    "channel",
    "impact_level",
    "signal_strength",
    "extraction_confidence",
    "magnitude_index",
    "price_delta_pct",
    "growth_delta_pct",
    "headcount_delta",
    "source_type",
    "provenance_status",
    "evidence_status",
    "independence_status",
    "generator_seed",
]

EXPECTED_COMPETITORS = {"Amazon", "Walmart", "eBay"}
EXPECTED_IMPACT_LEVELS = {"low", "medium", "high", "very_high"}


@pytest.fixture(scope="module")
def df():
    """Load dataset once for the test module in read-only mode."""
    assert DATASET_PATH.exists(), f"Dataset file does not exist at {DATASET_PATH}"
    dataset = pd.read_csv(DATASET_PATH, low_memory=False)
    return dataset


# ==============================================================================
# 1. Schema & Structural Tests
# ==============================================================================


def test_row_and_column_counts(df: pd.DataFrame):
    """Verify exactly 121,000 data rows and 21 columns."""
    assert len(df) == 121000, f"Expected 121000 rows, found {len(df)}"
    assert len(df.columns) == 21, f"Expected 21 columns, found {len(df.columns)}"


def test_column_names_match_expected(df: pd.DataFrame):
    """Check for missing, unexpected, or duplicate column names."""
    actual_cols = list(df.columns)
    assert actual_cols == EXPECTED_COLUMNS, "Columns do not match expected schema order"
    assert len(actual_cols) == len(set(actual_cols)), "Duplicate column headers detected"


def test_event_id_uniqueness_and_validity(df: pd.DataFrame):
    """Verify that event_id values are 100% unique, non-null, and non-empty."""
    assert df["event_id"].notna().all(), "Null event_id values detected"
    assert (df["event_id"].str.strip() != "").all(), "Empty string event_id values detected"
    assert df["event_id"].is_unique, "Duplicate event_id values found"
    assert df["event_id"].str.startswith("SYN-").all(), "Invalid event_id prefix format"


def test_event_date_parseability_and_bounds(df: pd.DataFrame):
    """Check event_date parseability and enforce expected range [2018-01-01, 2026-10-09]."""
    parsed_dates = pd.to_datetime(df["event_date"], errors="coerce")
    assert parsed_dates.notna().all(), "Unparseable event_date values detected"

    min_date = parsed_dates.min()
    max_date = parsed_dates.max()
    assert min_date >= pd.Timestamp("2018-01-01"), f"Date before 2018-01-01: {min_date}"
    assert max_date <= pd.Timestamp("2026-10-09"), f"Date after 2026-10-09: {max_date}"


# ==============================================================================
# 2. Data Quality & Hygiene Tests
# ==============================================================================


def test_zero_missing_or_empty_values(df: pd.DataFrame):
    """Verify that no column contains missing (NaN) or blank string values."""
    for col in df.columns:
        null_count = df[col].isna().sum()
        assert null_count == 0, f"Column '{col}' has {null_count} missing values"

        if df[col].dtype == object:
            empty_count = (df[col].astype(str).str.strip() == "").sum()
            assert empty_count == 0, f"Column '{col}' has {empty_count} empty string values"


def test_no_exact_duplicate_rows(df: pd.DataFrame):
    """Verify there are zero exact duplicate rows across all 21 columns."""
    assert df.duplicated().sum() == 0, "Exact duplicate rows detected in dataset"


def test_categorical_domains(df: pd.DataFrame):
    """Verify valid values for categorical columns."""
    assert set(df["competitor"].unique()) == EXPECTED_COMPETITORS
    assert set(df["impact_level"].unique()) == EXPECTED_IMPACT_LEVELS
    assert df["event_type"].nunique() == 30
    assert df["strategic_area"].nunique() == 20
    assert df["market"].nunique() == 10
    assert df["channel"].nunique() == 8


def test_numeric_ranges_and_validity(df: pd.DataFrame):
    """Check numeric columns for valid boundaries, no NaNs, and no infinities."""
    numeric_columns = [
        "signal_strength",
        "extraction_confidence",
        "magnitude_index",
        "price_delta_pct",
        "growth_delta_pct",
        "headcount_delta",
    ]

    for col in numeric_columns:
        series = pd.to_numeric(df[col], errors="coerce")
        assert series.notna().all(), f"Non-numeric values found in '{col}'"
        assert not np.isinf(series).any(), f"Infinity found in '{col}'"

    assert df["signal_strength"].between(0.0, 1.0).all()
    assert df["extraction_confidence"].between(0.0, 1.0).all()
    assert (df["magnitude_index"] >= 0.0).all()
    assert df["price_delta_pct"].between(-100.0, 100.0).all()
    assert df["growth_delta_pct"].between(-100.0, 500.0).all()


# ==============================================================================
# 3. Synthetic Data Integrity Tests
# ==============================================================================


def test_synthetic_metadata_labels(df: pd.DataFrame):
    """Verify all records are explicitly labeled with synthetic provenance."""
    assert (df["source_type"] == "synthetic").all()
    assert (df["provenance_status"] == "synthetic_generated").all()
    assert (df["evidence_status"] == "not_applicable").all()
    assert (df["independence_status"] == "independent_record").all()


def test_generator_seed_consistency(df: pd.DataFrame):
    """Verify generator_seed is consistent across all records."""
    assert df["generator_seed"].nunique() == 1
    assert df["generator_seed"].iloc[0] == 20261009


def test_description_template_prefix(df: pd.DataFrame):
    """Verify description adherence to synthetic template convention."""
    prefix = "Synthetic event for benchmark training:"
    assert df["description"].str.startswith(prefix).all(), (
        "Some descriptions do not follow the expected synthetic benchmark template"
    )
