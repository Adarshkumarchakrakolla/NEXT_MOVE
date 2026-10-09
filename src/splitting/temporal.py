"""Deterministic temporal utilities for NextMove prediction target and splitting contracts.

Implements Convention A (Start-of-Day Cutoff, Monday 00:00:00 UTC), cutoff generation,
forward label window definitions, feature isolation masks, and cluster derivation.
"""

from typing import List, Optional, Tuple, Union
import pandas as pd

from configs.prediction_config import (
    EVENT_TYPE_TO_CLUSTER,
    PredictionConfig,
    config as default_config,
)


def normalize_to_date(ts: Union[str, pd.Timestamp]) -> pd.Timestamp:
    """Normalize input date or string to UTC midnight timestamp (date level)."""
    return pd.to_datetime(ts).tz_localize(None).floor("D")


def map_event_type_to_cluster(event_type: str, cfg: PredictionConfig = default_config) -> int:
    """Map a single event_type string to its strategic cluster ID (1-5)."""
    try:
        return cfg.event_type_to_cluster[event_type]
    except KeyError:
        raise ValueError(f"Unknown event_type '{event_type}'. Not found in approved 30-category taxonomy.")


def assign_strategic_clusters(
    df: pd.DataFrame,
    event_type_col: str = "event_type",
    output_col: str = "strategic_cluster",
    cfg: PredictionConfig = default_config,
) -> pd.DataFrame:
    """Derive strategic cluster ID column from event_type column.

    Does not assume a pre-existing cluster column exists in raw data.
    """
    if event_type_col not in df.columns:
        raise KeyError(f"Column '{event_type_col}' not found in dataframe.")

    result_df = df.copy()
    unmapped = set(result_df[event_type_col].dropna().unique()) - set(cfg.event_type_to_cluster.keys())
    if unmapped:
        raise ValueError(f"Found unmapped event types in data: {unmapped}")

    result_df[output_col] = result_df[event_type_col].map(cfg.event_type_to_cluster)
    return result_df


def generate_weekly_monday_cutoffs(
    start_date: Union[str, pd.Timestamp],
    end_date: Union[str, pd.Timestamp],
    max_dataset_date: Optional[Union[str, pd.Timestamp]] = None,
    horizon_days: int = 30,
) -> List[pd.Timestamp]:
    """Generate deterministic weekly Monday cutoff grid between start and end dates.

    Rejects or raises ValueError if any requested cutoff's forward window extends beyond max_dataset_date.
    """
    start = normalize_to_date(start_date)
    end = normalize_to_date(end_date)

    if start > end:
        raise ValueError(f"start_date ({start}) cannot be after end_date ({end})")

    # Generate Mondays
    mondays = pd.date_range(start=start, end=end, freq="W-MON")
    cutoff_list = [normalize_to_date(m) for m in mondays]

    # Verify right-censoring rule if max_dataset_date provided
    if max_dataset_date is not None:
        max_limit = normalize_to_date(max_dataset_date)
        for cutoff in cutoff_list:
            label_end = cutoff + pd.Timedelta(days=horizon_days - 1)
            if label_end > max_limit:
                raise ValueError(
                    f"Cutoff {cutoff.strftime('%Y-%m-%d')} has forward label window ending on "
                    f"{label_end.strftime('%Y-%m-%d')}, which extends beyond max dataset date "
                    f"{max_limit.strftime('%Y-%m-%d')}."
                )

    return cutoff_list


def get_partition_cutoffs(
    partition: str,
    cfg: PredictionConfig = default_config,
) -> List[pd.Timestamp]:
    """Retrieve verified weekly Monday cutoffs for a specific partition.

    Enforces exact boundary intervals and expected counts from PredictionConfig.
    """
    partition_lower = partition.lower().strip()
    max_limit = normalize_to_date(cfg.dataset_max_date)

    if partition_lower == "train":
        cutoffs = generate_weekly_monday_cutoffs(
            cfg.train_cutoff_start,
            cfg.train_cutoff_end,
            max_dataset_date=max_limit,
            horizon_days=cfg.label_window_days,
        )
        if len(cutoffs) != cfg.expected_train_cutoffs:
            raise AssertionError(f"Expected {cfg.expected_train_cutoffs} train cutoffs, got {len(cutoffs)}")
        return cutoffs

    elif partition_lower in ("validation", "val"):
        cutoffs = generate_weekly_monday_cutoffs(
            cfg.val_cutoff_start,
            cfg.val_cutoff_end,
            max_dataset_date=max_limit,
            horizon_days=cfg.label_window_days,
        )
        if len(cutoffs) != cfg.expected_val_cutoffs:
            raise AssertionError(f"Expected {cfg.expected_val_cutoffs} validation cutoffs, got {len(cutoffs)}")
        return cutoffs

    elif partition_lower == "test":
        cutoffs = generate_weekly_monday_cutoffs(
            cfg.test_cutoff_start,
            cfg.test_cutoff_end,
            max_dataset_date=max_limit,
            horizon_days=cfg.label_window_days,
        )
        if len(cutoffs) != cfg.expected_test_cutoffs:
            raise AssertionError(f"Expected {cfg.expected_test_cutoffs} test cutoffs, got {len(cutoffs)}")
        return cutoffs

    else:
        raise ValueError(f"Unknown partition '{partition}'. Must be 'train', 'validation', or 'test'.")


def get_label_window_bounds(
    cutoff_date: Union[str, pd.Timestamp],
    horizon_days: int = 30,
    max_dataset_date: Optional[Union[str, pd.Timestamp]] = None,
) -> Tuple[pd.Timestamp, pd.Timestamp]:
    """Define the 30-day inclusive label window [cutoff_date, cutoff_date + 29 days].

    Rejects cutoffs whose full forward window exceeds max_dataset_date.
    """
    start_date = normalize_to_date(cutoff_date)
    end_date = start_date + pd.Timedelta(days=horizon_days - 1)

    if max_dataset_date is not None:
        max_limit = normalize_to_date(max_dataset_date)
        if end_date > max_limit:
            raise ValueError(
                f"Cutoff {start_date.strftime('%Y-%m-%d')} has label window ending on "
                f"{end_date.strftime('%Y-%m-%d')}, exceeding max dataset date {max_limit.strftime('%Y-%m-%d')}."
            )

    return start_date, end_date


def filter_feature_events(
    events_df: pd.DataFrame,
    cutoff_date: Union[str, pd.Timestamp],
    date_col: str = "event_date",
) -> pd.DataFrame:
    """Enforce Convention A: Features may use strictly completed days prior to cutoff_date.

    Asserts event_date < cutoff_date (i.e. event_date <= cutoff_date - 1 day).
    Excludes all events occurring on or after cutoff_date.
    """
    cutoff = normalize_to_date(cutoff_date)
    event_dates = pd.to_datetime(events_df[date_col]).dt.floor("D")
    mask = event_dates < cutoff
    return events_df[mask].copy()


def compute_binary_window_label(
    events_df: pd.DataFrame,
    competitor: str,
    cluster_id: int,
    cutoff_date: Union[str, pd.Timestamp],
    horizon_days: int = 30,
    max_dataset_date: Optional[Union[str, pd.Timestamp]] = None,
    date_col: str = "event_date",
    competitor_col: str = "competitor",
    event_type_col: str = "event_type",
    cfg: PredictionConfig = default_config,
) -> int:
    """Compute binary move label for a specific competitor and strategic cluster.

    - Validates label window [cutoff_date, cutoff_date + 29 days] independently of event presence.
    - If 0 matching events occur in window, returns 0.
    - If >= 1 matching event occurs in window, returns 1.
    """
    start_window, end_window = get_label_window_bounds(
        cutoff_date=cutoff_date,
        horizon_days=horizon_days,
        max_dataset_date=max_dataset_date,
    )

    if events_df.empty:
        return 0

    # Ensure event dates are comparable
    event_dates = pd.to_datetime(events_df[date_col]).dt.floor("D")

    # Filter by date window [start_window, end_window] inclusive
    in_window = (event_dates >= start_window) & (event_dates <= end_window)
    if not in_window.any():
        return 0

    window_subset = events_df[in_window]

    # Filter competitor
    comp_mask = window_subset[competitor_col] == competitor
    if not comp_mask.any():
        return 0

    comp_subset = window_subset[comp_mask]

    # Derive cluster on the fly if not already present
    if "strategic_cluster" in comp_subset.columns:
        matching_cluster = comp_subset["strategic_cluster"] == cluster_id
    else:
        clusters = comp_subset[event_type_col].map(cfg.event_type_to_cluster)
        matching_cluster = clusters == cluster_id

    return 1 if matching_cluster.any() else 0
