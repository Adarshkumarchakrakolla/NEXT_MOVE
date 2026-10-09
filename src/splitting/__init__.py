"""Splitting package for NextMove competitive intelligence platform."""

from src.splitting.temporal import (
    assign_strategic_clusters,
    compute_binary_window_label,
    filter_feature_events,
    generate_weekly_monday_cutoffs,
    get_label_window_bounds,
    get_partition_cutoffs,
    map_event_type_to_cluster,
    normalize_to_date,
)

__all__ = [
    "assign_strategic_clusters",
    "compute_binary_window_label",
    "filter_feature_events",
    "generate_weekly_monday_cutoffs",
    "get_label_window_bounds",
    "get_partition_cutoffs",
    "map_event_type_to_cluster",
    "normalize_to_date",
]
