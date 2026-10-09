"""Configuration package for NextMove competitive intelligence platform."""

from configs.prediction_config import (
    RAW_CSV_PATH,
    COMPETITORS,
    EVENT_TYPE_TO_CLUSTER,
    CLUSTER_NAMES,
    CLUSTER_TO_EVENT_TYPES,
    PredictionConfig,
    config,
)

__all__ = [
    "RAW_CSV_PATH",
    "COMPETITORS",
    "EVENT_TYPE_TO_CLUSTER",
    "CLUSTER_NAMES",
    "CLUSTER_TO_EVENT_TYPES",
    "PredictionConfig",
    "config",
]
