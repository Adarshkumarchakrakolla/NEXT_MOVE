"""Label generation package for NextMove.

Provides binary target label generation (target_30d) and feature-label
integration with strict temporal leakage prevention and integrity validation.
"""

from src.labels.label_builder import (
    LabelPipelineMetrics,
    TargetLabelBuilder,
    integrate_features_and_labels,
)

__all__ = [
    "TargetLabelBuilder",
    "LabelPipelineMetrics",
    "integrate_features_and_labels",
]
