"""Label generation package for NextMove.

Provides binary target label generation (target_30d) and modeling dataset
assembly with strict temporal leakage prevention and integrity validation.
"""

from src.labels.label_builder import LabelPipelineMetrics, TargetLabelBuilder

__all__ = [
    "TargetLabelBuilder",
    "LabelPipelineMetrics",
]
