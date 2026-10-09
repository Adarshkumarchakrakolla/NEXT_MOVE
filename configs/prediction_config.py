"""Centralized configuration for NextMove prediction target and temporal boundaries.

Defines the dataset path, taxonomy mapping, partition boundaries, cutoff conventions,
and validation assertions for the 30-day forward move prediction task.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Set


# ==============================================================================
# 1. Dataset & Competitor Constants
# ==============================================================================

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
RAW_CSV_PATH: Path = PROJECT_ROOT / "data" / "synthetic" / "raw" / "recall_intel_121000_independent_synthetic_events.csv"

COMPETITORS: List[str] = ["Amazon", "Walmart", "eBay"]

# ==============================================================================
# 2. Event Taxonomy to Strategic Cluster Mapping (All 30 Event Types)
# ==============================================================================

CLUSTER_NAMES: Dict[int, str] = {
    1: "Pricing, Promotions & Retail Media",
    2: "Logistics, Fulfillment & Expansion",
    3: "Product, AI & Platform Ecosystem",
    4: "Marketplace & Seller Operations",
    5: "Corporate Strategy, M&A & Workforce",
}

EVENT_TYPE_TO_CLUSTER: Dict[str, int] = {
    # Cluster 1: Pricing, Promotions & Retail Media (5 types)
    "pricing_change": 1,
    "promotion": 1,
    "membership": 1,
    "advertising": 1,
    "retail_media": 1,
    # Cluster 2: Logistics, Fulfillment & Expansion (5 types)
    "delivery": 2,
    "fulfillment": 2,
    "logistics_expansion": 2,
    "store_expansion": 2,
    "international_expansion": 2,
    # Cluster 3: Product, AI & Platform Ecosystem (7 types)
    "product_launch": 3,
    "feature_launch": 3,
    "ai_launch": 3,
    "app_update": 3,
    "live_commerce": 3,
    "data_service": 3,
    "payments": 3,
    # Cluster 4: Marketplace & Seller Operations (4 types)
    "marketplace_change": 4,
    "seller_program": 4,
    "trust_safety": 4,
    "authentication": 4,
    # Cluster 5: Corporate Strategy, M&A & Workforce (9 types)
    "earnings": 5,
    "hiring": 5,
    "restructuring": 5,
    "leadership_change": 5,
    "acquisition": 5,
    "investment": 5,
    "regulatory": 5,
    "sustainability": 5,
    "partnership": 5,
}

CLUSTER_TO_EVENT_TYPES: Dict[int, List[str]] = {
    1: ["pricing_change", "promotion", "membership", "advertising", "retail_media"],
    2: ["delivery", "fulfillment", "logistics_expansion", "store_expansion", "international_expansion"],
    3: ["product_launch", "feature_launch", "ai_launch", "app_update", "live_commerce", "data_service", "payments"],
    4: ["marketplace_change", "seller_program", "trust_safety", "authentication"],
    5: ["earnings", "hiring", "restructuring", "leadership_change", "acquisition", "investment", "regulatory", "sustainability", "partnership"],
}

# ==============================================================================
# 3. Temporal Conventions & Partition Schedule (Convention A)
# ==============================================================================

CUTOFF_CONVENTION: str = "Convention A (Start-of-Day Cutoff, Monday 00:00:00 UTC)"
TIMEZONE: str = "UTC"

LABEL_WINDOW_DAYS: int = 30  # [T, T + 29 days] inclusive
DATASET_START_DATE: str = "2018-01-01"
DATASET_MAX_DATE: str = "2026-10-09"
MAX_ELIGIBLE_CUTOFF: str = "2026-09-07"  # Cutoff + 29 days <= 2026-10-09

# Partition Date Ranges (Weekly Mondays)
TRAIN_CUTOFF_START: str = "2018-01-01"
TRAIN_CUTOFF_END: str = "2023-11-27"
EMBARGO_1_START: str = "2023-11-28"
EMBARGO_1_END: str = "2023-12-31"

VAL_CUTOFF_START: str = "2024-01-01"
VAL_CUTOFF_END: str = "2025-05-26"
EMBARGO_2_START: str = "2025-05-27"
EMBARGO_2_END: str = "2025-07-06"

TEST_CUTOFF_START: str = "2025-07-07"
TEST_CUTOFF_END: str = "2026-09-07"

# Expected Counts
EXPECTED_TRAIN_CUTOFFS: int = 309
EXPECTED_VAL_CUTOFFS: int = 74
EXPECTED_TEST_CUTOFFS: int = 62
EXPECTED_TOTAL_CUTOFFS: int = 445

TARGETS_PER_CUTOFF: int = 15  # 3 competitors * 5 strategic clusters
EXPECTED_TOTAL_INSTANCES: int = 6675


# ==============================================================================
# 4. Central Configuration Dataclass & Validator
# ==============================================================================

@dataclass(frozen=True)
class PredictionConfig:
    """Immutable prediction configuration object."""

    project_root: Path = PROJECT_ROOT
    raw_csv_path: Path = RAW_CSV_PATH
    competitors: List[str] = field(default_factory=lambda: list(COMPETITORS))
    cluster_names: Dict[int, str] = field(default_factory=lambda: dict(CLUSTER_NAMES))
    event_type_to_cluster: Dict[str, int] = field(default_factory=lambda: dict(EVENT_TYPE_TO_CLUSTER))
    cluster_to_event_types: Dict[int, List[str]] = field(default_factory=lambda: {k: list(v) for k, v in CLUSTER_TO_EVENT_TYPES.items()})

    cutoff_convention: str = CUTOFF_CONVENTION
    timezone: str = TIMEZONE
    label_window_days: int = LABEL_WINDOW_DAYS
    dataset_start_date: str = DATASET_START_DATE
    dataset_max_date: str = DATASET_MAX_DATE
    max_eligible_cutoff: str = MAX_ELIGIBLE_CUTOFF

    train_cutoff_start: str = TRAIN_CUTOFF_START
    train_cutoff_end: str = TRAIN_CUTOFF_END
    val_cutoff_start: str = VAL_CUTOFF_START
    val_cutoff_end: str = VAL_CUTOFF_END
    test_cutoff_start: str = TEST_CUTOFF_START
    test_cutoff_end: str = TEST_CUTOFF_END

    expected_train_cutoffs: int = EXPECTED_TRAIN_CUTOFFS
    expected_val_cutoffs: int = EXPECTED_VAL_CUTOFFS
    expected_test_cutoffs: int = EXPECTED_TEST_CUTOFFS
    expected_total_cutoffs: int = EXPECTED_TOTAL_CUTOFFS
    targets_per_cutoff: int = TARGETS_PER_CUTOFF
    expected_total_instances: int = EXPECTED_TOTAL_INSTANCES

    def validate(self) -> bool:
        """Validate internal consistency of the configuration."""
        # 1. Exactly 30 event types mapped
        assert len(self.event_type_to_cluster) == 30, (
            f"Expected exactly 30 event types, found {len(self.event_type_to_cluster)}"
        )

        # 2. Each maps to an approved cluster ID (1 to 5)
        valid_cluster_ids = set(self.cluster_names.keys())
        assert valid_cluster_ids == {1, 2, 3, 4, 5}, f"Invalid cluster IDs: {valid_cluster_ids}"

        for event_type, cluster_id in self.event_type_to_cluster.items():
            assert cluster_id in valid_cluster_ids, (
                f"Event type '{event_type}' maps to invalid cluster ID {cluster_id}"
            )

        # 3. Bi-directional mapping consistency
        reconstructed_mapping = {}
        for cid, etypes in self.cluster_to_event_types.items():
            for et in etypes:
                assert et not in reconstructed_mapping, f"Duplicate assignment for event type '{et}'"
                reconstructed_mapping[et] = cid

        assert reconstructed_mapping == self.event_type_to_cluster, (
            "Discrepancy between forward and reverse taxonomy mappings"
        )

        # 4. Partition count consistency
        total = self.expected_train_cutoffs + self.expected_val_cutoffs + self.expected_test_cutoffs
        assert total == self.expected_total_cutoffs == 445, (
            f"Partition cutoffs ({total}) do not match expected total ({self.expected_total_cutoffs})"
        )

        # 5. Total instances calculation
        assert (
            self.expected_total_cutoffs * len(self.competitors) * len(valid_cluster_ids)
            == self.expected_total_instances
            == 6675
        ), "Discrepancy in expected supervised instance count"

        return True


# Default validated singleton
config = PredictionConfig()
config.validate()
