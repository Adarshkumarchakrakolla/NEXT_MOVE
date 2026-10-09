"""Point-in-Time Feature Engineering Pipeline for NextMove.

Implements deterministic lookback feature generation under Convention A (Monday 00:00 UTC).
Guarantees zero future-event leakage by strictly querying events with event_date < cutoff_date.
Modular design operates interchangeably on synthetic benchmark data or real evidence logs.
"""

from dataclasses import dataclass
import hashlib
from pathlib import Path
import time
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from configs.prediction_config import (
    COMPETITORS,
    CLUSTER_NAMES,
    PredictionConfig,
    config as default_config,
)
from src.splitting.temporal import (
    assign_strategic_clusters,
    get_partition_cutoffs,
    normalize_to_date,
)


@dataclass
class FeaturePipelineMetrics:
    """Telemetry and metadata for a feature generation run."""

    row_count: int
    column_count: int
    runtime_seconds: float
    output_path: str
    sha256_checksum: str
    train_rows: int
    val_rows: int
    test_rows: int


class PointInTimeFeatureBuilder:
    """Builds point-in-time feature representations for competitor-cluster-cutoff pairs.

    Enforces Convention A: features for cutoff Monday T strictly reference completed days
    up to Sunday T - 1 (t_e <= T - 1). Disallows all events where t_e >= T.
    """

    def __init__(
        self,
        config: PredictionConfig = default_config,
        competitors: Optional[List[str]] = None,
        decay_rate: float = 0.05,
    ):
        self.config = config
        self.competitors = competitors or list(self.config.competitors)
        self.decay_rate = decay_rate
        self.cluster_ids = sorted(list(self.config.cluster_names.keys()))

        # Internal index structures (populated on fit_events)
        self._indexed_series: Dict[Tuple[str, int], Tuple[np.ndarray, np.ndarray]] = {}
        self._is_fitted: bool = False

    def fit_events(self, events_df: pd.DataFrame) -> "PointInTimeFeatureBuilder":
        """Index the event log by competitor and cluster for fast O(log N) temporal lookups.

        Accepts any event ledger with columns:
        ['competitor', 'event_date', 'event_type' (or 'strategic_cluster'), 'magnitude_index'].
        """
        required_cols = {"competitor", "event_date", "magnitude_index"}
        missing = required_cols - set(events_df.columns)
        if missing:
            raise KeyError(f"Events dataframe is missing required columns: {missing}")

        df = events_df.copy()

        # Derive cluster if not already present
        if "strategic_cluster" not in df.columns:
            if "event_type" not in df.columns:
                raise KeyError("Must provide either 'strategic_cluster' or 'event_type' column.")
            df = assign_strategic_clusters(df, cfg=self.config)

        # Normalize date to UTC midnight
        df["event_date"] = pd.to_datetime(df["event_date"]).dt.floor("D")

        # Build sorted arrays per (competitor, cluster_id)
        self._indexed_series = {}
        for comp in self.competitors:
            comp_mask = df["competitor"] == comp
            for cluster_id in self.cluster_ids:
                subset = df[comp_mask & (df["strategic_cluster"] == cluster_id)].sort_values("event_date")
                dates = subset["event_date"].values.astype("datetime64[D]")
                mags = subset["magnitude_index"].values.astype(np.float64)
                self._indexed_series[(comp, cluster_id)] = (dates, mags)

        self._is_fitted = True
        return self

    def compute_features_for_cutoff(
        self,
        cutoff_date: Union[str, pd.Timestamp],
        partition: str = "unassigned",
    ) -> List[Dict]:
        """Compute all 15 competitor-cluster feature rows for a single cutoff date.

        Enforces Convention A: Features strictly query history where t_e <= T - 1 day.
        Events occurring on or after cutoff_date are strictly excluded.
        """
        if not self._is_fitted:
            raise RuntimeError("FeatureBuilder must be fitted with events data via fit_events() first.")

        cutoff = normalize_to_date(cutoff_date)

        # Convention A feature boundary: T - 1 day (Sunday 23:59:59 UTC)
        t_minus_1 = cutoff - pd.Timedelta(days=1)
        t_minus_1_np = np.datetime64(t_minus_1.strftime("%Y-%m-%d"))

        # Lookback boundaries: [T - L, T - 1]
        t_minus_14_np = np.datetime64((cutoff - pd.Timedelta(days=14)).strftime("%Y-%m-%d"))
        t_minus_30_np = np.datetime64((cutoff - pd.Timedelta(days=30)).strftime("%Y-%m-%d"))
        t_minus_60_np = np.datetime64((cutoff - pd.Timedelta(days=60)).strftime("%Y-%m-%d"))
        t_minus_90_np = np.datetime64((cutoff - pd.Timedelta(days=90)).strftime("%Y-%m-%d"))
        t_minus_365_np = np.datetime64((cutoff - pd.Timedelta(days=365)).strftime("%Y-%m-%d"))

        # Precompute cluster-level counts for rivals over [T - 14, T - 1]
        cluster_total_counts_14d: Dict[int, int] = {}
        for cluster_id in self.cluster_ids:
            total_14d = 0
            for comp in self.competitors:
                dates, _ = self._indexed_series[(comp, cluster_id)]
                i_start = int(np.searchsorted(dates, t_minus_14_np, side="left"))
                i_end = int(np.searchsorted(dates, t_minus_1_np, side="right"))
                total_14d += (i_end - i_start)
            cluster_total_counts_14d[cluster_id] = total_14d

        rows: List[Dict] = []
        quarter = int(cutoff.quarter)
        month = int(cutoff.month)
        is_q4 = 1 if quarter == 4 else 0

        for comp in self.competitors:
            for cluster_id in self.cluster_ids:
                dates, mags = self._indexed_series[(comp, cluster_id)]

                # Upper bound index: strictly <= T - 1
                idx_end = int(np.searchsorted(dates, t_minus_1_np, side="right"))

                # 1. Recency & indicator
                if idx_end > 0:
                    last_event_date = dates[idx_end - 1]
                    delta_days = (t_minus_1_np - last_event_date).astype(int)
                    days_since = float(delta_days)
                    has_prior = 1
                else:
                    days_since = -1.0
                    has_prior = 0

                # 2. Velocity counts: [T - L, T - 1]
                idx_30 = int(np.searchsorted(dates, t_minus_30_np, side="left"))
                count_30d = int(idx_end - idx_30)

                idx_90 = int(np.searchsorted(dates, t_minus_90_np, side="left"))
                count_90d = int(idx_end - idx_90)

                idx_365 = int(np.searchsorted(dates, t_minus_365_np, side="left"))
                count_365d = int(idx_end - idx_365)

                # 3. Smoothed velocity ratio
                vel_ratio = float((count_30d + 1.0) / ((count_90d / 3.0) + 1.0))

                # 4. Rival event count over [T - 14, T - 1]
                idx_14 = int(np.searchsorted(dates, t_minus_14_np, side="left"))
                self_14d = int(idx_end - idx_14)
                rival_14d = int(cluster_total_counts_14d[cluster_id] - self_14d)

                # 5. Decayed magnitude mass over [T - 60, T - 1]
                idx_60 = int(np.searchsorted(dates, t_minus_60_np, side="left"))
                if idx_end > idx_60:
                    w_dates = dates[idx_60:idx_end]
                    w_mags = mags[idx_60:idx_end]
                    age_days = (t_minus_1_np - w_dates).astype(float)
                    weights = np.exp(-self.decay_rate * age_days)
                    decayed_mass = float(np.sum(w_mags * weights))
                else:
                    decayed_mass = 0.0

                rows.append({
                    "cutoff_date": cutoff.strftime("%Y-%m-%d"),
                    "competitor": comp,
                    "strategic_cluster": cluster_id,
                    "cluster_name": self.config.cluster_names[cluster_id],
                    "partition": partition,
                    "days_since_last_cluster_event": round(days_since, 4),
                    "has_prior_cluster_event": has_prior,
                    "cluster_event_count_30d": count_30d,
                    "cluster_event_count_90d": count_90d,
                    "cluster_event_count_365d": count_365d,
                    "velocity_ratio_30_90": round(vel_ratio, 6),
                    "rival_event_count_14d": rival_14d,
                    "decayed_magnitude_mass_60d": round(decayed_mass, 6),
                    "quarter": quarter,
                    "month": month,
                    "is_q4": is_q4,
                })

        return rows

    def build_feature_matrix(
        self,
        cutoffs: Optional[List[Tuple[str, Union[str, pd.Timestamp]]]] = None,
    ) -> pd.DataFrame:
        """Compute the complete feature matrix across all specified cutoffs and partitions.

        If cutoffs is None, generates all eligible cutoffs across train, val, and test partitions.
        """
        if cutoffs is None:
            train_c = [("train", c) for c in get_partition_cutoffs("train", cfg=self.config)]
            val_c = [("validation", c) for c in get_partition_cutoffs("val", cfg=self.config)]
            test_c = [("test", c) for c in get_partition_cutoffs("test", cfg=self.config)]
            cutoffs = train_c + val_c + test_c

        all_rows: List[Dict] = []
        for partition, cutoff in cutoffs:
            rows = self.compute_features_for_cutoff(cutoff, partition=partition)
            all_rows.extend(rows)

        feature_df = pd.DataFrame(all_rows)

        # Enforce column schema order
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
        return feature_df[expected_columns]

    def build_and_save(
        self,
        output_csv_path: Union[str, Path],
        events_df: Optional[pd.DataFrame] = None,
        cutoffs: Optional[List[Tuple[str, Union[str, pd.Timestamp]]]] = None,
    ) -> Tuple[pd.DataFrame, FeaturePipelineMetrics]:
        """Execute the feature engineering pipeline and serialize the artifact.

        Saves strictly under the specified path. Computes telemetry and SHA-256 checksum.
        """
        start_time = time.time()
        output_path = Path(output_csv_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if events_df is not None:
            self.fit_events(events_df)
        elif not self._is_fitted:
            raw_path = self.config.raw_csv_path
            raw_df = pd.read_csv(raw_path, low_memory=False)
            self.fit_events(raw_df)

        feature_df = self.build_feature_matrix(cutoffs=cutoffs)

        # Serialize to CSV
        feature_df.to_csv(output_path, index=False)
        runtime = time.time() - start_time

        # Calculate file checksum
        hasher = hashlib.sha256()
        with open(output_path, "rb") as f:
            while chunk := f.read(65536):
                hasher.update(chunk)
        checksum = hasher.hexdigest()

        # Partition breakdown
        train_count = int((feature_df["partition"] == "train").sum())
        val_count = int((feature_df["partition"] == "validation").sum())
        test_count = int((feature_df["partition"] == "test").sum())

        metrics = FeaturePipelineMetrics(
            row_count=len(feature_df),
            column_count=len(feature_df.columns),
            runtime_seconds=round(runtime, 3),
            output_path=str(output_path.as_posix()),
            sha256_checksum=checksum,
            train_rows=train_count,
            val_rows=val_count,
            test_rows=test_count,
        )

        return feature_df, metrics
