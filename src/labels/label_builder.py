"""Target Label Generation and Feature-Label Integration Pipeline for NextMove.

Implements binary move label generation (target_30d) for the approved 30-day forward window:
    cutoff_date <= event_date <= cutoff_date + 29 days (inclusive).

Enforces strict temporal isolation:
- Labels reference ONLY events occurring within the forward label window.
- Historical features reference ONLY events strictly preceding the cutoff date (event_date < cutoff_date).
- Joins features and labels on (cutoff_date, competitor, strategic_cluster) with full integrity checks.
"""

from dataclasses import dataclass
import hashlib
from pathlib import Path
import time
from typing import Dict, List, Optional, Set, Tuple, Union

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
    get_label_window_bounds,
    get_partition_cutoffs,
    normalize_to_date,
)


@dataclass
class LabelPipelineMetrics:
    """Telemetry and dataset distribution summary for target label generation."""

    row_count: int
    column_count: int
    positive_count: int
    negative_count: int
    positive_prevalence: float
    runtime_seconds: float
    output_path: str
    sha256_checksum: str
    train_rows: int
    train_positive_count: int
    train_positive_prevalence: float
    val_rows: int
    val_positive_count: int
    val_positive_prevalence: float
    test_rows: int
    test_positive_count: int
    test_positive_prevalence: float


class TargetLabelBuilder:
    """Builds binary target labels (target_30d) and joins them to point-in-time features.

    Forward window is strictly [cutoff_date, cutoff_date + 29 days] inclusive.
    Target label is 1 if at least one qualifying event occurs for the competitor and
    cluster in that window; otherwise 0.
    """

    def __init__(
        self,
        config: PredictionConfig = default_config,
        competitors: Optional[List[str]] = None,
        max_dataset_date: Optional[Union[str, pd.Timestamp]] = None,
    ):
        self.config = config
        self.competitors = competitors or list(self.config.competitors)
        self.cluster_ids = sorted(list(self.config.cluster_names.keys()))
        self.label_window_days = self.config.label_window_days
        self.max_dataset_date = (
            normalize_to_date(max_dataset_date)
            if max_dataset_date is not None
            else normalize_to_date(self.config.dataset_max_date)
        )

        # Internal sorted date arrays per (competitor, cluster_id)
        self._indexed_dates: Dict[Tuple[str, int], np.ndarray] = {}
        self._is_fitted: bool = False
        self._observed_max_event_date: Optional[pd.Timestamp] = None

    def fit_events(self, events_df: pd.DataFrame) -> "TargetLabelBuilder":
        """Index the event log by competitor and cluster for fast binary-search lookups.

        Accepts any event ledger with columns:
        ['competitor', 'event_date', 'event_type' (or 'strategic_cluster')].
        Never modifies the input DataFrame in-place.
        """
        required_cols = {"competitor", "event_date"}
        missing = required_cols - set(events_df.columns)
        if missing:
            raise KeyError(f"Events dataframe is missing required columns: {missing}")

        df = events_df.copy()

        # Handle unmapped event types if event_type is present
        if "event_type" in df.columns:
            unmapped = set(df["event_type"].dropna().unique()) - set(self.config.event_type_to_cluster.keys())
            if unmapped:
                raise ValueError(f"Found unmapped event types in data: {unmapped}")

        # Derive cluster if not already present
        if "strategic_cluster" not in df.columns:
            if "event_type" not in df.columns:
                raise KeyError("Must provide either 'strategic_cluster' or 'event_type' column.")
            df = assign_strategic_clusters(df, cfg=self.config)
        else:
            # Validate strategic_cluster domain (1-5)
            valid_clusters = set(self.cluster_ids)
            provided_clusters = set(df["strategic_cluster"].dropna().unique())
            invalid_clusters = provided_clusters - valid_clusters
            if invalid_clusters:
                raise ValueError(f"Found invalid strategic cluster IDs in data: {invalid_clusters}")

        # Normalize dates to UTC midnight
        df["event_date"] = pd.to_datetime(df["event_date"]).dt.floor("D")
        if not df.empty:
            self._observed_max_event_date = df["event_date"].max()

        # Index sorted arrays per (competitor, cluster_id)
        self._indexed_dates = {}
        for comp in self.competitors:
            comp_mask = df["competitor"] == comp
            for cluster_id in self.cluster_ids:
                subset = df[comp_mask & (df["strategic_cluster"] == cluster_id)].sort_values("event_date")
                dates = subset["event_date"].values.astype("datetime64[D]")
                self._indexed_dates[(comp, cluster_id)] = dates

        self._is_fitted = True
        return self

    def compute_labels_for_cutoff(
        self,
        cutoff_date: Union[str, pd.Timestamp],
        partition: str = "unassigned",
        include_diagnostic_count: bool = True,
    ) -> List[Dict]:
        """Compute target_30d labels for all 15 competitor-cluster pairs at a single cutoff.

        Forward window is [cutoff_date, cutoff_date + 29 days] inclusive.
        Rejects cutoffs whose forward window extends past the maximum dataset date.
        """
        if not self._is_fitted:
            raise RuntimeError("TargetLabelBuilder must be fitted with events data via fit_events() first.")

        cutoff = normalize_to_date(cutoff_date)
        start_window, end_window = get_label_window_bounds(
            cutoff_date=cutoff,
            horizon_days=self.label_window_days,
            max_dataset_date=self.max_dataset_date,
        )

        t_start_np = np.datetime64(start_window.strftime("%Y-%m-%d"))
        t_end_np = np.datetime64(end_window.strftime("%Y-%m-%d"))

        rows: List[Dict] = []
        for comp in self.competitors:
            for cluster_id in self.cluster_ids:
                dates = self._indexed_dates.get((comp, cluster_id), np.empty(0, dtype="datetime64[D]"))
                i_start = int(np.searchsorted(dates, t_start_np, side="left"))
                i_end = int(np.searchsorted(dates, t_end_np, side="right"))
                count = i_end - i_start
                label = 1 if count > 0 else 0

                row_dict = {
                    "cutoff_date": cutoff.strftime("%Y-%m-%d"),
                    "competitor": comp,
                    "strategic_cluster": cluster_id,
                    "partition": partition,
                    "target_30d": label,
                }
                if include_diagnostic_count:
                    row_dict["event_count_in_window"] = count
                rows.append(row_dict)

        return rows

    def build_label_matrix(
        self,
        cutoffs: Optional[List[Tuple[str, Union[str, pd.Timestamp]]]] = None,
        include_diagnostic_count: bool = True,
    ) -> pd.DataFrame:
        """Compute the complete target label matrix across all specified cutoffs.

        If cutoffs is None, generates all eligible cutoffs across train, val, and test partitions.
        """
        if cutoffs is None:
            train_c = [("train", c) for c in get_partition_cutoffs("train", cfg=self.config)]
            val_c = [("validation", c) for c in get_partition_cutoffs("val", cfg=self.config)]
            test_c = [("test", c) for c in get_partition_cutoffs("test", cfg=self.config)]
            cutoffs = train_c + val_c + test_c

        all_rows: List[Dict] = []
        for partition, cutoff in cutoffs:
            rows = self.compute_labels_for_cutoff(
                cutoff, partition=partition, include_diagnostic_count=include_diagnostic_count
            )
            all_rows.extend(rows)

        label_df = pd.DataFrame(all_rows)
        expected_columns = [
            "cutoff_date",
            "competitor",
            "strategic_cluster",
            "partition",
            "target_30d",
        ]
        if include_diagnostic_count:
            expected_columns.append("event_count_in_window")
        return label_df[expected_columns]

    def build_and_save_labels(
        self,
        output_csv_path: Union[str, Path] = "data/synthetic/processed/synthetic_labels_445_cutoffs_6675_instances.csv",
        events_df: Optional[pd.DataFrame] = None,
        cutoffs: Optional[List[Tuple[str, Union[str, pd.Timestamp]]]] = None,
        include_diagnostic_count: bool = True,
    ) -> Tuple[pd.DataFrame, LabelPipelineMetrics]:
        """Generate and serialize the label matrix artifact to CSV.

        Saves artifact under data/synthetic/processed/ with telemetry and SHA-256 checksum.
        """
        start_time = time.time()
        output_path = Path(output_csv_path)
        if not output_path.is_absolute() and hasattr(self.config, "project_root"):
            output_path = self.config.project_root / output_path
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # 1. Fit events if provided or load raw dataset
        if events_df is not None:
            self.fit_events(events_df)
        elif not self._is_fitted:
            raw_path = self.config.raw_csv_path
            raw_df = pd.read_csv(raw_path, low_memory=False)
            self.fit_events(raw_df)

        # 2. Build label matrix
        labels_df = self.build_label_matrix(cutoffs=cutoffs, include_diagnostic_count=include_diagnostic_count)

        # 3. Serialize artifact to CSV
        labels_df.to_csv(output_path, index=False)
        runtime = time.time() - start_time

        # 4. Compute SHA-256 checksum
        hasher = hashlib.sha256()
        with open(output_path, "rb") as f:
            while chunk := f.read(65536):
                hasher.update(chunk)
        checksum = hasher.hexdigest()

        # 5. Compute metrics
        metrics = self._compute_metrics(labels_df, output_path, checksum, runtime)
        return labels_df, metrics

    def join_features_and_labels(
        self,
        features_df: pd.DataFrame,
        labels_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Join point-in-time features with target labels on (cutoff_date, competitor, strategic_cluster).

        Enforces strict dataset integrity:
        - Exactly one label per feature row (1:1 bijection).
        - No duplicate keys in features or labels.
        - Normalizes key types and date formats before joining.
        - Reject unexpected competitors or invalid strategic cluster IDs.
        - Reject missing labels or extra orphan labels.
        - Verify partition consistency between features and labels.
        - Verify feature values remain completely unaltered (by key).
        - Strictly isolates target_30d; excludes any future-window diagnostics.
        - Assert binary labels only (in {0, 1}) with zero nulls.
        """
        key_cols = ["cutoff_date", "competitor", "strategic_cluster"]

        # Validate required columns in both dataframes
        for col in key_cols:
            if col not in features_df.columns:
                raise KeyError(f"Features dataframe missing required join key: '{col}'")
            if col not in labels_df.columns:
                raise KeyError(f"Labels dataframe missing required join key: '{col}'")

        if "target_30d" not in labels_df.columns:
            raise KeyError("Labels dataframe missing 'target_30d' column.")

        # Normalize key types and formats
        features_norm = features_df.copy()
        labels_norm = labels_df.copy()

        features_norm["cutoff_date"] = pd.to_datetime(features_norm["cutoff_date"]).dt.strftime("%Y-%m-%d")
        labels_norm["cutoff_date"] = pd.to_datetime(labels_norm["cutoff_date"]).dt.strftime("%Y-%m-%d")
        features_norm["competitor"] = features_norm["competitor"].astype(str).str.strip()
        labels_norm["competitor"] = labels_norm["competitor"].astype(str).str.strip()
        features_norm["strategic_cluster"] = features_norm["strategic_cluster"].astype(int)
        labels_norm["strategic_cluster"] = labels_norm["strategic_cluster"].astype(int)

        # Check for duplicate keys
        features_keys = list(zip(features_norm["cutoff_date"], features_norm["competitor"], features_norm["strategic_cluster"]))
        if len(features_keys) != len(set(features_keys)):
            raise ValueError("Duplicate join keys detected in features dataframe!")

        labels_keys = list(zip(labels_norm["cutoff_date"], labels_norm["competitor"], labels_norm["strategic_cluster"]))
        if len(labels_keys) != len(set(labels_keys)):
            raise ValueError("Duplicate join keys detected in labels dataframe!")

        # Validate competitor domain
        valid_competitors = set(self.competitors)
        feat_comps = set(features_norm["competitor"].unique())
        label_comps = set(labels_norm["competitor"].unique())
        if not feat_comps.issubset(valid_competitors):
            raise ValueError(f"Features contain unexpected competitors: {feat_comps - valid_competitors}")
        if not label_comps.issubset(valid_competitors):
            raise ValueError(f"Labels contain unexpected competitors: {label_comps - valid_competitors}")

        # Validate strategic cluster IDs (1-5)
        valid_clusters = set(self.cluster_ids)
        feat_clusters = set(features_norm["strategic_cluster"].unique())
        label_clusters = set(labels_norm["strategic_cluster"].unique())
        if not feat_clusters.issubset(valid_clusters):
            raise ValueError(f"Features contain invalid cluster IDs: {feat_clusters - valid_clusters}")
        if not label_clusters.issubset(valid_clusters):
            raise ValueError(f"Labels contain invalid cluster IDs: {label_clusters - valid_clusters}")

        # Validate key set equality (1:1 bijection)
        feat_key_set = set(features_keys)
        label_key_set = set(labels_keys)
        missing_in_labels = feat_key_set - label_key_set
        missing_in_features = label_key_set - feat_key_set
        if missing_in_labels:
            raise ValueError(f"Found {len(missing_in_labels)} feature instances missing target labels!")
        if missing_in_features:
            raise ValueError(f"Found {len(missing_in_features)} target labels with no matching feature row!")

        # Validate target_30d binary values and non-null
        if labels_norm["target_30d"].isna().any():
            raise ValueError("Target label column contains NaN / null values!")
        invalid_labels = ~labels_norm["target_30d"].isin([0, 1])
        if invalid_labels.any():
            raise ValueError("Target label column contains non-binary values!")

        # Check partition consistency if present in both
        if "partition" in features_norm.columns and "partition" in labels_norm.columns:
            features_norm["partition"] = features_norm["partition"].astype(str).str.strip()
            labels_norm["partition"] = labels_norm["partition"].astype(str).str.strip()
            feat_part_dict = dict(zip(features_keys, features_norm["partition"]))
            label_part_dict = dict(zip(labels_keys, labels_norm["partition"]))
            mismatches = [k for k in feat_part_dict if feat_part_dict[k] != label_part_dict.get(k)]
            if mismatches:
                raise ValueError(f"Partition mismatch between features and labels for instances: {mismatches[:5]}")
            join_cols = ["cutoff_date", "competitor", "strategic_cluster", "partition"]
        else:
            join_cols = key_cols

        # Perform exact join - strictly select ONLY join keys and target_30d from labels
        # Excludes any diagnostic columns like event_count_in_window to prevent leakage
        label_cols_to_merge = join_cols + ["target_30d"]
        merged_df = pd.merge(
            features_norm,
            labels_norm[label_cols_to_merge],
            on=join_cols,
            how="inner",
        )

        # Integrity assertions on joined dataset
        if len(merged_df) != len(features_norm):
            raise AssertionError(
                f"Row count mismatch after join: features had {len(features_norm)}, merged has {len(merged_df)}"
            )

        if merged_df.isna().sum().sum() != 0:
            null_cols = merged_df.columns[merged_df.isna().any()].tolist()
            raise AssertionError(f"Joined modeling dataset contains null values in columns: {null_cols}")

        # Ensure future-window diagnostics never leak into merged dataset
        if "event_count_in_window" in merged_df.columns:
            raise AssertionError("Diagnostic column 'event_count_in_window' leaked into integrated dataset!")

        # Verify that all original feature columns and values are strictly unchanged (checked by key)
        merged_sorted = merged_df.sort_values(key_cols).reset_index(drop=True)
        feat_sorted = features_norm.sort_values(key_cols).reset_index(drop=True)
        for col in features_norm.columns:
            if not np.array_equal(merged_sorted[col].values, feat_sorted[col].values):
                raise AssertionError(f"Feature column '{col}' was modified during the label join!")

        # Ensure target_30d is integer
        merged_df["target_30d"] = merged_df["target_30d"].astype(int)

        return merged_df

    def build_and_save_training_matrix(
        self,
        output_csv_path: Union[str, Path] = "data/synthetic/processed/synthetic_training_matrix_445_cutoffs_6675_instances.csv",
        features_input: Union[str, Path, pd.DataFrame] = "data/synthetic/processed/synthetic_features_445_cutoffs_6675_instances.csv",
        labels_input: Optional[Union[str, Path, pd.DataFrame]] = None,
        events_df: Optional[pd.DataFrame] = None,
        cutoffs: Optional[List[Tuple[str, Union[str, pd.Timestamp]]]] = None,
    ) -> Tuple[pd.DataFrame, LabelPipelineMetrics]:
        """Execute feature-label integration, validate integrity, and serialize training matrix artifact.

        Saves artifact under data/synthetic/processed/ with telemetry and SHA-256 checksum.
        """
        start_time = time.time()
        output_path = Path(output_csv_path)
        if not output_path.is_absolute() and hasattr(self.config, "project_root"):
            output_path = self.config.project_root / output_path
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # 1. Load features
        if isinstance(features_input, (str, Path)):
            feat_path = Path(features_input)
            if not feat_path.is_absolute() and hasattr(self.config, "project_root"):
                feat_path = self.config.project_root / feat_path
            features_df = pd.read_csv(feat_path)
        else:
            features_df = features_input.copy()

        # 2. Obtain labels
        if labels_input is not None:
            if isinstance(labels_input, (str, Path)):
                lbl_path = Path(labels_input)
                if not lbl_path.is_absolute() and hasattr(self.config, "project_root"):
                    lbl_path = self.config.project_root / lbl_path
                labels_df = pd.read_csv(lbl_path)
            else:
                labels_df = labels_input.copy()
        else:
            default_label_path = self.config.project_root / "data" / "synthetic" / "processed" / "synthetic_labels_445_cutoffs_6675_instances.csv"
            if default_label_path.exists():
                labels_df = pd.read_csv(default_label_path)
            else:
                if events_df is not None:
                    self.fit_events(events_df)
                elif not self._is_fitted:
                    raw_path = self.config.raw_csv_path
                    raw_df = pd.read_csv(raw_path, low_memory=False)
                    self.fit_events(raw_df)
                labels_df = self.build_label_matrix(cutoffs=cutoffs)

        # 3. Join and validate
        matrix_df = self.join_features_and_labels(features_df, labels_df)

        # 4. Serialize artifact to CSV
        matrix_df.to_csv(output_path, index=False)
        runtime = time.time() - start_time

        # 5. Compute SHA-256 checksum
        hasher = hashlib.sha256()
        with open(output_path, "rb") as f:
            while chunk := f.read(65536):
                hasher.update(chunk)
        checksum = hasher.hexdigest()

        metrics = self._compute_metrics(matrix_df, output_path, checksum, runtime)
        return matrix_df, metrics

    def build_and_save_modeling_dataset(
        self,
        output_csv_path: Union[str, Path],
        features_input: Union[str, Path, pd.DataFrame],
        events_df: Optional[pd.DataFrame] = None,
        cutoffs: Optional[List[Tuple[str, Union[str, pd.Timestamp]]]] = None,
    ) -> Tuple[pd.DataFrame, LabelPipelineMetrics]:
        """Backward-compatible alias for build_and_save_training_matrix."""
        return self.build_and_save_training_matrix(
            output_csv_path=output_csv_path,
            features_input=features_input,
            events_df=events_df,
            cutoffs=cutoffs,
        )

    @staticmethod
    def _compute_metrics(
        df: pd.DataFrame,
        output_path: Path,
        checksum: str,
        runtime: float,
    ) -> LabelPipelineMetrics:
        """Compute partition and label distribution telemetry."""
        train_mask = df["partition"] == "train"
        val_mask = df["partition"] == "validation"
        test_mask = df["partition"] == "test"

        train_rows = int(train_mask.sum())
        val_rows = int(val_mask.sum())
        test_rows = int(test_mask.sum())

        pos_count = int((df["target_30d"] == 1).sum())
        neg_count = int((df["target_30d"] == 0).sum())
        total_rows = len(df)
        pos_prev = float(pos_count / total_rows) if total_rows > 0 else 0.0

        train_pos = int((df.loc[train_mask, "target_30d"] == 1).sum())
        val_pos = int((df.loc[val_mask, "target_30d"] == 1).sum())
        test_pos = int((df.loc[test_mask, "target_30d"] == 1).sum())

        train_prev = float(train_pos / train_rows) if train_rows > 0 else 0.0
        val_prev = float(val_pos / val_rows) if val_rows > 0 else 0.0
        test_prev = float(test_pos / test_rows) if test_rows > 0 else 0.0

        return LabelPipelineMetrics(
            row_count=total_rows,
            column_count=len(df.columns),
            positive_count=pos_count,
            negative_count=neg_count,
            positive_prevalence=round(pos_prev, 6),
            runtime_seconds=round(runtime, 3),
            output_path=str(output_path.as_posix()),
            sha256_checksum=checksum,
            train_rows=train_rows,
            train_positive_count=train_pos,
            train_positive_prevalence=round(train_prev, 6),
            val_rows=val_rows,
            val_positive_count=val_pos,
            val_positive_prevalence=round(val_prev, 6),
            test_rows=test_rows,
            test_positive_count=test_pos,
            test_positive_prevalence=round(test_prev, 6),
        )


def integrate_features_and_labels(
    features_df: pd.DataFrame,
    labels_df: pd.DataFrame,
    config: PredictionConfig = default_config,
) -> pd.DataFrame:
    """Convenience functional interface for joining features and labels with full verification."""
    builder = TargetLabelBuilder(config=config)
    return builder.join_features_and_labels(features_df, labels_df)
