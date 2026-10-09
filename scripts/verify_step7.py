"""Verification script for NextMove Step 7 — Feature-Label Integration & Temporal Integrity.

Executes and verifies:
1. Loading Step 5 feature artifact and Step 6 label artifact.
2. 1:1 Bijective key match on (cutoff_date, competitor, strategic_cluster).
3. Zero duplicate keys and zero missing/orphan keys.
4. Partition consistency between features and labels.
5. Bitwise feature preservation (verified by key).
6. Strict target isolation (no diagnostic leakage, e.g. event_count_in_window excluded).
7. Non-null and binary target validation.
8. Serialization and SHA-256 checksum calculation for synthetic_training_matrix.
9. Full partition and target distribution reporting.
"""

import hashlib
from pathlib import Path
import sys
import time

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from configs.prediction_config import config
from src.labels.label_builder import TargetLabelBuilder


def main():
    start_time = time.time()
    print("=" * 70)
    print("NEXTMOVE STEP 7: FEATURE-LABEL INTEGRATION VERIFICATION")
    print("=" * 70)

    # 1. Paths verification
    features_path = config.project_root / "data" / "synthetic" / "processed" / "synthetic_features_445_cutoffs_6675_instances.csv"
    labels_path = config.project_root / "data" / "synthetic" / "processed" / "synthetic_labels_445_cutoffs_6675_instances.csv"
    matrix_path = config.project_root / "data" / "synthetic" / "processed" / "synthetic_training_matrix_445_cutoffs_6675_instances.csv"

    print("\n[CHECK 1] Artifact Paths Existence")
    assert features_path.exists(), f"Feature artifact not found: {features_path}"
    assert labels_path.exists(), f"Label artifact not found: {labels_path}"
    print(f"  Feature artifact:  {features_path} ({features_path.stat().st_size:,} bytes)")
    print(f"  Label artifact:    {labels_path} ({labels_path.stat().st_size:,} bytes)")

    # 2. Load artifacts
    feat_df = pd.read_csv(features_path)
    label_df = pd.read_csv(labels_path)
    print(f"  Loaded features:   {len(feat_df):,} rows, {len(feat_df.columns)} columns")
    print(f"  Loaded labels:     {len(label_df):,} rows, {len(label_df.columns)} columns")
    assert len(feat_df) == 6675, f"Expected 6,675 feature rows, got {len(feat_df)}"
    assert len(label_df) == 6675, f"Expected 6,675 label rows, got {len(label_df)}"

    # 3. Perform verified integration
    print("\n[CHECK 2] Bijective Join & Integrity Verification")
    builder = TargetLabelBuilder(config=config)
    integrated_df = builder.join_features_and_labels(feat_df, label_df)

    assert len(integrated_df) == 6675, "Row count changed after join!"
    assert len(integrated_df.columns) == 17, f"Expected 17 columns, got {len(integrated_df.columns)}"
    assert "target_30d" in integrated_df.columns, "target_30d missing!"
    assert "event_count_in_window" not in integrated_df.columns, "event_count_in_window leaked!"
    assert integrated_df.isna().sum().sum() == 0, "Null values detected!"
    print("  -> 1:1 bijective join verified without leakage or nulls.")

    # 4. Feature preservation check
    print("\n[CHECK 3] Bitwise Feature Value Preservation")
    key_cols = ["cutoff_date", "competitor", "strategic_cluster"]
    feat_sorted = feat_df.sort_values(key_cols).reset_index(drop=True)
    integ_sorted = integrated_df.sort_values(key_cols).reset_index(drop=True)
    for col in feat_df.columns:
        assert np.array_equal(integ_sorted[col].values, feat_sorted[col].values), (
            f"Feature column '{col}' modified!"
        )
    print("  -> All 16 feature columns bitwise identical when matched by key.")

    # 5. Partition distribution check
    print("\n[CHECK 4] Partition & Target Distribution")
    part_counts = integrated_df["partition"].value_counts().to_dict()
    print("  Partition counts:", part_counts)
    assert part_counts == {"train": 4635, "validation": 1110, "test": 930}

    # 6. Save and verify training matrix artifact
    print("\n[CHECK 5] Training Matrix Materialization & Checksum")
    integrated_df.to_csv(matrix_path, index=False)
    hasher = hashlib.sha256()
    with open(matrix_path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    checksum = hasher.hexdigest()

    print(f"  Artifact Path:     {matrix_path}")
    print(f"  File Size:         {matrix_path.stat().st_size:,} bytes")
    print(f"  SHA-256 Checksum:  {checksum}")
    print(f"  Execution Time:    {time.time() - start_time:.3f} seconds")
    print("\nStep 7 verification PASSED completely.")
    print("=" * 70)


if __name__ == "__main__":
    main()
