#!/usr/bin/env bash

LABEL_DIR="${1}"

if [ -z "$LABEL_DIR" ]; then
    echo "Usage: $0 /path/to/labelsTr"
    exit 1
fi

python <<EOF
import nibabel as nib
import numpy as np
import os
from pathlib import Path

label_dir = Path("${LABEL_DIR}")

expected_labels = {0, 1, 2, 3}

n_cases = 0
n_empty = 0
n_bad_labels = 0

print("=" * 100)
print(f"Checking labels in: {label_dir}")
print("=" * 100)

for f in sorted(label_dir.glob("*.nii.gz")):
    n_cases += 1

    try:
        label = nib.load(str(f)).get_fdata()

        unique = np.unique(label).astype(int)
        unique_set = set(unique.tolist())

        has_tumor = np.any(label > 0)

        unexpected = unique_set - expected_labels

        msg = (
            f"{f.name:35s} "
            f"shape={str(label.shape):20s} "
            f"labels={unique.tolist()}"
        )

        if not has_tumor:
            msg += "  <-- EMPTY LABEL"
            n_empty += 1

        if unexpected:
            msg += f"  <-- UNEXPECTED LABELS {sorted(unexpected)}"
            n_bad_labels += 1

        print(msg)

    except Exception as e:
        print(f"{f.name}: ERROR -> {e}")

print()
print("=" * 100)
print("SUMMARY")
print("=" * 100)
print(f"Cases checked      : {n_cases}")
print(f"Empty labels       : {n_empty}")
print(f"Bad label values   : {n_bad_labels}")
print("=" * 100)
EOF