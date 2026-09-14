#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATASET_DIR="$ROOT_DIR/nnUNet_raw/Dataset001_BraTS2021_all"

mkdir -p "$DATASET_DIR/imagesTr" "$DATASET_DIR/imagesTs" "$DATASET_DIR/labelsTr"
find "$DATASET_DIR/imagesTs" -maxdepth 1 -type f -delete

# echo "Number of image files:"
# find nnUNet_raw/Dataset*_BraTS2021_site_*/imagesTr -type f | wc -l

# echo "Number of label files:"
# find nnUNet_raw/Dataset*_BraTS2021_site_*/labelsTr -type f | wc -l

# echo "Number of test files:"
# find nnUNet_raw/Dataset*_BraTS2021_site_*/imagesTs -type f | wc -l

for d in "$ROOT_DIR"/nnUNet_raw/Dataset*_BraTS2021_site_*; do
    cp "$d"/imagesTr/* "$DATASET_DIR/imagesTr"
    cp "$d"/labelsTr/* "$DATASET_DIR/labelsTr"
    cp "$d"/imagesTs/* "$DATASET_DIR/imagesTs"
done

NUM=$(find "$DATASET_DIR/labelsTr" -maxdepth 1 -name "*.nii.gz" | wc -l)

cat > "$DATASET_DIR/dataset.json" <<EOL
{
    "channel_names": {
        "0": "T1",
        "1": "T1ce",
        "2": "T2",
        "3": "FLAIR"
    },
    "labels": {
        "background": 0,
        "ED": 1,
        "NCR": 2,
        "ET": 3
    },
    "numTraining": $NUM,
    "file_ending": ".nii.gz"
}
EOL