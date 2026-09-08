"""Create train, validation, and test splits for each originating site."""

import csv
import json
from collections import defaultdict
from pathlib import Path

from sklearn.model_selection import train_test_split

SEED = 42

DATASET_PATH = Path("raw/dataset")
MAPPING_PATH = Path("splits/BraTS21-17_Mapping.csv")
OUTPUT_PATH = Path("splits/sites")

CASE_COLUMN = "BraTS2021"
SITE_COLUMN = "Site No (represents the originating institution)"

OUTPUT_PATH.mkdir(exist_ok=True, parents=True)

patients = sorted(
    p.name
    for p in DATASET_PATH.iterdir()
    if p.is_dir() and p.name.startswith("BraTS")
)

print(f"Found {len(patients)} patients in the dataset.")

patients_by_site = defaultdict(list)
patient_set = set(patients)

with MAPPING_PATH.open(newline="", encoding="utf-8-sig") as mapping_file:
    mapping_reader = csv.DictReader(mapping_file)
    missing_columns = {CASE_COLUMN, SITE_COLUMN} - set(mapping_reader.fieldnames or [])
    if missing_columns:
        raise ValueError(f"Mapping is missing required columns: {sorted(missing_columns)}")

    mapped_patients = set()
    for row in mapping_reader:
        case_id = row[CASE_COLUMN].strip()
        site_id = row[SITE_COLUMN].strip()
        if not case_id or case_id == "N/A":
            continue
        if case_id in patient_set:
            patients_by_site[site_id].append(case_id)
            mapped_patients.add(case_id)

unmapped_patients = patient_set - mapped_patients
if unmapped_patients:
    raise ValueError(
        f"{len(unmapped_patients)} dataset cases are missing from {MAPPING_PATH}: "
        f"{sorted(unmapped_patients)}"
    )


def make_entry(case_id):
    folder = DATASET_PATH / case_id

    return {
        "image": [
            str(folder / f"{case_id}_t1.nii.gz"),
            str(folder / f"{case_id}_t1ce.nii.gz"),
            str(folder / f"{case_id}_t2.nii.gz"),
            str(folder / f"{case_id}_flair.nii.gz"),
        ],
        "label": str(folder / f"{case_id}_seg.nii.gz"),
    }


def split_site_patients(site_patients):
    train_ids, temp_ids = train_test_split(
        site_patients, test_size=0.2, random_state=SEED
    )
    if len(temp_ids) < 2:
        return train_ids, [], temp_ids
    val_ids, test_ids = train_test_split(
        temp_ids, test_size=0.5, random_state=SEED
    )
    return train_ids, val_ids, test_ids


for site_id, site_patients in sorted(patients_by_site.items(), key=lambda item: int(item[0])):
    site_patients = sorted(site_patients)
    train_ids, val_ids, test_ids = split_site_patients(site_patients)

    site_output_path = OUTPUT_PATH / f"site_{int(site_id):02d}"
    site_output_path.mkdir(exist_ok=True, parents=True)

    for split_name, split_ids in {
        "train": train_ids,
        "val": val_ids,
        "test": test_ids,
    }.items():
        (site_output_path / f"{split_name}.txt").write_text(
            "\n".join(split_ids) + "\n", encoding="utf-8"
        )

    dataset = {
        "training": [make_entry(case_id) for case_id in train_ids],
        "validation": [make_entry(case_id) for case_id in val_ids],
        "test": [make_entry(case_id) for case_id in test_ids],
    }
    (site_output_path / "dataset.json").write_text(
        json.dumps(dataset, indent=4) + "\n", encoding="utf-8"
    )

    print(
        f"Site {site_id}: train={len(train_ids)}, "
        f"validation={len(val_ids)}, test={len(test_ids)}"
    )
