from pathlib import Path
from sklearn.model_selection import train_test_split
import json

SEED = 42

DATASET_PATH = Path("dataset")
OUTPUT_PATH = Path("splits")

OUTPUT_PATH.mkdir(exist_ok=True, parents=True)

patients = sorted([p.name for p in DATASET_PATH.iterdir() if p.is_dir() & (p.name.startswith("BraTS"))])

print(f"Found {len(patients)} patients in the dataset.")

train_ids, temp_ids = train_test_split(patients, test_size=0.2, random_state=SEED)
val_ids, test_ids = train_test_split(temp_ids, test_size=0.5, random_state=SEED)

# print(f"Train: {len(train_ids)}, Validation: {len(val_ids)}, Test: {len(test_ids)}")

print(f"Train IDs: {train_ids[:5]}")

for split_name, split_ids in {"train": train_ids, "val": val_ids, "test": test_ids}.items():

    with open(OUTPUT_PATH / f"{split_name}.txt", "w") as f:

        f.write("\n".join(split_ids))


dataset = {"training":[],
           "validation":[],
           "test":[]}

def make_entry(case_id):
    folder = DATASET_PATH / case_id

    return{
        "image": [
            str(folder / f"{case_id}_t1.nii.gz"),
            str(folder / f"{case_id}_t1ce.nii.gz"),
            str(folder / f"{case_id}_t2.nii.gz"),
            str(folder / f"{case_id}_flair.nii.gz")
        ],
        "label": str(folder / f"{case_id}_seg.nii.gz")
    }

dataset["training"] = [make_entry(case_id) for case_id in train_ids]
dataset["validation"] = [make_entry(case_id) for case_id in val_ids]
dataset["test"] = [make_entry(case_id) for case_id in test_ids]

with open(OUTPUT_PATH / "dataset.json", "w") as f:
    json.dump(dataset, f, indent=4)

print(f"Saved dataset splits to {OUTPUT_PATH / 'dataset.json'}")