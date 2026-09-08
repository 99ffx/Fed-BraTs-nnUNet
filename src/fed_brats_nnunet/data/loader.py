"""Load BraTS train, validation, and test cases."""

from pathlib import Path

def make_case(dataset_dir, case_id):
    case_dir = Path(dataset_dir) / case_id
    return {
        "id": case_id,
        "image": [
            str(case_dir / f"{case_id}_t1.nii.gz"),
            str(case_dir / f"{case_id}_t1ce.nii.gz"),
            str(case_dir / f"{case_id}_t2.nii.gz"),
            str(case_dir / f"{case_id}_flair.nii.gz"),
        ],
        "label": str(case_dir / f"{case_id}_seg.nii.gz"),
    }

def load_split(split, site=None, dataset_dir="raw/dataset", splits_dir="splits/sites"):
    if split not in ("train", "val", "test"):
        raise ValueError("split must be 'train', 'val', or 'test'")

    splits_dir = Path(splits_dir)
    dataset_dir = Path(dataset_dir)

    if site is None:
        site_dirs = sorted(splits_dir.glob("site_*"))
    else:
        site_dirs = [splits_dir / f"site_{int(site):02d}"]  # 2 digit site code

    cases = []
    for site_dir in site_dirs:
        split_file = site_dir / f"{split}.txt"
        if not split_file.exists():
            raise FileNotFoundError(f"File not found: {split_file}")

        case_ids = split_file.read_text(encoding="utf-8").splitlines()
        for case_id in case_ids:
            if case_id.strip():
                cases.append(make_case(dataset_dir, case_id.strip()))

    return cases

def load_data(site=None, dataset_dir="raw/dataset", splits_dir="splits/sites"):
    return {
        "train": load_split("train", site, dataset_dir, splits_dir),
        "val": load_split("val", site, dataset_dir, splits_dir),
        "test": load_split("test", site, dataset_dir, splits_dir),
    }

if __name__ == "__main__":
    data = load_data()
    print("Train:", len(data["train"]))
    print("Validation:", len(data["val"]))
    print("Test:", len(data["test"]))