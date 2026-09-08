import argparse
import json
import shutil
from pathlib import Path

MODALITIES = ["t1", "t1ce", "t2", "flair"]


def read_split(site_dir, split):
    file = site_dir / f"{split}.txt"
    if not file.exists():
        raise FileNotFoundError(f"Missing file: {file}")
    return [line.strip() for line in file.read_text().splitlines() if line.strip()]


def add_file(source, target, copy_files):
    if not source.exists():
        raise FileNotFoundError(f"Missing file: {source}")
    if target.exists():
        return
    if copy_files:
        shutil.copy2(source, target)
    else:
        target.hardlink_to(source)


def prepare_site(site_dir, source_dir, output_dir, copy_files):
    train = read_split(site_dir, "train")
    val = read_split(site_dir, "val")
    test = read_split(site_dir, "test")
    case_ids = train + val + test

    images = output_dir / "imagesTr"
    labels = output_dir / "labelsTr"
    images.mkdir(parents=True, exist_ok=True)
    labels.mkdir(parents=True, exist_ok=True)

    for case_id in case_ids:
        case_dir = source_dir / case_id
        for channel, modality in enumerate(MODALITIES):
            add_file(
                case_dir / f"{case_id}_{modality}.nii.gz",
                images / f"{case_id}_{channel:04d}.nii.gz",
                copy_files,
            )
        add_file(
            case_dir / f"{case_id}_seg.nii.gz",
            labels / f"{case_id}.nii.gz",
            copy_files,
        )

    dataset = {
        "channel_names": {"0": "T1", "1": "T1ce", "2": "T2", "3": "FLAIR"},
        "labels": {"background": 0, "NCR": 1, "ED": 2, "ET": 3},
        "numTraining": len(case_ids),
        "file_ending": ".nii.gz",
    }
    (output_dir / "dataset.json").write_text(json.dumps(dataset, indent=2) + "\n")
    (output_dir / "splits_final.json").write_text(
        json.dumps([{"train": train, "val": val}], indent=2) + "\n"
    )
    (output_dir / "test_cases.json").write_text(json.dumps(test, indent=2) + "\n")

    print(f"{site_dir.name}: train={len(train)}, val={len(val)}, test={len(test)}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="raw/dataset", type=Path)
    parser.add_argument("--splits", default="splits/sites", type=Path)
    parser.add_argument(
        "--output", default="nnUNet_raw/Dataset001_BraTS", type=Path
    )
    parser.add_argument("--site", type=int, help="Prepare only one site.")
    parser.add_argument("--copy", action="store_true", help="Copy instead of link.")
    args = parser.parse_args()

    if args.site is None:
        sites = sorted(path for path in args.splits.glob("site_*") if path.is_dir())
    else:
        sites = [args.splits / f"site_{args.site:02d}"]

    if not sites:
        raise FileNotFoundError(f"No site directories found in {args.splits}")

    for site_dir in sites:
        if not site_dir.is_dir():
            raise FileNotFoundError(f"Site directory not found: {site_dir}")
        output = (
            args.output
            if args.site is not None
            else args.output.parent / f"{args.output.name}_{site_dir.name}"
        )
        prepare_site(site_dir, args.source, output, args.copy)


if __name__ == "__main__":
    main()
