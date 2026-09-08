"""Convert the site-specific BraTS splits to nnUNet raw datasets. It not supppose to be this complicated but I separate the site-specific splits into separate folders to avoid having to deal with the different site-specific splits in a single dataset.json file. This is because nnUNet does not support multiple sites in a single dataset.json file."""

import argparse
import json
import multiprocessing
import shutil
from pathlib import Path

import numpy as np
import SimpleITK as sitk

MODALITIES = ("T1", "T1ce", "T2", "FLAIR")
BRAts_LABELS = {0, 1, 2, 4}


def copy_BraTS_segmentation_and_convert_labels_to_nnUNet(
    in_file: str | Path, out_file: str | Path
) -> None:
    """Convert BraTS labels 0, 1, 2, 4 to nnUNet labels 0, 1, 2, 3. For segmnentation onlyyyy"""
    image = sitk.ReadImage(str(in_file))
    array = sitk.GetArrayFromImage(image)
    unexpected = set(np.unique(array).tolist()) - BRAts_LABELS
    if unexpected:
        raise ValueError(
            f"Unexpected labels {sorted(unexpected)} in {in_file}; "
            "expected only 0, 1, 2, and 4."
        )

    converted = np.zeros_like(array)
    converted[array == 2] = 1  # edema
    converted[array == 1] = 2  # necrotic/non-enhancing tumor
    converted[array == 4] = 3  # enhancing tumor
    output = sitk.GetImageFromArray(converted)
    output.CopyInformation(image)
    sitk.WriteImage(output, str(out_file))


def convert_labels_back_to_BraTS(seg: np.ndarray) -> np.ndarray:
    new_seg = np.zeros_like(seg)
    new_seg[seg == 1] = 2
    new_seg[seg == 2] = 1
    new_seg[seg == 3] = 4
    return new_seg


def load_convert_labels_back_to_BraTS(
    filename: str, input_folder: str, output_folder: str
) -> None:
    image = sitk.ReadImage(str(Path(input_folder) / filename))
    converted = convert_labels_back_to_BraTS(sitk.GetArrayFromImage(image))
    output = sitk.GetImageFromArray(converted)
    output.CopyInformation(image)
    sitk.WriteImage(output, str(Path(output_folder) / filename))


def convert_folder_with_predictions_back_to_BraTS(
    input_folder: str, output_folder: str, num_processes: int = 12
) -> None:
    """Convert all nnUNet prediction labels in a folder back to BraTS labels."""
    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)
    files = [path.name for path in Path(input_folder).glob("*.nii.gz")]
    with multiprocessing.get_context("spawn").Pool(num_processes) as pool:
        pool.starmap(
            load_convert_labels_back_to_BraTS,
            [(name, input_folder, output_folder) for name in files],
        )


def read_site_cases(site_json: Path) -> tuple[list[dict], list[dict], list[dict]]:
    data = json.loads(site_json.read_text(encoding="utf-8"))
    return data.get("training", []), data.get("validation", []), data.get("test", [])


def resolve_path(path: str, project_root: Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else project_root / candidate


def case_id(case: dict, project_root: Path) -> str:
    first_image = resolve_path(case["image"][0], project_root)
    return first_image.name.removesuffix(".nii.gz").removesuffix("_t1")


def copy_case_images(case: dict, destination: Path, project_root: Path) -> None:
    identifier = case_id(case, project_root)
    for channel, image_path in enumerate(case["image"]):
        source = resolve_path(image_path, project_root)
        if not source.exists():
            raise FileNotFoundError(f"Missing image: {source}")
        shutil.copy2(source, destination / f"{identifier}_{channel:04d}.nii.gz")


def convert_site(
    site_json: Path, output_root: Path, task_id: int, task_name: str
) -> None:
    project_root = Path.cwd()
    train, validation, test = read_site_cases(site_json)
    training_cases = train + validation
    site_name = site_json.parent.name
    site_number = int(site_name.removeprefix("site_"))
    dataset_id = task_id + site_number - 1
    output = output_root / f"Dataset{dataset_id:03d}_{task_name}_{site_name}"
    images_tr, images_ts, labels_tr = (
        output / "imagesTr",
        output / "imagesTs",
        output / "labelsTr",
    )
    for directory in (images_tr, images_ts, labels_tr):
        directory.mkdir(parents=True, exist_ok=True)
        for old_file in directory.glob("*.nii.gz"):
            old_file.unlink()

    for case in training_cases:
        copy_case_images(case, images_tr, project_root)
        identifier = case_id(case, project_root)
        label = resolve_path(case["label"], project_root)
        if not label.exists():
            raise FileNotFoundError(f"Missing label: {label}")
        copy_BraTS_segmentation_and_convert_labels_to_nnUNet(
            label, labels_tr / f"{identifier}.nii.gz"
        )

    for case in test:
        copy_case_images(case, images_ts, project_root)

    dataset = {
        "channel_names": {str(i): name for i, name in enumerate(MODALITIES)},
        "labels": {"background": 0, "ED": 1, "NCR": 2, "ET": 3},
        "numTraining": len(training_cases),
        "file_ending": ".nii.gz",
    }
    (output / "dataset.json").write_text(json.dumps(dataset, indent=2) + "\n")
    (output / "splits_final.json").write_text(
        json.dumps(
            [
                {
                    "train": [case_id(item, project_root) for item in train],
                    "val": [case_id(item, project_root) for item in validation],
                }
            ],
            indent=2,
        )
        + "\n"
    )
    (output / "test_cases.json").write_text(
        json.dumps([case_id(item, project_root) for item in test], indent=2) + "\n"
    )
    print(
        f"{site_name}: imagesTr={len(training_cases)}, imagesTs={len(test)}, "
        f"train={len(train)}, val={len(validation)}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--splits", type=Path, default=Path("splits/sites"))
    parser.add_argument("--output", type=Path, default=Path("nnUNet_raw"))
    parser.add_argument("--task-id", type=int, default=137)
    parser.add_argument("--task-name", default="BraTS2021")
    parser.add_argument("--site", type=int)
    args = parser.parse_args()

    sites = (
        [args.splits / f"site_{args.site:02d}"]
        if args.site is not None
        else sorted(path for path in args.splits.glob("site_*") if path.is_dir())
    )
    if not sites:
        raise FileNotFoundError(f"No site directories found in {args.splits}")
    for site in sites:
        site_json = site / "dataset.json"
        if not site_json.exists():
            raise FileNotFoundError(f"Missing split dataset: {site_json}")
        convert_site(site_json, args.output, args.task_id, args.task_name)


if __name__ == "__main__":
    main()
