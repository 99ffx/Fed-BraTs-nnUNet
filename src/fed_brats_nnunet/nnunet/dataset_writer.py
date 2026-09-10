"""Writes one nnUNet raw dataset folder from a list of BraTSCase objects."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from fed_brats_nnunet.data.case import BraTSCase
from fed_brats_nnunet.data.labels import BraTSLabelConverter

CHANNEL_NAMES = {"0": "T1", "1": "T1ce", "2": "T2", "3": "FLAIR"}


class NnUNetDatasetWriter:
    """Materializes imagesTr/labelsTr/imagesTs/dataset.json for one nnUNet dataset."""

    def __init__(
        self,
        output_dir: Path,
        *,
        copy_files: bool = False,
        label_converter: BraTSLabelConverter | None = None,
    ):
        self.output_dir = Path(output_dir)
        self.copy_files = copy_files
        self.label_converter = label_converter or BraTSLabelConverter()
        self.images_tr = self.output_dir / "imagesTr"
        self.images_ts = self.output_dir / "imagesTs"
        self.labels_tr = self.output_dir / "labelsTr"

    def reset(self) -> None:
        """Create (or clear) the output subfolders."""
        for directory in (self.images_tr, self.images_ts, self.labels_tr):
            directory.mkdir(parents=True, exist_ok=True)
            for old_file in directory.glob("*.nii.gz"):
                old_file.unlink()

    def write_training_cases(self, cases: list[BraTSCase]) -> None:
        for case in cases:
            self._place_images(case, self.images_tr)
            self.label_converter.convert_file_to_nnunet(
                case.label_path, self.labels_tr / f"{case.case_id}.nii.gz"
            )

    def write_test_cases(self, cases: list[BraTSCase]) -> None:
        for case in cases:
            self._place_images(case, self.images_ts)

    def write_metadata(
        self,
        *,
        train_ids: list[str],
        val_ids: list[str],
        test_ids: list[str],
        num_training: int,
    ) -> None:
        dataset = {
            "channel_names": CHANNEL_NAMES,
            "labels": self.label_converter.NNUNET_LABELS,
            "numTraining": num_training,
            "file_ending": ".nii.gz",
        }
        (self.output_dir / "dataset.json").write_text(json.dumps(dataset, indent=2) + "\n")
        (self.output_dir / "splits_final.json").write_text(
            json.dumps([{"train": train_ids, "val": val_ids}], indent=2) + "\n"
        )
        (self.output_dir / "test_cases.json").write_text(
            json.dumps(test_ids, indent=2) + "\n"
        )

    def _place_images(self, case: BraTSCase, destination: Path) -> None:
        for channel, source in enumerate(case.image_paths):
            if not source.exists():
                raise FileNotFoundError(f"Missing image: {source}")
            target = destination / f"{case.case_id}_{channel:04d}.nii.gz"
            if target.exists():
                continue
            if self.copy_files:
                shutil.copy2(source, target)
            else:
                target.hardlink_to(source)
