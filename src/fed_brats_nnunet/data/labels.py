"""Conversion between BraTS and nnUNet segmentation label spaces.

BraTS labels:   0 background, 1 NCR, 2 ED, 4 ET
nnUNet labels:  0 background, 1 ED,  2 NCR, 3 ET
"""

from __future__ import annotations

import multiprocessing
from pathlib import Path

import numpy as np
import SimpleITK as sitk


class BraTSLabelConverter:
    """Remaps segmentation label values, preserving image geometry."""

    NNUNET_LABELS = {"background": 0, "ED": 1, "NCR": 2, "ET": 3}
    _TO_NNUNET = {0: 0, 2: 1, 1: 2, 4: 3}
    _TO_BRATS = {value: key for key, value in _TO_NNUNET.items()}

    def convert_file_to_nnunet(self, source: Path, target: Path) -> None:
        image = sitk.ReadImage(str(source))
        array = sitk.GetArrayFromImage(image)
        self._validate(array, source, self._TO_NNUNET)
        self._write_like(self._remap(array, self._TO_NNUNET), image, target)

    def convert_file_to_brats(self, source: Path, target: Path) -> None:
        image = sitk.ReadImage(str(source))
        array = sitk.GetArrayFromImage(image)
        self._validate(array, source, self._TO_BRATS)
        self._write_like(self._remap(array, self._TO_BRATS), image, target)

    def convert_folder_to_brats(
        self, input_dir: Path, output_dir: Path, num_processes: int = 12
    ) -> None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        files = sorted(Path(input_dir).glob("*.nii.gz"))
        with multiprocessing.get_context("spawn").Pool(num_processes) as pool:
            pool.starmap(
                _convert_file_to_brats,
                [(f, output_dir / f.name) for f in files],
            )

    @staticmethod
    def _remap(array: np.ndarray, mapping: dict[int, int]) -> np.ndarray:
        converted = np.zeros_like(array)
        for source_value, target_value in mapping.items():
            if source_value:
                converted[array == source_value] = target_value
        return converted

    @staticmethod
    def _validate(array: np.ndarray, source: Path, mapping: dict[int, int]) -> None:
        unexpected = set(np.unique(array).tolist()) - set(mapping)
        if unexpected:
            raise ValueError(
                f"Unexpected labels {sorted(unexpected)} in {source}; "
                f"expected only {sorted(mapping)}."
            )

    @staticmethod
    def _write_like(array: np.ndarray, reference_image: "sitk.Image", target: Path) -> None:
        output = sitk.GetImageFromArray(array)
        output.CopyInformation(reference_image)
        sitk.WriteImage(output, str(target))


def _convert_file_to_brats(source: Path, target: Path) -> None:
    BraTSLabelConverter().convert_file_to_brats(source, target)
