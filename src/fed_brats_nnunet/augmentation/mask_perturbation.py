"""Simulate inter-site segmentation inconsistency via morphological noise.

Federated BraTS sites rarely annotate with identical care: some over-segment
(dilate), some under-segment (erode). `MorphologicalMaskPerturber` grows or
shrinks each label of a segmentation mask by an exact target volume-change
percentage (5-10% by default) to synthesize that kind of inconsistency.

A single full morphological dilation/erosion step already changes the volume
of a typical tumor-sized structure by well over 10% (a 1-voxel boundary
shell is a large fraction of a small blob's volume), so hitting a precise
5-10% target requires only *partially* including the outermost shell rather
than adding/removing it wholesale. That's what `perturb_binary_mask` does:
it grows/shrinks in whole 1-voxel layers, then randomly keeps just enough
voxels of the final layer to land on the target voxel count exactly.
"""

from __future__ import annotations

import multiprocessing
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from scipy.ndimage import binary_dilation, binary_erosion, generate_binary_structure


@dataclass(frozen=True)
class LabelPerturbationReport:
    """What happened to one label during a perturbation."""

    label: int
    operation: str
    original_voxels: int
    perturbed_voxels: int

    @property
    def volume_change(self) -> float:
        if self.original_voxels == 0:
            return 0.0
        return (self.perturbed_voxels - self.original_voxels) / self.original_voxels


class MorphologicalMaskPerturber:
    """Grows or shrinks each label of a segmentation mask by a target volume change."""

    def __init__(
        self,
        volume_change_range: tuple[float, float] = (0.05, 0.10),
        max_layers: int = 200,
        seed: int | None = None,
    ):
        self.volume_change_range = volume_change_range
        self.max_layers = max_layers
        self.rng = np.random.default_rng(seed)
        self._structure = generate_binary_structure(rank=3, connectivity=1)

    def perturb_binary_mask(
        self, binary_mask: np.ndarray, operation: str = "random"
    ) -> tuple[np.ndarray, str, float]:
        """Grow/shrink one binary mask by an exact target voxel count.

        Returns (perturbed_mask, operation_used, target_change_used).
        """
        if operation == "random":
            operation = "dilate" if self.rng.random() < 0.5 else "erode"
        if operation not in ("erode", "dilate"):
            raise ValueError(f"operation must be 'erode', 'dilate', or 'random', got {operation!r}")

        original_voxels = int(binary_mask.sum())
        if original_voxels == 0:
            return binary_mask, operation, 0.0

        target_change = float(self.rng.uniform(*self.volume_change_range))
        target_count = max(1, round(target_change * original_voxels))
        if operation == "erode":
            target_count = min(target_count, original_voxels - 1)  # never erase the structure

        changed = np.zeros_like(binary_mask, dtype=bool)
        frontier = binary_mask
        remaining = target_count
        for _ in range(self.max_layers):
            if remaining <= 0:
                break
            if operation == "dilate":
                grown = binary_dilation(frontier, structure=self._structure)
                layer = grown & ~binary_mask & ~changed
            else:
                shrunk = binary_erosion(frontier, structure=self._structure)
                layer = frontier & ~shrunk

            layer_count = int(layer.sum())
            if layer_count == 0:
                break  # dilation/erosion has nowhere left to grow/shrink into

            if layer_count <= remaining:
                changed |= layer
                remaining -= layer_count
                frontier = grown if operation == "dilate" else shrunk
            else:
                layer_coords = np.argwhere(layer)
                chosen = self.rng.choice(len(layer_coords), size=remaining, replace=False)
                changed[tuple(layer_coords[chosen].T)] = True
                remaining = 0

        perturbed_mask = (binary_mask | changed) if operation == "dilate" else (binary_mask & ~changed)
        return perturbed_mask, operation, target_change

    def perturb_label_map(
        self,
        label_array: np.ndarray,
        labels: list[int] | None = None,
        operation: str = "random",
    ) -> tuple[np.ndarray, list[LabelPerturbationReport]]:
        """Perturb every requested label of a multi-label mask independently.

        Labels are repainted largest-original-volume-first, so a smaller,
        nested structure (e.g. enhancing tumor inside edema) always wins any
        overlap created by a neighboring label's dilation.
        """
        if labels is None:
            labels = sorted(int(v) for v in np.unique(label_array) if v != 0)

        voxel_counts = {label: int((label_array == label).sum()) for label in labels}
        paint_order = sorted(labels, key=lambda label: voxel_counts[label], reverse=True)

        perturbed = np.zeros_like(label_array)
        reports = []
        for label in paint_order:
            binary_mask = label_array == label
            perturbed_mask, used_operation, _target_change = self.perturb_binary_mask(binary_mask, operation)
            perturbed[perturbed_mask] = label
            reports.append(
                LabelPerturbationReport(
                    label=label,
                    operation=used_operation,
                    original_voxels=voxel_counts[label],
                    perturbed_voxels=int(perturbed_mask.sum()),
                )
            )
        return perturbed, sorted(reports, key=lambda report: report.label)

    def perturb_file(
        self,
        source: Path,
        target: Path,
        labels: list[int] | None = None,
        operation: str = "random",
    ) -> list[LabelPerturbationReport]:
        image = sitk.ReadImage(str(source))
        array = sitk.GetArrayFromImage(image)
        perturbed, reports = self.perturb_label_map(array, labels, operation)

        output = sitk.GetImageFromArray(perturbed.astype(array.dtype))
        output.CopyInformation(image)
        Path(target).parent.mkdir(parents=True, exist_ok=True)
        sitk.WriteImage(output, str(target))
        return reports

    def perturb_folder(
        self,
        input_dir: Path,
        output_dir: Path,
        labels: list[int] | None = None,
        operation: str = "random",
        num_processes: int = 8,
    ) -> dict[str, list[LabelPerturbationReport]]:
        """Perturb every `*.nii.gz` in `input_dir`, each with its own derived seed."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        files = sorted(Path(input_dir).glob("*.nii.gz"))
        base_seed = int(self.rng.integers(0, 2**32 - 1))

        jobs = [
            (f, output_dir / f.name, labels, operation, self.volume_change_range, self.max_layers, base_seed + i)
            for i, f in enumerate(files)
        ]
        with multiprocessing.get_context("spawn").Pool(num_processes) as pool:
            results = pool.starmap(_perturb_one_file, jobs)

        return {f.name: reports for f, reports in zip(files, results)}


def _perturb_one_file(
    source: Path,
    target: Path,
    labels: list[int] | None,
    operation: str,
    volume_change_range: tuple[float, float],
    max_layers: int,
    seed: int,
) -> list[LabelPerturbationReport]:
    perturber = MorphologicalMaskPerturber(volume_change_range, max_layers, seed)
    return perturber.perturb_file(source, target, labels, operation)
