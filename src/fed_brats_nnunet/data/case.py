"""A single BraTS case and how to locate its files on disk."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

MODALITIES = ("t1", "t1ce", "t2", "flair")


@dataclass(frozen=True)
class BraTSCase:
    """One patient case, stored under `dataset_dir/case_id/`."""

    case_id: str
    dataset_dir: Path

    @property
    def case_dir(self) -> Path:
        return self.dataset_dir / self.case_id

    @property
    def image_paths(self) -> list[Path]:
        return [
            self.case_dir / f"{self.case_id}_{modality}.nii.gz"
            for modality in MODALITIES
        ]

    @property
    def label_path(self) -> Path:
        return self.case_dir / f"{self.case_id}_seg.nii.gz"


class BraTSDataset:
    """Resolves case IDs to `BraTSCase` objects rooted at a raw dataset directory."""

    def __init__(self, dataset_dir: Path | str):
        self.dataset_dir = Path(dataset_dir)

    def case(self, case_id: str) -> BraTSCase:
        return BraTSCase(case_id, self.dataset_dir)

    def cases(self, case_ids: list[str]) -> list[BraTSCase]:
        return [self.case(case_id) for case_id in case_ids]

    def discover_case_ids(self) -> list[str]:
        """List every case folder present on disk (used to build splits)."""
        return sorted(
            path.name
            for path in self.dataset_dir.iterdir()
            if path.is_dir() and path.name.startswith("BraTS")
        )
