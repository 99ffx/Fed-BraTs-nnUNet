"""Converts one federated site's split into a standalone nnUNet raw dataset."""

from __future__ import annotations

from pathlib import Path

from fed_brats_nnunet.data.case import BraTSDataset
from fed_brats_nnunet.data.site_split import SiteSplit, SiteSplitRepository
from fed_brats_nnunet.nnunet.dataset_writer import NnUNetDatasetWriter


class SiteDatasetConverter:
    """Builds `DatasetXXX_<task>_site_YY` for one site from its BraTSCase split."""

    def __init__(
        self,
        dataset: BraTSDataset,
        output_root: Path,
        task_id: int,
        task_name: str,
        *,
        copy_files: bool = False,
    ):
        self.dataset = dataset
        self.output_root = Path(output_root)
        self.task_id = task_id
        self.task_name = task_name
        self.copy_files = copy_files

    def output_dir(self, split: SiteSplit) -> Path:
        dataset_id = self.task_id + split.site_number - 1
        return self.output_root / f"Dataset{dataset_id:03d}_{self.task_name}_{split.name}"

    def convert(self, split: SiteSplit) -> None:
        writer = NnUNetDatasetWriter(self.output_dir(split), copy_files=self.copy_files)
        writer.reset()

        training_cases = self.dataset.cases(split.training_ids)
        test_cases = self.dataset.cases(split.test_ids)

        writer.write_training_cases(training_cases)
        writer.write_test_cases(test_cases)
        writer.write_metadata(
            train_ids=split.train_ids,
            val_ids=split.val_ids,
            test_ids=split.test_ids,
            num_training=len(training_cases),
        )
        print(
            f"{split.name}: imagesTr={len(training_cases)}, imagesTs={len(test_cases)}, "
            f"train={len(split.train_ids)}, val={len(split.val_ids)}"
        )


def convert_all_sites(
    *,
    splits_root: Path,
    dataset_dir: Path,
    output_root: Path,
    task_id: int,
    task_name: str,
    site_number: int | None = None,
    copy_files: bool = False,
) -> None:
    repository = SiteSplitRepository(splits_root)
    splits = (
        [repository.load_site(site_number)]
        if site_number is not None
        else repository.load_all()
    )

    converter = SiteDatasetConverter(
        BraTSDataset(dataset_dir), output_root, task_id, task_name, copy_files=copy_files
    )
    for split in splits:
        converter.convert(split)
