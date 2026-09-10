"""Build per-site train/val/test splits from the site mapping CSV."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

from sklearn.model_selection import train_test_split

from fed_brats_nnunet.data.case import BraTSDataset
from fed_brats_nnunet.data.site_split import SiteSplit

CASE_COLUMN = "BraTS2021"
SITE_COLUMN = "Site No (represents the originating institution)"


class SiteSplitBuilder:
    """Groups BraTS cases by originating site and splits each into train/val/test."""

    def __init__(
        self,
        dataset: BraTSDataset,
        mapping_csv: Path,
        *,
        seed: int = 42,
        test_size: float = 0.2,
    ):
        self.dataset = dataset
        self.mapping_csv = Path(mapping_csv)
        self.seed = seed
        self.test_size = test_size

    def build(self) -> list[SiteSplit]:
        case_ids_by_site = self._group_cases_by_site()
        return [
            self._split_site(site_number, case_ids)
            for site_number, case_ids in sorted(case_ids_by_site.items())
        ]

    def _group_cases_by_site(self) -> dict[int, list[str]]:
        known_case_ids = set(self.dataset.discover_case_ids())
        print(f"Found {len(known_case_ids)} patients in the dataset.")

        case_ids_by_site: dict[int, list[str]] = defaultdict(list)
        mapped_case_ids = set()
        with self.mapping_csv.open(newline="", encoding="utf-8-sig") as mapping_file:
            reader = csv.DictReader(mapping_file)
            missing_columns = {CASE_COLUMN, SITE_COLUMN} - set(reader.fieldnames or [])
            if missing_columns:
                raise ValueError(
                    f"Mapping is missing required columns: {sorted(missing_columns)}"
                )

            for row in reader:
                case_id = row[CASE_COLUMN].strip()
                site_id = row[SITE_COLUMN].strip()
                if not case_id or case_id == "N/A" or case_id not in known_case_ids:
                    continue
                case_ids_by_site[int(site_id)].append(case_id)
                mapped_case_ids.add(case_id)

        unmapped = known_case_ids - mapped_case_ids
        if unmapped:
            raise ValueError(
                f"{len(unmapped)} dataset cases are missing from {self.mapping_csv}: "
                f"{sorted(unmapped)}"
            )
        return case_ids_by_site

    def _split_site(self, site_number: int, case_ids: list[str]) -> SiteSplit:
        case_ids = sorted(case_ids)
        train_ids, remaining_ids = train_test_split(
            case_ids, test_size=self.test_size, random_state=self.seed
        )
        if len(remaining_ids) < 2:
            return SiteSplit(site_number, train_ids, [], remaining_ids)
        val_ids, test_ids = train_test_split(
            remaining_ids, test_size=0.5, random_state=self.seed
        )
        return SiteSplit(site_number, train_ids, val_ids, test_ids)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=Path("raw/dataset"), type=Path)
    parser.add_argument(
        "--mapping", default=Path("splits/BraTS21-17_Mapping.csv"), type=Path
    )
    parser.add_argument("--output", default=Path("splits/sites"), type=Path)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    builder = SiteSplitBuilder(BraTSDataset(args.dataset), args.mapping, seed=args.seed)
    for split in builder.build():
        site_dir = split.save(args.output)
        print(
            f"{split.name}: train={len(split.train_ids)}, "
            f"val={len(split.val_ids)}, test={len(split.test_ids)} -> {site_dir}"
        )


if __name__ == "__main__":
    main()
