"""CLI: convert per-site BraTS splits into nnUNet raw datasets."""

from __future__ import annotations

import argparse
from pathlib import Path

from fed_brats_nnunet.nnunet.converter import convert_all_sites


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default=Path("raw/dataset"), type=Path)
    parser.add_argument("--splits", default=Path("splits/sites"), type=Path)
    parser.add_argument("--output", default=Path("nnUNet_raw"), type=Path)
    parser.add_argument("--task-id", type=int, default=137)
    parser.add_argument("--task-name", default="BraTS2021")
    parser.add_argument("--site", type=int, help="Convert only one site.")
    parser.add_argument("--copy", action="store_true", help="Copy instead of hardlink.")
    args = parser.parse_args()

    convert_all_sites(
        splits_root=args.splits,
        dataset_dir=args.source,
        output_root=args.output,
        task_id=args.task_id,
        task_name=args.task_name,
        site_number=args.site,
        copy_files=args.copy,
    )


if __name__ == "__main__":
    main()
