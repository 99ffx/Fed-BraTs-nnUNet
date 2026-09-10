"""CLI: convert nnUNet prediction labels in a folder back to BraTS labels."""

from __future__ import annotations

import argparse
from pathlib import Path

from fed_brats_nnunet.data.labels import BraTSLabelConverter


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Folder of nnUNet prediction .nii.gz files.")
    parser.add_argument("output", type=Path, help="Folder to write BraTS-label files to.")
    parser.add_argument("--num-processes", type=int, default=12)
    args = parser.parse_args()

    BraTSLabelConverter().convert_folder_to_brats(
        args.input, args.output, num_processes=args.num_processes
    )


if __name__ == "__main__":
    main()
