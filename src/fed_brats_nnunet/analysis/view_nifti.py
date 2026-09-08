import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from matplotlib.colors import ListedColormap


def _show_slice(image, label, axis, index, title, axes):
    image_slice = np.take(image, index, axis=axis)
    label_slice = np.take(label, index, axis=axis)
    image_slice = np.rot90(image_slice)
    label_slice = np.rot90(label_slice)

    foreground = image_slice[image_slice > 0]
    if foreground.size:
        low, high = np.percentile(foreground, [1, 99])
    else:
        low, high = float(image_slice.min()), float(image_slice.max())
    axes.imshow(image_slice, cmap="gray", vmin=low, vmax=high)
    axes.imshow(
        np.ma.masked_where(label_slice == 0, label_slice),
        cmap=ListedColormap(["red", "lime", "blue"]),
        alpha=0.45,
        vmin=1,
        vmax=3,
    )
    axes.set_title(f"{title} (slice {index})")
    axes.axis("off")


def plot_nifti_with_segmentation(
    image_path,
    label_path,
    output_path="plots/overlay_.png",
):
    # Save center axial, coronal, and sagittal views with labels overlaid.
    image = np.asarray(nib.load(image_path).get_fdata())
    label = np.asarray(nib.load(label_path).get_fdata())

    if image.shape != label.shape:
        raise ValueError(
            f"Image and label shapes differ: {image.shape} versus {label.shape}"
        )

    figure, axes = plt.subplots(1, 3, figsize=(15, 5))
    for axis, title, plot_axis in zip(
        axes,
        ("Sagittal", "Coronal", "Axial"),
        (0, 1, 2),
    ):
        _show_slice(
            image,
            label,
            plot_axis,
            image.shape[plot_axis] // 2,
            title,
            axis,
        )

    figure.suptitle(Path(image_path).stem)
    figure.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)
    return output_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path, help="Image NIfTI file.")
    parser.add_argument("label", type=Path, help="Segmentation NIfTI file.")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("plots/nifti_overlay.png"),
    )
    args = parser.parse_args()
    output = plot_nifti_with_segmentation(args.image, args.label, args.output)
    print(f"Saved overlay to {output}")


if __name__ == "__main__":
    main()
