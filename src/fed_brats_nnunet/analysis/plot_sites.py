"""Plot BraTS case distribution across originating sites."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

SITE_COLUMN = "Site No (represents the originating institution)"
COHORT_COLUMN = "Cohort Name (if publicly available)"
UNKNOWN_SITE = "Unknown site"
UNKNOWN_COHORT = "Unknown cohort"


def plot_site_distribution(
    csv_path: Path,
    output_path: Path,
    *,
    stacked_by_cohort: bool = False,
) -> pd.DataFrame:
    """Create a site distribution plot and return the grouped counts."""
    data = pd.read_csv(csv_path, na_values=["N/A", "NA", ""])

    missing_columns = {SITE_COLUMN, COHORT_COLUMN} - set(data.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"CSV is missing required column(s): {missing}")

    data[SITE_COLUMN] = data[SITE_COLUMN].fillna(UNKNOWN_SITE).astype(str).str.strip()
    data[COHORT_COLUMN] = (
        data[COHORT_COLUMN].fillna(UNKNOWN_COHORT).astype(str).str.strip()
    )

    if stacked_by_cohort:
        counts = pd.crosstab(data[SITE_COLUMN], data[COHORT_COLUMN])
        counts = counts.loc[counts.sum(axis=1).sort_values(ascending=False).index]
        ax = counts.plot(kind="bar", stacked=True, figsize=(14, 7), width=0.85)
        ax.set_ylabel("Number of cases")
        ax.legend(title="Cohort", bbox_to_anchor=(1.02, 1), loc="upper left")
    else:
        counts = data[SITE_COLUMN].value_counts().rename("cases").to_frame()
        ax = counts["cases"].plot(kind="bar", figsize=(14, 7), width=0.85)
        ax.set_ylabel("Number of cases")

    ax.set_xlabel("Originating site")
    ax.set_title("BraTS case distribution across originating sites")
    ax.tick_params(axis="x", labelrotation=45)
    for label in ax.get_xticklabels():
        label.set_horizontalalignment("right")
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure = ax.get_figure()
    figure.tight_layout()
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)

    return counts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Distribution of BraTS cases across sites."
    )
    parser.add_argument(
        "csv_path",
        nargs="?",
        type=Path,
        default=Path("splits/BraTS21-17_Mapping.csv"),
        help="Path to the BraTS mapping CSV.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("plots/site_distribution.png"),
        help="Output image path.",
    )
    parser.add_argument(
        "--stacked-by-cohort",
        action="store_true",
        help="Stack each site's bar by cohort.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    counts = plot_site_distribution(
        args.csv_path,
        args.output,
        stacked_by_cohort=args.stacked_by_cohort,
    )
    print(f"Saved plot to {args.output}")
    print(f"Plotted {int(counts.to_numpy().sum())} cases across {len(counts)} sites.")


if __name__ == "__main__":
    main()
