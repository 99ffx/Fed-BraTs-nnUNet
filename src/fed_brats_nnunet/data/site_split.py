"""Per-site train/val/test case-ID splits.

Splits are stored as plain case-ID lists (`train.txt`, `val.txt`, `test.txt`)
under `splits/sites/site_XX/`. IDs are the single source of truth; file
paths are always resolved later from `BraTSDataset`, so renaming raw files
never leaves a split pointing at stale paths.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SiteSplit:
    """The train/val/test case IDs collected for one originating site."""

    site_number: int
    train_ids: list[str]
    val_ids: list[str]
    test_ids: list[str]

    @property
    def name(self) -> str:
        return f"site_{self.site_number:02d}"

    @property
    def training_ids(self) -> list[str]:
        return self.train_ids + self.val_ids

    @classmethod
    def load(cls, site_dir: Path) -> "SiteSplit":
        site_number = int(site_dir.name.removeprefix("site_"))
        return cls(
            site_number=site_number,
            train_ids=_read_ids(site_dir / "train.txt"),
            val_ids=_read_ids(site_dir / "val.txt"),
            test_ids=_read_ids(site_dir / "test.txt"),
        )

    def save(self, splits_root: Path) -> Path:
        site_dir = splits_root / self.name
        site_dir.mkdir(parents=True, exist_ok=True)
        _write_ids(site_dir / "train.txt", self.train_ids)
        _write_ids(site_dir / "val.txt", self.val_ids)
        _write_ids(site_dir / "test.txt", self.test_ids)
        return site_dir


class SiteSplitRepository:
    """Finds and loads every site's split under a `splits/sites`-style root."""

    def __init__(self, splits_root: Path | str):
        self.splits_root = Path(splits_root)

    def load_all(self) -> list[SiteSplit]:
        site_dirs = sorted(p for p in self.splits_root.glob("site_*") if p.is_dir())
        if not site_dirs:
            raise FileNotFoundError(f"No site directories found in {self.splits_root}")
        return [SiteSplit.load(site_dir) for site_dir in site_dirs]

    def load_site(self, site_number: int) -> SiteSplit:
        site_dir = self.splits_root / f"site_{site_number:02d}"
        if not site_dir.is_dir():
            raise FileNotFoundError(f"Site directory not found: {site_dir}")
        return SiteSplit.load(site_dir)


def _read_ids(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(f"Missing split file: {path}")
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_ids(path: Path, ids: list[str]) -> None:
    path.write_text("\n".join(ids) + "\n" if ids else "", encoding="utf-8")
