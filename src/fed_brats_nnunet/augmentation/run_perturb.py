"""CLI for percentage-based tumour-core perturbation (uses perturb_simple.py).

  1. DEMO -- no data needed, checks the install
       python run_perturbation.py demo

  2. SINGLE -- one case, to eyeball before touching the dataset
       python run_perturbation.py single --input case.nii.gz --pct 40 --out out.nii.gz

  3. AUDIT -- is this percentage achievable on your data? Writes nothing.
       python run_perturbation.py audit --labels-dir .../labelsTr --pct 40

  4. DATASET -- build a perturbed nnU-Net dataset (labels + dataset.json)
       python run_perturbation.py dataset \\
           --source-dir $nnUNet_raw/Dataset501_FeTS2022 \\
           --out-dir    $nnUNet_raw/Dataset601_FeTS2022_pct40 --pct 40

     per-site instead of uniform:
       python run_perturbation.py dataset --source-dir ... --out-dir ... \\
           --csv partitioning_1.csv --site-config site_pct.json

ALWAYS run `audit` first. Morphology moves in whole voxel layers, so small
percentages land on "do nothing" for many cases -- audit counts those.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np

from perturb_simple import detect_convention, perturb_core

CHANNELS = {"0": "T1", "1": "T1ce", "2": "T2", "3": "Flair"}


# ---------------------------------------------------------------------------
# io
# ---------------------------------------------------------------------------

def read_seg(path):
    import SimpleITK as sitk
    img = sitk.ReadImage(str(path))
    return sitk.GetArrayFromImage(img), img


def write_seg(arr, reference, path):
    import SimpleITK as sitk
    out = sitk.GetImageFromArray(arr.astype(np.uint8))
    out.CopyInformation(reference)       # keeps spacing / origin / direction
    import SimpleITK as _s
    _s.WriteImage(out, str(path))


def write_dataset_json(out_dir: Path, n_cases: int, output: str, note: str):
    """nnU-Net v2 dataset.json matching whichever label scheme we wrote."""
    if output == "binary":
        labels = {"background": 0, "tumor core": 1}
        extra = {}
    else:
        labels = {"background": 0, "whole tumor": [1, 2], "tumor core": [2]}
        extra = {"regions_class_order": [1, 2]}

    ds = {
        "channel_names": CHANNELS,
        "labels": labels,
        "numTraining": n_cases,
        "file_ending": ".nii.gz",
        "description": note,
        **extra,
    }
    (out_dir / "dataset.json").write_text(json.dumps(ds, indent=4))
    return ds


def link_images(source_dir: Path, out_dir: Path):
    """Symlink imagesTr -- the scans are unchanged, only labels are perturbed."""
    src, dst = source_dir / "imagesTr", out_dir / "imagesTr"
    if not src.exists():
        print(f"  note: {src} not found, skipping imagesTr")
        return 0
    dst.mkdir(parents=True, exist_ok=True)
    n = 0
    for f in src.glob("*.nii.gz"):
        t = dst / f.name
        if not t.exists() and not t.is_symlink():
            t.symlink_to(f.resolve())
            n += 1
    return n


def load_site_map(csv_path, site_col=None, subj_col=None):
    import pandas as pd
    df = pd.read_csv(csv_path)
    site_col = site_col or next((c for c in
        ["Partition_ID", "partition_id", "site", "Site", "site_id"] if c in df.columns), None)
    subj_col = subj_col or next((c for c in
        ["Subject_ID", "subject_id", "subject", "case_id", "patient_id"] if c in df.columns), None)
    if not site_col or not subj_col:
        raise SystemExit(f"Could not find site/subject columns in {list(df.columns)}. "
                         f"Pass --site-col and --subject-col.")
    print(f"site column '{site_col}', subject column '{subj_col}'")
    return {str(r[subj_col]): int(r[site_col]) for _, r in df.iterrows()}


def make_structure(radius):
    if not radius:
        return None
    from skimage.morphology import ball
    return ball(radius)


# ---------------------------------------------------------------------------

def cmd_demo(args):
    sh = (90, 90, 90)
    c = [s // 2 for s in sh]
    z, y, x = np.ogrid[:sh[0], :sh[1], :sh[2]]
    d = np.sqrt((z - c[0]) ** 2 + (y - c[1]) ** 2 + (x - c[2]) ** 2)
    seg = np.zeros(sh, np.uint8)
    seg[d <= 26] = 2
    seg[d <= 12] = 4
    seg[d <= 6] = 1

    print(f"convention : {detect_convention(seg)}")
    print(f"core       : {int(np.isin(seg,[1,4]).sum())} voxels\n")
    print(f"{'requested':>10} {'achieved':>10} {'layers':>7} {'core':>8}")
    print("-" * 40)
    for pct in [-60, -40, -20, 0, 20, 40, 60]:
        _, i = perturb_core(seg, float(pct), output=args.output)
        print(f"{pct:>+9d}% {i['volume_change_pct']:>+9.1f}% "
              f"{i['layers_used']:>7d} {i['core_after']:>8d}")
    print("\nWorking. achieved != requested is grid quantisation, not a bug.")
    print("layers=0 means the case was left untouched.")


def cmd_single(args):
    seg, ref = read_seg(args.input)
    print(f"file       : {args.input}")
    print(f"shape      : {seg.shape}")
    print(f"values     : {np.unique(seg)}")
    print(f"convention : {detect_convention(seg)}\n")

    out, info = perturb_core(seg, args.pct,
                             preserve_wt=not args.grow_wt,
                             output=args.output,
                             structure=make_structure(args.ball))
    for k, v in info.items():
        print(f"  {k:<20} {v}")
    if info["layers_used"] == 0 and args.pct != 0:
        print("\n  WARNING: 0 layers -- this case was NOT perturbed.")

    if args.out:
        write_seg(out, ref, args.out)
        print(f"\nwritten: {args.out}")
        print("Overlay both on FLAIR in ITK-SNAP before trusting it.")


def cmd_audit(args):
    files = sorted(Path(args.labels_dir).glob("*.nii.gz"))
    if args.limit:
        files = files[:args.limit]
    if not files:
        raise SystemExit(f"No .nii.gz in {args.labels_dir}")

    print(f"Auditing {len(files)} cases at {args.pct:+g}%\n")
    struct = make_structure(args.ball)
    rows = []
    for i, f in enumerate(files, 1):
        seg, _ = read_seg(f)
        _, info = perturb_core(seg, args.pct, preserve_wt=not args.grow_wt,
                               output=args.output, structure=struct)
        rows.append(info)
        if i % 50 == 0 or i == len(files):
            print(f"  {i}/{len(files)}")

    ach = np.array([r["volume_change_pct"] for r in rows])
    n0 = np.array([r["core_before"] for r in rows])
    lay = np.array([r["layers_used"] for r in rows])
    err = ach - args.pct
    noop = int((lay == 0).sum())
    floored = sum(r["core_floored"] for r in rows)

    print(f"\n{'=' * 54}")
    print(f"achieved      median {np.median(ach):+6.1f}%   IQR "
          f"[{np.percentile(ach,25):+.1f}, {np.percentile(ach,75):+.1f}]")
    print(f"error         median {np.median(err):+6.1f}pp  90th "
          f"{np.percentile(np.abs(err),90):5.1f}pp  max {np.abs(err).max():.1f}pp")
    print(f"NOT PERTURBED {noop}/{len(rows)} cases (0 layers)")
    print(f"core floored  {floored}/{len(rows)} cases")
    print(f"layers used   {dict(zip(*np.unique(lay, return_counts=True)))}")
    print(f"core size     median {np.median(n0):.0f} voxels, "
          f"min {n0.min():.0f}, max {n0.max():.0f}")

    small = n0 < np.percentile(n0, 25)
    if small.any():
        print(f"\nsmallest quartile: {int((lay[small]==0).sum())}/{int(small.sum())} "
              f"not perturbed, median error {np.median(np.abs(err[small])):.1f}pp")
        print(f"largest quartile : {int((lay[~small]==0).sum())}/{int((~small).sum())} "
              f"not perturbed, median error {np.median(np.abs(err[~small])):.1f}pp")

    if noop > 0.05 * len(rows):
        print(f"\nWARNING: {100*noop/len(rows):.0f}% of cases untouched. Use a larger")
        print(f"|pct| -- one voxel layer is the smallest change possible.")

    if args.out_csv:
        import pandas as pd
        df = pd.DataFrame(rows)
        df.insert(0, "case_id", [f.name.replace(".nii.gz", "") for f in files])
        df.to_csv(args.out_csv, index=False)
        print(f"\nwritten: {args.out_csv}")


def cmd_dataset(args):
    src_dir = Path(args.source_dir)
    out_dir = Path(args.out_dir)
    src_labels = src_dir / "labelsTr"
    out_labels = out_dir / "labelsTr"
    if not src_labels.exists():
        raise SystemExit(f"{src_labels} not found")
    out_labels.mkdir(parents=True, exist_ok=True)

    files = sorted(src_labels.glob("*.nii.gz"))
    if args.limit:
        files = files[:args.limit]

    site_map, site_cfg = {}, {}
    if args.site_config:
        if not args.csv:
            raise SystemExit("--site-config needs --csv")
        site_map = load_site_map(args.csv, args.site_col, args.subject_col)
        site_cfg = json.loads(Path(args.site_config).read_text())
        print(f"per-site mode, {len(site_cfg)} sites configured")
    else:
        print(f"uniform mode, {args.pct:+g}%")
    print(f"{len(files)} cases -> {out_labels}\n")

    struct = make_structure(args.ball)
    rows = []
    for i, f in enumerate(files, 1):
        case_id = f.name.replace(".nii.gz", "")
        if site_cfg:
            site = site_map.get(case_id)
            pct = float(site_cfg.get(str(site), {}).get("pct", 0.0))
        else:
            site, pct = None, args.pct

        seg, ref = read_seg(f)
        out, info = perturb_core(seg, pct, preserve_wt=not args.grow_wt,
                                 output=args.output, structure=struct)
        write_seg(out, ref, out_labels / f.name)
        info.update(case_id=case_id, site=site)
        rows.append(info)
        if i % 50 == 0 or i == len(files):
            print(f"  {i}/{len(files)}")

    n_linked = link_images(src_dir, out_dir)
    note = (f"per-site perturbation" if site_cfg else f"uniform {args.pct:+g}% perturbation")
    write_dataset_json(out_dir, len(files), args.output, note)

    import pandas as pd
    df = pd.DataFrame(rows)
    cols = ["case_id", "site", "pct_requested", "volume_change_pct", "layers_used",
            "core_before", "core_after", "core_floored"]
    df[[c for c in cols if c in df.columns]].to_csv(out_dir / "perturbation_log.csv",
                                                    index=False)

    noop = int((df["layers_used"] == 0).sum())
    print(f"\nmean achieved {df['volume_change_pct'].mean():+.1f}%")
    print(f"not perturbed {noop}/{len(df)} cases")
    print(f"symlinked     {n_linked} image files")
    print(f"dataset.json  labels = "
          f"{'0 bg / 1 core' if args.output=='binary' else '0 bg / 1 edema / 2 core'}")

    if "site" in df and df["site"].notna().any():
        print("\nper site:")
        g = df.groupby("site").agg(n=("case_id", "count"),
                                   requested=("pct_requested", "first"),
                                   achieved=("volume_change_pct", "mean"),
                                   untouched=("layers_used", lambda s: int((s == 0).sum())))
        print(g.to_string())

    print(f"\nlog: {out_dir/'perturbation_log.csv'}")
    print(f"Next: nnUNetv2_plan_and_preprocess -d <id> --verify_dataset_integrity")


# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    def shared(sp, with_pct=True):
        if with_pct:
            sp.add_argument("--pct", type=float, default=40.0,
                            help="signed percent. +40 over-segments, -40 under-segments")
        sp.add_argument("--output", choices=["binary", "merged"], default="binary")
        sp.add_argument("--grow-wt", action="store_true",
                        help="let the lesion grow instead of fencing the core inside it")
        sp.add_argument("--ball", type=int, default=None,
                        help="use a spherical element of this radius instead of "
                             "scipy's 6-connected default")
        sp.add_argument("--limit", type=int, default=None)

    d0 = sub.add_parser("demo"); shared(d0, with_pct=False)
    d0.set_defaults(pct=None)

    s = sub.add_parser("single"); s.add_argument("--input", required=True)
    s.add_argument("--out", default=None); shared(s)

    a = sub.add_parser("audit"); a.add_argument("--labels-dir", required=True)
    a.add_argument("--out-csv", default=None); shared(a)

    d = sub.add_parser("dataset")
    d.add_argument("--source-dir", required=True,
                   help="nnU-Net dataset folder containing labelsTr and imagesTr")
    d.add_argument("--out-dir", required=True)
    d.add_argument("--csv", default=None)
    d.add_argument("--site-config", default=None)
    d.add_argument("--site-col", default=None)
    d.add_argument("--subject-col", default=None)
    shared(d)

    args = p.parse_args()
    {"demo": cmd_demo, "single": cmd_single,
     "audit": cmd_audit, "dataset": cmd_dataset}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())