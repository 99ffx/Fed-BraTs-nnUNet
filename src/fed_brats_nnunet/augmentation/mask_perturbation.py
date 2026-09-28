"""CLI for percentage-based tumour-core perturbation.

Four ways to run it:

  1. DEMO -- no data needed, checks the install works
       python run_perturbation.py demo

  2. ONE CASE -- eyeball a single file before touching the dataset
       python run_perturbation.py single --input case.nii.gz --pct 20 --out out.nii.gz

  3. AUDIT -- how achievable is a given percentage on your data? Writes nothing.
       python run_perturbation.py audit --labels-dir .../labelsTr --pct 20

  4. DATASET -- build a perturbed nnU-Net dataset
       python run_perturbation.py dataset --source-labels .../Dataset501_FeTS2022/labelsTr \\
           --out-labels .../Dataset601_FeTS2022_pct20/labelsTr --pct 20

     per-site instead of uniform:
       python run_perturbation.py dataset --source-labels ... --out-labels ... \\
           --csv partitioning_1.csv --site-config site_pct.json

Always run `audit` before `dataset`. On a 1mm grid, volume advances in whole
shells, so small cores overshoot -- audit tells you by how much before you spend
hours preprocessing something unusable.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from perturb_pct import detect_convention, perturb_tc_pct, tumor_core_mask


# ---------------------------------------------------------------------------
# io
# ---------------------------------------------------------------------------

def read_seg(path):
    """Returns (array, spacing_in_array_order, reference_image).

    SimpleITK reports spacing as (x,y,z) but GetArrayFromImage returns (z,y,x),
    so the spacing tuple is reversed to match the array axes. Getting this wrong
    silently gives anisotropic radii on anisotropic data.
    """
    import SimpleITK as sitk
    img = sitk.ReadImage(str(path))
    return sitk.GetArrayFromImage(img), tuple(reversed(img.GetSpacing())), img


def write_seg(arr, reference, path):
    import SimpleITK as sitk
    out = sitk.GetImageFromArray(arr.astype(np.uint8))
    out.CopyInformation(reference)      # preserves spacing/origin/direction
    sitk.WriteImage(out, str(path))


def load_site_map(csv_path, site_col=None, subj_col=None):
    import pandas as pd
    df = pd.read_csv(csv_path)
    site_col = site_col or next(
        (c for c in ["Partition_ID", "partition_id", "site", "Site", "site_id"]
         if c in df.columns), None)
    subj_col = subj_col or next(
        (c for c in ["Subject_ID", "subject_id", "subject", "case_id", "patient_id"]
         if c in df.columns), None)
    if not site_col or not subj_col:
        raise SystemExit(f"Could not find site/subject columns in {list(df.columns)}. "
                         f"Pass --site-col and --subject-col.")
    print(f"site column '{site_col}', subject column '{subj_col}'")
    return {str(r[subj_col]): int(r[site_col]) for _, r in df.iterrows()}


# ---------------------------------------------------------------------------
# 1. demo
# ---------------------------------------------------------------------------

def cmd_demo(args):
    def phantom(shape, core_r, edema_r):
        c = [s // 2 for s in shape]
        z, y, x = np.ogrid[:shape[0], :shape[1], :shape[2]]
        d = np.sqrt((z - c[0]) ** 2 + (y - c[1]) ** 2 + (x - c[2]) ** 2)
        s = np.zeros(shape, np.uint8)
        s[d <= edema_r] = 2      # edema
        s[d <= core_r] = 4       # enhancing tumour
        s[d <= core_r / 2] = 1   # necrotic core
        return s

    seg = phantom((90, 90, 90), 12, 26)
    spacing = (1.0, 1.0, 1.0)

    print(f"convention detected : {detect_convention(seg)}")
    print(f"core volume         : {int(tumor_core_mask(seg).sum())} voxels\n")
    print(f"{'requested':>10} {'achieved':>10} {'radius':>9} {'core voxels':>13}")
    print("-" * 46)
    for pct in [-40, -20, 0, 20, 40]:
        _, info = perturb_tc_pct(seg, float(pct), spacing)
        print(f"{pct:>+9d}% {info['achieved_pct']:>+9.1f}% "
              f"{info['radius_mm_used']:>8.2f}mm {info['tc_after_voxels']:>13d}")
    print("\nWorking. Note achieved != requested -- that is grid quantisation,")
    print("not a bug. Always log achieved_pct.")


# ---------------------------------------------------------------------------
# 2. single case
# ---------------------------------------------------------------------------

def cmd_single(args):
    seg, spacing, ref = read_seg(args.input)
    print(f"file      : {args.input}")
    print(f"shape     : {seg.shape}")
    print(f"spacing   : {spacing} mm (array order)")
    print(f"values    : {np.unique(seg)}")
    print(f"convention: {detect_convention(seg)}\n")

    out, info = perturb_tc_pct(
        seg, args.pct, spacing,
        pct_mode=args.pct_mode,
        reassemble_mode=args.reassemble_mode,
    )
    for k, v in info.items():
        print(f"  {k:<20} {v}")

    if args.out:
        write_seg(out, ref, args.out)
        print(f"\nwritten: {args.out}")
        print("Open both in ITK-SNAP overlaid on FLAIR before trusting it.")


# ---------------------------------------------------------------------------
# 3. audit
# ---------------------------------------------------------------------------

def cmd_audit(args):
    files = sorted(Path(args.labels_dir).glob("*.nii.gz"))
    if args.limit:
        files = files[:args.limit]
    if not files:
        raise SystemExit(f"No .nii.gz files in {args.labels_dir}")

    print(f"Auditing {len(files)} cases at {args.pct:+g}%\n")
    rows = []
    for i, f in enumerate(files, 1):
        seg, spacing, _ = read_seg(f)
        _, info = perturb_tc_pct(seg, args.pct, spacing,
                                 pct_mode=args.pct_mode,
                                 reassemble_mode=args.reassemble_mode)
        rows.append(info)
        if i % 50 == 0 or i == len(files):
            print(f"  {i}/{len(files)}")

    ach = np.array([r["achieved_pct"] for r in rows])
    err = ach - args.pct
    n0 = np.array([r["tc_before_voxels"] for r in rows])
    floored = sum(r["core_floored"] for r in rows)

    print(f"\n{'=' * 52}")
    print(f"achieved   median {np.median(ach):+6.1f}%   IQR "
          f"[{np.percentile(ach, 25):+.1f}, {np.percentile(ach, 75):+.1f}]")
    print(f"error      median {np.median(err):+6.1f}pp  90th pct "
          f"{np.percentile(np.abs(err), 90):5.1f}pp  max {np.abs(err).max():.1f}pp")
    print(f"off by >10pp : {(np.abs(err) > 10).sum()}/{len(rows)} cases")
    print(f"core floored : {floored}/{len(rows)} cases")
    print(f"core size    : median {np.median(n0):.0f} voxels, "
          f"min {n0.min():.0f}, max {n0.max():.0f}")

    small = n0 < np.percentile(n0, 25)
    if small.any():
        print(f"\nsmallest quartile error: median {np.median(np.abs(err[small])):.1f}pp")
        print(f"largest  quartile error: median {np.median(np.abs(err[~small])):.1f}pp")
        print("Small cores overshoot more -- one shell is a bigger fraction of them.")

    if np.percentile(np.abs(err), 90) > 10:
        print(f"\nWARNING: 90th-percentile error above 10pp. Consider a larger |pct|;")
        print(f"at small percentages you are measuring grid artefacts, not bias.")

    if args.out_csv:
        import pandas as pd
        df = pd.DataFrame(rows)
        df.insert(0, "case_id", [f.name.replace(".nii.gz", "") for f in files])
        df.to_csv(args.out_csv, index=False)
        print(f"\nwritten: {args.out_csv}")


# ---------------------------------------------------------------------------
# 4. dataset
# ---------------------------------------------------------------------------

def cmd_dataset(args):
    src = Path(args.source_labels)
    dst = Path(args.out_labels)
    dst.mkdir(parents=True, exist_ok=True)

    files = sorted(src.glob("*.nii.gz"))
    if args.limit:
        files = files[:args.limit]
    if not files:
        raise SystemExit(f"No .nii.gz files in {src}")

    site_map, site_cfg = {}, {}
    if args.site_config:
        if not args.csv:
            raise SystemExit("--site-config needs --csv too")
        site_map = load_site_map(args.csv, args.site_col, args.subject_col)
        site_cfg = json.loads(Path(args.site_config).read_text())
        print(f"per-site mode, {len(site_cfg)} sites configured")
    else:
        print(f"uniform mode, {args.pct:+g}%")

    print(f"{len(files)} cases -> {dst}\n")

    rows = []
    for i, f in enumerate(files, 1):
        case_id = f.name.replace(".nii.gz", "")
        if site_cfg:
            site = site_map.get(case_id)
            pct = float(site_cfg.get(str(site), {}).get("pct", 0.0))
        else:
            site, pct = None, args.pct

        seg, spacing, ref = read_seg(f)
        out, info = perturb_tc_pct(seg, pct, spacing,
                                   pct_mode=args.pct_mode,
                                   reassemble_mode=args.reassemble_mode)
        write_seg(out, ref, dst / f.name)

        info.update(case_id=case_id, site=site)
        rows.append(info)
        if i % 50 == 0 or i == len(files):
            print(f"  {i}/{len(files)}")

    import pandas as pd
    df = pd.DataFrame(rows)
    cols = ["case_id", "site", "pct_requested", "achieved_pct", "radius_mm_used",
            "tc_before_voxels", "tc_after_voxels", "core_floored", "wt_changed"]
    df = df[[c for c in cols if c in df.columns]]
    log = dst.parent / "perturbation_log.csv"
    df.to_csv(log, index=False)

    print(f"\nmean achieved {df['achieved_pct'].mean():+.1f}%, "
          f"{int(df['core_floored'].sum())} floored")
    if "site" in df and df["site"].notna().any():
        print("\nper site:")
        g = df.groupby("site").agg(n=("case_id", "count"),
                                   requested=("pct_requested", "first"),
                                   achieved=("achieved_pct", "mean"))
        print(g.to_string())
    print(f"\nlog: {log}")
    print("\nNext: copy dataset.json (labels 0/1/2, regions whole tumor (1,2) /")
    print("tumor core (2,)), symlink imagesTr, then plan_and_preprocess.")


# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    def shared(sp):
        sp.add_argument("--pct", type=float, default=20.0,
                        help="signed percent. +20 over-segments, -20 under-segments")
        sp.add_argument("--pct-mode", choices=["volume", "radius"], default="volume")
        sp.add_argument("--reassemble-mode",
                        choices=["preserve_wt", "grow_wt"], default="preserve_wt")
        sp.add_argument("--limit", type=int, default=None)

    sub.add_parser("demo", help="synthetic check, no data needed")

    s = sub.add_parser("single", help="one case")
    s.add_argument("--input", required=True)
    s.add_argument("--out", default=None)
    shared(s)

    a = sub.add_parser("audit", help="is this percentage achievable on your data?")
    a.add_argument("--labels-dir", required=True)
    a.add_argument("--out-csv", default=None)
    shared(a)

    d = sub.add_parser("dataset", help="build a perturbed dataset")
    d.add_argument("--source-labels", required=True)
    d.add_argument("--out-labels", required=True)
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