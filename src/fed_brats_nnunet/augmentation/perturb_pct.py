"""Percentage-based tumour-core perturbation.

WHY PERCENT INSTEAD OF MILLIMETRES
----------------------------------
A fixed mm radius is a surface-to-volume effect: +2mm might grow a small core by
80% and a large one by 15%. Tumour size varies systematically across BraTS sites,
so a uniform mm radius silently hands smaller-tumour sites a harder perturbation,
and "site has annotation bias" becomes inseparable from "site has small tumours".

Specifying the perturbation as a percentage makes the insult comparable across
cases and across sites.

TWO MEANINGS OF PERCENT -- pick deliberately
--------------------------------------------
mode="volume"   (default)  pct is the target CHANGE IN CORE VOLUME.
                           +20 means the core ends up ~20% larger.
                           This is the one that removes the size confound and
                           the one you can state plainly in a paper.

mode="radius"              pct scales each tumour's own equivalent-sphere radius.
                           +20 means dilate by 0.20 x r_eq. Cheaper conceptually,
                           but the achieved volume change still varies with shape
                           (a spiky core gains more than a compact one).

HOW THE RADIUS IS FOUND WITHOUT SEARCHING
-----------------------------------------
Dilation by r is exactly {distance_to_mask <= r}, and that set grows monotonically
with r. So one Euclidean distance transform gives every candidate radius at once:
sort the distances of the outside voxels, and the radius that adds N voxels is the
N-th smallest. Same trick inverted for erosion. One EDT, no iteration.

A CAVEAT WORTH KNOWING
----------------------
On a 1mm isotropic grid the distance values are heavily quantised (1.0, 1.414,
1.732, ...), so whole shells of voxels share a distance and volume advances in
jumps. Small cores therefore cannot hit an arbitrary target exactly. The functions
here return a true morphological result and REPORT the achieved percentage, which
is always the number you should log and plot -- never the requested one.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import distance_transform_edt

_TC_VALUES = {
    "raw_brats": {1, 4},
    "nnunet": {2, 3},
    "merged": {2},
    "binary": {1},
}


def detect_convention(seg: np.ndarray) -> str:
    vals = {int(v) for v in np.unique(seg)}
    if 4 in vals:
        return "raw_brats"
    if 3 in vals:
        return "nnunet"
    if vals <= {0, 1, 2} and 2 in vals:
        return "merged"
    if vals <= {0, 1}:
        return "binary"
    raise ValueError(f"Unrecognised label convention, values {sorted(vals)}")


def tumor_core_mask(seg: np.ndarray, convention: str | None = None) -> np.ndarray:
    conv = convention or detect_convention(seg)
    return np.isin(seg, sorted(_TC_VALUES[conv]))


def whole_tumor_mask(seg: np.ndarray) -> np.ndarray:
    return seg > 0


# ---------------------------------------------------------------------------
# percent -> radius
# ---------------------------------------------------------------------------

def equivalent_sphere_radius_mm(mask: np.ndarray, spacing: tuple[float, ...]) -> float:
    """Radius of a sphere with the same volume as `mask`. A size scalar that,
    unlike a bounding box, does not blow up for elongated or multifocal cores."""
    vol = float(mask.sum()) * float(np.prod(spacing))
    if vol <= 0:
        return 0.0
    return (3.0 * vol / (4.0 * np.pi)) ** (1.0 / 3.0)


def morph_by_volume_pct(
    mask: np.ndarray,
    pct: float,
    spacing: tuple[float, ...],
    bound: np.ndarray | None = None,
) -> tuple[np.ndarray, float]:
    """Grow or shrink `mask` to change its volume by roughly `pct` percent.

    `bound` optionally constrains dilation (e.g. the whole-tumour mask), so the
    core can only take territory from inside the lesion. The radius search
    accounts for the bound, so the achieved percentage stays honest instead of
    silently falling short after a later clip.

    Returns (new_mask, radius_mm_used).
    """
    n0 = int(mask.sum())
    if pct == 0 or n0 == 0:
        return mask.copy(), 0.0

    if pct > 0:
        dist = distance_transform_edt(~mask, sampling=spacing)
        outside = ~mask if bound is None else (~mask & bound)
        cand = dist[outside]
        if cand.size == 0:
            return mask.copy(), 0.0

        n_add = int(round(n0 * pct / 100.0))
        n_add = min(n_add, cand.size)
        if n_add <= 0:
            return mask.copy(), 0.0

        # radius that admits exactly the n_add nearest outside voxels
        radius = float(np.partition(cand, n_add - 1)[n_add - 1])
        grown = dist <= radius
        if bound is not None:
            grown = grown & bound
        return (grown | mask), radius

    # erosion
    dist = distance_transform_edt(mask, sampling=spacing)
    cand = dist[mask]
    n_remove = int(round(n0 * abs(pct) / 100.0))
    n_remove = min(n_remove, cand.size - 1) if cand.size > 1 else 0
    if n_remove <= 0:
        return mask.copy(), 0.0

    # the removed voxels are the shallowest ones
    radius = float(np.partition(cand, n_remove - 1)[n_remove - 1])
    return (dist > radius), radius


def morph_by_radius_pct(
    mask: np.ndarray,
    pct: float,
    spacing: tuple[float, ...],
) -> tuple[np.ndarray, float]:
    """Dilate/erode by pct percent of the mask's own equivalent-sphere radius."""
    if pct == 0 or not mask.any():
        return mask.copy(), 0.0

    r_eq = equivalent_sphere_radius_mm(mask, spacing)
    radius = abs(pct) / 100.0 * r_eq
    if radius <= 0:
        return mask.copy(), 0.0

    if pct > 0:
        return (distance_transform_edt(~mask, sampling=spacing) <= radius), radius
    return (distance_transform_edt(mask, sampling=spacing) > radius), radius


# ---------------------------------------------------------------------------
# reassembly
# ---------------------------------------------------------------------------

def reassemble(tc_new: np.ndarray, seg_original: np.ndarray,
               mode: str = "preserve_wt") -> np.ndarray:
    """Merge a perturbed core back into a 0/1/2 segmentation.

    preserve_wt : new core voxels come out of edema, lesion volume unchanged
    grow_wt     : core may push past the lesion, which grows to contain it
    """
    wt = whole_tumor_mask(seg_original)

    if mode == "preserve_wt":
        tc_final, wt_final = tc_new & wt, wt
    elif mode == "grow_wt":
        tc_final, wt_final = tc_new, wt | tc_new
    else:
        raise ValueError(f"mode must be 'preserve_wt' or 'grow_wt', got {mode!r}")

    out = np.zeros_like(seg_original, dtype=np.uint8)
    out[wt_final] = 1
    out[tc_final] = 2
    return out


# ---------------------------------------------------------------------------
# main entry point
# ---------------------------------------------------------------------------

def perturb_tc_pct(
    seg: np.ndarray,
    pct: float,
    spacing: tuple[float, ...],
    pct_mode: str = "volume",
    reassemble_mode: str = "preserve_wt",
    protect_empty: bool = True,
    min_core_voxels: int = 10,
) -> tuple[np.ndarray, dict]:
    """Perturb the tumour core by a percentage.

    Parameters
    ----------
    pct              signed. +20 over-segments, -20 under-segments.
    pct_mode         "volume" (target volume change) or "radius" (scale r_eq).
    reassemble_mode  "preserve_wt" or "grow_wt".
    min_core_voxels  erosion will not take the core below this. A handful of
                     voxels is not a tumour core and nnU-Net trains badly on
                     near-empty labels.

    Returns (perturbed_seg, info). ALWAYS log info["achieved_pct"], not `pct`:
    quantised distances mean small cores cannot hit every target exactly.
    """
    conv = detect_convention(seg)
    tc = tumor_core_mask(seg, conv)
    wt = whole_tumor_mask(seg)
    n0 = int(tc.sum())

    bound = wt if reassemble_mode == "preserve_wt" else None

    if pct_mode == "volume":
        tc_new, radius = morph_by_volume_pct(tc, pct, spacing, bound=bound)
    elif pct_mode == "radius":
        tc_new, radius = morph_by_radius_pct(tc, pct, spacing)
    else:
        raise ValueError(f"pct_mode must be 'volume' or 'radius', got {pct_mode!r}")

    floored = False
    if protect_empty and n0 > 0 and int(tc_new.sum()) < min(min_core_voxels, n0):
        tc_new, radius, floored = tc, 0.0, True

    out = reassemble(tc_new, seg, mode=reassemble_mode)

    n1 = int((out == 2).sum())
    if protect_empty and n0 > 0 and n1 < min(min_core_voxels, n0):
        out = reassemble(tc, seg, mode=reassemble_mode)
        n1, radius, floored = n0, 0.0, True

    voxel_vol = float(np.prod(spacing))
    return out, {
        "convention": conv,
        "pct_requested": pct,
        "achieved_pct": 100.0 * (n1 - n0) / n0 if n0 else 0.0,
        "radius_mm_used": radius,
        "r_eq_mm": equivalent_sphere_radius_mm(tc, spacing),
        "tc_before_voxels": n0,
        "tc_after_voxels": n1,
        "tc_before_mm3": n0 * voxel_vol,
        "tc_after_mm3": n1 * voxel_vol,
        "wt_changed": bool(int(wt.sum()) != int((out > 0).sum())),
        "core_floored": floored,
    }