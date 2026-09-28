"""Perturb the BraTS tumour core by a percentage, using scipy.ndimage morphology.

    perturb_core(seg, +30)   core ends up ~30% bigger   (over-segmentation)
    perturb_core(seg, -30)   core ends up ~30% smaller  (under-segmentation)
    perturb_core(seg,   0)   unchanged

Labels in  : raw BraTS (0,1,2,4), nnU-Net (0,1,2,3) or merged (0,1,2)
Labels out : "binary" (default) -> 0 background, 1 core
             "merged"           -> 0 background, 1 edema, 2 core

HOW PERCENT WORKS
-----------------
Morphology grows or shrinks one voxel layer at a time, so volume moves in jumps.
This applies layers one at a time and keeps whichever count lands CLOSEST to the
requested percentage. On small cores the closest may still be far off -- always
read info["volume_change_pct"], never assume you got what you asked for.
"""

import numpy as np
from scipy import ndimage

_TC = {"raw_brats": [1, 4], "nnunet": [2, 3], "merged": [2], "binary": [1]}


def detect_convention(seg):
    v = {int(x) for x in np.unique(seg)}
    if 4 in v:
        return "raw_brats"
    if 3 in v:
        return "nnunet"
    if 2 in v:
        return "merged"
    return "binary"


def perturb_core(seg, pct, preserve_wt=True, min_core_voxels=10,
                 output="binary", structure=None, max_layers=40):
    """Change the tumour core volume by roughly `pct` percent.

    pct > 0 dilates, pct < 0 erodes.
    preserve_wt=True keeps the core inside the original lesion boundary.
    """
    tc = np.isin(seg, _TC[detect_convention(seg)])
    wt = seg > 0                       # lesion, read from the INPUT
    n0 = int(tc.sum())

    # --- apply layers one at a time, keep the closest to target ---
    tc_new = tc.copy()
    layers = 0
    if pct != 0 and n0 > 0:
        target = n0 * (1.0 + pct / 100.0)
        best, best_err, best_n = tc.copy(), abs(n0 - target), 0
        cur = tc
        for n in range(1, max_layers + 1):
            if pct > 0:
                cur = ndimage.binary_dilation(cur, structure=structure)
                if preserve_wt:
                    cur = cur & wt
            else:
                cur = ndimage.binary_erosion(cur, structure=structure)
            err = abs(int(cur.sum()) - target)
            if err < best_err:
                best, best_err, best_n = cur.copy(), err, n
            else:
                break                  # volume is monotone, so we passed the optimum
            if not cur.any():
                break
        tc_new, layers = best, best_n

    if preserve_wt:
        tc_new = tc_new & wt
        wt_new = wt
    else:
        wt_new = wt | tc_new

    # never hand nnU-Net an empty core
    floored = False
    if n0 > 0 and int(tc_new.sum()) < min(min_core_voxels, n0):
        tc_new = (tc & wt) if preserve_wt else tc
        layers, floored = 0, True

    out = np.zeros_like(seg, dtype=np.uint8)
    if output == "binary":
        out[tc_new] = 1                # edema falls into background
    elif output == "merged":
        out[wt_new] = 1                # lesion -> edema
        out[tc_new] = 2                # core overwrites
    else:
        raise ValueError(f"output must be 'binary' or 'merged', got {output!r}")

    n1 = int((out == (1 if output == "binary" else 2)).sum())
    return out, {
        "pct_requested": pct,
        "volume_change_pct": 100.0 * (n1 - n0) / n0 if n0 else 0.0,
        "layers_used": layers,
        "core_before": n0,
        "core_after": n1,
        "core_floored": floored,
        "output": output,
    }