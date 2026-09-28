"""Simple binary morphology on the tumour core, using scipy.ndimage directly.

    +n  dilate n iterations   (over-segment)
    -n  erode n iterations    (under-segment)
     0  unchanged

Labels in  : raw BraTS (0,1,2,4), nnU-Net (0,1,2,3) or merged (0,1,2)
Labels out : chosen with `output`
    "binary" (default)  0 background (edema included), 1 tumour core
    "merged"            0 background, 1 edema, 2 tumour core

NOTE ON BINARY OUTPUT
---------------------
Binary output changes the nnU-Net task from three nested regions to one region.
That means:
  * dataset.json becomes  {"background": 0, "tumor core": 1}
    with NO regions_class_order
  * your existing 3-region baseline is no longer comparable -- you need a
    binary centralized baseline to compare the perturbation runs against
Edema is still read from the input and still used as the fence for clipping
(preserve_wt); it is just not written to the output.
"""

import numpy as np
from scipy import ndimage

# which values mean "tumour core" in each convention
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


def perturb_core(seg, iterations, preserve_wt=True, min_core_voxels=10,
                 output="binary", structure=None):
    """Dilate (+) or erode (-) the tumour core by `iterations` voxels.

    preserve_wt=True   core cannot grow past the original lesion boundary
    preserve_wt=False  core may grow into background
    output             "binary" -> 0/1   |   "merged" -> 0/1/2
    structure          optional structuring element, e.g. skimage ball(2).
                       Pass it WITHOUT iterations>1 or you compound the radius.

    Returns (new_seg, info).
    """
    tc = np.isin(seg, _TC[detect_convention(seg)])
    wt = seg > 0                      # lesion, read from the INPUT
    n0 = int(tc.sum())

    if iterations > 0:
        tc_new = ndimage.binary_dilation(tc, structure=structure,
                                         iterations=iterations)
    elif iterations < 0:
        tc_new = ndimage.binary_erosion(tc, structure=structure,
                                        iterations=-iterations)
    else:
        tc_new = tc.copy()

    # the core may not escape the lesion it came from
    if preserve_wt:
        tc_new = tc_new & wt
        wt_new = wt
    else:
        wt_new = wt | tc_new

    # never hand nnU-Net an empty core
    floored = False
    if n0 > 0 and int(tc_new.sum()) < min(min_core_voxels, n0):
        tc_new = (tc & wt) if preserve_wt else tc
        floored = True

    out = np.zeros_like(seg, dtype=np.uint8)
    if output == "binary":
        out[tc_new] = 1               # edema falls into background
    elif output == "merged":
        out[wt_new] = 1               # lesion -> edema
        out[tc_new] = 2               # core overwrites
    else:
        raise ValueError(f"output must be 'binary' or 'merged', got {output!r}")

    core_value = 1 if output == "binary" else 2
    n1 = int((out == core_value).sum())
    return out, {
        "iterations": iterations,
        "output": output,
        "core_before": n0,
        "core_after": n1,
        "volume_change_pct": 100.0 * (n1 - n0) / n0 if n0 else 0.0,
        "core_floored": floored,
    }


def perturb_core_pct(seg, pct, max_iter=30, **kw):
    """Same thing, but keep adding iterations until the core volume has changed
    by about `pct` percent. Stops at the first iteration that reaches the target.
    """
    if pct == 0:
        return perturb_core(seg, 0, **kw)

    step = 1 if pct > 0 else -1
    best = perturb_core(seg, 0, **kw)
    for i in range(1, max_iter + 1):
        out, info = perturb_core(seg, i * step, **kw)
        best = (out, info)
        if abs(info["volume_change_pct"]) >= abs(pct) or info["core_floored"]:
            break
    return best