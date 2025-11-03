import numpy as np
from scipy.spatial import cKDTree

def estimate_sectional_lift_from_cp(coords, cp, probe_y=0.0, tol=0.02):
    # coords: (N,2) array, cp: (N,) array
    coords = np.asarray(coords)
    x = coords[:, 0]
    y = coords[:, 1]
    cp = np.asarray(cp)

    # try selecting points near probe_y
    mask = np.abs(y - probe_y) < max(tol, 1e-6)
    if mask.sum() < 6:
        mask = np.abs(y - probe_y) < (tol * 5)
    if mask.sum() < 6:
        # fallback: split by median y
        median_y = np.median(y)
        upper_mask = y > median_y
        lower_mask = y < median_y
    else:
        upper_mask = mask & (y > probe_y)
        lower_mask = mask & (y < probe_y)

    if upper_mask.sum() < 3 or lower_mask.sum() < 3:
        return {"error": "Not enough data near probe line to estimate lift."}

    xu = x[upper_mask]; cpu = cp[upper_mask]
    xl = x[lower_mask]; cpl = cp[lower_mask]

    # build x-grid over overlapping x-range
    x_min = max(np.min(xu), np.min(xl))
    x_max = min(np.max(xu), np.max(xl))
    if x_max <= x_min:
        return {"error": "Upper and lower surfaces do not overlap in x; can't estimate."}

    x_grid = np.linspace(x_min, x_max, 200)

    # nearest-neighbour sampling via KDTree (1D)
    tu = cKDTree(xu.reshape(-1,1))
    tl = cKDTree(xl.reshape(-1,1))
    _, iu = tu.query(x_grid.reshape(-1,1), k=1)
    _, il = tl.query(x_grid.reshape(-1,1), k=1)
    cp_u = cpu[iu]
    cp_l = cpl[il]

    cp_diff = cp_l - cp_u
    lift_per_unit_span = np.trapz(cp_diff, x_grid)

    return {
        "x_grid": x_grid,
        "cp_upper": cp_u,
        "cp_lower": cp_l,
        "cp_diff": cp_diff,
        "lift_per_unit_span": float(lift_per_unit_span)
    }

def compare_cp_arrays(cp1, cp2):
    cp1 = np.asarray(cp1).flatten()
    cp2 = np.asarray(cp2).flatten()
    n = min(cp1.size, cp2.size)
    cp1 = cp1[:n]
    cp2 = cp2[:n]
    diff = cp2 - cp1
    return {
        "diff": diff,
        "mean_diff": float(np.mean(diff)),
        "max_diff": float(np.max(diff)),
        "min_diff": float(np.min(diff)),
        "rmse": float(np.sqrt(np.mean(diff**2)))
    }
