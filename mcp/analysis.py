import numpy as np
from scipy.spatial import cKDTree

def estimate_sectional_lift_from_cp(coords, cp, probe_y=0.0, tol=0.02):
    coords = np.asarray(coords)
    x = coords[:, 0]
    y = coords[:, 1]
    cp = np.asarray(cp)

    near = np.abs(y - probe_y) < max(tol, 1e-6)
    if near.sum() < 6:
        near = np.abs(y - probe_y) < (tol * 5)

    if near.sum() < 6:
        med = np.median(y)
        top_mask = y > med
        bot_mask = y < med
    else:
        top_mask = near & (y > probe_y)
        bot_mask = near & (y < probe_y)

    if top_mask.sum() < 3 or bot_mask.sum() < 3:
        return {"error": "Not enough points near probe to estimate lift."}

    xu, cpu = x[top_mask], cp[top_mask]
    xl, cpl = x[bot_mask], cp[bot_mask]

    x_min = max(xu.min(), xl.min())
    x_max = min(xu.max(), xl.max())
    if x_max <= x_min:
        return {"error": "No overlapping x-range between top and bottom data."}

    x_grid = np.linspace(x_min, x_max, 200)

    tu = cKDTree(xu.reshape(-1, 1))
    tl = cKDTree(xl.reshape(-1, 1))
    _, iu = tu.query(x_grid.reshape(-1, 1), k=1)
    _, il = tl.query(x_grid.reshape(-1, 1), k=1)

    cp_u = cpu[iu]
    cp_l = cpl[il]

    cp_diff = cp_l - cp_u
    lift_per_unit_span = np.trapz(cp_diff, x_grid)

    return {
        "x": x_grid,
        "cp_top": cp_u,
        "cp_bot": cp_l,
        "cp_diff": cp_diff,
        "lift_per_unit_span": float(lift_per_unit_span)
    }

def compare_cp_arrays(cp1, cp2):
    a = np.asarray(cp1).flatten()
    b = np.asarray(cp2).flatten()
    n = min(a.size, b.size)
    a = a[:n]
    b = b[:n]
    d = b - a
    return {
        "diff": d,
        "mean_diff": float(d.mean()),
        "max_diff": float(d.max()),
        "min_diff": float(d.min()),
        "rmse": float(np.sqrt((d**2).mean()))
    }
