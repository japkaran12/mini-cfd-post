import numpy as np
from scipy.interpolate import griddata

def compute_cp(p, p_inf, q_inf):
    if q_inf == 0:
        return np.zeros_like(p)
    return (p - p_inf) / q_inf

def velocity_magnitude(u, v, w=None):
    if w is None:
        return np.sqrt(u**2 + v**2)
    return np.sqrt(u**2 + v**2 + w**2)

def mach_number(u, v, w=None, a=343.0):
    V = velocity_magnitude(u, v, w)
    return V / a

def vorticity_2d(x, y, u, v, nx=200, ny=200, method="cubic"):
    x = np.asarray(x)
    y = np.asarray(y)
    u = np.asarray(u)
    v = np.asarray(v)

    xi = np.linspace(x.min(), x.max(), nx)
    yi = np.linspace(y.min(), y.max(), ny)
    Xi, Yi = np.meshgrid(xi, yi)

    pts = np.column_stack([x, y])
    Ui = griddata(pts, u, (Xi, Yi), method=method)
    Vi = griddata(pts, v, (Xi, Yi), method=method)

    if np.isnan(Ui).any() or np.isnan(Vi).any():
        Ui = griddata(pts, u, (Xi, Yi), method="nearest")
        Vi = griddata(pts, v, (Xi, Yi), method="nearest")

    dx = xi[1] - xi[0] if nx > 1 else 1.0
    dy = yi[1] - yi[0] if ny > 1 else 1.0

    dVdx = np.gradient(Vi, dx, axis=1)
    dUdy = np.gradient(Ui, dy, axis=0)
    omega = dVdx - dUdy

    return Xi, Yi, omega, Ui, Vi
