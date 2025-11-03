import argparse
import sys
import os
import numpy as np
import matplotlib.pyplot as plt

from .io import read_dataset
from .fields import vorticity_2d, velocity_magnitude, compute_cp


def cmd_info(args):
    print("Mini CFD Post-Processor - CLI")
    print("Use this tool to process small CFD datasets (npz or csv).")
    print("Example: python -m mcp.cli plot-vort data.npz --out out.png")


def plot_vorticity(path, out=None, nx=200, ny=200, method="cubic"):
    data = read_dataset(path)
    coords = data["coords"]
    fields = data["fields"]

    if "u" not in fields or "v" not in fields:
        print("Error: dataset must contain 'u' and 'v' velocity components.")
        return 1

    x = coords[:, 0]
    y = coords[:, 1]
    u = fields["u"]
    v = fields["v"]

    Xi, Yi, omega, Ui, Vi = vorticity_2d(x, y, u, v, nx=nx, ny=ny, method=method)

    fig, ax = plt.subplots(figsize=(6, 6))
    cf = ax.contourf(Xi, Yi, omega, levels=60, cmap="RdBu_r")
    fig.colorbar(cf, ax=ax, label="Vorticity")
    ax.set_title("Vorticity")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal")
    fig.tight_layout()

    if out:
        fig.savefig(out, dpi=200)
        print(f"Saved vorticity plot -> {out}")
    else:
        plt.show()
    plt.close(fig)
    return 0


def plot_velocity_magnitude(path, out=None, nx=200, ny=200, method="cubic"):
    data = read_dataset(path)
    coords = data["coords"]
    fields = data["fields"]

    if "u" not in fields or "v" not in fields:
        print("Error: dataset must contain 'u' and 'v' velocity components.")
        return 1

    x = coords[:, 0]
    y = coords[:, 1]
    u = fields["u"]
    v = fields["v"]
    Vm = velocity_magnitude(u, v)

    Xi, Yi, _, Ui, Vi = vorticity_2d(x, y, u, v, nx=nx, ny=ny, method=method)

    pts = np.column_stack([x, y])
    from scipy.interpolate import griddata
    Vm_grid = griddata(pts, Vm, (Xi, Yi), method=method)
    if np.isnan(Vm_grid).any():
        Vm_grid = griddata(pts, Vm, (Xi, Yi), method="nearest")

    fig, ax = plt.subplots(figsize=(6, 6))
    cf = ax.contourf(Xi, Yi, Vm_grid, levels=60, cmap="viridis")
    fig.colorbar(cf, ax=ax, label="Velocity magnitude")
    ax.set_title("Velocity magnitude")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal")
    fig.tight_layout()

    if out:
        fig.savefig(out, dpi=200)
        print(f"Saved velocity magnitude plot -> {out}")
    else:
        plt.show()
    plt.close(fig)
    return 0


def compute_and_save_cp(path, out_csv, p_inf=101325.0, q_inf=500.0):
    data = read_dataset(path)
    coords = data["coords"]
    fields = data["fields"]

    if "p" not in fields:
        print("Error: dataset must contain pressure field 'p' to compute Cp.")
        return 1

    p = fields["p"]
    cp = compute_cp(p, p_inf, q_inf)

    arr = np.column_stack([coords, p, cp])
    header = "x,y,p,cp"
    np.savetxt(out_csv, arr, delimiter=",", header=header, comments="")
    print(f"Saved processed CSV -> {out_csv} (columns: {header})")
    return 0


def build_parser():
    p = argparse.ArgumentParser(prog="mcp-cli", description="Mini CFD Post-Processor CLI")
    sub = p.add_subparsers(dest="cmd")

    sub.add_parser("info", help="Show basic info about this tool")

    pv = sub.add_parser("plot-vort", help="Plot vorticity from dataset")
    pv.add_argument("path", help="Path to .npz or .csv dataset")
    pv.add_argument("--out", help="Output image file (png)", default=None)
    pv.add_argument("--nx", type=int, default=200)
    pv.add_argument("--ny", type=int, default=200)
    pv.add_argument("--method", default="cubic", choices=["nearest", "linear", "cubic"])

    pvm = sub.add_parser("plot-vel", help="Plot velocity magnitude")
    pvm.add_argument("path", help="Path to .npz or .csv dataset")
    pvm.add_argument("--out", help="Output image file (png)", default=None)
    pvm.add_argument("--nx", type=int, default=200)
    pvm.add_argument("--ny", type=int, default=200)
    pvm.add_argument("--method", default="cubic", choices=["nearest", "linear", "cubic"])

    cp = sub.add_parser("save-cp", help="Compute Cp and save CSV")
    cp.add_argument("path", help="Path to .npz or .csv dataset")
    cp.add_argument("--out", help="Output CSV filename", required=True)
    cp.add_argument("--pinf", type=float, default=101325.0)
    cp.add_argument("--qinf", type=float, default=500.0)

    return p


def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.cmd == "info" or args.cmd is None:
        return cmd_info(args)

    if args.cmd == "plot-vort":
        return plot_vorticity(args.path, out=args.out, nx=args.nx, ny=args.ny, method=args.method)

    if args.cmd == "plot-vel":
        return plot_velocity_magnitude(args.path, out=args.out, nx=args.nx, ny=args.ny, method=args.method)

    if args.cmd == "save-cp":
        return compute_and_save_cp(args.path, out_csv=args.out, p_inf=args.pinf, q_inf=args.qinf)

    print("Unknown command. Use -h for help.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
