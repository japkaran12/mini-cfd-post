# streamlit_app.py
import streamlit as st
import numpy as np
import pandas as pd
from io import BytesIO

from mcp.io import read_dataset
from mcp.fields import vorticity_2d, compute_cp, velocity_magnitude, mach_number
from mcp.plotting import plot_contour_grid, plot_quiver, plot_streamlines, plot_scatter_coords
from mcp.analysis import estimate_sectional_lift_from_cp, compare_cp_arrays

st.set_page_config(page_title="Mini CFD Post-Processor  ", layout="wide")
st.title("Mini CFD Post-Processor (vorticity, Cp, velocity, Mach, streamlines, compare)")

# Sidebar - inputs
with st.sidebar:
    st.header("Settings")
    mode = st.selectbox("Data source", ["Synthetic", "Upload file (single)", "Upload 2 files (compare)"])
    n_points = st.slider("Synthetic points", 200, 5000, 1500, step=100)
    grid_n = st.slider("Grid resolution (nx=ny)", 50, 300, 200, step=50)
    interp_method = st.selectbox("Interpolation method", ["cubic", "linear", "nearest"])
    show_streamlines = st.checkbox("Show streamlines", value=True)
    show_quiver = st.checkbox("Show quiver", value=False)
    quantity = st.selectbox("Plot quantity", ["Vorticity", "Cp", "Velocity magnitude", "Mach number"])
    p_inf = st.number_input("p_infty (Pa)", value=101325.0)
    q_inf = st.number_input("q_infty (Pa)", value=500.0)
    probe_y = st.number_input("Cp probe y (for Cp vs x)", value=0.0)
    st.markdown("---")
    st.caption("Upload .npz (coords keys: 'coords', u,v,p) or CSV with x,y,u,v,p columns.")

uploaded_1 = None
uploaded_2 = None
if mode == "Upload file (single)":
    uploaded_1 = st.file_uploader("Upload file (.npz or .csv)", type=["npz", "csv"])
elif mode == "Upload 2 files (compare)":
    uploaded_1 = st.file_uploader("Upload file 1 (.npz or .csv)", type=["npz", "csv"], key="f1")
    uploaded_2 = st.file_uploader("Upload file 2 (.npz or .csv)", type=["npz", "csv"], key="f2")

def load_data_from_streamlit_file(f):
    # f is UploadedFile
    if f is None:
        return None, None
    name = f.name.lower()
    try:
        if name.endswith(".npz"):
            data = np.load(f, allow_pickle=True)
            coords = data["coords"]
            fields = {k: data[k] for k in data.files if k != "coords"}
            return coords, fields
        else:
            df = pd.read_csv(f)
            if not {"x","y"}.issubset(set([c.lower() for c in df.columns])):
                st.error("CSV must have x and y columns")
                return None, None
            # find columns
            cols = {c.lower(): c for c in df.columns}
            coords = df[[cols["x"], cols["y"]]].to_numpy()
            fields = {}
            for k in ("u","v","p","rho","t","mach","ux","uy"):
                if k in cols:
                    fields[k] = df[cols[k]].to_numpy()
            if "ux" in fields and "uy" in fields and ("u" not in fields or "v" not in fields):
                fields["u"] = fields.pop("ux")
                fields["v"] = fields.pop("uy")
            return coords, fields
    except Exception as e:
        st.error(f"Failed to read file: {e}")
        return None, None

def make_synthetic(n):
    theta = np.random.rand(n) * 2 * np.pi
    r = 0.05 + np.sqrt(np.random.rand(n))
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    omega0 = 2.0
    u = -0.5 * omega0 * y + 0.02 * (np.random.rand(n) - 0.5)
    v =  0.5 * omega0 * x + 0.02 * (np.random.rand(n) - 0.5)
    p = p_inf - (omega0 * 5.0) / (1.0 + r) * 100.0
    coords = np.column_stack([x, y])
    fields = {"u": u, "v": v, "p": p}
    return coords, fields

# load/generate based on mode
if mode == "Synthetic":
    coords, fields = make_synthetic(n_points)
elif mode == "Upload file (single)":
    coords, fields = load_data_from_streamlit_file(uploaded_1)
    if coords is None:
        st.stop()
else:  # compare mode
    coords1, fields1 = load_data_from_streamlit_file(uploaded_1)
    coords2, fields2 = load_data_from_streamlit_file(uploaded_2)
    if coords1 is None or coords2 is None:
        st.info("Upload both files for comparison")
        st.stop()

def process_case(coords, fields):
    if coords is None or fields is None:
        return None
    if "u" not in fields or "v" not in fields:
        st.error("Velocity fields 'u' and 'v' required")
        return None
    # compute grids and fields
    Xi, Yi, omega, Ui, Vi = vorticity_2d(coords[:,0], coords[:,1], fields["u"], fields["v"],
                                        nx=grid_n, ny=grid_n, method=interp_method)
    vel_mag = velocity_magnitude(fields["u"], fields["v"])
    mach = mach_number(fields["u"], fields["v"], a=343.0)
    cp_points = None
    if "p" in fields:
        cp_points = compute_cp(fields["p"], p_inf, q_inf)
    else:
        cp_points = np.zeros(coords.shape[0])
    return {
        "coords": coords, "fields": fields,
        "Xi": Xi, "Yi": Yi, "omega": omega, "Ui": Ui, "Vi": Vi,
        "vel_mag": vel_mag, "mach": mach, "cp_points": cp_points
    }

if mode != "Upload 2 files (compare)":
    case = process_case(coords, fields)
    if case is None:
        st.stop()

# UI layout
left, right = st.columns([1,2])

with left:
    st.subheader("Controls")
    st.write("Quantity:", quantity)
    st.write("Streamlines:", show_streamlines)
    if st.button("Download processed CSV"):
        # prepare csv from last case
        if mode == "Synthetic" or mode == "Upload file (single)":
            dfout = pd.DataFrame({
                "x": case["coords"][:,0],
                "y": case["coords"][:,1],
                "u": case["fields"]["u"],
                "v": case["fields"]["v"],
                "Cp": case["cp_points"]
            })
            st.download_button("Download CSV", dfout.to_csv(index=False).encode(), "processed.csv", "text/csv")
        else:
            st.info("Not available in compare mode")

with right:
    st.subheader("Results")
    if mode != "Upload 2 files (compare)":
        # pick quantity to show
        if quantity == "Vorticity":
            fig, ax = plot_contour_grid(case["Xi"], case["Yi"], case["omega"], title="Vorticity (ω_z)")
            st.pyplot(fig)
        elif quantity == "Cp":
            fig, ax = plot_scatter_coords(case["coords"], case["cp_points"], title="Cp (scatter)")
            st.pyplot(fig)
            # cp line at probe
            res = estimate_sectional_lift_from_cp(case["coords"], case["cp_points"], probe_y=probe_y, tol=0.02)
            if "error" not in res:
                st.markdown(f"**Estimated lift per unit span (non-dim):** {res['lift_per_unit_span']:.4f}")
                # plot cp line if available
                import matplotlib.pyplot as plt
                fig2, ax2 = plt.subplots(figsize=(7,2.5))
                ax2.plot(res["x_grid"], res["cp_diff"], "-o", markersize=3)
                ax2.set_title("Cp_lower - Cp_upper vs x (used for lift estimate)")
                ax2.set_xlabel("x"); ax2.set_ylabel("Cp diff")
                st.pyplot(fig2)
            else:
                st.info(res["error"])
        elif quantity == "Velocity magnitude":
            Xi, Yi = case["Xi"], case["Yi"]
            # interpolate vel magnitude to grid
            Vm_grid = case["vel_mag"]
            # Vm_grid is pointwise on coords; we will regrid:
            from scipy.interpolate import griddata
            pts = case["coords"][:, :2]
            Vm = case["vel_mag"]
            Vm_grid = griddata(pts, Vm, (Xi, Yi), method=interp_method)
            fig, ax = plot_contour_grid(Xi, Yi, Vm_grid, title="Velocity magnitude")
            st.pyplot(fig)
        elif quantity == "Mach number":
            pts = case["coords"][:, :2]
            M = case["mach"]
            Xi, Yi = case["Xi"], case["Yi"]
            from scipy.interpolate import griddata
            M_grid = griddata(pts, M, (Xi, Yi), method=interp_method)
            fig, ax = plot_contour_grid(Xi, Yi, M_grid, title="Mach number")
            st.pyplot(fig)

        # optional streamlines/quiver
        if show_quiver:
            figq, axq = plot_quiver(case["Xi"], case["Yi"], case["Ui"], case["Vi"])
            st.pyplot(figq)
        if show_streamlines:
            figs, axs = plot_streamlines(case["Xi"], case["Yi"], case["Ui"], case["Vi"])
            st.pyplot(figs)

    else:
        # compare mode
        st.markdown("### Comparison: Cp difference and stats")
        if coords1 is not None and coords2 is not None:
            c1 = process_case(coords1, fields1)
            c2 = process_case(coords2, fields2)
            if c1 is None or c2 is None:
                st.error("Processing failed for one of the files")
            else:
                # compute cp arrays (align by nearest points)
                cp1 = c1["cp_points"]; cp2 = c2["cp_points"]
                stats = compare_cp_arrays(cp1, cp2)
                st.write("Mean diff:", stats["mean_diff"])
                st.write("RMSE:", stats["rmse"])
                st.write("Max diff:", stats["max_diff"])
                # show scatter of difference on coords of file1 (if shapes equal otherwise show hist)
                try:
                    diff = (cp2 - cp1)
                    figd, axd = plot_scatter_coords(c1["coords"], diff, title="Cp difference (file2 - file1)")
                    st.pyplot(figd)
                except Exception:
                    st.write("Cannot plot pointwise difference (arrays different length). Showing histogram instead.")
                    import matplotlib.pyplot as plt
                    fig, ax = plt.subplots()
                    ax.hist(stats["diff"].flatten(), bins=40)
                    st.pyplot(fig)
        else:
            st.info("Upload two files to compare")

st.success("Finished processing. Change settings or upload new files.")
