# mcp/io.py
from pathlib import Path
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any

def read_dataset(path_or_file) -> Dict[str, Any]:
    """
    Read .npz or .csv from disk (path str/Path) or from a file-like object (Streamlit upload).
    Returns {"coords": ndarray(N,2), "fields": { 'u':..., 'v':..., 'p':... } }
    """
    def _read_npz(f) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        data = np.load(f, allow_pickle=True)
        if "coords" not in data.files:
            raise ValueError("NPZ must contain 'coords' array.")
        coords = data["coords"]
        fields = {k: data[k] for k in data.files if k != "coords"}
        return coords, fields

    def _read_csv(f) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        df = pd.read_csv(f)
        cols_lower = [c.lower() for c in df.columns]
        if not {"x", "y"}.issubset(set(cols_lower)):
            raise ValueError("CSV must contain columns named 'x' and 'y'.")
        col_map = {c.lower(): c for c in df.columns}
        xcol = col_map["x"]; ycol = col_map["y"]
        coords = df[[xcol, ycol]].to_numpy()
        fields = {}
        for cname in df.columns:
            cl = cname.lower()
            if cl in ("u","v","p","rho","t","mach","ux","uy"):
                fields[cl] = df[cname].to_numpy()
        if "ux" in fields and "uy" in fields and ("u" not in fields or "v" not in fields):
            fields["u"] = fields.pop("ux")
            fields["v"] = fields.pop("uy")
        return coords, fields

    # If path string or Path -> read from disk
    if isinstance(path_or_file, (str, Path)):
        p = Path(path_or_file)
        if not p.exists():
            raise FileNotFoundError(f"File not found: {p}")
        if p.suffix.lower() == ".npz":
            coords, fields = _read_npz(str(p))
            return {"coords": coords, "fields": fields}
        elif p.suffix.lower() == ".csv":
            coords, fields = _read_csv(str(p))
            return {"coords": coords, "fields": fields}
        else:
            raise ValueError("Unsupported extension. Use .npz or .csv")

    # If file-like (e.g. Streamlit UploadedFile)
    if hasattr(path_or_file, "read"):
        # try npz then csv
        try:
            path_or_file.seek(0)
            coords, fields = _read_npz(path_or_file)
            return {"coords": coords, "fields": fields}
        except Exception:
            try:
                path_or_file.seek(0)
                coords, fields = _read_csv(path_or_file)
                return {"coords": coords, "fields": fields}
            except Exception as e:
                raise ValueError(f"Failed to read uploaded file as NPZ or CSV: {e}")

    raise ValueError("Unsupported input for read_dataset()")
