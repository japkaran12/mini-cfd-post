# diag.py
import importlib, inspect, sys

importlib.invalidate_caches()

try:
    import mcp
    print("mcp package file:", getattr(mcp, "__file__", "package dir"))
    import mcp.fields as fields
    print("mcp.fields file:", fields.__file__)
    print("vorticity_2d exists?:", hasattr(fields, "vorticity_2d"))
    print("list of names in mcp.fields (filtered):")
    names = [n for n in dir(fields) if not n.startswith("_")]
    print(", ".join(n for n in names if "vort" in n.lower() or "cp" in n.lower() or "velocity" in n.lower()) )
    print("\n--- file snippet (first 300 chars) ---")
    with open(fields.__file__, "r", encoding="utf-8") as fh:
        s = fh.read(2000)
        print(s[:1000])
except Exception:
    import traceback
    traceback.print_exc()
    sys.exit(1)
print("\nDIAG OK")
