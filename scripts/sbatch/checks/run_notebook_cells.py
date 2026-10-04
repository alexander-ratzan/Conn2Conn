"""Execute a notebook's code cells in order, headless, in one namespace (no nbconvert in kraken_env).

    python scripts/sbatch/checks/run_notebook_cells.py <notebook.ipynb> [NAME=python_literal ...]

Runs from the notebook's directory (as Jupyter would), with the Agg backend and `plt.show()` closing figures.
Overrides are applied right after the first cell that assigns NAME (e.g. SAVE_FIGURES=True). Prints per-cell
timings; exits nonzero at the first failing cell. Outputs are not written back to the notebook.
"""
import ast
import json
import os
import sys
import time
import traceback

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

path = os.path.abspath(sys.argv[1])
overrides = dict(arg.split("=", 1) for arg in sys.argv[2:])
os.chdir(os.path.dirname(path))
plt.show = lambda *a, **k: plt.close("all")
cells = [c for c in json.load(open(path))["cells"] if c["cell_type"] == "code"]
ns = {"__name__": "__main__"}
for i, cell in enumerate(cells):
    src = "".join(cell["source"])
    t0 = time.time()
    try:
        exec(compile(src, f"<cell {i}>", "exec"), ns)
    except Exception:
        traceback.print_exc()
        print(f"CELL {i} FAILED:\n{src[:400]}")
        sys.exit(1)
    assigned = {t.id for node in ast.walk(ast.parse(src)) if isinstance(node, ast.Assign)
                for t in node.targets if isinstance(t, ast.Name)}
    for name in list(overrides):
        if name in assigned:
            ns[name] = ast.literal_eval(overrides.pop(name))
            print(f"  override {name}={ns[name]!r}")
    plt.close("all")
    print(f"cell {i:2d} ok ({time.time() - t0:6.1f}s): {src.strip().splitlines()[0][:80] if src.strip() else ''}", flush=True)
print(f"ALL {len(cells)} CODE CELLS RAN")
