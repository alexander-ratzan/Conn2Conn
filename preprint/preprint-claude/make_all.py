#!/usr/bin/env python3
"""Regenerate every main figure. Run with the dev-env interpreter:

    /Users/user/dev-env/bin/python make_all.py

Each figure script is standalone and writes PNG + PDF to ./figures/.
"""
import runpy
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

SCRIPTS = ["make_fig1.py", "make_fig2.py", "make_fig3.py",
           "make_fig4.py", "make_fig5.py", "make_supp.py", "make_supp_extra.py",
           "make_tables.py", "make_robustness.py", "make_robustness_bayes.py"]

for s in SCRIPTS:
    print(f"== {s} ==")
    runpy.run_path(str(HERE / s), run_name="__main__")
print("\nDone. Figures in", HERE / "figures")
