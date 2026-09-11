"""
demo1a_lp_app.py -- standalone entry point for Demo 1a (LP, graphical method).

Used to build a self-contained executable with PyInstaller, so the demo can be
run on a classroom machine that has no Python installed:

    pyinstaller --onefile --name LP-Graphical-Demo --paths . demo1a_lp_app.py

(see build_exe_demo1.py for the full command). Running this file directly also works.
"""
import os
import sys

# make sure sibling modules are importable, both when run normally and when
# frozen by PyInstaller
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
if getattr(sys, "frozen", False):
    sys.path.insert(0, os.path.dirname(sys.executable))
    sys.path.insert(0, getattr(sys, "_MEIPASS", ""))

from grb_frozen_support import prepare_gurobi

prepare_gurobi()                    # MUST run before gurobipy is imported

import matplotlib

# DEMO_SMOKE lets us verify a frozen build works without opening a window
_SMOKE = bool(os.environ.get("DEMO_SMOKE"))
if _SMOKE:
    _backend = "Agg"
elif sys.platform == "darwin":
    _backend = "macosx"        # native macOS GUI -- no tkinter needed
else:
    _backend = "TkAgg"         # Windows / Linux
matplotlib.use(_backend, force=True)

import demo1a_lp_graphical as demo       # builds the figure (module-level code)
import matplotlib.pyplot as plt

if _SMOKE:
    print("LP max:", demo.solve_lp(4, 5, 10, 10))          # exercises Gurobi
    print("LP min:", demo.solve_lp(4, 5, 10, 10, maximize=False))
    demo.play(None)
    demo.toggle_sense(None)
    print("SMOKE OK")
    sys.exit(0)

plt.show()
