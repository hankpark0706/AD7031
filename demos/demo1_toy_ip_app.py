"""
demo1_toy_ip_app.py -- standalone entry point for Demo 1 (the toy integer program).

Used to build a self-contained executable with PyInstaller, so the demo can be
run on a classroom machine that has no Python installed:

    pyinstaller --onefile --name Toy-IP-Demo --paths . demo1_toy_ip_app.py

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

import demo1_toy_ip as demo             # builds the figure (module-level code)
import matplotlib.pyplot as plt

if _SMOKE:
    args = (demo.s_c1.val, demo.s_c2.val, demo.s_r1.val, demo.s_r2.val)
    print("LP relaxation:", demo.solve(*args, integer=False))   # exercises Gurobi
    print("IP optimum   :", demo.solve(*args, integer=True))
    demo.play(None)
    demo.toggle_sense(None)
    print("SMOKE OK")
    sys.exit(0)

plt.show()
