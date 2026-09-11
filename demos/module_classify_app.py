"""
module_classify_app.py — standalone entry point for the classification module.

Used to build a self-contained executable with PyInstaller, so the dashboard can
be run on a classroom machine that has no Python installed:

    pyinstaller --onefile --name Classify-Dashboard --paths . module_classify_app.py

(see build_exe.py for the full command). Running this file directly also works.
"""
import os
import sys

# make sure sibling modules are importable, both when run normally and when
# frozen by PyInstaller
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
if getattr(sys, "frozen", False):
    sys.path.insert(0, os.path.dirname(sys.executable))
    sys.path.insert(0, getattr(sys, "_MEIPASS", ""))

import matplotlib

# CLASSIFY_SMOKE lets us verify a frozen build works without opening a window
_SMOKE = bool(os.environ.get("CLASSIFY_SMOKE"))
if _SMOKE:
    _backend = "Agg"
elif sys.platform == "darwin":
    _backend = "macosx"        # native macOS GUI — no tkinter needed
else:
    _backend = "TkAgg"         # Windows / Linux
matplotlib.use(_backend, force=True)

import module_classify as app          # builds the figure (module-level code)
import matplotlib.pyplot as plt

if _SMOKE:
    app.on_opt(None)                    # exercise the scipy/HiGHS MILP solve path
    app.on_data(None)
    print("SMOKE OK")
    sys.exit(0)

plt.show()
