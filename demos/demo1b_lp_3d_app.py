"""
demo1b_lp_3d_app.py -- standalone entry point for Demo 1b (the 3D LP polytope).

Used to build a self-contained executable with PyInstaller, so the demo can be
run on a classroom machine that has no Python installed:

    pyinstaller --onefile --name LP-3D-Demo --paths . demo1b_lp_3d_app.py

(see build_exe_demo1.py for the full command). Running this file directly also works.

Unlike the other Lecture-1 demos this one never calls a solver -- the optimum is
argmax c.x over the polytope's vertices, computed with numpy -- so there is no
gurobipy import and no license to prepare. It does need scipy, for the
half-space intersection that turns the inequalities into vertices and faces.
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

# DEMO_SMOKE lets us verify a frozen build works without opening a window
_SMOKE = bool(os.environ.get("DEMO_SMOKE"))
if _SMOKE:
    _backend = "Agg"
elif sys.platform == "darwin":
    _backend = "macosx"        # native macOS GUI -- no tkinter needed
else:
    _backend = "TkAgg"         # Windows / Linux
matplotlib.use(_backend, force=True)

import demo1b_lp_3d as demo            # builds the figure (module-level code)
import matplotlib.pyplot as plt

if _SMOKE:
    import numpy as np
    # the polytope itself: a unit cube with the (1,1,1) corner sliced off
    # -> 10 vertices, 7 faces.  Exercises scipy's HalfspaceIntersection.
    print("vertices:", len(demo.VERTS), " faces:", len(demo.FACES))
    for cz in (1.0, -1.0):                     # two objective directions
        demo.s_cz.set_val(cz)                  # fires redraw()
        c = np.array([demo.s_cx.val, demo.s_cy.val, demo.s_cz.val])
        v = demo.VERTS[int(np.argmax(demo.VERTS @ c))]
        print(f"c = ({c[0]:.2f}, {c[1]:.2f}, {c[2]:.2f})  ->  vertex "
              f"({v[0]:.2f}, {v[1]:.2f}, {v[2]:.2f})  value {(demo.VERTS @ c).max():.3f}")
    print("SMOKE OK")
    sys.exit(0)

plt.show()
