"""
demo1c_lp_slide_app.py -- standalone entry point for Demo 1c (the sliding
objective plane: the animated version of the 3D LP picture).

Used to build a self-contained executable with PyInstaller, so the demo can be
run on a classroom machine that has no Python installed:

    pyinstaller --onefile --name LP-3D-Slide-Demo --paths . demo1c_lp_slide_app.py

(see build_exe_demo1.py for the full command). Running this file directly also works.

Like demo 1b this one never calls a solver -- the optimum is argmax c.x over the
polytope's vertices -- so there is no gurobipy import and no license to prepare.
It does need scipy, for the half-space intersection and the convex hull that
give the polytope its vertices, faces and edges.
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

import demo1c_lp_slide_animation as demo    # builds the figure (module-level code)
import matplotlib.pyplot as plt

if _SMOKE:
    # geometry: unit cube with the (1,1,1) corner sliced off -> 10 vertices,
    # 7 faces.  Exercises scipy's HalfspaceIntersection and ConvexHull.
    print("vertices:", len(demo.VERTS), " faces:", len(demo.FACES),
          " edges:", len(demo.EDGES))
    c = demo.current_c()
    lvl0 = c @ demo.CENTROID
    lvlmax = (demo.VERTS @ c).max()
    # walk the slide by hand (FuncAnimation needs a GUI event loop to advance)
    for frac in (0.0, 0.5, 1.0):
        level = lvl0 + frac * (lvlmax - lvl0)
        sec = demo.cross_section(level, c)
        demo.draw(level, c, final=(frac == 1.0))
        print(f"  c.x = {level:6.3f}   slice = "
              f"{0 if sec is None else len(sec)} point(s)")
    demo.play(None)                 # builds the FuncAnimation
    demo.stop()
    print("SMOKE OK")
    sys.exit(0)

plt.show()
