"""
demo1d_qp_app.py -- standalone entry point for Demo 1d (QP, graphical method).

Used to build a self-contained executable with PyInstaller, so the demo can be
run on a classroom machine that has no Python installed:

    pyinstaller --onefile --name QP-Graphical-Demo --paths . demo1d_qp_app.py

(see build_exe_demo1.py for the full command). Running this file directly also works.

Like demos 1b/1c this one never calls a solver -- for Q = 2I the QP optimum is
the Euclidean projection of the target onto the polygon, computed exactly with
numpy -- so there is no gurobipy import and no license to prepare.
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

import demo1d_qp_graphical as demo       # builds the figure (module-level code)
import matplotlib.pyplot as plt

if _SMOKE:
    # the lecture-note case: target (4,2) -> tangent to the weight wall
    print("edge    :", demo.solve_qp(4, 2, 10, 10))     # (2.8, 1.6, 1.6, edge)
    print("interior:", demo.solve_qp(1, 1, 10, 10))     # (1, 1, 0, interior)
    print("boundary:", demo.solve_qp(2.5, 2.5, 10, 10))  # feasible AT a vertex
    print("vertex  :", demo.solve_qp(5, -2, 10, 10))    # (10/3, 0, ., vertex)
    poly = demo.feasible_region_polygon(3, 9)           # triple point deduped
    print("degenerate poly:", len(poly), "vertices")    # 3, not 5
    demo.render(radius=None, reveal=True)               # full reveal draw
    fig_w, fig_h = demo.fig.canvas.get_width_height()
    demo.fig.canvas.draw()                              # exercises both panels
    print(f"canvas {fig_w}x{fig_h}")
    demo.draw_movers(0.9)                               # animation fast path
    demo.fig.canvas.draw()
    demo.draw_movers(None)
    demo.play(None)                                     # animation constructs
    print("SMOKE OK")
    sys.exit(0)

plt.show()
