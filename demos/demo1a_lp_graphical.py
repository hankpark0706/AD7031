"""
Demo 1a -- Solving a LINEAR program graphically (no integers yet!)
==================================================================

    max  c1*x + c2*y
    s.t. x + 3y <= r1
         3x +  y <= r2
         x, y >= 0            <-- NO integrality constraint

This is the LP-ONLY twin of demo1_toy_ip.py, meant to be shown FIRST in
lecture.  It teaches the graphical method on its own terms:

  * the feasible region is a polygon (draw it),
  * the objective c.x is a family of parallel lines (level sets),
  * sliding the line in the improving direction, the LAST touch is a VERTEX,
  * so the graphical method = evaluate the corners, keep the best.

No lattice points, no rounding, no IP -- that is demo1_toy_ip.py, shown
right after this one, where adding "x, y integer" breaks the clean picture.

HOW TO PLAY
-----------
Run it:  python demo1a_lp_graphical.py
A window opens.  Drag the sliders:
  * c1, c2  -- the objective coefficients (tilts the "uphill" direction)
  * r1, r2  -- the right-hand sides (reshapes the feasible region)
Press Play to animate the objective line sliding from the worst vertex to
the optimum; at the end every vertex is labeled with its objective value
(the graphical method in one picture) and the optimum gets the star.
The "sense: MAX/MIN" button toggles between maximizing and minimizing.
"""

import itertools
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from matplotlib.animation import FuncAnimation
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import gurobipy as gp
from gurobipy import GRB

# Fixed plot window (same frame as demo1_toy_ip.py so the two demos look alike).
AXIS_LO, AXIS_HI = -0.8, 7.5


def solve_lp(c1, c2, r1, r2, maximize=True):
    """Solve the LP (continuous variables only)."""
    m = gp.Model()
    m.Params.OutputFlag = 0
    x = m.addVar(lb=0, vtype=GRB.CONTINUOUS, name="x")
    y = m.addVar(lb=0, vtype=GRB.CONTINUOUS, name="y")
    m.addConstr(x + 3 * y <= r1)
    m.addConstr(3 * x + y <= r2)
    m.setObjective(c1 * x + c2 * y, GRB.MAXIMIZE if maximize else GRB.MINIMIZE)
    m.optimize()
    if m.Status != GRB.OPTIMAL:
        return None
    return (x.X, y.X, m.ObjVal)


def feasible_region_polygon(r1, r2):
    """Vertices of the feasible polygon, ordered, via half-plane intersection."""
    A = [(1, 3, r1), (3, 1, r2), (-1, 0, 0), (0, -1, 0)]
    pts = []
    for (a1, b1, c1_), (a2, b2, c2_) in itertools.combinations(A, 2):
        det = a1 * b2 - a2 * b1
        if abs(det) < 1e-12:
            continue
        px = (c1_ * b2 - c2_ * b1) / det
        py = (a1 * c2_ - a2 * c1_) / det
        if all(a * px + b * py <= c + 1e-9 for a, b, c in A):
            pts.append((px, py))
    if not pts:
        return np.empty((0, 2))
    pts = np.array(pts)
    centroid = pts.mean(axis=0)
    order = np.argsort(np.arctan2(pts[:, 1] - centroid[1], pts[:, 0] - centroid[0]))
    return pts[order]


# ----------------------------------------------------------------------------
# The interactive figure
# ----------------------------------------------------------------------------

fig = plt.figure(figsize=(13.5, 8))
gs = fig.add_gridspec(1, 2, width_ratios=[1.45, 0.55], wspace=0.04)
ax = fig.add_subplot(gs[0])        # the x-y plot
ax_f = fig.add_subplot(gs[1])      # the program in math form + numeric readout
plt.subplots_adjust(left=0.06, bottom=0.30, right=0.97, top=0.92)

ax_c1 = plt.axes([0.10, 0.21, 0.48, 0.025])
ax_c2 = plt.axes([0.10, 0.165, 0.48, 0.025])
ax_r1 = plt.axes([0.10, 0.12, 0.48, 0.025])
ax_r2 = plt.axes([0.10, 0.075, 0.48, 0.025])
ax_play = plt.axes([0.10, 0.02, 0.13, 0.04])
ax_sense = plt.axes([0.25, 0.02, 0.16, 0.04])
s_c1 = Slider(ax_c1, "c1 (obj x)", -10.0, 10.0, valinit=4.0, valstep=0.5)
s_c2 = Slider(ax_c2, "c2 (obj y)", -10.0, 10.0, valinit=5.0, valstep=0.5)
s_r1 = Slider(ax_r1, "r1 (x+3y<=)", 3.0, 16.0, valinit=10.0, valstep=1.0)
s_r2 = Slider(ax_r2, "r2 (3x+y<=)", 3.0, 16.0, valinit=10.0, valstep=1.0)
btn_play = Button(ax_play, "Play", color="#c8e6c9", hovercolor="#9ccc9c")
btn_sense = Button(ax_sense, "sense: MAX", color="#d6e4ff", hovercolor="#b8d0ff")

FRAMES = 60
state = {"anim": None, "maximize": True}

LEGEND = [
    Patch(facecolor="#bcd6f0", edgecolor="none", label="feasible region"),
    Line2D([0], [0], marker="o", linestyle="None", markersize=7,
           markerfacecolor="#1f4e79", markeredgecolor="white",
           label="vertex (corner point)"),
    Line2D([0], [0], marker="*", linestyle="None", markersize=11,
           markerfacecolor="gold", markeredgecolor="black",
           label="LP optimum"),
    Line2D([0], [0], color="#ff7f0e", lw=2.2, label="objective line"),
]


def best_worst(c1, c2, r1, r2, maximize):
    """(start_vertex, start_level, opt_level) for the Play slide."""
    poly = feasible_region_polygon(r1, r2)
    if not len(poly):
        return np.array([0.0, 0.0]), 0.0, 0.0
    vals = poly @ np.array([c1, c2])
    if maximize:
        start = poly[int(np.argmin(vals))]
        return start, float(vals.min()), float(vals.max())
    start = poly[int(np.argmax(vals))]
    return start, float(vals.max()), float(vals.min())


def obj_line(c1, c2, k):
    """Points spanning the plot box for the line c1*x + c2*y = k."""
    if abs(c2) >= abs(c1):
        xs = np.array([AXIS_LO, AXIS_HI])
        return xs, (k - c1 * xs) / c2
    ys = np.array([AXIS_LO, AXIS_HI])
    return (k - c2 * ys) / c1, ys


def draw_formulation(reveal, lp, level, alt_edge=None):
    """Right panel: the program in LaTeX + a numeric readout.  alt_edge, when
    not None, is the ((x1,y1), (x2,y2)) pair of tied vertices -- signals that
    every point on that edge is optimal, not just a single vertex."""
    c1, c2 = s_c1.val, s_c2.val
    r1, r2 = s_r1.val, s_r2.val
    maximize = state["maximize"]
    ax_f.clear()
    ax_f.set_xticks([]); ax_f.set_yticks([])
    for sp in ax_f.spines.values():
        sp.set_visible(False)
    ax_f.set_facecolor("#fff7d6")

    opt = r"\max" if maximize else r"\min"
    formulation = "\n".join([
        r"$\mathbf{Linear\ Program}$",
        "",
        rf"${opt}\ \ {c1:g}\,x + {c2:g}\,y$",
        rf"$\mathrm{{s.t.}}\ \ x + 3y \leq {r1:g}$",
        rf"$\qquad\ \ 3x + y \leq {r2:g}$",
        r"$\qquad\ \ x,\ y \geq 0$",
    ])
    ax_f.text(0.06, 0.98, formulation, transform=ax_f.transAxes, va="top",
              ha="left", fontsize=12, linespacing=1.5)

    if not reveal and lp:
        txt = ("OBJECTIVE LINE  (press Play)\n"
               f"  current  c.x = {level if level is not None else 0.0:.3f}\n"
               f"  optimum      = {lp[2]:.3f}")
    elif lp and alt_edge is not None:
        (x1, y1), (x2, y2) = alt_edge
        txt = (f"LP optimum: value = {lp[2]:.3f}\n"
               f"  achieved at EVERY point from\n"
               f"  ({x1:.3g}, {y1:.3g})  to  ({x2:.3g}, {y2:.3g})\n"
               "\nINFINITELY MANY OPTIMAL SOLUTIONS:\n"
               "  c is parallel to that edge, so\n"
               "  the whole segment ties -- not\n"
               "  just one vertex.")
    elif lp:
        txt = (f"LP optimum: x={lp[0]:.3g}, y={lp[1]:.3g}\n"
               f"  value = {lp[2]:.3f}\n"
               "\nGRAPHICAL METHOD:\n"
               "  the optimum is the vertex\n"
               "  with the best c.x value\n"
               "  (see the labels).")
    else:
        txt = ""
    ax_f.text(0.06, 0.50, txt, transform=ax_f.transAxes, va="top", ha="left",
              family="monospace", fontsize=9.5, linespacing=1.3)


def render(level=None, reveal=False):
    """Draw the scene.  reveal -> label every vertex with its objective value
    and star the optimum (only at the end of the Play animation)."""
    c1, c2 = s_c1.val, s_c2.val
    r1, r2 = s_r1.val, s_r2.val
    maximize = state["maximize"]
    cnorm = np.hypot(c1, c2)
    lp = solve_lp(c1, c2, r1, r2, maximize=maximize)

    ax.clear()
    poly = feasible_region_polygon(r1, r2)
    if len(poly):
        ax.fill(poly[:, 0], poly[:, 1], color="#bcd6f0", alpha=0.7, zorder=0)
        ax.scatter(poly[:, 0], poly[:, 1], s=45, c="#1f4e79",
                   edgecolors="white", linewidths=1.0, zorder=2)

    if cnorm < 1e-9:
        ax.text(0.5, 0.5, "c = 0:\nobjective is constant", transform=ax.transAxes,
                ha="center", va="center", fontsize=12, color="#888888")
        ax.set_xlim(AXIS_LO, AXIS_HI); ax.set_ylim(AXIS_LO, AXIS_HI)
        ax.set_aspect("equal"); ax.set_xlabel("x"); ax.set_ylabel("y")
        draw_formulation(False, None, 0.0)
        fig.canvas.draw_idle()
        return

    sv, lvl0, _ = best_worst(c1, c2, r1, r2, maximize)
    if level is None:
        level = lvl0

    # faint family of iso-objective ("level") lines
    corners = np.array([[AXIS_LO, AXIS_LO], [AXIS_LO, AXIS_HI],
                        [AXIS_HI, AXIS_LO], [AXIS_HI, AXIS_HI]])
    kvals = corners @ np.array([c1, c2])
    for k in np.linspace(kvals.min(), kvals.max(), 11):
        lx, ly = obj_line(c1, c2, k)
        ax.plot(lx, ly, color="#9aa0a6", lw=0.7, ls="--", zorder=0.5)
    # the current objective line (orange)
    lx, ly = obj_line(c1, c2, level)
    ax.plot(lx, ly, color="#ff7f0e", lw=2.6, zorder=3)

    # improving-direction arrow, tail on the start vertex
    sign = 1.0 if maximize else -1.0
    sx, sy = sv
    L = 1.6
    ax.annotate("", xy=(sx + sign * c1 / cnorm * L, sy + sign * c2 / cnorm * L),
                xytext=(sx, sy), zorder=7,
                arrowprops=dict(arrowstyle="-|>", color="black", lw=2.0,
                                mutation_scale=20))
    ax.text(sx + sign * c1 / cnorm * (L + 0.4), sy + sign * c2 / cnorm * (L + 0.4),
            r"$c$" if maximize else r"$-c$", fontsize=12, ha="center", va="center",
            bbox=dict(boxstyle="round", fc="white", ec="none", alpha=0.8))

    alt_edge = None
    if reveal and lp and len(poly):
        # THE GRAPHICAL METHOD: label every vertex with its objective value...
        vals = poly @ np.array([c1, c2])
        best = np.argmax(vals) if maximize else np.argmin(vals)
        best_val = vals[best]
        # tied vertices -> c is parallel to the edge between them, so every
        # point on that edge is equally optimal (a convex-polygon fact: the
        # optimal FACE of a linear objective is a vertex, an edge, or -- only
        # when c=0, handled separately above -- the whole region).
        tied = np.where(np.isclose(vals, best_val, rtol=1e-9, atol=1e-6))[0]
        n = len(poly)
        tied_edges = [(i, (i + 1) % n) for i in tied if (i + 1) % n in tied]

        for i, ((vx, vy), v) in enumerate(zip(poly, vals)):
            is_best = i in tied
            ax.annotate(f"c.x = {v:g}", xy=(vx, vy), xytext=(10, 10),
                        textcoords="offset points", fontsize=9.5,
                        fontweight="bold" if is_best else "normal",
                        color="#1f4e79",
                        bbox=dict(boxstyle="round,pad=0.25",
                                  fc="#ffe9a8" if is_best else "white",
                                  ec="#1f4e79", lw=0.8, alpha=0.95), zorder=6)

        if tied_edges:
            # highlight the optimal EDGE and star both its endpoints
            i, j = tied_edges[0]
            alt_edge = (tuple(poly[i]), tuple(poly[j]))
            ax.plot([poly[i, 0], poly[j, 0]], [poly[i, 1], poly[j, 1]],
                    color="gold", lw=6, solid_capstyle="round", zorder=4)
            ax.scatter(poly[[i, j], 0], poly[[i, j], 1], marker="*", s=420,
                       c="gold", edgecolors="black", linewidths=1.0, zorder=5)
            mx, my = (poly[i, 0] + poly[j, 0]) / 2, (poly[i, 1] + poly[j, 1]) / 2
            centroid = poly.mean(axis=0)
            out = np.array([mx, my]) - centroid
            out = out / (np.hypot(*out) or 1.0)
            ax.annotate("∞ many optimal solutions\n(every point on this edge ties)",
                        xy=(mx, my), xytext=(mx + out[0] * 1.6, my + out[1] * 1.6),
                        fontsize=10, fontweight="bold", ha="center", va="center",
                        color="#7a2e12", zorder=8,
                        bbox=dict(boxstyle="round,pad=0.45", fc="#ffe1b8",
                                  ec="#c0392b", lw=1.6),
                        arrowprops=dict(arrowstyle="-", color="#c0392b", lw=1.3))
        else:
            # the usual case: a single optimal vertex -> one star
            xlp, ylp, _ = lp
            ax.scatter([xlp], [ylp], marker="*", s=420, c="gold",
                       edgecolors="black", linewidths=1.0, zorder=5)

    ax.set_xlim(AXIS_LO, AXIS_HI); ax.set_ylim(AXIS_LO, AXIS_HI)
    ax.set_aspect("equal")
    ax.set_xlabel("x"); ax.set_ylabel("y")
    ax.set_title("Toy LP:  slide the objective line to the %s"
                 % ("maximum" if maximize else "minimum"))
    ax.legend(handles=LEGEND, loc="upper right", fontsize=8.5, framealpha=0.95)

    draw_formulation(reveal, lp, level, alt_edge)
    fig.canvas.draw_idle()


def redraw(_=None):
    render(level=None, reveal=False)


def play(_):
    anim = state.get("anim")
    if anim is not None and anim.event_source is not None:
        anim.event_source.stop()
    c1, c2 = s_c1.val, s_c2.val
    r1, r2 = s_r1.val, s_r2.val
    if np.hypot(c1, c2) < 1e-9:
        return
    _, lvl0, lvlopt = best_worst(c1, c2, r1, r2, state["maximize"])
    levels = np.linspace(lvl0, lvlopt, FRAMES)

    def update(k):
        render(level=levels[k], reveal=(k == FRAMES - 1))

    state["anim"] = FuncAnimation(fig, update, frames=FRAMES, interval=60,
                                  repeat=False)
    fig.canvas.draw_idle()


def toggle_sense(_):
    state["maximize"] = not state["maximize"]
    btn_sense.label.set_text("sense: MAX" if state["maximize"] else "sense: MIN")
    redraw()


for s in (s_c1, s_c2, s_r1, s_r2):
    s.on_changed(redraw)
btn_play.on_clicked(play)
btn_sense.on_clicked(toggle_sense)

redraw()

if __name__ == "__main__":
    plt.show()
