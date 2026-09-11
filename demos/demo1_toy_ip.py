"""
Demo 1 -- The Lecture-1 toy integer program
============================================

    max  c1*x + c2*y
    s.t.  x + 3y <= r1          (cargo volume)
         3x +  y <= r2          (payload weight)
         x, y >= 0
         (x, y) in Z^2          <-- the integrality constraint

This is the example from IE 631, Lecture 1, dressed as a resupply-truck
loading problem: x = crates of ammunition, y = crates of food rations.
Ammo crates are dense (heavy, not bulky) so they dominate the WEIGHT limit
(r2); ration crates are the opposite (bulky, not heavy) so they dominate the
VOLUME limit (r1) -- that opposite pull is what makes the two constraints a
genuinely coupled pair, not just two independent caps. The coefficients are
deliberately IDENTICAL to the LP twin demo1a_lp_graphical.py, shown right
before this one -- same objective, same two constraints, same slider ranges --
so the ONLY thing that changes between the two demos on screen is the line
"x, y integer".  Integrality isn't a modelling
nicety here -- you literally cannot load 2.5 crates. The point of the demo
is to *see* why "solve the LP and round" does NOT solve the integer program:

  * the LP optimum sits at a fractional vertex,
  * rounding it gives up to 2^d candidates,
  * some/all of those candidates are infeasible,
  * and the true integer optimum can be a lattice point far from the LP vertex.

HOW TO PLAY
-----------
Run it:  python demo1_toy_ip.py
A window opens.  Drag the sliders:
  * c1, c2  -- the objective coefficients (tilts the "uphill" direction)
  * r1, r2  -- the right-hand sides (reshapes the feasible region)
  * c1, c2 may now be NEGATIVE.
Press Play to animate the objective line sliding along the improving direction
from the worst vertex out to the optimum.  The "sense: MAX/MIN" button toggles
between maximizing and minimizing c.x -- a nice way to see that  max c.x  and
min (-c).x  reach the SAME optimal point (only the objective value flips sign).
The legend (right) names every marker, including the green square = a feasible
rounding of the LP optimum and the red x = an infeasible rounding.  The beige
panel shows the program in math form and reports the LP / IP values and the gap.
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

# Fixed plot window.  We deliberately DO NOT rescale the axes when the sliders
# move: a constant frame is the only way to actually *see* the feasible region
# grow or shrink.  The widest the region ever gets (over the slider ranges) is
# ~5.3 in each coordinate, so this frame always contains it with margin.
AXIS_LO, AXIS_HI = -0.8, 7.5

# per-crate volume and weight, in the same units as the LP twin
# (demo1a_lp_graphical.py): ammo is heavy/compact, rations are light/bulky, so
# the two constraints below pull in opposite directions.
VOL_AMMO, VOL_FOOD = 1, 3            # x + 3y <= r1   (cargo volume)
WT_AMMO, WT_FOOD = 3, 1              # 3x + y <= r2   (payload weight)

# ----------------------------------------------------------------------------
# The optimization models.  Both are built with Gurobi.  The ONLY difference
# between the LP relaxation and the integer program is the variable type:
# continuous vs. integer.  That one line is the whole story of this lecture.
# ----------------------------------------------------------------------------

def solve(c1, c2, r1, r2, integer, maximize=True):
    """Solve the toy problem. integer=False -> LP relaxation, True -> IP.
    maximize -> MAX c.x, else MIN c.x (same feasible region either way)."""
    m = gp.Model()
    m.Params.OutputFlag = 0  # stay quiet
    vtype = GRB.INTEGER if integer else GRB.CONTINUOUS
    x = m.addVar(lb=0, vtype=vtype, name="x")
    y = m.addVar(lb=0, vtype=vtype, name="y")
    m.addConstr(VOL_AMMO * x + VOL_FOOD * y <= r1)
    m.addConstr(WT_AMMO * x + WT_FOOD * y <= r2)
    m.setObjective(c1 * x + c2 * y, GRB.MAXIMIZE if maximize else GRB.MINIMIZE)
    m.optimize()
    if m.Status != GRB.OPTIMAL:
        return None
    return (x.X, y.X, m.ObjVal)


def feasible_region_polygon(r1, r2):
    """Vertices of the feasible polygon, ordered, via half-plane intersection.

    Constraints (as a*x + b*y <= c):
        VOL_AMMO x + VOL_FOOD y <= r1   (volume)
        WT_AMMO  x + WT_FOOD  y <= r2   (weight)
        -x <= 0   (i.e. x >= 0)
        -y <= 0   (i.e. y >= 0)
    We intersect every pair of boundary lines, keep the points that satisfy all
    four constraints, then sort them by angle to get a convex polygon.
    """
    A = [(VOL_AMMO, VOL_FOOD, r1), (WT_AMMO, WT_FOOD, r2), (-1, 0, 0), (0, -1, 0)]
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


def integer_points(r1, r2):
    """Integer (x,y) >= 0 inside the FIXED plot box, split feasible/infeasible."""
    hi = int(AXIS_HI)
    feas, infeas = [], []
    for xi in range(hi + 1):
        for yi in range(hi + 1):
            if (VOL_AMMO * xi + VOL_FOOD * yi <= r1 + 1e-9
                    and WT_AMMO * xi + WT_FOOD * yi <= r2 + 1e-9):
                feas.append((xi, yi))
            else:
                infeas.append((xi, yi))
    return np.array(feas), np.array(infeas)


def rounding_candidates(xlp, ylp, r1, r2):
    """floor/ceil combinations of the LP optimum, with a feasibility flag."""
    out = []
    for xi in {int(np.floor(xlp)), int(np.ceil(xlp))}:
        for yi in {int(np.floor(ylp)), int(np.ceil(ylp))}:
            ok = (xi >= 0 and yi >= 0
                  and VOL_AMMO * xi + VOL_FOOD * yi <= r1 + 1e-9
                  and WT_AMMO * xi + WT_FOOD * yi <= r2 + 1e-9)
            out.append((xi, yi, ok))
    return out


# ----------------------------------------------------------------------------
# The interactive figure
# ----------------------------------------------------------------------------

fig = plt.figure(figsize=(13.5, 8))
gs = fig.add_gridspec(1, 2, width_ratios=[1.45, 0.55], wspace=0.04)
ax = fig.add_subplot(gs[0])        # the x-y plot
ax_f = fig.add_subplot(gs[1])      # the program in math form + numeric readout
plt.subplots_adjust(left=0.06, bottom=0.30, right=0.97, top=0.92)

# sliders + Play/sense buttons along the bottom
ax_c1 = plt.axes([0.10, 0.21, 0.48, 0.025])
ax_c2 = plt.axes([0.10, 0.165, 0.48, 0.025])
ax_r1 = plt.axes([0.10, 0.12, 0.48, 0.025])
ax_r2 = plt.axes([0.10, 0.075, 0.48, 0.025])
ax_play = plt.axes([0.10, 0.02, 0.13, 0.04])
ax_sense = plt.axes([0.25, 0.02, 0.16, 0.04])
# c1, c2 may be negative now: max c.x with c=(4,5) and min c.x with c=(-4,-5)
# reach the SAME optimal point (only the objective value flips sign).
s_c1 = Slider(ax_c1, "c1 (ammo priority)", -10.0, 10.0, valinit=4.0, valstep=0.5)
s_c2 = Slider(ax_c2, "c2 (food priority)", -10.0, 10.0, valinit=5.0, valstep=0.5)
s_r1 = Slider(ax_r1, "r1 (volume: x+3y<=)", 3.0, 16.0, valinit=10.0, valstep=1.0)
s_r2 = Slider(ax_r2, "r2 (weight: 3x+y<=)", 3.0, 16.0, valinit=10.0, valstep=1.0)
btn_play = Button(ax_play, "Play", color="#c8e6c9", hovercolor="#9ccc9c")
btn_sense = Button(ax_sense, "sense: MAX", color="#d6e4ff", hovercolor="#b8d0ff")

FRAMES = 60
state = {"anim": None, "maximize": True}

# A custom legend with SMALL markers that also names the rounding symbols
# (the green square and the red x) that previously had no explanation.
LEGEND = [
    Patch(facecolor="#bcd6f0", edgecolor="none", label="LP feasible region"),
    Line2D([0], [0], marker="o", linestyle="None", markersize=7,
           markerfacecolor="none", markeredgecolor="#d62728",
           label="feasible integer points"),
    Line2D([0], [0], marker="*", linestyle="None", markersize=11,
           markerfacecolor="gold", markeredgecolor="black",
           label="LP optimum (fractional)"),
    Line2D([0], [0], marker="D", linestyle="None", markersize=7,
           markerfacecolor="magenta", markeredgecolor="black",
           label="IP optimum (integer)"),
    Line2D([0], [0], marker="s", linestyle="None", markersize=8,
           markerfacecolor="none", markeredgecolor="green",
           label="feasible rounding of LP opt"),
    Line2D([0], [0], marker="x", linestyle="None", markersize=8,
           markeredgecolor="red", markerfacecolor="red",
           label="infeasible rounding"),
    Line2D([0], [0], color="#ff7f0e", lw=2.2, label="objective line"),
]


def best_worst(c1, c2, r1, r2, maximize):
    """Return (start_vertex, start_level, opt_level).  The slide starts at the
    WORST vertex (arrow's tail) and ends at the optimum.  For max the worst is
    argmin c.v and the optimum is argmax c.v; for min it's the other way."""
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
    """Points spanning the plot box for the line c1*x + c2*y = k.  Switches
    orientation so it also works when c2 == 0 (a vertical line)."""
    if abs(c2) >= abs(c1):
        xs = np.array([AXIS_LO, AXIS_HI])
        return xs, (k - c1 * xs) / c2
    ys = np.array([AXIS_LO, AXIS_HI])
    return (k - c2 * ys) / c1, ys


def draw_formulation(reveal, lp, ip, level):
    """Right panel: the program in LaTeX + a numeric readout."""
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
        r"$\mathbf{Integer\ Program}$",
        r"$x=$ ammo crates, $y=$ food crates",
        "",
        rf"${opt}\ \ {c1:g}\,x + {c2:g}\,y$",
        rf"$\mathrm{{s.t.}}\ \ x + 3y \leq {r1:g}\ \ \mathrm{{(volume)}}$",
        rf"$\qquad\ \ 3x + y \leq {r2:g}\ \ \mathrm{{(weight)}}$",
        r"$\qquad\ \ x,\ y \geq 0$",
        r"$\qquad\ \ x,\ y \in \mathbb{Z}$",
    ])
    ax_f.text(0.06, 0.98, formulation, transform=ax_f.transAxes, va="top",
              ha="left", fontsize=12, linespacing=1.5)

    if not reveal and lp:
        txt = ("OBJECTIVE LINE  (press Play)\n"
               f"  current  c.x = {level if level is not None else 0.0:.3f}\n"
               f"  optimum      = {lp[2]:.3f}")
    elif lp:
        txt = (f"LP optimum: x={lp[0]:.3g}, y={lp[1]:.3g}\n"
               f"  value = {lp[2]:.3f}\n")
        if ip:
            txt += (f"IP optimum: x={ip[0]:.0f}, y={ip[1]:.0f}\n"
                    f"  value = {ip[2]:.3f}\n"
                    f"\nintegrality gap = {lp[2] - ip[2]:.3f}")
    else:
        txt = ""
    ax_f.text(0.06, 0.50, txt, transform=ax_f.transAxes, va="top", ha="left",
              family="monospace", fontsize=9.5, linespacing=1.3)


def render(level=None, reveal=False):
    """Draw the whole scene.  level=None -> park the hyperplane at the start
    (the arrow's tail); level=value -> the sliding objective line at that level.
    reveal -> show the LP optimum, rounding candidates, and IP optimum (only at
    the end of the Play animation, never before)."""
    c1, c2 = s_c1.val, s_c2.val
    r1, r2 = s_r1.val, s_r2.val
    maximize = state["maximize"]
    cnorm = np.hypot(c1, c2)
    lp = solve(c1, c2, r1, r2, integer=False, maximize=maximize)
    ip = solve(c1, c2, r1, r2, integer=True, maximize=maximize)

    ax.clear()
    poly = feasible_region_polygon(r1, r2)
    if len(poly):
        ax.fill(poly[:, 0], poly[:, 1], color="#bcd6f0", alpha=0.7, zorder=0)
    feas, infeas = integer_points(r1, r2)
    if len(infeas):
        ax.scatter(infeas[:, 0], infeas[:, 1], s=18, c="#cccccc", zorder=1)
    if len(feas):
        ax.scatter(feas[:, 0], feas[:, 1], s=60, facecolors="none",
                   edgecolors="#d62728", linewidths=1.6, zorder=2)

    if cnorm < 1e-9:
        # c = 0: the objective is constant, nothing to optimize -- just say so
        ax.text(0.5, 0.5, "c = 0:\nobjective is constant", transform=ax.transAxes,
                ha="center", va="center", fontsize=12, color="#888888")
        ax.set_xlim(AXIS_LO, AXIS_HI); ax.set_ylim(AXIS_LO, AXIS_HI)
        ax.set_aspect("equal")
        ax.set_xlabel("x  —  ammo crates"); ax.set_ylabel("y  —  food crates")
        draw_formulation(False, None, None, 0.0)
        fig.canvas.draw_idle()
        return

    sv, lvl0, _ = best_worst(c1, c2, r1, r2, maximize)
    if level is None:                            # initial view: park at the start
        level = lvl0

    # a faint family of iso-objective ("level") lines spanning the plot box
    corners = np.array([[AXIS_LO, AXIS_LO], [AXIS_LO, AXIS_HI],
                        [AXIS_HI, AXIS_LO], [AXIS_HI, AXIS_HI]])
    kvals = corners @ np.array([c1, c2])
    for k in np.linspace(kvals.min(), kvals.max(), 11):
        lx, ly = obj_line(c1, c2, k)
        ax.plot(lx, ly, color="#9aa0a6", lw=0.7, ls="--", zorder=0.5)
    # the current objective hyperplane (orange) at this level
    lx, ly = obj_line(c1, c2, level)
    ax.plot(lx, ly, color="#ff7f0e", lw=2.6, zorder=3)

    # the arrow shows the IMPROVING direction (the way the line slides): +c for
    # max, -c for min.  Its tail sits on the start line (vertex sv).
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

    if reveal and lp:
        xlp, ylp, _ = lp
        ax.scatter([xlp], [ylp], marker="*", s=420, c="gold",
                   edgecolors="black", linewidths=1.0, zorder=5)
        for xi, yi, ok in rounding_candidates(xlp, ylp, r1, r2):
            if ok:
                ax.scatter([xi], [yi], marker="s", s=130, facecolors="none",
                           edgecolors="green", linewidths=2.0, zorder=4)
            else:
                ax.scatter([xi], [yi], marker="x", s=130, c="red",
                           linewidths=2.5, zorder=4)
    if reveal and ip:
        ax.scatter([ip[0]], [ip[1]], marker="D", s=200, c="magenta",
                   edgecolors="black", linewidths=1.0, zorder=6)

    ax.set_xlim(AXIS_LO, AXIS_HI); ax.set_ylim(AXIS_LO, AXIS_HI)
    ax.set_aspect("equal")
    ax.set_xlabel("x  —  ammo crates"); ax.set_ylabel("y  —  food crates")
    ax.set_title("Resupply IP:  slide the objective line to the %s"
                 % ("maximum" if maximize else "minimum"))
    ax.legend(handles=LEGEND, loc="upper right", fontsize=8.5, framealpha=0.95)

    draw_formulation(reveal, lp, ip, level)
    fig.canvas.draw_idle()


def redraw(_=None):
    render(level=None, reveal=False)             # start state: no solution shown


def play(_):
    anim = state.get("anim")                     # cancel a running animation
    if anim is not None and anim.event_source is not None:
        anim.event_source.stop()
    c1, c2 = s_c1.val, s_c2.val
    r1, r2 = s_r1.val, s_r2.val
    if np.hypot(c1, c2) < 1e-9:
        return
    # Slide from the worst vertex (arrow's tail) through to the optimum.  For a
    # min, the level decreases; np.linspace handles either direction.
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
    redraw()                                     # new sense -> back to start state


for s in (s_c1, s_c2, s_r1, s_r2):
    s.on_changed(redraw)
btn_play.on_clicked(play)
btn_sense.on_clicked(toggle_sense)

redraw()

if __name__ == "__main__":
    plt.show()
