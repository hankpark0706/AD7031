"""
Demo 1d -- Solving a QUADRATIC program graphically (the circle inflates)
========================================================================

    min  (x - a)^2 + (y - b)^2
    s.t. x + 3y <= r1
         3x +  y <= r2
         x, y >= 0

Same feasible polygon as demo1a_lp_graphical.py, but the objective now CURVES:
its level sets are circles around the target (a, b) instead of parallel lines.

  * the feasible region is the SAME polygon P (draw it),
  * the objective is a family of concentric circles (level sets),
  * inflate the circle from the target until it FIRST touches P,
  * the touch point is the optimum -- and it need NOT be a vertex:
    it is interior (target feasible), on an edge (tangent), or at a vertex.

That is the one-rung-up story after the LP demo: keep the polyhedron, curve
the objective, and the "optimum sits at a vertex" fact is gone.

TWO PANELS
----------
Left:  the x-y plane -- polygon, circles, the growing level set.
Right: the same problem in 3D -- the bowl z = (x-a)^2 + (y-b)^2 over the
       plane, the polygon flat on the floor, and the slicing plane z = r^2
       rising with the animation.  The optimum is the SAME magenta point in
       both panels: position on the left, position + objective value on the
       right.  Drag the right panel to rotate.

HOW TO PLAY
-----------
Run it:  python demo1d_qp_graphical.py
A window opens.  Drag the sliders:
  * a, b    -- the target = center of the circles = bottom of the bowl
  * r1, r2  -- the right-hand sides (reshapes the feasible region)
Press Play to inflate the circle from radius 0 until it first touches P; at
the end the optimum gets the magenta star (left) and the magenta dot on the
bowl (right), every vertex is labeled with its objective value, and the
readout names the binding wall.  After one Play the reveal stays on, so
dragging the target slides the optimum around live -- interior, edge, vertex.

No solver is called: for Q = 2I the optimum is the Euclidean projection of
the target onto P, computed exactly with numpy (inside test + per-edge
projection).  Gurobi solves the very same QP in this week's notebook.
"""

import itertools
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.widgets import Slider, Button
from matplotlib.animation import FuncAnimation
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# Fixed plot window (same frame as demo1a_lp_graphical.py so the demos look alike).
AXIS_LO, AXIS_HI = -0.8, 7.5

MAGENTA = "#e520c8"          # the optimum, same color in both panels
TARGET_RED = "#c0392b"       # the target x
ORANGE = "#ff7f0e"           # the moving level set, like demo1a's objective line
RING_R = (0.8, 1.6, 2.4, 3.2)   # the faint always-on family of level circles


def halfplanes(r1, r2):
    """The constraints as (p, q, c) meaning p*x + q*y <= c."""
    return [(1, 3, r1), (3, 1, r2), (-1, 0, 0), (0, -1, 0)]


def feasible_region_polygon(r1, r2):
    """Vertices of the feasible polygon, ordered, via half-plane intersection."""
    A = halfplanes(r1, r2)
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
    # dedupe: where three walls meet at one point (e.g. r2 = 3*r1) the corner
    # is appended once per line pair -- keep it once
    pts = np.unique(np.round(np.array(pts), 9), axis=0)
    centroid = pts.mean(axis=0)
    order = np.argsort(np.arctan2(pts[:, 1] - centroid[1], pts[:, 0] - centroid[0]))
    return pts[order]


def solve_qp(a, b, r1, r2):
    """Exact QP optimum = Euclidean projection of the target (a, b) onto P.

    Returns (x*, y*, f*, kind) with kind in {"interior", "boundary", "edge",
    "vertex"} -- "boundary" is a FEASIBLE target sitting exactly on a wall,
    where "the circle never grew" is true but "interior" would be a lie.
    """
    A = halfplanes(r1, r2)
    if all(p * a + q * b <= c + 1e-9 for p, q, c in A):
        on_wall = any(abs(p * a + q * b - c) < 1e-7 for p, q, c in A)
        return float(a), float(b), 0.0, "boundary" if on_wall else "interior"
    poly = feasible_region_polygon(r1, r2)
    if not len(poly):
        return 0.0, 0.0, float(a * a + b * b), "vertex"
    t0 = np.array([a, b], dtype=float)
    best = None            # (x, y, f, kind)
    n = len(poly)
    for i in range(n):
        p, q = poly[i], poly[(i + 1) % n]
        d = q - p
        L2 = float(d @ d)
        t = 0.0 if L2 < 1e-12 else float(np.clip((t0 - p) @ d / L2, 0.0, 1.0))
        cand = p + t * d
        f = float((cand[0] - a) ** 2 + (cand[1] - b) ** 2)
        if best is None or f < best[2] - 1e-12:
            kind = "edge" if 1e-7 < t < 1 - 1e-7 else "vertex"
            best = (float(cand[0]), float(cand[1]), f, kind)
    return best


def binding_walls(xs, ys, r1, r2):
    """Names of the constraints tight at (xs, ys)."""
    names = [f"x + 3y <= {r1:g}  (volume)", f"3x + y <= {r2:g}  (weight)",
             "x >= 0", "y >= 0"]
    out = []
    for (p, q, c), name in zip(halfplanes(r1, r2), names):
        if abs(p * xs + q * ys - c) < 1e-6:
            out.append(name)
    return out


# ----------------------------------------------------------------------------
# The interactive figure
# ----------------------------------------------------------------------------

fig = plt.figure(figsize=(14.5, 7.9))
gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.0], wspace=0.02)
ax = fig.add_subplot(gs[0])                       # the x-y plane
ax3 = fig.add_subplot(gs[1], projection="3d")     # the bowl over the plane
plt.subplots_adjust(left=0.055, bottom=0.26, right=0.97, top=0.92)

ax_a = plt.axes([0.08, 0.185, 0.34, 0.025])
ax_b = plt.axes([0.08, 0.140, 0.34, 0.025])
ax_r1 = plt.axes([0.08, 0.095, 0.34, 0.025])
ax_r2 = plt.axes([0.08, 0.050, 0.34, 0.025])
ax_play = plt.axes([0.50, 0.11, 0.11, 0.05])
s_a = Slider(ax_a, "a (target x)", -2.0, 9.0, valinit=4.0, valstep=0.25)
s_b = Slider(ax_b, "b (target y)", -2.0, 9.0, valinit=2.0, valstep=0.25)
s_r1 = Slider(ax_r1, "r1 (x+3y<=)", 3.0, 16.0, valinit=10.0, valstep=1.0)
s_r2 = Slider(ax_r2, "r2 (3x+y<=)", 3.0, 16.0, valinit=10.0, valstep=1.0)
btn_play = Button(ax_play, "Play", color="#c8e6c9", hovercolor="#9ccc9c")

FRAMES = 40
state = {"anim": None, "revealed": False, "movers": [], "zcap": 4.0}

LEGEND = [
    Patch(facecolor="#bcd6f0", edgecolor="none", label="feasible region $P$"),
    Line2D([0], [0], marker="x", linestyle="None", markersize=8,
           markeredgewidth=2.2, color=TARGET_RED, label="target $(a, b)$"),
    Line2D([0], [0], color=ORANGE, lw=2.2,
           label="level set $(x{-}a)^2 + (y{-}b)^2 = z$"),
    Line2D([0], [0], marker="*", linestyle="None", markersize=12,
           markerfacecolor=MAGENTA, markeredgecolor="black", label="QP optimum"),
]


def sq_term(var, v):
    """'(x - 4)^2', '(x + 2)^2', or 'x^2' -- no 'x - -2' double negatives."""
    if v == 0:
        return f"{var}^2"
    return f"({var} - {v:g})^2" if v > 0 else f"({var} + {-v:g})^2"


def draw_readout(radius, reveal, sol, banner=None):
    """Monospace readout on the 2D panel: the program + the current numbers."""
    a, b = s_a.val, s_b.val
    r1, r2 = s_r1.val, s_r2.val
    xs, ys, fs, kind = sol
    lines = ["QUADRATIC PROGRAM",
             f"  min  {sq_term('x', a)} + {sq_term('y', b)}",
             f"  s.t. x + 3y <= {r1:g}",
             f"       3x + y <= {r2:g}",
             "       x, y >= 0", ""]
    if reveal:
        lines += [f"optimum  x* = {xs:.3g},  y* = {ys:.3g}",
                  f"value    f* = {fs:.3f}   (radius r* = {np.sqrt(fs):.3f})"]
        if kind == "interior":
            lines += ["", "target is FEASIBLE -> optimum = target,",
                      "INTERIOR of P: the circle never grew.",
                      "an LP can never do this."]
        elif kind == "boundary":
            lines += ["", "target is FEASIBLE -> optimum = target,",
                      "sitting exactly ON the boundary of P;",
                      "tight:"]
            lines += ["  " + w for w in binding_walls(xs, ys, r1, r2)]
        elif kind == "edge":
            wall = binding_walls(xs, ys, r1, r2)
            lines += ["", "tangent to " + (wall[0] if wall else "an edge"),
                      "on an EDGE of P -- NOT a vertex!"]
        else:
            lines += ["", "binding:"]
            lines += ["  " + w for w in binding_walls(xs, ys, r1, r2)]
            lines += ["at a VERTEX -- a corner can still win."]
    elif radius is not None:
        lines += ["inflating the level set...",
                  f"  radius r = {radius:.3f}",
                  f"  value  z = r^2 = {radius**2:.3f}"]
    elif banner is not None:
        lines += [banner]
    else:
        lines += ["press Play: the circle inflates from the",
                  "target until it FIRST touches P."]
    ax.text(0.02, 0.98, "\n".join(lines), transform=ax.transAxes, va="top",
            ha="left", family="monospace", fontsize=8.6, linespacing=1.35,
            bbox=dict(boxstyle="round,pad=0.45", fc="#fff7d6", ec="#c9b96a",
                      alpha=0.95), zorder=10)


def wall_labels(poly, r1, r2):
    """Label the two structural walls just outside their polygon edge."""
    for (p, q, c), txt in (((1, 3, r1), f"$x+3y={r1:g}$"),
                           ((3, 1, r2), f"$3x+y={r2:g}$")):
        on = [v for v in poly if abs(p * v[0] + q * v[1] - c) < 1e-6]
        if len(on) < 2:
            continue
        m = (on[0] + on[1]) / 2.0
        nrm = np.array([p, q], float)
        nrm /= np.hypot(*nrm)
        ax.text(m[0] + 0.62 * nrm[0], m[1] + 0.62 * nrm[1], txt, fontsize=8,
                color="#1f4e79", ha="center", va="center")


def render(radius=None, reveal=False, banner=None):
    """Draw both panels.  radius -> the moving level circle; reveal -> star the
    optimum, label every vertex with its objective value, name the binding wall."""
    a, b = s_a.val, s_b.val
    r1, r2 = s_r1.val, s_r2.val
    sol = solve_qp(a, b, r1, r2)
    xs, ys, fs, kind = sol
    rstar = float(np.sqrt(fs))
    at_target = kind in ("interior", "boundary")   # optimum IS the target
    if reveal and radius is None:
        radius = rstar
    poly = feasible_region_polygon(r1, r2)
    state["movers"] = []          # ax.clear() below kills any animation artist

    # ---------------- left: the x-y plane --------------------------------
    ax.clear()
    if len(poly):
        ax.fill(poly[:, 0], poly[:, 1], color="#bcd6f0", alpha=0.7, zorder=0)
        ax.scatter(poly[:, 0], poly[:, 1], s=45, c="#1f4e79",
                   edgecolors="white", linewidths=1.0, zorder=2)
        wall_labels(poly, r1, r2)

    # faint family of level circles (the QP twin of demo1a's dashed lines)
    for rr in RING_R:
        ax.add_patch(plt.Circle((a, b), rr, fill=False, color="#9aa0a6",
                                lw=0.7, ls="--", zorder=0.5))
    # the moving level set
    if radius is not None and radius > 1e-9:
        ax.add_patch(plt.Circle((a, b), radius, fill=False, color=ORANGE,
                                lw=2.6, zorder=3))
    # the target = circle center = bowl bottom
    ax.plot([a], [b], marker="x", ms=10, mew=2.4, color=TARGET_RED, zorder=4)
    if not (reveal and at_target):              # else the magenta label says it
        ax.annotate(f"target $({a:g}, {b:g})$", xy=(a, b), xytext=(8, 8),
                    textcoords="offset points", fontsize=9, color=TARGET_RED,
                    zorder=4)

    if reveal:
        # label every vertex with its objective value (none beats the star
        # unless the optimum IS a vertex -- then that label gets the gold box)
        if len(poly):
            fv = (poly[:, 0] - a) ** 2 + (poly[:, 1] - b) ** 2
            for (vx, vy), v in zip(poly, fv):
                is_best = kind == "vertex" and abs(v - fs) < 1e-9
                ax.annotate(f"f = {v:.2f}", xy=(vx, vy), xytext=(8, -13),
                            textcoords="offset points", fontsize=8.5,
                            fontweight="bold" if is_best else "normal",
                            color="#1f4e79",
                            bbox=dict(boxstyle="round,pad=0.22",
                                      fc="#ffe9a8" if is_best else "white",
                                      ec="#1f4e79", lw=0.8, alpha=0.95),
                            zorder=6)
        if not at_target:
            ax.plot([a, xs], [b, ys], color="gray", lw=1.0, zorder=3.5)
        ax.scatter([xs], [ys], marker="*", s=420, c=MAGENTA,
                   edgecolors="black", linewidths=1.0, zorder=5)
        opt_txt = (f"$({xs:.3g},\\ {ys:.3g})$ = target"
                   if at_target else f"$({xs:.3g},\\ {ys:.3g})$")
        ax.annotate(opt_txt, xy=(xs, ys), xytext=(10, 10),
                    textcoords="offset points", fontsize=10, fontweight="bold",
                    color=MAGENTA, zorder=6,
                    bbox=dict(boxstyle="round,pad=0.25", fc="white",
                              ec=MAGENTA, lw=0.9, alpha=0.95))

    draw_readout(radius, reveal, sol, banner)
    ax.set_xlim(AXIS_LO, AXIS_HI)
    ax.set_ylim(AXIS_LO, AXIS_HI)
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Toy QP:  inflate the circle until it first touches $P$")
    ax.legend(handles=LEGEND, loc="lower right", fontsize=8.5, framealpha=0.95)

    # ---------------- right: the bowl in 3D ------------------------------
    elev, azim = ax3.elev, ax3.azim       # keep the user's view angle
    ax3.clear()
    ax3.view_init(elev=elev, azim=azim)

    # cap the bowl so the interesting part (bowl meets polygon) fills the box
    if len(poly):
        fv = (poly[:, 0] - a) ** 2 + (poly[:, 1] - b) ** 2
        zcap = max(4.0, 1.15 * float(fv.max()))
    else:
        zcap = 40.0

    def frame_mask(x, y):
        return ((x < AXIS_LO) | (x > AXIS_HI) | (y < AXIS_LO) | (y > AXIS_HI))

    def ring3d(r, z, **kw):
        """A level circle at height z, clipped to the plot frame (matplotlib
        3D does not clip data to the axes box on its own)."""
        xs_ = a + r * np.cos(th)
        ys_ = b + r * np.sin(th)
        zs_ = np.where(frame_mask(xs_, ys_), np.nan, z)
        ax3.plot(xs_, ys_, zs_, **kw)

    # the bowl, on a POLAR grid around the target: the rim z = zcap is then a
    # clean circle instead of a ragged square-grid cut.  Kept coarse: at
    # alpha 0.42 every quad is depth-sorted per draw, and this redraws on
    # every slider step
    R_, TH_ = np.meshgrid(np.linspace(0.0, np.sqrt(zcap), 26),
                          np.linspace(0.0, 2.0 * np.pi, 49))
    X = a + R_ * np.cos(TH_)
    Y = b + R_ * np.sin(TH_)
    Z = np.where(frame_mask(X, Y), np.nan, R_ ** 2)
    ax3.plot_surface(X, Y, Z, cmap="Oranges", alpha=0.42, linewidth=0,
                     antialiased=True, rcount=49, ccount=26, vmin=0.0,
                     vmax=zcap)
    state["zcap"] = zcap          # for draw_movers' animation fast path
    th = np.linspace(0.0, 2.0 * np.pi, 181)
    ring3d(np.sqrt(zcap), zcap, color="#d97706", lw=1.2, alpha=0.6)

    # the feasible polygon, flat on the floor z = 0
    if len(poly):
        ax3.add_collection3d(Poly3DCollection(
            [[(vx, vy, 0.0) for vx, vy in poly]], facecolor="#bcd6f0",
            edgecolor="#1f4e79", linewidths=1.2, alpha=0.75))

    # faint level rings ON the bowl, matching the left panel's circles
    for rr in RING_R:
        if rr ** 2 <= zcap:
            ring3d(rr, rr ** 2, color="#9aa0a6", lw=0.7, ls="--")

    # the moving level set: the slicing plane z = r^2 and its ring on the bowl
    if radius is not None and radius > 1e-9 and radius ** 2 <= zcap:
        lvl = radius ** 2
        Px, Py = np.meshgrid([AXIS_LO, AXIS_HI], [AXIS_LO, AXIS_HI])
        ax3.plot_surface(Px, Py, np.full_like(Px, lvl, dtype=float),
                         color="#888888", alpha=0.12, shade=False)
        ring3d(radius, lvl, color=ORANGE, lw=2.6)

    # the target: bottom of the bowl, on the floor
    ax3.scatter([a], [b], [0.0], marker="x", s=70, color=TARGET_RED,
                linewidths=2.2, depthshade=False)

    if reveal:
        # SAME magenta point as the left panel: position on the floor, then
        # lifted to the bowl at height f* -- the objective value, made visible
        ax3.scatter([xs], [ys], [0.0], s=60, c=MAGENTA, edgecolors="black",
                    linewidths=0.8, depthshade=False)
        ax3.plot([xs, xs], [ys, ys], [0.0, fs], color=MAGENTA, lw=1.4,
                 ls="--")
        ax3.scatter([xs], [ys], [fs], s=130, c=MAGENTA, edgecolors="black",
                    linewidths=1.0, depthshade=False)
        ax3.text(xs, ys, fs + 0.06 * zcap, f"  f* = {fs:.3f}", color=MAGENTA,
                 fontsize=10, fontweight="bold",
                 path_effects=[pe.withStroke(linewidth=2.5,
                                             foreground="white")])

    ax3.set_xlim(AXIS_LO, AXIS_HI)
    ax3.set_ylim(AXIS_LO, AXIS_HI)
    ax3.set_zlim(0.0, zcap)
    ax3.set_box_aspect((1.0, 1.0, 0.72))
    ax3.set_xlabel("x")
    ax3.set_ylabel("y")
    ax3.set_zlabel("$f(x, y)$")
    ax3.set_title("the same QP in 3D:  bowl $z = (x{-}a)^2 + (y{-}b)^2$"
                  "   (drag to rotate)")

    fig.canvas.draw_idle()


def draw_movers(r):
    """Animation fast path: swap ONLY the moving level-set artists (2D circle,
    3D ring + slicing plane, radius readout) on top of the static scene drawn
    once by render().  A full render of both panels takes seconds-scale per
    frame; this keeps Play at animation speed."""
    a, b = s_a.val, s_b.val
    for art in state["movers"]:
        try:
            art.remove()
        except (ValueError, KeyError, AttributeError):
            pass
    state["movers"] = []
    if r is None or r <= 1e-9:
        return
    arts = state["movers"]
    arts.append(ax.add_patch(plt.Circle((a, b), r, fill=False, color=ORANGE,
                                        lw=2.6, zorder=3)))
    arts.append(ax.text(0.03, 0.64, f"r = {r:.3f}    z = r^2 = {r * r:.3f}",
                        transform=ax.transAxes, va="top", ha="left",
                        family="monospace", fontsize=8.6, color="#b45309"))
    lvl = r * r
    zcap = state["zcap"]
    if lvl <= zcap:
        th_ = np.linspace(0.0, 2.0 * np.pi, 121)
        xs_ = a + r * np.cos(th_)
        ys_ = b + r * np.sin(th_)
        zs_ = np.where((xs_ < AXIS_LO) | (xs_ > AXIS_HI)
                       | (ys_ < AXIS_LO) | (ys_ > AXIS_HI), np.nan, lvl)
        arts.extend(ax3.plot(xs_, ys_, zs_, color=ORANGE, lw=2.6))
        Px, Py = np.meshgrid([AXIS_LO, AXIS_HI], [AXIS_LO, AXIS_HI])
        arts.append(ax3.plot_surface(Px, Py,
                                     np.full_like(Px, lvl, dtype=float),
                                     color="#888888", alpha=0.12, shade=False))


def stop_anim():
    anim = state.get("anim")
    if anim is not None and anim.event_source is not None:
        anim.event_source.stop()
    state["anim"] = None


def redraw(_=None):
    # a slider move cancels a running Play: its radii were computed for the
    # OLD problem, and finishing them would draw a circle that is not tangent
    stop_anim()
    render(radius=None, reveal=state["revealed"])


def play(_):
    stop_anim()
    a, b = s_a.val, s_b.val
    r1, r2 = s_r1.val, s_r2.val
    _, _, fs, _ = solve_qp(a, b, r1, r2)
    rstar = float(np.sqrt(fs))
    if rstar < 1e-9:                       # target feasible: nothing to inflate
        state["revealed"] = True
        render(radius=None, reveal=True)
        return
    render(radius=None, reveal=False,      # static scene, drawn ONCE
           banner="inflating the level set...")
    radii = np.linspace(0.0, rstar, FRAMES)

    def update(k):
        if k == FRAMES - 1:
            draw_movers(None)              # the reveal renders its own circle
            state["revealed"] = True
            render(radius=radii[k], reveal=True)
        else:
            draw_movers(radii[k])

    state["anim"] = FuncAnimation(fig, update, frames=FRAMES, interval=40,
                                  repeat=False)
    fig.canvas.draw_idle()


for s in (s_a, s_b, s_r1, s_r2):
    s.on_changed(redraw)
btn_play.on_clicked(play)

redraw()

if __name__ == "__main__":
    plt.show()
