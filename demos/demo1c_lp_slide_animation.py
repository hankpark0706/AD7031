"""
Demo 1c -- Animated: sliding the objective plane to the LP optimum
==================================================================

This is the moving-picture version of Demo 1b.  It animates exactly what the
simplex method's geometry *means*:

  1. Pick a cost-coefficient vector  c = (cx, cy, cz)  with the sliders.
  2. Press PLAY.
  3. A plane perpendicular to c starts in the MIDDLE of the feasible region and
     slides in the direction of c.  At every position the plane is a "level set"
     -- every point on it has the same objective value c.x.
  4. The orange polygon is the slice of the feasible region cut by that plane;
     watch the objective value rise as the plane advances.
  5. The plane slides until it can go no further without leaving the region.
     That last contact is a single VERTEX -- the optimal solution.

That shrinking-slice-to-a-vertex is the whole idea of linear programming.

HOW TO PLAY
-----------
Run it:  python demo1c_lp_slide_animation.py
  * sliders cx, cy, cz -- set the objective direction (then press Play)
  * Play button        -- run the slide animation
  * Stop button        -- pause the slide wherever it currently is
  * drag with the mouse -- rotate at any time
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial import HalfspaceIntersection, ConvexHull

# ----------------------------------------------------------------------------
# Geometry: a unit cube [0,1]^3 with the single corner (1,1,1) sliced off.
# (Same simple polytope as Demo 1b, kept here so this file stands alone.)
# ----------------------------------------------------------------------------

def build_halfspaces(cut=2.3):
    rows = []
    for axis in range(3):
        e = np.zeros(3); e[axis] = 1.0
        rows.append((e.copy(), 1.0))            #  x_axis <= 1
        rows.append((-e.copy(), 0.0))           #  x_axis >= 0
    rows.append((np.array([1.0, 1.0, 1.0]), cut))   # x + y + z <= cut
    return rows


def order_loop(pts, normal):
    """Order coplanar points into a convex polygon (counter-clockwise)."""
    n = normal / np.linalg.norm(normal)
    ref = np.array([1.0, 0, 0]) if abs(n[0]) < 0.9 else np.array([0, 1.0, 0])
    u = np.cross(n, ref); u /= np.linalg.norm(u)
    w = np.cross(n, u)
    centroid = pts.mean(axis=0)
    ang = np.arctan2((pts - centroid) @ w, (pts - centroid) @ u)
    return pts[np.argsort(ang)]


def polytope_geometry(rows):
    H = np.array([[a[0], a[1], a[2], -c] for a, c in rows])
    verts = HalfspaceIntersection(H, np.array([0.5, 0.5, 0.5])).intersections
    faces = []
    for a, c in rows:
        a = np.asarray(a)
        on = verts[np.abs(verts @ a - c) < 1e-7]
        if len(on) >= 3:
            faces.append(order_loop(on, a))
    return verts, faces


def hull_edges(verts):
    """Unique edges (vertex-index pairs) of the convex polytope."""
    edges = set()
    for s in ConvexHull(verts).simplices:
        for a, b in ((s[0], s[1]), (s[1], s[2]), (s[2], s[0])):
            edges.add((min(a, b), max(a, b)))
    return list(edges)


ROWS = build_halfspaces()
VERTS, FACES = polytope_geometry(ROWS)
EDGES = hull_edges(VERTS)
CENTROID = VERTS.mean(axis=0)


def cross_section(level, c):
    """Polygon where the plane c.x = level cuts the polytope (or None)."""
    pts = []
    for i, j in EDGES:
        fi, fj = c @ VERTS[i] - level, c @ VERTS[j] - level
        if (fi <= 0 <= fj) or (fj <= 0 <= fi):
            if abs(fi - fj) < 1e-12:
                continue
            t = fi / (fi - fj)
            pts.append(VERTS[i] + t * (VERTS[j] - VERTS[i]))
    if len(pts) < 1:
        return None
    pts = np.array(pts)
    # drop near-duplicate points
    keep = [pts[0]]
    for p in pts[1:]:
        if min(np.linalg.norm(p - q) for q in keep) > 1e-7:
            keep.append(p)
    pts = np.array(keep)
    if len(pts) < 3:
        return pts                              # a point or an edge
    return order_loop(pts, c)


# ----------------------------------------------------------------------------
# Figure + widgets
# ----------------------------------------------------------------------------

fig = plt.figure(figsize=(9, 9))
ax = fig.add_subplot(111, projection="3d")
plt.subplots_adjust(left=0.0, bottom=0.24, right=1.0, top=0.92)

ax_cx = plt.axes([0.15, 0.15, 0.62, 0.025])
ax_cy = plt.axes([0.15, 0.10, 0.62, 0.025])
ax_cz = plt.axes([0.15, 0.05, 0.62, 0.025])
ax_play = plt.axes([0.80, 0.115, 0.12, 0.045])
ax_stop = plt.axes([0.80, 0.055, 0.12, 0.045])
s_cx = Slider(ax_cx, "cx", -1.0, 1.0, valinit=0.6, valstep=0.05)
s_cy = Slider(ax_cy, "cy", -1.0, 1.0, valinit=0.8, valstep=0.05)
s_cz = Slider(ax_cz, "cz", -1.0, 1.0, valinit=1.0, valstep=0.05)
btn_play = Button(ax_play, "Play")
btn_stop = Button(ax_stop, "Stop")

FRAMES = 80
state = {"anim": None}


def current_c():
    c = np.array([s_cx.val, s_cy.val, s_cz.val])
    return np.array([0.0, 0.0, 1.0]) if np.allclose(c, 0) else c


def draw(level, c, final=False):
    """Render one frame: polytope, sliding plane + slice, objective readout."""
    elev, azim = ax.elev, ax.azim
    ax.clear()
    ax.view_init(elev=elev, azim=azim)

    # the feasible polytope (flat translucent green)
    ax.add_collection3d(Poly3DCollection(list(FACES), facecolor="#5ab17a",
                        edgecolor="#1c5e36", linewidths=1.0, alpha=0.18))
    ax.scatter(VERTS[:, 0], VERTS[:, 1], VERTS[:, 2], s=12, c="#1c5e36")

    cn = c / np.linalg.norm(c)
    scores = VERTS @ c
    vstar = VERTS[int(np.argmax(scores))]
    lvl0, lvlmax = c @ CENTROID, scores.max()

    # faint full plane (so it reads as "a plane") + bold feasible slice
    centre = CENTROID + ((level - c @ CENTROID) / (c @ c)) * c
    half = 0.7
    ref = np.array([1.0, 0, 0]) if abs(cn[0]) < 0.9 else np.array([0, 1.0, 0])
    u = np.cross(cn, ref); u /= np.linalg.norm(u)
    w = np.cross(cn, u)
    sq = np.array([centre + s * half * u + t * half * w
                   for s, t in [(-1, -1), (1, -1), (1, 1), (-1, 1)]])
    ax.add_collection3d(Poly3DCollection([sq], facecolor="black", alpha=0.32,
                                         edgecolor="none"))
    sec = cross_section(level, c)
    if sec is not None and len(sec) >= 3:
        ax.add_collection3d(Poly3DCollection([sec], facecolor="black",
                            alpha=0.72, edgecolor="black", linewidths=2.0))

    # the cost-coefficient vector c, anchored at the centroid (the slide line)
    ax.quiver(*CENTROID, *cn, length=0.9, color="black", linewidth=2.5,
              arrow_length_ratio=0.18)
    ax.text(*(CENTROID + cn * 1.0), "  c", fontsize=12)

    # optimal vertex: faint until we arrive, then highlighted
    if final:
        ax.scatter(*vstar, s=200, c="#d62728", edgecolors="black",
                   depthshade=False, zorder=20)

    ax.set_xlim(-0.5, 1.5); ax.set_ylim(-0.5, 1.5); ax.set_zlim(-0.5, 1.5)
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
    ax.set_title("Sliding the objective plane along c   (drag to rotate)")

    frac = 0.0 if lvlmax <= lvl0 else (level - lvl0) / (lvlmax - lvl0)
    bar = "#" * int(round(20 * np.clip(frac, 0, 1)))
    tag = "\n*** OPTIMAL VERTEX REACHED ***" if final else ""
    ax.text2D(0.0, 1.0,
              f"objective  c . x = {level:6.3f}\n"
              f"start (centre)   = {lvl0:6.3f}\n"
              f"optimum (vertex) = {lvlmax:6.3f}\n"
              f"[{bar:<20}]{tag}",
              transform=ax.transAxes, va="top", ha="left", fontsize=10,
              family="monospace",
              bbox=dict(boxstyle="round", facecolor="#fff7d6", alpha=0.95))
    fig.canvas.draw_idle()


def static_preview(_=None):
    """Before pressing Play: show the plane parked in the middle of the region."""
    c = current_c()
    draw(c @ CENTROID, c)


def stop(_=None):
    """Halt the running animation, leaving the plane where it stopped."""
    anim = state.get("anim")
    if anim is not None and anim.event_source is not None:
        anim.event_source.stop()


def play(_):
    stop()                                      # cancel any running animation
    c = current_c()
    lvl0, lvlmax = c @ CENTROID, (VERTS @ c).max()
    levels = np.linspace(lvl0, lvlmax, FRAMES)

    def update(k):
        draw(levels[k], c, final=(k == FRAMES - 1))

    # keep a reference so the animation is not garbage-collected
    state["anim"] = FuncAnimation(fig, update, frames=FRAMES, interval=55,
                                  repeat=False)
    fig.canvas.draw_idle()


for s in (s_cx, s_cy, s_cz):
    s.on_changed(static_preview)
btn_play.on_clicked(play)
btn_stop.on_clicked(stop)

static_preview()

if __name__ == "__main__":
    plt.show()
