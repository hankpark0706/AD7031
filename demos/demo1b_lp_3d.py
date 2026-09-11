"""
Demo 1b -- 3D geometry of a linear program (the "cool polytope" picture)
========================================================================

A linear program's feasible region is a polyhedron -- the intersection of
half-spaces.  In 3D you can actually look at it.  This demo builds a simple
polytope -- a unit cube with one corner sliced off -- and shows the central
fact of linear programming geometry:

  * the feasible region is a convex polytope,
  * the objective c.x is constant on planes perpendicular to the vector c,
  * maximizing slides that plane along c until it last touches the polytope,
  * and that last contact point is always a VERTEX (corner) of the polytope.

This is the 3D analog of Demo 1's 2D picture, and it's why the simplex method
only ever has to look at corners.

HOW TO PLAY
-----------
Run it:  python demo1b_lp_3d.py
  * DRAG with the mouse to rotate the polytope in 3D.
  * Sliders cx, cy, cz set the objective direction c = (cx, cy, cz).
The green dot is the optimal vertex, the black arrow is c (the gradient), and
the translucent gray square is the supporting plane touching the polytope there.
Move the sliders and watch the optimum jump from corner to corner.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial import HalfspaceIntersection

# ----------------------------------------------------------------------------
# Build the polytope once: a unit cube [0,1]^3 with every corner truncated.
# Each half-space is written as  a . x <= c.
# ----------------------------------------------------------------------------

def build_halfspaces(cut=2.3):
    rows = []                                   # each row: (a_vector, c)
    # the 6 box faces of the unit cube [0,1]^3
    for axis in range(3):
        e = np.zeros(3); e[axis] = 1.0
        rows.append((e.copy(), 1.0))            #  x_axis <= 1
        rows.append((-e.copy(), 0.0))           # -x_axis <= 0  (x_axis >= 0)
    # one slanted cut that slices off the single corner (1,1,1)
    rows.append((np.array([1.0, 1.0, 1.0]), cut))   # x + y + z <= cut
    return rows


def polytope_geometry(rows):
    """Return (vertices, faces) where faces is a list of ordered vertex loops."""
    # scipy wants half-spaces as [a | b] meaning a.x + b <= 0, i.e. b = -c.
    H = np.array([[a[0], a[1], a[2], -c] for a, c in rows])
    interior = np.array([0.5, 0.5, 0.5])        # a point strictly inside
    verts = HalfspaceIntersection(H, interior).intersections

    faces = []
    for a, c in rows:
        a = np.asarray(a)
        on = verts[np.abs(verts @ a - c) < 1e-7]          # vertices on this face
        if len(on) < 3:
            continue
        faces.append(order_loop(on, a))
    return verts, faces


def order_loop(pts, normal):
    """Order coplanar points into a convex polygon (counter-clockwise)."""
    n = normal / np.linalg.norm(normal)
    ref = np.array([1.0, 0, 0]) if abs(n[0]) < 0.9 else np.array([0, 1.0, 0])
    u = np.cross(n, ref); u /= np.linalg.norm(u)
    w = np.cross(n, u)
    centroid = pts.mean(axis=0)
    ang = np.arctan2((pts - centroid) @ w, (pts - centroid) @ u)
    return pts[np.argsort(ang)]


ROWS = build_halfspaces()
VERTS, FACES = polytope_geometry(ROWS)


def supporting_square(center, normal, half=0.5):
    """A square patch lying in the plane through `center` with given `normal`."""
    n = normal / np.linalg.norm(normal)
    ref = np.array([1.0, 0, 0]) if abs(n[0]) < 0.9 else np.array([0, 1.0, 0])
    u = np.cross(n, ref); u /= np.linalg.norm(u)
    w = np.cross(n, u)
    return np.array([center + s * half * u + t * half * w
                     for s, t in [(-1, -1), (1, -1), (1, 1), (-1, 1)]])


# ----------------------------------------------------------------------------
# Interactive figure
# ----------------------------------------------------------------------------

fig = plt.figure(figsize=(9, 8.5))
ax = fig.add_subplot(111, projection="3d")
plt.subplots_adjust(left=0.0, bottom=0.22, right=1.0, top=0.90)

ax_cx = plt.axes([0.15, 0.13, 0.70, 0.025])
ax_cy = plt.axes([0.15, 0.08, 0.70, 0.025])
ax_cz = plt.axes([0.15, 0.03, 0.70, 0.025])
s_cx = Slider(ax_cx, "cx", -1.0, 1.0, valinit=0.6, valstep=0.05)
s_cy = Slider(ax_cy, "cy", -1.0, 1.0, valinit=0.8, valstep=0.05)
s_cz = Slider(ax_cz, "cz", -1.0, 1.0, valinit=1.0, valstep=0.05)


def redraw(_=None):
    c = np.array([s_cx.val, s_cy.val, s_cz.val])
    if np.allclose(c, 0):
        c = np.array([0.0, 0.0, 1.0])

    # keep the current view angle when sliders move
    elev, azim = ax.elev, ax.azim
    ax.clear()
    ax.view_init(elev=elev, azim=azim)

    # the polytope faces
    poly = Poly3DCollection(list(FACES), facecolor="#5ab17a", edgecolor="#1c5e36",
                            linewidths=1.2, alpha=0.30)
    ax.add_collection3d(poly)
    ax.scatter(VERTS[:, 0], VERTS[:, 1], VERTS[:, 2], s=18, c="#1c5e36")

    # optimal vertex = argmax c.x over the vertices (an LP optimum is a vertex)
    scores = VERTS @ c
    vstar = VERTS[int(np.argmax(scores))]
    ax.scatter(*vstar, s=160, c="#2ca02c", edgecolors="black", depthshade=False,
               zorder=10)

    # supporting plane touching the polytope at the optimum (perpendicular to c)
    sq = supporting_square(vstar, c)
    ax.add_collection3d(Poly3DCollection([sq], facecolor="black", alpha=0.55,
                                         edgecolor="black", linewidths=1.5))

    # the objective/gradient vector c, drawn from the optimal vertex
    cn = c / np.linalg.norm(c)
    ax.quiver(*vstar, *cn, length=0.6, color="black", linewidth=2.5,
              arrow_length_ratio=0.25)
    ax.text(*(vstar + cn * 0.75), "  c", fontsize=12)

    # wider frame than the polytope so it sits at ~half the block, not filling it
    ax.set_xlim(-0.5, 1.5); ax.set_ylim(-0.5, 1.5); ax.set_zlim(-0.5, 1.5)
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
    ax.set_title("Maximize  c . x  over the polytope   (drag to rotate)")
    ax.text2D(0.0, 1.0,
              f"c = ({c[0]:.2f}, {c[1]:.2f}, {c[2]:.2f})\n"
              f"optimal vertex = ({vstar[0]:.2f}, {vstar[1]:.2f}, {vstar[2]:.2f})\n"
              f"objective value = {scores.max():.3f}",
              transform=ax.transAxes, va="top", ha="left", fontsize=10,
              family="monospace",
              bbox=dict(boxstyle="round", facecolor="#fff7d6", alpha=0.95))
    fig.canvas.draw_idle()


for s in (s_cx, s_cy, s_cz):
    s.on_changed(redraw)
redraw()

if __name__ == "__main__":
    plt.show()
