"""
Two-Stage Data-Driven LAP -- interactive matplotlib dashboard
=============================================================

Same spirit as the toy TSP demo: one self-contained script, a graph on the left,
the optimization formulation on the right, sliders + buttons at the bottom.
No web server, no cvxpy -- the SAA model is solved with scipy/HiGHS in lap_core.

THE PROBLEM
-----------
Pre-position relief facilities BEFORE a disaster (stage 1), then ship supplies
AFTER demand is revealed (stage 2).  The plan is chosen by Sample Average
Approximation: the expected recourse cost E[Q] is replaced by an average over
N sampled disaster scenarios.

HOW TO PLAY
-----------
Run it:  python viz/lap_dashboard.py
  * n nodes slider   -- size of the network (2 = a tiny by-hand example)
  * facilities p     -- how many locations you may open  (adds  sum_j x_j = p)
  * train scenarios  -- how many sampled disasters the plan is fitted on
  * Solve            -- choose where to build (green) + how much to stock
  * Simulate         -- draw one UNSEEN disaster, ship supplies, show the cost
  * New network      -- resample node positions / costs / demands
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from matplotlib.lines import Line2D
from matplotlib.offsetbox import TextArea, HPacker, VPacker, AnnotationBbox

# colour scheme shared by the formulation and its symbol key
COL_DEC = "#1f5fd0"      # decision variables (x, r, y, q, s)
COL_UNC = "#c0392b"      # uncertainty (xi)
COL_PAR = "#1a1a1a"      # everything else (parameters, operators)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lap_core import (Instance, solve_saa, solve_ev, eval_oos,
                      oos_cost_distribution, DEFAULT_BUDGET_PER_NODE)

seed_rng = np.random.default_rng(0)


def letter(k):
    return chr(ord("A") + k) if k < 26 else chr(ord("A") + k // 26 - 1) + chr(ord("A") + k % 26)


# ── application state ─────────────────────────────────────────────────────────
# mode: "formula" -> right panel shows the model;  "dist" -> overlaid distributions
# state["dists"] accumulates one entry per decision so they can be compared; it is
# cleared only when the network (instance) changes, not when the decision changes.
state = {"inst": None, "sol": None, "sim": None, "dists": [], "curve": None,
         "mode": "formula"}

PALETTE = ["#ffb547", "#4f9dff", "#27c08a", "#d65db1", "#8a7dff"]
DEF_C, DEF_F = 100.0, 10.0       # default uniform facility cost c_j and reserve cost f_j


def regenerate(n):
    state["inst"] = Instance(n_nodes=int(n), n_scenarios=1200,
                             seed=int(seed_rng.integers(1, 99999)))
    state["inst"].set_economics(facility_cost=DEF_C, reserve_cost=DEF_F)   # uniform c, f
    state["sol"] = None
    state["sim"] = None
    state["dists"] = []           # new instance -> distributions no longer comparable
    state["curve"] = None
    state["mode"] = "formula"


regenerate(8)

# ── figure / panels ───────────────────────────────────────────────────────────
fig = plt.figure(figsize=(14.5, 8))
fig.canvas.manager.set_window_title("Two-Stage Data-Driven LAP — SAA dashboard")
gs = fig.add_gridspec(1, 2, width_ratios=[1.2, 0.8], wspace=0.06)
ax = fig.add_subplot(gs[0])      # network
ax_f = fig.add_subplot(gs[1])    # formulation
plt.subplots_adjust(left=0.03, bottom=0.205, right=0.985, top=0.94)
# lift the network panel's bottom to leave a strip for its legend (clear of the sliders)
ax.set_position([0.035, 0.35, 0.575, 0.57])
# small histogram axes used only by the "View data" mode (lower-right area)
ax_hist = fig.add_axes([0.665, 0.235, 0.30, 0.20])
ax_hist.set_visible(False)

# In the "formula" mode the right panel splits into a MODEL box (top, formulation +
# symbol key) and a RESULTS box (bottom, the live plan / disaster readout). The other
# modes (distribution / data / curve) use the full ax_f and hide ax_res.
AXF_FULL = tuple(ax_f.get_position().bounds)
_fx, _fy, _fw, _fh = AXF_FULL
_RES_H, _GAP = 0.235, 0.022
AXF_RES = (_fx, _fy, _fw, _RES_H)
AXF_MODEL = (_fx, _fy + _RES_H + _GAP, _fw, _fh - _RES_H - _GAP)
ax_res = fig.add_axes(AXF_RES)
ax_res.set_visible(False)


def _box(a, fc, ec):
    a.clear(); a.set_facecolor(fc); a.set_xticks([]); a.set_yticks([])
    a.set_xlim(0, 1); a.set_ylim(0, 1)
    for sp in a.spines.values():
        sp.set_visible(True); sp.set_edgecolor(ec); sp.set_linewidth(1.4)

# widgets ----------------------------------------------------------------------
# left block: 4 rows x 2 columns of sliders.
#   left  column = structure & budget (n, p, training data, B)
#   right column = unit costs          (facility c, reserve f, shortage u, surplus o)
_R = (0.225, 0.18, 0.135, 0.09)            # four row y-positions
s_n = Slider(plt.axes([0.085, _R[0], 0.15, 0.02]), "n nodes", 2, 20, valinit=8, valstep=1)
s_p = Slider(plt.axes([0.085, _R[1], 0.15, 0.02]), "max facilities", 1, 8, valinit=2, valstep=1)
s_t = Slider(plt.axes([0.085, _R[2], 0.15, 0.02]), "training data", 1, 200, valinit=15, valstep=1)
s_b = Slider(plt.axes([0.085, _R[3], 0.15, 0.02]), "budget B", 40, 760, valinit=160, valstep=10)
s_c = Slider(plt.axes([0.37, _R[0], 0.15, 0.02]), "facility cost c", 0, 8000, valinit=DEF_C, valstep=100)
s_f = Slider(plt.axes([0.37, _R[1], 0.15, 0.02]), "reserve cost f", 10, 200, valinit=DEF_F, valstep=5)
s_u = Slider(plt.axes([0.37, _R[2], 0.15, 0.02]), "shortage u", 10, 200, valinit=60, valstep=5)
s_o = Slider(plt.axes([0.37, _R[3], 0.15, 0.02]), "surplus o", 0, 30, valinit=1, valstep=1)
# bottom row: action buttons + trials
b_solve = Button(plt.axes([0.05, 0.03, 0.078, 0.05]), "Solve", color="#bcd6ff", hovercolor="#8fbcff")
b_sim = Button(plt.axes([0.133, 0.03, 0.142, 0.05]), "Simulate disaster", color="#ffd79a", hovercolor="#ffc46b")
b_new = Button(plt.axes([0.28, 0.03, 0.10, 0.05]), "New network")
b_data = Button(plt.axes([0.385, 0.03, 0.083, 0.05]), "View data", color="#ffe3b0", hovercolor="#ffd089")
b_cmp = Button(plt.axes([0.473, 0.03, 0.093, 0.05]), "SAA vs mean", color="#cfe8d6", hovercolor="#a8d8b6")
for _b in (b_sim, b_new, b_data, b_cmp):
    _b.label.set_fontsize(8.5)
s_k = Slider(plt.axes([0.61, 0.115, 0.14, 0.022]), "trials", 1, 1000, valinit=300, valstep=1)
b_dist = Button(plt.axes([0.61, 0.035, 0.14, 0.058]), "Run trials", color="#cdbcff", hovercolor="#b39bff")
b_curve = Button(plt.axes([0.77, 0.035, 0.175, 0.058]), "Cost vs N curve", color="#bfe3c8", hovercolor="#98d4a8")
BAND_MODES = ["IQR (25-75%)", "±1 std", "mean only"]
BAND_LABELS = ["band: IQR 25-75", "band: ±1 std", "band: mean only"]
band_state = {"i": 0}
b_band = Button(plt.axes([0.77, 0.10, 0.175, 0.05]), BAND_LABELS[0],
                color="#e7e0c0", hovercolor="#d8cfa0")
b_band.label.set_fontsize(9)

YLORRD = plt.cm.YlOrRd
GREEN, FLOW, SHORT, CAND = "#27c08a", "#1f77b4", "#ff3b3b", "#6b7a93"
RESERVE_REF = 250.0          # reserve (units) at which an opened-facility node hits max size


def p_value():
    return min(int(s_p.val), int(s_n.val))


# ── progress overlay (so long computations don't look frozen) ─────────────────
progress_txt = fig.text(0.82, 0.57, "", ha="center", va="center", fontsize=14,
                        color="#143a5a", zorder=60, visible=False,
                        bbox=dict(boxstyle="round,pad=0.8", fc="#fff3c4",
                                  ec="#d9b24a", lw=1.6))


def show_progress(msg):
    progress_txt.set_text(msg)
    progress_txt.set_visible(True)
    try:
        fig.canvas.draw()          # force an immediate repaint mid-callback
        fig.canvas.flush_events()
    except Exception:
        pass


def hide_progress():
    progress_txt.set_visible(False)


def _dot(color, label, ec="black", hollow=False, line=False):
    if line:
        return Line2D([0], [0], color=color, lw=3.2, label=label)
    return Line2D([0], [0], marker="o", linestyle="None", markersize=11,
                  markerfacecolor="none" if hollow else color, markeredgecolor=ec,
                  markeredgewidth=2.4 if hollow else 0.7, label=label)


def legend_handles():
    """Context-aware legend on the network panel explaining node colours/symbols."""
    sol, sim = state["sol"], state["sim"]
    if sim:
        handles = [
            _dot(GREEN, "facility (depot)"),
            _dot(YLORRD(0.25), "low demand"),
            _dot(YLORRD(0.55), "medium demand"),
            _dot(YLORRD(0.90), "high demand"),
            _dot(SHORT, "unmet demand (shortage)", ec=SHORT, hollow=True),
            _dot(FLOW, "relief flow", line=True),
        ]
    elif sol:
        handles = [_dot(GREEN, "facility (opened, size ∝ reserve)"),
                   _dot(CAND, "candidate site")]
    else:
        handles = [_dot(CAND, "candidate site")]
    # placed BELOW the network axes (outside the plot) so it never hides the graph
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.02),
              ncol=min(3, len(handles)), fontsize=8.5, framealpha=0.95,
              borderpad=0.6, columnspacing=1.4, handletextpad=0.5)


# ── drawing ───────────────────────────────────────────────────────────────────
def draw_network():
    inst, sol, sim = state["inst"], state["sol"], state["sim"]
    n, pos = inst.I, inst.pos
    ax.clear()

    # arcs (directed) ----------------------------------------------------------
    label = n <= 10
    seen = set()
    for k, (a, b) in enumerate(inst.line_mat):
        flow = sim["flows"][k] if sim else 0.0
        on = flow > 0.05
        ax.annotate("", xy=pos[b], xytext=pos[a],
                    arrowprops=dict(arrowstyle="-|>",
                                    color="#1f77b4" if on else "#cdd5e2",
                                    lw=(1.3 + 4 * min(flow / 40, 1)) if on else 0.8,
                                    shrinkA=11, shrinkB=13,
                                    connectionstyle="arc3,rad=0.09"),
                    zorder=4 if on else 1)
        key = frozenset((int(a), int(b)))
        if label and key not in seen:
            seen.add(key)
            mx, my = (pos[a] + pos[b]) / 2
            ax.text(mx, my, f"{inst.trans[k]:.1f}", ha="center", va="center",
                    fontsize=7.5, color="#7a869c", zorder=2,
                    bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="#dde3ec", lw=0.5))

    # node colours / sizes -----------------------------------------------------
    opened = set(i for i in range(n) if sol and sol["x"][i])
    sizes, colors = [], []
    dmax = max(sim["demand"]) if sim else 1.0
    for i in range(n):
        if i in opened:
            # ABSOLUTE reserve scale (not ÷ budget) so more reserve = bigger node,
            # consistently across different budgets. RESERVE_REF caps the growth.
            sizes.append(280 + 1500 * min(sol["r"][i] / RESERVE_REF, 1.0))
            colors.append("#27c08a")
        elif sim:
            sizes.append(300)
            colors.append(YLORRD(0.25 + 0.6 * sim["demand"][i] / dmax))
        else:
            sizes.append(300)
            colors.append("#6b7a93")
    ax.scatter(pos[:, 0], pos[:, 1], s=sizes, c=colors,
               edgecolors="black", linewidths=0.7, zorder=5)

    # shortage rings -----------------------------------------------------------
    if sim:
        sh = [i for i in range(n) if sim["shortages"][i] > 0.05]
        if sh:
            ax.scatter(pos[sh, 0], pos[sh, 1], s=[sizes[i] + 380 for i in sh],
                       facecolors="none", edgecolors="#ff3b3b", linewidths=2.6, zorder=6)

    # labels -------------------------------------------------------------------
    for i in range(n):
        ax.text(pos[i, 0], pos[i, 1], letter(i), color="white", ha="center",
                va="center", fontsize=11, fontweight="bold", zorder=7)
        sub = ""
        if sim and sim["shortages"][i] > 0.05:
            sub = f"short {sim['shortages'][i]:.0f}"
        elif sim:
            sub = f"d={sim['demand'][i]:.0f}"
        elif i in opened:
            sub = f"r={sol['r'][i]:.0f}"
        if sub:
            ax.text(pos[i, 0], pos[i, 1] - 0.058, sub, ha="center", va="top",
                    fontsize=8.5, color="#b03030" if "short" in sub else "#33405a",
                    fontweight="bold", zorder=7)

    ax.set_xlim(-0.05, 1.05); ax.set_ylim(-0.08, 1.05)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    legend_handles()
    if state["mode"] == "data":
        ax.set_title(f"The {int(s_t.val)} training scenarios are shown on the right →",
                     fontsize=12)
    elif state["mode"] == "curve" and state["curve"]:
        ax.set_title(f"Plan fitted on N={state['curve']['Ns'][-1]} scenarios  —  "
                     "convergence shown on the right", fontsize=12)
    elif state["mode"] == "dist" and state["dists"]:
        ax.set_title("Each location decision → its own out-of-sample cost "
                     "distribution (right)", fontsize=12)
    elif sim:
        ax.set_title(f"Disaster #{sim['scenario']}  —  total cost = {sim['total_cost']:.0f},"
                     f"  shortage = {sim['total_shortage']:.1f}", fontsize=12)
    elif sol:
        ax.set_title(f"Facility plan  —  opened {', '.join(letter(i) for i in sorted(opened))}"
                     f"   ·   reserve {sum(sol['r']):.0f}/{inst.budget:.0f}", fontsize=12)
    else:
        ax.set_title("Candidate network & transport costs  —  press  Solve", fontsize=12)


def _minop(sub, fs):
    """A 'min' operator with its decision variables stacked UNDERNEATH (bold blue)."""
    return VPacker(align="center", pad=0, sep=0, children=[
        TextArea(r"$\min$", textprops=dict(color=COL_PAR, fontsize=fs)),
        TextArea(sub, textprops=dict(color=COL_DEC, fontsize=fs * 0.7)),
    ])


def _eqline(items, fs):
    """One equation line: horizontally pack colour-coded mathtext segments.
    Each item is either a (text, colour) tuple or an already-built offsetbox."""
    children = []
    for it in items:
        if isinstance(it, tuple):
            t, c = it
            children.append(TextArea(t, textprops=dict(color=c, fontsize=fs)))
        else:
            children.append(it)
    return HPacker(align="baseline", pad=0, sep=0, children=children)


def draw_formulation():
    _box(ax_f, "#fff7d6", "#d9c179")
    B, D, U = COL_PAR, COL_DEC, COL_UNC
    fs = 8.5
    hdr = lambda s: TextArea(s, textprops=dict(fontsize=9, fontweight="bold", color="#222"))
    gap = lambda: TextArea(" ", textprops=dict(fontsize=3))
    rows = [
        TextArea("Two-stage LAP   (Sample Average Approximation)",
                 textprops=dict(fontsize=10, fontweight="bold", color="#222")),
        gap(),
        hdr("Stage 1  (here-and-now, before the disaster)"),
        _eqline([_minop(r"$\mathbf{x},\mathbf{r}$", fs),
                 (r"$\ \sum_j c_j$", B), (r"$x_j$", D),
                 (r"$+\sum_j f_j$", B), (r"$r_j$", D),
                 (r"$+\frac{1}{N}\sum_{n=1}^{N} Q($", B), (r"$\mathbf{x},\mathbf{r}$", D),
                 (r"$;$", B), (r"$\hat{\boldsymbol{\xi}}^{\,n}$", U), (r"$)$", B)], fs),
        _eqline([(r"$\mathrm{s.t.}\ \sum_{j\in\mathcal{N}}$", B), (r"$r_j$", D),
                 (r"$\leq B,\ \ \sum_{j\in\mathcal{N}}$", B), (r"$x_j$", D),
                 (r"$\leq p$", B)], fs),
        _eqline([(r"$\quad\ \ $", B), (r"$r_j$", D), (r"$\leq B\,$", B), (r"$x_j$", D),
                 (r"$,\ $", B), (r"$x_j$", D), (r"$\in\{0,1\},\ $", B), (r"$r_j$", D),
                 (r"$\geq0\ \ \forall j\!\in\!\mathcal{N}$", B)], fs),
        gap(),
        hdr("Stage 2  (recourse for scenario n, after demand is seen)"),
        _eqline([(r"$Q($", B), (r"$\mathbf{x},\mathbf{r}$", D), (r"$;$", B),
                 (r"$\hat{\boldsymbol{\xi}}^{\,n}$", U), (r"$)=$", B),
                 _minop(r"$\mathbf{y},\mathbf{q},\mathbf{s}$", fs),
                 (r"$\ \sum_a d_a$", B), (r"$y_a$", D),
                 (r"$+\sum_j u_j$", B), (r"$q_j$", D),
                 (r"$+\sum_j o_j$", B), (r"$s_j$", D)], fs),
        _eqline([(r"$\mathrm{s.t.}\ (\mathrm{in}_j\!-\!\mathrm{out}_j)+$", B), (r"$q_j$", D),
                 (r"$-$", B), (r"$s_j$", D), (r"$=$", B), (r"$\hat{\xi}^{\,n}_j$", U),
                 (r"$-$", B), (r"$r_j$", D), (r"$\ \ \forall j\!\in\!\mathcal{N}$", B)], fs),
        _eqline([(r"$\quad 0\leq$", B), (r"$y_a$", D),
                 (r"$\leq\bar y\ \forall a\!\in\!\mathcal{A};\ \ 0\leq$", B),
                 (r"$q_j$", D), (r"$\leq$", B), (r"$\hat{\xi}^{\,n}_j$", U), (r"$,\ $", B),
                 (r"$s_j$", D), (r"$\geq0$", B)], fs),
    ]
    box = VPacker(align="left", pad=0, sep=2, children=rows)
    ax_f.add_artist(AnnotationBbox(box, (0.03, 0.985), xycoords="axes fraction",
                                   box_alignment=(0, 1), frameon=False))

    # ── description / symbol key (mathtext, colour-consistent with the model) ──
    dfs = 7.5
    ax_f.text(0.03, 0.345,
              "DECISION VARIABLES  (what we choose)\n"
              r"  s1: $\mathbf{x}$ open facilities, $\mathbf{r}$ reserves;   "
              r"s2: $\mathbf{y}$ flows, $\mathbf{q}$ shortages, $\mathbf{s}$ surpluses",
              transform=ax_f.transAxes, va="top", ha="left", fontsize=dfs, color=COL_DEC)
    ax_f.text(0.03, 0.265,
              "UNCERTAINTY  (seen only after stage 1)\n"
              r"  $\hat{\boldsymbol{\xi}}^{\,n}$ = demand vector in scenario $n$"
              r"   ($\hat{\xi}^{\,n}_j$ at node $j$)",
              transform=ax_f.transAxes, va="top", ha="left", fontsize=dfs, color=COL_UNC)
    ax_f.text(0.03, 0.185,
              "PARAMETERS  (given numbers)\n"
              r"  $c_j$ open, $f_j$ reserve, $B$ budget, $p$ cap;  "
              r"$d_a$ transport, $u_j$ short, $o_j$ surplus, $N$ #scen",
              transform=ax_f.transAxes, va="top", ha="left", fontsize=dfs, color=COL_PAR)
    ax_f.text(0.03, 0.105,
              "SETS & INDICES\n"
              r"  graph $G=(\mathcal{N},\mathcal{A})$:  $\mathcal{N}$ nodes,  $\mathcal{A}$ arcs;   "
              r"$j\in\mathcal{N}$, $a\in\mathcal{A}$, $n=1,\dots,N$",
              transform=ax_f.transAxes, va="top", ha="left", fontsize=dfs, color=COL_PAR)


def draw_results_panel():
    """Bottom-right RESULTS box (formula mode): the live plan or disaster outcome,
    styled with colour-tinted boxes + a green/red shortage chip (like Module 1)."""
    inst, sol, sim = state["inst"], state["sol"], state["sim"]
    _box(ax_res, "#eef4fb", "#9bb8de")

    def chip(y, short):
        if short < 0.05:
            txt, c = "✓ no shortage", "#1a8a3a"
        else:
            txt, c = "⚠ short %.0f" % short, "#c0392b"
        ax_res.text(0.965, y, txt, fontsize=8.5, fontweight="bold", va="top", ha="right",
                    color="white", bbox=dict(boxstyle="round,pad=0.3", fc=c, ec=c, lw=1.2))

    if sim:
        x = np.asarray(sol["x"], float); r = np.asarray(sol["r"], float)
        flows = np.asarray(sim["flows"], float)
        shorts = np.asarray(sim["shortages"], float)
        surp = np.asarray(sim["surplus"], float)
        c_fr = float(inst.c @ x + inst.f @ r)         # facility opening + reserves
        c_tr = float(inst.trans @ flows)              # transport
        c_sh = float(inst.v_under @ shorts)           # shortage penalty
        c_su = float(inst.v_over @ surp)              # surplus
        ax_res.text(0.035, 0.94, "STAGE 2 — disaster #%d" % sim["scenario"],
                    fontsize=9, fontweight="bold", va="top", color=COL_UNC)
        chip(0.94, sim["total_shortage"])
        ax_res.text(0.04, 0.73, "Out-of-sample cost   %.0f" % sim["total_cost"],
                    fontsize=10.5, fontweight="bold", va="top", color="#7a2e12",
                    bbox=dict(boxstyle="round,pad=0.35", fc="#ffe7c4", ec="#e0992f", lw=1.6))
        bd = [(r"Facility + reserve   $\sum_j (c_j x_j + f_j r_j)$", c_fr),
              (r"Transport   $\sum_a d_a y_a$", c_tr),
              (r"Shortage   $\sum_j u_j q_j$", c_sh),
              (r"Surplus   $\sum_j o_j s_j$", c_su)]
        yb = 0.49
        for lbl, val in bd:
            ax_res.text(0.05, yb, lbl, fontsize=7.7, va="top", color="#333")
            ax_res.text(0.95, yb, "%.0f" % val, fontsize=8.4, fontweight="bold",
                        va="top", ha="right", color="#16324f")
            yb -= 0.122
    elif sol:
        opened = [i for i in range(inst.I) if sol["x"][i]]
        sites = ", ".join("%s(%.0f)" % (letter(i), sol["r"][i]) for i in opened) or "none"
        ax_res.text(0.035, 0.92, "STAGE-1 PLAN   (p≤%d, N=%d)" % (p_value(), int(s_t.val)),
                    fontsize=9, fontweight="bold", va="top", color=COL_DEC)
        ax_res.text(0.04, 0.66, "build  %s" % sites, fontsize=10, fontweight="bold",
                    va="top", color="#16324f",
                    bbox=dict(boxstyle="round,pad=0.4", fc="#dCE9FB", ec="#5b8fd0", lw=1.6))
        ax_res.text(0.04, 0.34,
                    "first-stage %.0f   +   expected recourse %.0f"
                    % (sol["first_stage_cost"], sol["expected_recourse"]),
                    fontsize=8.3, va="top", color="#333")
        ax_res.text(0.04, 0.2, "SAA objective  %.0f" % sol["obj"], fontsize=10.5,
                    fontweight="bold", va="top", color="#1a5f33",
                    bbox=dict(boxstyle="round,pad=0.35", fc="#e3f3e9", ec="#7bc59a", lw=1.6))
    else:
        ax_res.text(0.5, 0.55, "press  Solve  to choose where to build\nand how much to stock",
                    fontsize=9.5, va="center", ha="center", color="#566", style="italic")


def draw_distribution():
    """Right panel: OVERLAID out-of-sample cost distributions, one per decision,
    as normalized densities so they are directly comparable."""
    dists = state["dists"]
    ax_f.clear()
    ax_f.set_facecolor("white")
    for sp in ax_f.spines.values():
        sp.set_visible(True)

    # shared bins across all decisions (clip extreme tail for readability)
    allc = np.concatenate([d["costs"] for d in dists])
    lo, hi = allc.min(), np.percentile(allc, 99)
    if hi <= lo:
        hi = lo + 1.0
    bins = np.linspace(lo, hi, 36)

    for d in dists:
        mean = float(d["costs"].mean())
        sh = 100.0 * float(np.mean(d["shorts"] > 0.05))
        lbl = f"{d['label']}\n     mean {mean:.0f},  P(short) {sh:.0f}%"
        ax_f.hist(d["costs"], bins=bins, density=True, histtype="stepfilled",
                  alpha=0.4, color=d["color"], edgecolor=d["color"],
                  linewidth=1.5, label=lbl, zorder=2)
        ax_f.axvline(mean, color=d["color"], lw=1.8, ls="--", zorder=3)

    ax_f.set_xlabel("out-of-sample total cost", fontsize=10)
    ax_f.set_ylabel("probability density", fontsize=10)
    ax_f.tick_params(labelsize=8)
    ax_f.set_title("Out-of-sample cost distribution — by decision", fontsize=12)
    ax_f.legend(loc="upper right", title="dashed line = mean", title_fontsize=8,
                framealpha=0.95, prop={"size": 8})


def draw_data():
    """Right panel: the training scenarios shown as the data structure itself —
    a table of demand-per-node vectors (≈ one {node: demand} dict per scenario),
    plus a histogram of the per-scenario total demand Σ."""
    inst = state["inst"]
    N = int(s_t.val)
    I = inst.I
    M = [np.asarray(inst.demand[s], float) for s in range(N)]
    nodes = [letter(i) for i in range(I)]
    ax_f.clear()
    ax_f.axis("off")
    ax_f.set_facecolor("white")
    RED, OK = "#c0392b", "#16324f"

    ax_f.text(0.0, 1.0,
              "Training data = a list of N demand scenarios.\n"
              "Each scenario gives the demand at every node (region):",
              transform=ax_f.transAxes, va="top", ha="left", fontsize=9.5, color="#333")
    ex = ", ".join(f"{nodes[i]}:{M[0][i]:.0f}" for i in range(min(I, 8)))
    if I > 8:
        ex += ", ..."
    ax_f.text(0.0, 0.905, f"e.g.  scenario #1 = {{ {ex} }}",
              transform=ax_f.transAxes, va="top", ha="left", fontsize=7.5,
              color="#a05a00", family="monospace")

    colw = 4
    fs = 8.5 if I <= 10 else (7.5 if I <= 14 else 6.5)
    line_h = fs * 1.35 / (fig.get_figheight() * 72 * 0.735)   # axes units per row
    header = " scen │" + "".join(f"{n:>{colw}}" for n in nodes) + "  │   Σ"
    sep = "──────┼" + "─" * (colw * I) + "──┼──────"
    y = 0.82
    ax_f.text(0.0, y, header, transform=ax_f.transAxes, va="top", ha="left",
              family="monospace", fontsize=fs, color="#555"); y -= line_h
    ax_f.text(0.0, y, sep, transform=ax_f.transAxes, va="top", ha="left",
              family="monospace", fontsize=fs, color="#aaa"); y -= line_h
    show = min(N, 9)
    for s in range(show):
        vals = "".join(f"{M[s][i]:>{colw}.0f}" for i in range(I))
        tot = float(M[s].sum())
        over = tot > inst.budget
        row = f" #{s+1:<3} │{vals}  │{tot:>5.0f}"
        ax_f.text(0.0, y, row, transform=ax_f.transAxes, va="top", ha="left",
                  family="monospace", fontsize=fs, color=RED if over else OK,
                  fontweight="bold" if over else "normal")
        y -= line_h
    if N > show:
        ax_f.text(0.0, y, f"  ...  ({N - show} more scenarios)",
                  transform=ax_f.transAxes, va="top", ha="left",
                  family="monospace", fontsize=fs, color="#aaa")
        y -= line_h
    ax_f.text(0.0, y - 0.012,
              f"red = scenario total Σ exceeds budget {inst.budget:.0f}  "
              f"(shortage unavoidable).",
              transform=ax_f.transAxes, va="top", ha="left", fontsize=8, color=RED)

    # ── histogram of the per-scenario total demand Σ ──────────────────────────
    totals = np.array([float(M[s].sum()) for s in range(N)])
    ax_hist.clear()
    ax_hist.set_visible(True)
    for sp in ax_hist.spines.values():
        sp.set_visible(True)
    nb = int(np.clip(N // 3, 6, 20))
    counts, edges, patches = ax_hist.hist(totals, bins=nb, edgecolor="white")
    half = (edges[1] - edges[0]) / 2 if len(edges) > 1 else 0.0
    for patch, left in zip(patches, edges[:-1]):       # red bars beyond budget
        patch.set_facecolor("#e8635a" if left + half > inst.budget else "#ffb547")
    ax_hist.axvline(inst.budget, color=RED, lw=2, label=f"budget {inst.budget:.0f}")
    ax_hist.axvline(totals.mean(), color="#1f77b4", lw=1.6, ls="--",
                    label=f"mean {totals.mean():.0f}")
    pct = 100.0 * float(np.mean(totals > inst.budget))
    ax_hist.set_title(f"Distribution of total demand Σ   "
                      f"({pct:.0f}% exceed budget)", fontsize=8.5)
    ax_hist.set_xlabel("total demand per scenario  Σ", fontsize=8)
    ax_hist.set_ylabel("count", fontsize=8)
    ax_hist.tick_params(labelsize=7)
    ax_hist.legend(fontsize=7, loc="upper right", framealpha=0.9)


def draw_curve():
    """Right panel: out-of-sample cost vs amount of training data N."""
    cv = state["curve"]
    Ns, oos = cv["Ns"], cv["oos"]
    ax_f.clear()
    ax_f.set_facecolor("white")
    for sp in ax_f.spines.values():
        sp.set_visible(True)
    mode = BAND_MODES[band_state["i"]]
    if mode == "IQR (25-75%)":
        lo, hi, blabel = cv["p25"], cv["p75"], "middle 50% of outcomes (25–75th pct)"
    elif mode == "±1 std":
        lo = [max(0.0, m - s) for m, s in zip(oos, cv["std"])]
        hi = [m + s for m, s in zip(oos, cv["std"])]
        blabel = "out-of-sample cost  ±1 std"
    else:
        lo = hi = None
    if lo is not None:
        ax_f.fill_between(Ns, lo, hi, color="#d9772b", alpha=0.18,
                          linewidth=0, label=blabel)
    ax_f.plot(Ns, oos, "o-", color="#d9772b", lw=2.4, ms=7,
              label="out-of-sample cost (mean)")
    ax_f.set_xscale("log")
    ax_f.set_xticks(Ns); ax_f.set_xticklabels([str(n) for n in Ns], fontsize=7.5)
    ax_f.minorticks_off()
    ax_f.tick_params(labelsize=8)
    ax_f.set_xlabel("training scenarios  N   (log scale)", fontsize=10)
    ax_f.set_ylabel("total cost", fontsize=10)
    ax_f.set_title(f"More data → better:  OOS cost vs N   (p={cv['p']})", fontsize=12)
    ax_f.grid(True, alpha=0.25)
    ax_f.legend(fontsize=9, loc="upper right", framealpha=0.95)
    ax_f.text(0.5, 0.03,
              "N=1 = deterministic plan.  Line = mean out-of-sample cost; the shaded\n"
              "band shows the spread of realized costs (pick the band type below the plot).",
              transform=ax_f.transAxes, ha="center", va="bottom", fontsize=8,
              color="#777", style="italic")


def redraw():
    draw_network()
    is_formula = not (state["mode"] == "data"
                      or (state["mode"] == "curve" and state["curve"])
                      or (state["mode"] == "dist" and state["dists"]))
    ax_f.set_position(AXF_MODEL if is_formula else AXF_FULL)
    ax_res.set_visible(is_formula)
    if state["mode"] == "data":
        draw_data()
    else:
        ax_hist.set_visible(False)
        if state["mode"] == "curve" and state["curve"]:
            draw_curve()
        elif state["mode"] == "dist" and state["dists"]:
            draw_distribution()
        else:
            draw_formulation()
            draw_results_panel()
    fig.canvas.draw_idle()


# ── callbacks ─────────────────────────────────────────────────────────────────
def apply_econ():
    """Push the budget / cost sliders onto the current instance."""
    state["inst"].set_economics(budget=s_b.val, shortage_penalty=s_u.val,
                                surplus_cost=s_o.val, facility_cost=s_c.val,
                                reserve_cost=s_f.val)


def on_econ(_):
    apply_econ()
    state["sol"] = None
    state["sim"] = None
    state["dists"] = []              # economics changed -> a different problem
    state["curve"] = None
    state["mode"] = "formula"
    redraw()


def on_n(_):
    regenerate(int(s_n.val))
    # auto-rescale the budget default for the new network size; set_val fires
    # on_econ, which re-applies B/u/o to the fresh instance and redraws.
    s_b.set_val(round(DEFAULT_BUDGET_PER_NODE * int(s_n.val)))


def on_decision(_):
    state["sol"] = None
    state["sim"] = None
    state["curve"] = None            # p/train changed -> curve no longer valid
    if state["mode"] != "data":      # stay in data view so it refreshes as N changes
        state["mode"] = "formula"    # keep state["dists"] so decisions can be compared
    redraw()


def on_data(_):
    state["mode"] = "data"
    redraw()


def on_solve(_):
    show_progress("Solving the\ninteger program…")
    try:
        inst = state["inst"]
        train = list(range(int(s_t.val)))
        state["sol"] = solve_saa(inst, train, n_facilities=p_value())
        state["sim"] = None
        state["mode"] = "formula"        # keep prior distributions for comparison
    finally:
        hide_progress()
    redraw()


def on_sim(_):
    if state["sol"] is None:
        ax.set_title("Press  Solve  first, then Simulate a disaster", fontsize=12)
        fig.canvas.draw_idle()
        return
    inst = state["inst"]
    test = range(int(s_t.val), inst.num_data)
    sid = int(seed_rng.choice(list(test)))
    state["sim"] = eval_oos(inst, state["sol"]["x"], state["sol"]["r"], sid)
    state["sim"]["scenario"] = sid
    state["mode"] = "formula"
    redraw()


def on_trials(_):
    if state["sol"] is None:
        ax.set_title("Press  Solve  first, then Run trials", fontsize=12)
        fig.canvas.draw_idle()
        return
    inst = state["inst"]
    K = int(s_k.val)
    pool = np.arange(int(s_t.val), inst.num_data)        # unseen (out-of-sample)
    ids = seed_rng.choice(pool, size=K, replace=K > len(pool))
    show_progress(f"Running {K}\nout-of-sample trials…")
    try:
        costs, shorts = oos_cost_distribution(inst, state["sol"]["x"],
                                              state["sol"]["r"], ids)
    finally:
        hide_progress()

    # Decision identity = location AND reserve (two plans with the same sites but
    # different reserve are genuinely different decisions -> separate curves).
    opened = [i for i in range(inst.I) if state["sol"]["x"][i]]
    label = f"p={p_value()}: " + ", ".join(
        f"{letter(i)}(r={state['sol']['r'][i]:.0f})" for i in opened)
    # replace any earlier run of the SAME decision, then append (cap at 5 overlays)
    state["dists"] = [d for d in state["dists"] if d["label"] != label]
    color = PALETTE[len(state["dists"]) % len(PALETTE)]
    state["dists"].append({"costs": costs, "shorts": shorts, "K": K,
                           "label": label, "color": color})
    state["dists"] = state["dists"][-5:]
    state["sim"] = None
    state["mode"] = "dist"
    redraw()


def on_curve(_):
    if state["inst"] is None:
        return
    inst = state["inst"]
    p = p_value()
    Ns = [1, 2, 3, 6, 12, 30, 60]
    # Large FIXED out-of-sample test set -> precise per-solution cost estimate.
    # Several training-subset repeats per N: the line is the average performance,
    # the band is ±1.5 std of that performance ACROSS training samples — i.e. how
    # much your result depends on which data you happened to get. It shrinks with N.
    pool = np.arange(0, 600)                                  # training pool
    test = list(range(600, min(inst.num_data, 950)))          # large fixed disjoint OOS set
    oos, std, p25, p75, last = [], [], [], [], None
    try:
        for k, N in enumerate(Ns, 1):
            show_progress(f"Computing learning curve…\n"
                          f"step {k} / {len(Ns)}   (N = {N})")
            reps = 4 if N <= 6 else 1
            all_costs = []
            for _r in range(reps):
                train = seed_rng.choice(pool, size=N, replace=False)
                sol = solve_saa(inst, train, n_facilities=p)
                c, _ = oos_cost_distribution(inst, sol["x"], sol["r"], test)
                all_costs.append(c); last = sol
            allc = np.concatenate(all_costs)        # the realized OOS costs themselves
            q1, q3 = np.percentile(allc, [25, 75])
            oos.append(float(allc.mean())); std.append(float(allc.std()))
            p25.append(float(q1)); p75.append(float(q3))
    finally:
        hide_progress()
    state["sol"] = last              # show a representative large-N plan
    state["sim"] = None
    state["curve"] = {"Ns": Ns, "oos": oos, "std": std,
                      "p25": p25, "p75": p75, "p": p}
    state["mode"] = "curve"
    redraw()


def on_compare(_):
    """Solve SAA and the deterministic 'plan-for-the-mean' model on the same
    training data, then overlay their out-of-sample cost distributions."""
    inst = state["inst"]
    p = p_value()
    N = int(s_t.val)
    train = list(range(N))
    show_progress("Solving SAA  and  the\ndeterministic (mean) model…")
    try:
        sol_saa = solve_saa(inst, train, n_facilities=p)
        sol_ev = solve_ev(inst, train, n_facilities=p)
        test = list(range(N, min(inst.num_data, N + 700)))     # unseen disasters
        c_saa, s_saa = oos_cost_distribution(inst, sol_saa["x"], sol_saa["r"], test)
        c_ev, s_ev = oos_cost_distribution(inst, sol_ev["x"], sol_ev["r"], test)
    finally:
        hide_progress()
    state["sol"] = sol_saa            # show the SAA plan on the network
    state["sim"] = None

    def _dec(sol):                    # opened sites + reserves, like Run trials
        return ", ".join(f"{letter(i)}(r={sol['r'][i]:.0f})"
                         for i in range(inst.I) if sol["x"][i])
    state["dists"] = [
        {"costs": c_ev, "shorts": s_ev, "K": len(test), "color": "#8a7dff",
         "label": f"mean-value model:  {_dec(sol_ev)}"},
        {"costs": c_saa, "shorts": s_saa, "K": len(test), "color": "#ffb547",
         "label": f"SAA (N={N}):  {_dec(sol_saa)}"},
    ]
    state["mode"] = "dist"
    redraw()


def on_new(_):
    regenerate(int(s_n.val))
    apply_econ()                     # keep the current B/u/o on the fresh network
    redraw()


s_n.on_changed(on_n)
s_p.on_changed(on_decision)
s_t.on_changed(on_decision)
s_b.on_changed(on_econ)
s_c.on_changed(on_econ)
s_f.on_changed(on_econ)
s_u.on_changed(on_econ)
s_o.on_changed(on_econ)
b_solve.on_clicked(on_solve)
b_sim.on_clicked(on_sim)
b_new.on_clicked(on_new)
b_data.on_clicked(on_data)
b_cmp.on_clicked(on_compare)
b_dist.on_clicked(on_trials)
b_curve.on_clicked(on_curve)


def on_band(_):
    band_state["i"] = (band_state["i"] + 1) % len(BAND_MODES)
    b_band.label.set_text(BAND_LABELS[band_state["i"]])
    if state["mode"] == "curve" and state["curve"]:
        redraw()
    else:
        fig.canvas.draw_idle()


b_band.on_clicked(on_band)

redraw()

if __name__ == "__main__":
    plt.show()
