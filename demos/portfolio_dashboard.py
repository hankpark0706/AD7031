"""Single-stage portfolio optimization under uncertainty -- teaching dashboard.

Run with::

    python portfolio_dashboard.py

Companion to PORTFOLIO_MODEL_NOTE.pdf and a sibling of the IP course's LAP and
classification dashboards (same house style: blue = decisions, red =
uncertainty, pale-yellow formulation panel, pale-blue results panel).

The organizing idea: ONE feasible set (the budget simplex) and ONE random
return, optimized under FOUR risk measures -- this is a single-stage
stochastic program, not a multi-stage one, and the dashboard exists to make
CVaR, VaR, and chance constraints concrete.

Workflow is deliberately one model at a time: pick a model (top-right tab),
read its formulation, press Solve, and read its own solution -- the sector
weights as a bar chart on the left, and the decision vector / objective
breakdown on the right. Comparing all four models side by side (cost or
weights) is a secondary, later step, tucked into a small button group.
"""

from __future__ import annotations

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.offsetbox import AnnotationBbox, HPacker, OffsetImage, TextArea, VPacker
from matplotlib.widgets import Button, Slider

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from portfolio_core import (  # noqa: E402
    SectorMarket,
    empirical_var_cvar,
    evaluate_portfolio,
    portfolio_losses,
    solve_portfolio,
)

# ── house-style colours ─────────────────────────────────────────────────────
COL_DEC = "#1f5fd0"   # decision variables (blue)
COL_UNC = "#c0392b"   # uncertainty / data (red)
COL_PAR = "#1a1a1a"   # parameters / operators (near-black)
COL_GOOD = "#1a8a3a"  # "below the running mean" during a live simulation
INK = "#1a1a1a"
PANEL_MODEL = ("#fff7d6", "#d9c179")   # pale yellow / gold
PANEL_RES = ("#eef4fb", "#9bb8de")     # pale blue / steel

MODEL_KEYS = ("mean_variance", "cvar", "var", "chance")
MODEL_LABEL = {"mean_variance": "Mean–variance", "cvar": "CVaR",
               "var": "VaR", "chance": "Chance constraint"}
MODEL_CLASS = {"mean_variance": "convex QP", "cvar": "convex LP",
               "var": "mixed integer program", "chance": "mixed integer program"}
MODEL_COLOR = {"mean_variance": "#2f6fd0", "cvar": "#1a8a3a",
               "var": "#e8820c", "chance": "#7e3ff2"}

market = SectorMarket()
seed_rng = np.random.default_rng(20260722)


def _new_seed() -> int:
    return int(seed_rng.integers(1, 2_000_000_000))


state = {
    "focus": "cvar",
    "train_seed": 41,
    "train_bank": market.sample(1000, 41),
    "test_bank": market.sample(6000, 9041),
    "solutions": {k: None for k in MODEL_KEYS},   # key -> PortfolioSolution|None
    "errors": {k: None for k in MODEL_KEYS},      # key -> str|None
    "solved_config": {k: None for k in MODEL_KEYS},  # key -> config-tuple|None
    "snapshots": [None, None],  # two pinned (model, params, weights, OOS metrics) for Compare
    "sweep": None,  # {param, key, grid, step, orig_valstep, results:[(v,weights|None,err|None)]}
    "mode": "solution",  # solution | distribution | vsN | data | sim | compare2
    "test_metrics": None,
    "learning": None,
    "message": "Press  Solve  to see this model's optimal portfolio.",
    "trial_stage": "idle",  # idle | armed | running | paused
    "sim": None,            # {"key", "i", "total"} while trial_stage != idle
    "_timer": None,
}


def is_stale(key: str) -> bool:
    """True if `key` has never been solved, or was solved under old inputs."""
    cfg = state["solved_config"].get(key)
    return cfg is None or cfg != current_config(key)


# ── input readers ────────────────────────────────────────────────────────────
def training_returns() -> np.ndarray:
    return state["train_bank"][: int(s_n.val)]


def epsilon() -> float:
    return float(s_eps.val) / 100.0


def target_return() -> float:
    return float(s_target.val) / 100.0


def loss_limit() -> float:
    return float(s_gamma.val) / 100.0


def current_config(key: str) -> tuple:
    """The slider inputs that actually matter for `key` -- e.g. the return
    target R only affects mean-variance; CVaR/VaR/Chance are pure risk
    problems and don't depend on it."""
    seed = int(state["train_seed"])
    if key == "mean_variance":
        return (round(target_return(), 5), seed)
    if key in ("cvar", "var"):
        return (int(s_n.val), round(epsilon(), 4), seed)
    return (int(s_n.val), round(epsilon(), 4), round(loss_limit(), 4), seed)


# ── figure & panels (classify-style explicit rectangles) ─────────────────────
fig = plt.figure(figsize=(14, 7.6))
fig.canvas.manager.set_window_title("Portfolio Selection — OUU dashboard")
AX_MAIN_RECT = [0.055, 0.30, 0.52, 0.60]
# distribution view needs a taller x-axis label + bigger tick labels than the
# ~30pt gap below AX_MAIN_RECT leaves before the slider row -- shrink the
# axes for that view only (redraw() restores AX_MAIN_RECT for every other mode).
AX_MAIN_RECT_DIST = [0.10, 0.36, 0.475, 0.54]
ax_main = fig.add_axes(AX_MAIN_RECT)
ax_model = fig.add_axes([0.635, 0.458, 0.34, 0.472])
ax_res = fig.add_axes([0.635, 0.115, 0.34, 0.335])
# two-snapshot comparison view: stacked inside AX_MAIN_RECT's footprint,
# hidden except in "compare2" mode (see redraw())
ax_cmp_a = fig.add_axes([0.055, 0.63, 0.52, 0.27])
ax_cmp_b = fig.add_axes([0.055, 0.30, 0.52, 0.27])
ax_cmp_a.set_visible(False)
ax_cmp_b.set_visible(False)


def _box(a, fc, ec):
    a.clear(); a.set_facecolor(fc); a.set_xticks([]); a.set_yticks([])
    a.set_xlim(0, 1); a.set_ylim(0, 1)
    for sp in a.spines.values():
        sp.set_visible(True); sp.set_edgecolor(ec); sp.set_linewidth(1.4)


# ── model selector buttons (top of the right column) ─────────────────────────
model_buttons = {}
for i, key in enumerate(MODEL_KEYS):
    bx = fig.add_axes([0.635 + i * 0.085, 0.945, 0.079, 0.038])
    b = Button(bx, MODEL_LABEL[key], color="#e7ebf2", hovercolor="#cdd8ea")
    b.label.set_fontsize(9.6)
    b.label.set_fontweight("bold")
    b.label.set_color(MODEL_COLOR[key])
    model_buttons[key] = b


# ── sliders (two columns, bottom-left) ───────────────────────────────────────
# The 4 sliders that actually feed solve_portfolio() (see current_config()) each
# get a small ▶ sweep button carved out of their right edge; OOS trials/sim
# speed don't affect solving, so they stay full-width with no button.
s_n = Slider(fig.add_axes([0.10, 0.225, 0.125, 0.02]), "training N",
             20, 1000, valinit=120, valstep=10, valfmt="%d")
s_target = Slider(fig.add_axes([0.10, 0.18, 0.125, 0.02]), "target return R",
                  0.30, 0.68, valinit=0.55, valstep=0.01, valfmt="%.2f%%")
s_eps = Slider(fig.add_axes([0.10, 0.135, 0.125, 0.02]), "tail prob ε",
               5, 95, valinit=5, valstep=5, valfmt="%d%%")
s_gamma = Slider(fig.add_axes([0.41, 0.225, 0.125, 0.02]), "chance constraint γ",
                 -1.0, 10, valinit=6.0, valstep=0.1, valfmt="%.1f%%")
s_trials = Slider(fig.add_axes([0.41, 0.18, 0.15, 0.02]), "OOS trials",
                  100, 2000, valinit=600, valstep=100, valfmt="%d")
s_speed = Slider(fig.add_axes([0.41, 0.135, 0.15, 0.02]), "sim speed",
                 1, 60, valinit=15, valstep=1, valfmt="%d/tick")

SWEEP_SLIDERS = {"n": s_n, "target": s_target, "eps": s_eps, "gamma": s_gamma}
SWEEP_LABELS = {"n": "training N", "target": "target return R",
                "eps": "tail prob ε", "gamma": "chance constraint γ"}
_sweep_btn_specs = {"n": [0.27, 0.225], "target": [0.27, 0.18],
                    "eps": [0.27, 0.135], "gamma": [0.585, 0.225]}
SWEEP_BUTTONS = {}
for _pk, (_bx, _by) in _sweep_btn_specs.items():
    _b = Button(fig.add_axes([_bx, _by, 0.018, 0.02]), "▶",
               color="#d9c9f2", hovercolor="#c2a8e8")
    _b.label.set_fontsize(9)
    SWEEP_BUTTONS[_pk] = _b


# ── action buttons (bottom row) ──────────────────────────────────────────────
# Primary, per-model workflow (left group) vs. secondary, two-snapshot
# comparison (right group, muted colours -- pin a result, change sliders,
# pin a second, then compare their out-of-sample distributions side by side).
_specs_primary = [
    ("Solve", "#bcd6ff", "#8fbcff"),
    ("Run trials", "#cdbcff", "#b39bff"),
    ("Perf vs N", "#bfe3c8", "#98d4a8"),
    ("View data", "#ffe3b0", "#ffd089"),
    ("New sample", "#e3e7ee", "#cfd6e0"),
]
_specs_secondary = [
    ("Snapshot", "#e3e5e8", "#d0d3d8"),
    ("Compare", "#e3e5e8", "#d0d3d8"),
]
action_buttons = {}
_bx = 0.045
for label, fc, hc in _specs_primary:
    w = 0.076
    b = Button(fig.add_axes([_bx, 0.05, w, 0.05]), label, color=fc, hovercolor=hc)
    b.label.set_fontsize(8.3)
    action_buttons[label] = b
    _bx += w + 0.003
_bx += 0.02
for label, fc, hc in _specs_secondary:
    w = 0.076
    b = Button(fig.add_axes([_bx, 0.05, w, 0.05]), label, color=fc, hovercolor=hc)
    b.label.set_fontsize(7.6)
    action_buttons[label] = b
    _bx += w + 0.003


progress = fig.text(0.315, 0.60, "", ha="center", va="center", fontsize=13,
                    color="#143a5a", zorder=90, visible=False,
                    bbox=dict(boxstyle="round,pad=0.7", fc="#fff3c4",
                              ec="#d9b24a", lw=1.8))


def show_progress(msg):
    progress.set_text(msg); progress.set_visible(True)
    try:
        fig.canvas.draw(); fig.canvas.flush_events()
    except Exception:
        pass


def hide_progress():
    progress.set_visible(False)


# ── formulation panel (color-coded, offsetbox packing) ───────────────────────
def _minop(symbol, sub, fs):
    return VPacker(align="center", pad=0, sep=0, children=[
        TextArea(symbol, textprops=dict(color=COL_PAR, fontsize=fs)),
        TextArea(sub, textprops=dict(color=COL_DEC, fontsize=fs * 0.62))])


def _param_line(label, pre_tex, xi_tex, post_tex, fs):
    """'{label}  {pre}[xi]{post}', with xi in the uncertainty colour and
    everything else (label, operators, mu/Sigma) in the parameter colour --
    a parameter is DEFINED FROM the random return, so xi keeps its red."""
    kids = [
        TextArea(label, textprops=dict(color=COL_PAR, fontsize=fs)),
        TextArea(f"  ${pre_tex}$", textprops=dict(color=COL_PAR, fontsize=fs)),
        TextArea(f"${xi_tex}$", textprops=dict(color=COL_UNC, fontsize=fs)),
        TextArea(f"${post_tex}$", textprops=dict(color=COL_PAR, fontsize=fs)),
    ]
    return HPacker(align="baseline", pad=0, sep=0, children=kids)


def _eqline(items, fs):
    kids = []
    for it in items:
        if isinstance(it, tuple):
            kids.append(TextArea(it[0], textprops=dict(color=it[1], fontsize=fs)))
        else:
            kids.append(it)
    return HPacker(align="baseline", pad=0, sep=0, children=kids)


def _formula_rows(key, fs=10):
    B, D, U = COL_PAR, COL_DEC, COL_UNC
    if key == "mean_variance":
        return [
            _eqline([_minop(r"$\min$", r"$\mathbf{x}$", fs), (r"$\ \ $", B),
                     (r"$\mathbf{x}^{\!\top}$", D), (r"$\Sigma\,$", B),
                     (r"$\mathbf{x}$", D)], fs),
            _eqline([(r"$\mathrm{s.t.}\ \ \boldsymbol{\mu}^{\!\top}$", B),
                     (r"$\mathbf{x}$", D), (r"$\ \geq R$", B)], fs),
            _eqline([(r"$\qquad \mathbf{e}^{\!\top}$", B), (r"$\mathbf{x}$", D),
                     (r"$=1,\ $", B), (r"$\mathbf{x}$", D), (r"$\geq 0$", B)], fs),
        ]
    if key == "cvar":
        return [
            _eqline([_minop(r"$\min$", r"$\mathbf{x},\beta,\mathbf{u}$", fs),
                     (r"$\ \ \beta+\dfrac{1}{\epsilon N}\!\sum_s$", B),
                     (r"$u_s$", D)], fs),
            _eqline([(r"$\mathrm{s.t.}\ \ $", B), (r"$u_s$", D), (r"$\geq -$", B),
                     (r"$(\boldsymbol{\xi}^{s})^{\!\top}$", U), (r"$\mathbf{x}$", D),
                     (r"$-\beta,\ \ $", B), (r"$u_s$", D), (r"$\geq 0$", B)], fs),
            _eqline([(r"$\qquad \mathbf{e}^{\!\top}$", B), (r"$\mathbf{x}$", D),
                     (r"$=1,\ $", B), (r"$\mathbf{x}$", D), (r"$\geq 0$", B)], fs),
        ]
    if key == "var":
        return [
            _eqline([_minop(r"$\min$", r"$\mathbf{x},\gamma,\mathbf{z}$", fs),
                     (r"$\ \ \gamma$", D)], fs),
            _eqline([(r"$\mathrm{s.t.}\ -$", B), (r"$(\boldsymbol{\xi}^{s})^{\!\top}$", U),
                     (r"$\mathbf{x}$", D), (r"$\leq$", B), (r"$\gamma$", D),
                     (r"$+M$", B), (r"$z_s$", D)], fs),
            _eqline([(r"$\qquad \sum_s$", B), (r"$z_s$", D),
                     (r"$\leq \lfloor\epsilon N\rfloor,\ $", B), (r"$z_s$", D),
                     (r"$\in\{0,1\}$", B)], fs),
            _eqline([(r"$\qquad \mathbf{e}^{\!\top}$", B), (r"$\mathbf{x}$", D),
                     (r"$=1,\ $", B), (r"$\mathbf{x}$", D), (r"$\geq 0$", B)], fs),
        ]
    return [
        _eqline([_minop(r"$\min$", r"$\mathbf{x},\mathbf{z}$", fs),
                 (r"$\ \ -\hat{\boldsymbol{\mu}}^{\!\top}$", B), (r"$\mathbf{x}$", D)], fs),
        _eqline([(r"$\mathrm{s.t.}\ -$", B), (r"$(\boldsymbol{\xi}^{s})^{\!\top}$", U),
                 (r"$\mathbf{x}$", D), (r"$\leq \gamma+M$", B), (r"$z_s$", D)], fs),
        _eqline([(r"$\qquad \sum_s$", B), (r"$z_s$", D),
                 (r"$\leq \lfloor\epsilon N\rfloor,\ $", B), (r"$z_s$", D),
                 (r"$\in\{0,1\}$", B)], fs),
        _eqline([(r"$\qquad \mathbf{e}^{\!\top}$", B), (r"$\mathbf{x}$", D),
                 (r"$=1,\ $", B), (r"$\mathbf{x}$", D), (r"$\geq 0$", B)], fs),
    ]


_MODEL_TITLE = {
    "mean_variance": "Mean–variance portfolio",
    "cvar": "CVaR portfolio",
    "var": "VaR portfolio",
    "chance": "Chance-constrained portfolio",
}


def draw_model_panel():
    _box(ax_model, *PANEL_MODEL)
    key = state["focus"]
    ax_model.text(0.04, 0.965, _MODEL_TITLE[key], transform=ax_model.transAxes,
                  va="top", fontsize=13, fontweight="bold", color="#222")
    ax_model.text(0.965, 0.895, MODEL_CLASS[key].upper(), transform=ax_model.transAxes,
                  va="top", ha="right", fontsize=8.4, fontweight="bold",
                  color="#8a4b08",
                  bbox=dict(boxstyle="round,pad=0.25", fc="#ffe7a8", ec="#d19a2a"))
    rows = _formula_rows(key, fs=12.5)
    box = VPacker(align="left", pad=0, sep=4, children=rows)
    ax_model.add_artist(AnnotationBbox(
        box, (0.045, 0.86), xycoords="axes fraction",
        box_alignment=(0, 1), frameon=False))

    ax_model.text(0.04, 0.40, "DECISION", transform=ax_model.transAxes,
                  fontsize=10.2, fontweight="bold", color=COL_DEC)
    ax_model.text(0.04, 0.34,
                  r"$\mathbf{x}$ = sector weights"
                  + (r";  $\beta,u_s$ = CVaR aux." if key == "cvar" else
                     (r";  $\gamma,z_s$ = tail aux." if key in ("var", "chance") else "")),
                  transform=ax_model.transAxes, fontsize=9.2, color=COL_DEC)
    if key == "mean_variance":
        ax_model.text(0.04, 0.265, "PARAMETERS", transform=ax_model.transAxes,
                      fontsize=10.2, fontweight="bold", color=COL_PAR)
        mean_row = _param_line("Mean:", r"\boldsymbol{\mu} = \mathbb{E}[",
                               r"\boldsymbol{\xi}", "]", 9.2)
        ax_model.add_artist(AnnotationBbox(
            mean_row, (0.04, 0.205), xycoords="axes fraction",
            box_alignment=(0, 0.5), frameon=False))
        cov_row = _param_line("Covariance:", r"\Sigma = \mathbb{E}[",
                              r"\boldsymbol{\xi}\boldsymbol{\xi}^{\top}", "]", 9.2)
        ax_model.add_artist(AnnotationBbox(
            cov_row, (0.04, 0.145), xycoords="axes fraction",
            box_alignment=(0, 0.5), frameon=False))
        note = f"Uses population μ, Σ (training-N doesn't apply).  Target R = {target_return()*100:.2f}%."
        ax_model.text(0.04, 0.085, note, transform=ax_model.transAxes, fontsize=8.6,
                      color="#555", style="italic")
    else:
        ax_model.text(0.04, 0.265, "UNCERTAINTY", transform=ax_model.transAxes,
                      fontsize=10.2, fontweight="bold", color=COL_UNC)
        ax_model.text(0.04, 0.205,
                      r"$\boldsymbol{\xi}^{s}$ = monthly sector returns, scenario $s$",
                      transform=ax_model.transAxes, fontsize=9.2, color=COL_UNC)

        # Same PARAMETERS treatment as mean-variance's -- bold header, plain
        # "label:  value" content -- just N/epsilon/gamma instead of a
        # mu/Sigma formula (no xi to colour here, so plain text is enough).
        # Kept to a single combined line so the pitch below UNCERTAINTY
        # matches mean-variance's own (≈0.06) instead of cramming two.
        ax_model.text(0.04, 0.145, "PARAMETERS", transform=ax_model.transAxes,
                      fontsize=10.2, fontweight="bold", color=COL_PAR)
        if key == "chance":
            line = (f"Scenarios:  N = {int(s_n.val)},  ε = {epsilon():.0%}     "
                    f"Loss limit:  γ = {loss_limit()*100:.1f}%")
        else:
            line = f"Scenarios:  N = {int(s_n.val)}     Tail probability:  ε = {epsilon():.0%}"
        ax_model.text(0.04, 0.085, line, transform=ax_model.transAxes,
                      fontsize=9.2, color=COL_PAR)

    lesson = {
        "mean_variance": "Variance is symmetric — it also penalizes upside.",
        "cvar": "Convex repair of VaR: an LP, and CVaR ≥ VaR always.",
        "chance": "P(loss ≤ γ) ≥ 1−ε  is exactly a VaR constraint.",
    }.get(key)
    if lesson:
        ax_model.text(0.04, 0.032, lesson, transform=ax_model.transAxes, fontsize=8.8,
                      color=MODEL_COLOR[key], style="italic", fontweight="bold")


def _invest_str(weights: np.ndarray) -> str:
    return "Invest  x* = [" + ", ".join(f"{100*w:.0f}%" for w in weights) + "]"


def _perf_name(key: str, eps: float) -> str:
    """Mathtext for the objective's name (with tail-probability subscript where relevant)."""
    if key == "mean_variance":
        return r"$\mathbb{V}$"  # "Var" reads too easily as VaR
    if key == "cvar":
        return rf"$\mathrm{{CVaR}}_{{{100*eps:.0f}\%}}$"
    if key == "var":
        return rf"$\mathrm{{VaR}}_{{{100*eps:.0f}\%}}$"
    return r"$\mathbb{E}$"


def _perf_note(key: str, sol) -> str:
    if key == "mean_variance":
        return "in-sample = out-of-sample — solved on the true μ, Σ, not a sample"
    if key == "cvar":
        return f"optimized VaR threshold β = {100*sol.threshold:.2f}%   (CVaR ≥ VaR always)"
    if key == "var":
        return f"breaches {sol.breaches}/{sol.allowed_breaches} allowed  (⌊εN⌋)"
    return ""  # chance: see _chance_insample_row, placed beside the Mean box instead


def _chance_insample_row(sol, fs=12.5, color="#16324f"):
    """'Prob[ −ξᵀx* ≥ γ=X% ] = breaches/N = pct%' -- the in-sample breach rate
    written out as the actual probability statement, in place of the old
    prose caption ("breaches 18/18" read like a budget count, not a
    probability -- and used the allowed-breach budget as its denominator,
    not the scenario count N). Same font size/weight/colour as the Mean box's
    _formula_row text (fs=12.5, navy, bold) since this is meant to read as a
    second metric at the same level as Mean, not a footnote."""
    n_train = len(sol.train_losses)
    g_pct = 100 * sol.threshold
    pct = 100 * sol.breaches / n_train
    kids = [
        TextArea("Prob[ ", textprops=dict(color=color, fontsize=fs, fontweight="bold")),
        *_xi_x_star_kids(fs, negate=True, minus_color=color),
        TextArea(f" ≥ γ={g_pct:.1f}% ]  =  {sol.breaches}/{n_train}  =  {pct:.2f}%",
                 textprops=dict(color=color, fontsize=fs, fontweight="bold")),
    ]
    return HPacker(align="baseline", pad=0, sep=0.5, children=kids)


def _xi_x_star_kids(fs: float, rotation: float = 0, negate: bool = False,
                     minus_color: str = "#16324f"):
    """The (optionally negated) ξᵀx* triplet as plain TextAreas -- ξ in the
    uncertainty colour, x* in the decision colour, star never auto-shrunk the
    way mathtext '^\\star' is. `rotation` lets the same triplet be packed into
    a sideways axis label. `negate` prepends a plain '−' for '−ξᵀx*'."""
    rp = dict(rotation=rotation) if rotation else {}
    kids = []
    if negate:
        kids.append(TextArea("−", textprops=dict(color=minus_color, fontsize=fs,
                                                        fontweight="bold", **rp)))
    kids += [
        TextArea(r"$\boldsymbol{\xi}^{\top}$",
                 textprops=dict(color=COL_UNC, fontsize=fs, fontweight="bold", **rp)),
        TextArea("x", textprops=dict(color=COL_DEC, fontsize=fs, fontweight="bold",
                                     fontstyle="italic", **rp)),
        TextArea("*", textprops=dict(color=COL_DEC, fontsize=fs, fontweight="bold", **rp)),
    ]
    return kids


def _formula_row(name_tex: str, value_str: str, fs: float = 12.5, name_color: str = "#16324f",
                 negate: bool = False):
    """'{NAME}[ ξᵀ x* ] = value', with ξ in the uncertainty colour, x* in the
    decision colour, and {NAME}/brackets/value in `name_color`."""
    B = name_color
    kids = [
        TextArea(name_tex, textprops=dict(color=B, fontsize=fs, fontweight="bold")),
        TextArea("  [ ", textprops=dict(color=B, fontsize=fs, fontweight="bold")),
        *_xi_x_star_kids(fs, negate=negate, minus_color=B),
        TextArea(f" ]  =  {value_str}", textprops=dict(color=B, fontsize=fs, fontweight="bold")),
    ]
    return HPacker(align="baseline", pad=0, sep=0.5, children=kids)


def _compact_row(name_tex: str, value_str: str, fs: float = 10.8, name_color: str = "#16324f"):
    """'{NAME} = value' -- the half-width sibling of _formula_row, used when
    in-sample and out-of-sample sit side by side and there isn't room for the
    full [ ξᵀx* ] notation in each column."""
    B = name_color
    kids = [
        TextArea(name_tex, textprops=dict(color=B, fontsize=fs, fontweight="bold")),
        TextArea(f"  =  {value_str}", textprops=dict(color=B, fontsize=fs, fontweight="bold")),
    ]
    return HPacker(align="baseline", pad=0, sep=0.5, children=kids)


def _boxed_row(ax, row, xy, fc, ec, box_alignment=(0, 0.5), pad=0.45):
    ax.add_artist(AnnotationBbox(
        row, xy, xycoords="axes fraction", box_alignment=box_alignment,
        frameon=True, pad=pad,
        bboxprops=dict(boxstyle=f"round,pad={pad}", fc=fc, ec=ec, lw=1.4)))


def _set_wealth_xlabel(ax, prefix: str, fs: float = 15, negate: bool = False,
                       xybox=(0, -32)):
    """x-axis label '{prefix},  ξᵀx*' with ξ red / x* blue, replacing the
    plain-string xlabel so the ξᵀx* notation matches the formulation panel."""
    ax.set_xlabel(" ")  # reserve the normal label row so tight_layout still leaves room
    kids = [TextArea(prefix, textprops=dict(color=COL_PAR, fontsize=fs)),
            *_xi_x_star_kids(fs, negate=negate, minus_color=COL_PAR)]
    row = HPacker(align="baseline", pad=0, sep=1, children=kids)
    ax.add_artist(AnnotationBbox(
        row, (0.5, 0), xycoords="axes fraction", box_alignment=(0.5, 1),
        xybox=xybox, boxcoords="offset points", frameon=False, annotation_clip=False))


def _render_kids_to_rgba(kids, dpi):
    """Render an unrotated HPacker of `kids` to a tightly-cropped RGBA array.
    OffsetBox layout does not measure individually-rotated TextAreas
    correctly (children overlap instead of stacking) -- rotating the
    rasterized pixels instead of the text objects sidesteps that entirely."""
    import io
    import matplotlib.transforms as mtransforms
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    tmp_fig = plt.figure(dpi=dpi)
    tmp_ax = tmp_fig.add_axes([0, 0, 1, 1])
    tmp_ax.axis("off")
    row = HPacker(align="baseline", pad=0, sep=1, children=kids)
    ab = AnnotationBbox(row, (0.5, 0.5), xycoords="axes fraction", frameon=False)
    tmp_ax.add_artist(ab)
    canvas = FigureCanvasAgg(tmp_fig)
    canvas.draw()
    bbox_px = ab.get_window_extent(canvas.get_renderer())
    pad_px = 2
    bbox_in = mtransforms.Bbox([[(bbox_px.x0 - pad_px) / dpi, (bbox_px.y0 - pad_px) / dpi],
                                 [(bbox_px.x1 + pad_px) / dpi, (bbox_px.y1 + pad_px) / dpi]])
    buf = io.BytesIO()
    tmp_fig.savefig(buf, format="png", transparent=True, bbox_inches=bbox_in, dpi=dpi)
    plt.close(tmp_fig)
    buf.seek(0)
    return plt.imread(buf)


def _set_wealth_ylabel(ax, prefix: str, suffix: str, fs: float = 15, negate: bool = False,
                       xybox=(-27, 0), zoom: float = 0.497):
    """y-axis label '{prefix} ξᵀx* {suffix}' with ξ red / x* blue, rotated 90°
    and positioned beside the axis like a normal ylabel (see _render_kids_to_rgba
    for why this is a rotated raster, not rotated text objects)."""
    ax.set_ylabel(" ")
    kids = [TextArea(prefix, textprops=dict(color=COL_PAR, fontsize=fs)),
            *_xi_x_star_kids(fs, negate=negate, minus_color=COL_PAR)]
    if suffix:
        kids.append(TextArea(suffix, textprops=dict(color=COL_PAR, fontsize=fs)))
    rgba = _render_kids_to_rgba(kids, dpi=fig.dpi)
    rotated = np.rot90(rgba, k=1)
    oi = OffsetImage(rotated, zoom=zoom)
    ax.add_artist(AnnotationBbox(
        oi, (0, 0.5), xycoords="axes fraction", box_alignment=(1, 0.5),
        xybox=xybox, boxcoords="offset points", frameon=False, annotation_clip=False))


def _solve_time_lines(sol):
    t = sol.solve_time
    slow = sol.status.startswith("Feasible incumbent")
    if slow:
        return f"{t:.1f}s (time limit)", "feasible incumbent"
    return (f"{t*1000:.0f} ms" if t < 1 else f"{t:.2f}s"), None


def draw_results_panel():
    _box(ax_res, *PANEL_RES)
    key = state["focus"]
    ax_res.text(0.035, 0.955, f"{MODEL_LABEL[key].upper()} SOLUTION",
                transform=ax_res.transAxes, va="top", fontsize=9.3,
                fontweight="bold", color=COL_DEC)
    if state["sweep"] is not None and state["sweep"]["key"] == key:
        ax_res.text(0.5, 0.5, "Sweep mode —\nwatch the allocation\nchart as you drag\nthe slider.",
                    transform=ax_res.transAxes, ha="center", va="center", fontsize=11,
                    color="#5b3a94", style="italic", wrap=True)
        return
    sol = state["solutions"].get(key)
    err = state["errors"].get(key)
    stale = is_stale(key)
    if sol is None or stale:
        if err:
            # drop any solver-internal detail in parentheses (e.g. HiGHS status
            # codes) -- the point of this box is "infeasible", not why the
            # underlying LP solver thinks so
            msg = err.split("(")[0].strip()
            ax_res.text(0.5, 0.5, msg, transform=ax_res.transAxes, ha="center",
                        va="center", fontsize=15, color="#a8330a", fontweight="bold",
                        wrap=True,
                        bbox=dict(boxstyle="round,pad=0.7", fc="#fde3df",
                                  ec="#a8330a", lw=2.0))
            return
        if sol is not None and stale:
            msg = "Inputs changed — press  Solve  to refresh."
        else:
            msg = f"press  Solve  to see the {MODEL_LABEL[key]} portfolio."
        ax_res.text(0.5, 0.5, msg, transform=ax_res.transAxes, ha="center",
                    va="center", fontsize=8.8, color="#5b6770", style="italic", wrap=True)
        return

    # decision box (blue), matching the house "build ..." box style
    ax_res.text(0.035, 0.79, _invest_str(sol.weights), transform=ax_res.transAxes,
                va="center", fontsize=10.5, fontweight="bold", color=COL_DEC,
                bbox=dict(boxstyle="round,pad=0.4", fc="#dce9ff", ec=COL_DEC, lw=1.4))

    eps = epsilon()
    tm = state["test_metrics"]
    has_oos = key != "mean_variance" and tm is not None and state["mode"] == "distribution"
    RIGHT_X = 0.53

    ax_res.text(0.035, 0.62, "IN-SAMPLE PERFORMANCE", transform=ax_res.transAxes,
                va="center", fontsize=9.3, fontweight="bold", color=COL_DEC)
    if has_oos:
        ax_res.text(RIGHT_X, 0.62, "OUT-OF-SAMPLE PERFORMANCE", transform=ax_res.transAxes,
                    va="center", fontsize=9.3, fontweight="bold", color="#8a3f16")

    def _row_pair(name_tex, in_value, oos_value, y, negate=False):
        if has_oos:
            _boxed_row(ax_res, _compact_row(name_tex, in_value), (0.04, y),
                      "#e1f5e6", "#4fa868")
            _boxed_row(ax_res, _compact_row(name_tex, oos_value), (RIGHT_X, y),
                      "#ffe9d6", "#c9752e")
        else:
            _boxed_row(ax_res, _formula_row(name_tex, in_value, negate=negate), (0.04, y),
                      "#e1f5e6", "#4fa868")

    # 𝔼[ξᵀx*] is exactly as valid to report positive, but the whole dashboard
    # (x-axis, histogram legend, VaR/CVaR brackets above) is framed in negative
    # wealth -- so Mean is negated here too, purely for uniform framing.
    if key == "mean_variance":
        # 𝕍(ξᵀx*) == 𝕍(−ξᵀx*): sign-symmetric, so the negate here changes
        # only the bracket's label, not the value -- kept for uniform framing
        # with the rest of the dashboard.
        _row_pair(_perf_name(key, eps), f"{100*sol.variance:.3f} %²", None, 0.49,
                  negate=True)
        _row_pair(r"$\mathbb{E}$", f"{-100*sol.expected_return:.2f}%", None, 0.36,
                  negate=True)
        note_y = 0.25
    elif key == "chance":
        oos_val = f"{-100*tm['expected_return']:.2f}%" if has_oos else None
        _row_pair(_perf_name(key, eps), f"{-100*sol.expected_return:.2f}%", oos_val, 0.49,
                  negate=True)
        if not has_oos:
            # chance is the only model with just ONE _row_pair call, so the
            # y=0.36 slot every other model uses for a second metric is free
            # here -- boxed the same green as the Mean row above it, same
            # size/weight, so it reads as a second metric at the same level,
            # not a footnote
            _boxed_row(ax_res, _chance_insample_row(sol), (0.04, 0.36),
                      "#e1f5e6", "#4fa868")
        note_y = None
    else:
        # sol.cvar / sol.var are computed on the loss (−ξᵀx*), so the bracket
        # must say [−ξᵀx*] -- the number itself is already correctly signed
        # (a positive tail-loss magnitude, matching the histogram's own legend).
        in_val = {"cvar": f"{100*sol.cvar:.2f}%", "var": f"{100*sol.var:.2f}%"}[key]
        oos_val = ({"cvar": f"{100*tm['cvar']:.2f}%", "var": f"{100*tm['var']:.2f}%"}[key]
                   if has_oos else None)
        _row_pair(_perf_name(key, eps), in_val, oos_val, 0.49, negate=True)
        oos_mean = f"{-100*tm['expected_return']:.2f}%" if has_oos else None
        _row_pair(r"$\mathbb{E}$", f"{-100*sol.expected_return:.2f}%", oos_mean, 0.36,
                  negate=True)
        note_y = 0.25
    if note_y is not None:
        ax_res.text(0.04, note_y, _perf_note(key, sol), transform=ax_res.transAxes,
                    va="center", fontsize=7.8, style="italic", color="#5b6770", wrap=True)

    # SOLVE TIME -- the caption lives INSIDE the box now (not a floating
    # label above it): the box's height changes with 1 vs 2 lines of content,
    # so a label positioned outside/above it left a gap for the short case
    # and risked collision for the tall one. Packing the caption as the
    # box's own first line means there is never a gap, by construction.
    time_str, sub_str = _solve_time_lines(sol)
    time_row = HPacker(align="baseline", pad=0, sep=6, children=[
        TextArea("SOLVE TIME", textprops=dict(color="#5b3a94", fontsize=12.5,
                                               fontweight="bold")),
        TextArea(time_str, textprops=dict(color="#5b3a94", fontsize=12.5,
                                           fontweight="bold")),
    ])
    if sub_str:
        sub_area = TextArea(sub_str, textprops=dict(color="#5b3a94", fontsize=7.6,
                                                      fontstyle="italic"))
        time_box = VPacker(align="left", pad=0, sep=2, children=[time_row, sub_area])
    else:
        time_box = time_row
    _boxed_row(ax_res, time_box, (0.04, 0.015), "#ece3f7", "#8c5fc9",
              box_alignment=(0, 0), pad=0.3)


# ── main-panel views ─────────────────────────────────────────────────────────
def _draw_weights_bars(weights, title, title_color="#1a1a1a", title_fontsize=12,
                       title_fontweight="normal"):
    n = market.n_assets
    xs = np.arange(n)
    ax_main.bar(xs, 100 * weights, width=0.55, color=market.colors,
               edgecolor="white", linewidth=0.8, zorder=3)
    for xi, w in zip(xs, weights):
        # size scales with the weight itself -- a 71% allocation should look
        # bigger on the page than a 0% one, not just taller
        fs = 8 + 10 * max(w, 0.0)
        ax_main.text(xi, 100 * w, f"{100 * w:.1f}%", ha="center", va="bottom",
                     fontsize=fs, color="#16324f", fontweight="bold")
    ax_main.set_xticks(xs, market.sectors, fontsize=12, fontweight="bold")  # 12pt is
    # the actual ceiling for 6 names sharing this axis width before "Manufacturing"
    # collides with its neighbors -- verified by measuring rendered label extents,
    # not by eye (see chat: 13pt already overlaps by ~5px, 18pt by ~55px)
    ax_main.set_ylabel("share of wealth (%)", fontsize=12)
    ax_main.set_title(title, fontsize=title_fontsize, color=title_color,
                      fontweight=title_fontweight)
    ax_main.set_ylim(0, max(100 * weights.max() * 1.2, 10))
    ax_main.grid(axis="y", alpha=0.22)


def _draw_sweep_solution():
    sw = state["sweep"]
    param = sw["param"]
    v = SWEEP_SLIDERS[param].val
    v_near, weights, err = min(sw["results"], key=lambda r: abs(r[0] - v))
    label = (f"{MODEL_LABEL[sw['key']]} — sweeping {SWEEP_LABELS[param]} = "
             f"{_fmt_sweep_val(param, v_near)}")
    if weights is None:
        ax_main.text(0.5, 0.5, "The problem is infeasible.", transform=ax_main.transAxes,
                     ha="center", va="center", fontsize=15, color="#a8330a",
                     fontweight="bold",
                     bbox=dict(boxstyle="round,pad=0.7", fc="#fde3df",
                               ec="#a8330a", lw=2.0))
        ax_main.set_title(label, fontsize=13, color="#7b3fa0", fontweight="bold")
        return
    _draw_weights_bars(weights, label, title_color="#7b3fa0", title_fontsize=13,
                       title_fontweight="bold")


def draw_solution():
    ax_main.clear()
    key = state["focus"]
    if state["sweep"] is not None and state["sweep"]["key"] == key:
        _draw_sweep_solution()
        return
    sol = state["solutions"].get(key)
    if sol is None or is_stale(key):
        prompt = ("Inputs changed — press  Solve" if sol is not None
                  else f"press  Solve  to see the {MODEL_LABEL[key]} portfolio")
        ax_main.text(0.5, 0.5, prompt, transform=ax_main.transAxes,
                     ha="center", va="center", fontsize=13, color="#8a97a8",
                     style="italic")
        return
    _draw_weights_bars(sol.weights,
                       f"{MODEL_LABEL[key]} portfolio — optimal portfolio allocation")
    ax_main.tick_params(axis="y", labelsize=8)  # axis="y" only -- this used to
    # blanket-override BOTH axes, silently resetting the x-tick sector-name
    # fontsize set just above back down to 8 every time


def _draw_snapshot_panel(ax, snap, show_xlabel, xrange):
    """One compact half of the two-snapshot Compare view -- same histogram +
    risk-measure-lines idea as draw_distribution(), simplified (no rotated
    y-label raster, smaller one-line legend) to fit half the vertical space.
    `xrange` is shared across both panels (see draw_compare2) so the two
    histograms sit on the same scale -- otherwise a tight distribution and a
    long-tailed one can look deceptively similar, each auto-scaled to fill
    its own panel."""
    ax.clear()
    key = snap["key"]
    tm = snap["tm"]
    eps_pct = snap["eps_pct"]
    neg_wealth = 100 * np.asarray(tm["losses"])

    weights = np.full_like(neg_wealth, 1.0 / len(neg_wealth))
    counts, bins, patches = ax.hist(neg_wealth, bins=30, range=xrange, weights=weights,
                                     color="#9ec2e8", edgecolor="white", alpha=0.9)

    if key == "chance":
        g_nw = snap["gamma_pct"]          # frozen at snapshot time, not the live slider
        _chance_color_bars(patches, bins, g_nw)
        ax.axvline(g_nw, color="#111", lw=2.0, ls=":")
        legend = _chance_legend(neg_wealth, g_nw, eps_pct / 100, fs=8.5)
    else:
        var_nw, cvar_nw = 100 * float(tm["var"]), 100 * float(tm["cvar"])
        mean_nw = -100 * float(tm["expected_return"])
        _chance_color_bars(patches, bins, var_nw)
        ax.axvline(mean_nw, color=COL_GOOD, lw=2.2, ls="-.")
        ax.axvline(var_nw, color="#8b1a1a", lw=2.2)
        ax.axvline(cvar_nw, color="#6924a8", lw=2.2, ls="--")
        rows = [
            _formula_row(r"$\mathbb{E}$", f"{mean_nw:.2f}%", fs=8.5, name_color=COL_GOOD,
                         negate=True),
            _formula_row(rf"$\mathrm{{VaR}}_{{{eps_pct:.0f}\%}}$", f"{var_nw:.2f}%",
                         fs=8.5, name_color="#8b1a1a", negate=True),
            _formula_row(rf"$\mathrm{{CVaR}}_{{{eps_pct:.0f}\%}}$", f"{cvar_nw:.2f}%",
                         fs=8.5, name_color="#6924a8", negate=True),
        ]
        legend = HPacker(align="baseline", pad=0, sep=10, children=rows)

    ax.set_xlim(*xrange)
    ax.set_ylim(0, counts.max() * 1.18)

    ax.set_title(f"{MODEL_LABEL[key]}  —  {snap['label']}", fontsize=11.5,
                color=MODEL_COLOR[key], fontweight="bold", pad=6)
    ax.tick_params(labelsize=8.5)
    ax.grid(axis="y", alpha=0.18)
    ax.set_ylabel("Prob", fontsize=9)
    if show_xlabel:
        ax.set_xlabel("Negative wealth (i.e., Loss)", fontsize=9.5)

    ax.add_artist(AnnotationBbox(
        legend, (0.985, 0.97), xycoords="axes fraction", box_alignment=(1, 1),
        frameon=True, pad=0.35,
        bboxprops=dict(boxstyle="round,pad=0.3", fc="#ffffff", ec="#999999", lw=1.0)))


def draw_compare2():
    a, b = state["snapshots"]
    both_losses = np.concatenate([100 * np.asarray(a["tm"]["losses"]),
                                   100 * np.asarray(b["tm"]["losses"])])
    lo, hi = both_losses.min(), both_losses.max()
    pad = 0.06 * (hi - lo)
    xrange = (lo - pad, hi + pad)
    _draw_snapshot_panel(ax_cmp_a, a, show_xlabel=False, xrange=xrange)
    _draw_snapshot_panel(ax_cmp_b, b, show_xlabel=True, xrange=xrange)


def _chance_color_bars(patches, bins, g_nw):
    """Colour bars by the ONE threshold that actually matters for a
    chance-constrained portfolio -- γ, not a risk measure it never optimized
    (VaR/CVaR/mean are all off-target here: this model doesn't shape the tail,
    it only bounds how OFTEN losses may cross γ)."""
    for patch, left in zip(patches, bins[:-1]):
        if left >= g_nw:
            patch.set_facecolor("#e45d55")


def _chance_legend(neg_wealth, g_nw, eps, fs):
    """'Required P(loss≤γ)≥1-ε' vs. 'Observed (OOS) P(loss≤γ)' -- whether the
    constraint's promised coverage actually held up on unseen scenarios is
    the only question this model's distribution plot needs to answer."""
    required_pct = 100 * (1 - eps)
    observed_pct = 100 * float(np.mean(neg_wealth <= g_nw))
    satisfied = observed_pct >= required_pct
    obs_color = COL_GOOD if satisfied else "#b23b3b"
    mark = "✓" if satisfied else "✗"

    def _row(prefix, value_str, color):
        kids = [
            TextArea(f"{prefix}  P( ",
                     textprops=dict(color=color, fontsize=fs, fontweight="bold")),
            *_xi_x_star_kids(fs, negate=True, minus_color=color),
            TextArea(f" ≤ γ={g_nw:.1f}% )  {value_str}",
                     textprops=dict(color=color, fontsize=fs, fontweight="bold")),
        ]
        return HPacker(align="baseline", pad=0, sep=0.5, children=kids)

    rows = [
        _row("Required:", f"≥ 1-ε={required_pct:.0f}%", COL_PAR),
        _row("Actual (OOS):", f"= {observed_pct:.1f}%   {mark}", obs_color),
    ]
    return VPacker(align="left", pad=0, sep=6, children=rows)


def draw_distribution():
    """Histogram of out-of-sample negative wealth (−ξᵀx*, i.e. loss -- right =
    bad, left = good, matching the course notes' VaR/CVaR figure directly, so
    "loss" never needs translating to "negative wealth" in a student's head).
    Mean-variance/CVaR/VaR get the model's risk measures as vertical lines and
    a custom in-plot legend (ax.legend can't colour a substring, so each row
    is built the same colour-coded way as the results-panel boxes). Chance-
    constrained gets a DIFFERENT, simpler view: it never targets a risk
    measure, only P(loss>γ)≤ε, so the only threshold worth drawing is γ, and
    the only number worth reporting is whether that coverage held OOS."""
    ax_main.clear()
    tm = state["test_metrics"]
    key = state["focus"]
    if tm is None:
        draw_solution(); return
    ax_main.set_position(AX_MAIN_RECT_DIST)
    neg_wealth = 100 * np.asarray(tm["losses"])       # −ξᵀx*, i.e. loss
    eps_pct = 100 * epsilon()

    weights = np.full_like(neg_wealth, 1.0 / len(neg_wealth))
    counts, bins, patches = ax_main.hist(neg_wealth, bins=40, weights=weights,
                                          color="#9ec2e8", edgecolor="white", alpha=0.9)

    if key == "chance":
        g_nw = 100 * loss_limit()
        _chance_color_bars(patches, bins, g_nw)
        ax_main.axvline(g_nw, color="#111", lw=2.8, ls=":")
        legend = _chance_legend(neg_wealth, g_nw, epsilon(), fs=12.5)
    else:
        var_nw, cvar_nw = 100 * float(tm["var"]), 100 * float(tm["cvar"])
        mean_nw = -100 * float(tm["expected_return"])     # 𝔼[−ξᵀx*] = −𝔼[ξᵀx*]
        _chance_color_bars(patches, bins, var_nw)          # same left/right split, var-based
        ax_main.axvline(mean_nw, color=COL_GOOD, lw=3.2, ls="-.")
        ax_main.axvline(var_nw, color="#8b1a1a", lw=3.2)
        ax_main.axvline(cvar_nw, color="#6924a8", lw=3.2, ls="--")
        rows = [
            _formula_row(r"$\mathbb{E}$", f"{mean_nw:.2f}%", fs=10.5, name_color=COL_GOOD,
                         negate=True),
            _formula_row(rf"$\mathrm{{VaR}}_{{{eps_pct:.0f}\%}}$", f"{var_nw:.2f}%",
                         fs=10.5, name_color="#8b1a1a", negate=True),
            _formula_row(rf"$\mathrm{{CVaR}}_{{{eps_pct:.0f}\%}}$", f"{cvar_nw:.2f}%",
                         fs=10.5, name_color="#6924a8", negate=True),
        ]
        legend = HPacker(align="baseline", pad=0, sep=12, children=rows)

    ax_main.set_ylim(0, counts.max() * 1.12)  # headroom so the tallest bar
    # doesn't touch the top spine -- otherwise it reads as a stray line under
    # the legend box, which sits just a few points below that spine
    ax_main.add_artist(AnnotationBbox(
        legend, (0.5, 1), xycoords="axes fraction", box_alignment=(0.5, 1),
        xybox=(0, -6), boxcoords="offset points", frameon=True, pad=0.5,
        bboxprops=dict(boxstyle="round,pad=0.2", fc="#ffffff", ec="#999999", lw=1.4),
        annotation_clip=False))

    _set_wealth_xlabel(ax_main, "Negative wealth (i.e., Loss),  ", negate=True, xybox=(0, -32))
    _set_wealth_ylabel(ax_main, "Prob[ ", " ]", negate=True, zoom=0.65, xybox=(-45, 0))
    ax_main.set_title(f"{MODEL_LABEL[key]} portfolio — out-of-sample test performance, "
                      f"{len(neg_wealth):,} months", fontsize=15, pad=34)
    ax_main.grid(axis="y", alpha=0.18)
    ax_main.tick_params(labelsize=13)


def draw_sim():
    """Animated reveal of out-of-sample scenarios, one batch at a time -- the
    LAP-competition style: a thick magenta break-even line at zero separates
    making from losing money, dots are green/red by which side of THAT line
    they land on, a black running-mean line, and live mean/CVaR stats in the
    title. Armed-but-not-started (i == 0) just shows the empty axes with a
    prompt, mirroring the same state before ▶ Simulate has been pressed."""
    ax_main.clear()
    key = state["focus"]
    sim = state["sim"]
    total = sim["total"] if sim else int(s_trials.val)
    ax_main.set_xlim(0.5, total + 0.5)
    ax_main.set_xlabel("simulation #", fontsize=13)
    ax_main.set_ylabel("out-of-sample wealth (%)", fontsize=13)
    ax_main.grid(True, alpha=0.22)
    ax_main.tick_params(labelsize=8)
    i = sim["i"] if sim else 0
    ax_main.set_title(f"{MODEL_LABEL[key]} portfolio — out-of-sample gain per scenario",
                      fontsize=14, color=COL_PAR)
    if i == 0:
        ax_main.text(0.5, 0.5, f"press  ▶ Simulate  to reveal {total} out-of-sample "
                     "scenarios", transform=ax_main.transAxes, ha="center",
                     va="center", fontsize=12.5, color="#8a97a8", style="italic")
        return
    sol = state["solutions"][key]
    losses = 100 * portfolio_losses(state["test_bank"][:i], sol.weights)
    gains = -losses
    xs = np.arange(1, i + 1)
    running_mean = np.cumsum(gains) / xs
    ax_main.axhline(0, color="#c800a0", lw=3.4, zorder=2)
    made_money = gains >= 0
    ax_main.scatter(xs[made_money], gains[made_money], s=13, color=COL_GOOD, alpha=0.65,
                    zorder=3, label="gain")
    ax_main.scatter(xs[~made_money], gains[~made_money], s=13, color=COL_UNC, alpha=0.65,
                    zorder=3, label="loss")
    ax_main.plot(xs, running_mean, "-", color=COL_PAR, lw=2.6, zorder=4,
                label="running mean")
    eps = epsilon()
    stats = f"{i}/{total} · running mean {running_mean[-1]:.2f}%"
    if i >= 5:
        _, running_cvar = empirical_var_cvar(losses, eps)
        stats += f" · running CVaR{100*eps:.0f}% {running_cvar:.2f}%"
    ax_main.set_title(f"{MODEL_LABEL[key]} portfolio — {stats}", fontsize=14,
                      color=COL_PAR)
    ax_main.legend(fontsize=12, loc="upper right", framealpha=0.9)


def draw_data():
    ax_main.clear()
    train = training_returns()
    corr = np.corrcoef(train, rowvar=False)
    ax_main.imshow(corr, vmin=-1, vmax=1, cmap="RdBu_r", aspect="auto")
    ax_main.set_xticks(range(market.n_assets), market.sectors, rotation=25,
                       ha="right", fontsize=8.5)
    ax_main.set_yticks(range(market.n_assets), market.sectors, fontsize=8.5)
    for i in range(market.n_assets):
        for j in range(market.n_assets):
            ax_main.text(j, i, f"{corr[i, j]:.2f}", ha="center", va="center",
                         fontsize=8, color="white" if abs(corr[i, j]) > 0.55 else "#222")
    ax_main.set_title(f"Training sample correlations — N={len(train)} months", fontsize=12)


def draw_learning():
    ax_main.clear()
    c = state["learning"]
    if c is None:
        draw_solution(); return
    Ns = np.asarray(c["Ns"])
    ax_main.fill_between(Ns, c["lo"], c["hi"], color=MODEL_COLOR[c["key"]], alpha=0.18,
                         label="range over training samples")
    ax_main.plot(Ns, c["mean"], "o-", color=MODEL_COLOR[c["key"]], lw=2.4, label=c["metric"])
    if c.get("ref") is not None:
        ax_main.axhline(c["ref"], color="#355f8a", lw=1.8, ls="--", label=c["ref_label"])
    ax_main.set_xscale("log"); ax_main.set_xticks(Ns, [str(n) for n in Ns])
    ax_main.minorticks_off()
    ax_main.set_xlabel("training scenarios N (log scale)", fontsize=10)
    ax_main.set_ylabel(c["ylabel"], fontsize=10)
    ax_main.set_title(f"Out-of-sample performance vs N — {MODEL_LABEL[c['key']]}", fontsize=12)
    ax_main.grid(alpha=0.22); ax_main.legend(fontsize=8.4); ax_main.tick_params(labelsize=8)


def redraw():
    draw_model_panel()
    draw_results_panel()
    is_compare2 = state["mode"] == "compare2"
    ax_main.set_visible(not is_compare2)
    ax_cmp_a.set_visible(is_compare2)
    ax_cmp_b.set_visible(is_compare2)
    if is_compare2:
        draw_compare2()
    else:
        ax_main.set_aspect("auto", adjustable="box")
        ax_main.set_position(AX_MAIN_RECT)
        {"solution": draw_solution, "distribution": draw_distribution,
         "vsN": draw_learning, "data": draw_data, "sim": draw_sim}[state["mode"]]()
    for key, b in model_buttons.items():
        b.ax.set_facecolor("#9ec5ff" if key == state["focus"] else "#e7ebf2")
    fig.canvas.draw_idle()


# ── solving ──────────────────────────────────────────────────────────────────
def on_solve_focused(_):
    """Primary action: solve ONLY the currently focused model."""
    _cancel_sim()
    key = state["focus"]
    show_progress(f"Solving\n{MODEL_LABEL[key]}…")
    try:
        try:
            state["solutions"][key] = solve_portfolio(
                key, market, training_returns(), target_return(),
                epsilon(), loss_limit())
            state["errors"][key] = None
            state["message"] = f"{MODEL_LABEL[key]} solved."
        except Exception as exc:
            state["solutions"][key] = None
            state["errors"][key] = str(exc)
            state["message"] = f"{MODEL_LABEL[key]} failed to solve."
        state["solved_config"][key] = current_config(key)
    finally:
        hide_progress()
    state["mode"] = "solution"
    state["test_metrics"] = None
    redraw()


def on_focus(key):
    def cb(_):
        _cancel_sim()
        state["focus"] = key
        state["mode"] = "solution"
        state["test_metrics"] = None
        redraw()
    return cb


# ── parameter sweep (▶ next to N / R / ε / γ) ────────────────────────────────
def _sweep_grid_and_step(slider, max_points=25):
    """Evenly spaced grid aligned to the slider's own valstep (so every grid
    point is one the slider could already snap to on its own) -- capped to
    `max_points` by using an integer multiple of valstep, so N/R/ε/γ all stay
    at clean, round values instead of arbitrary floats from a raw linspace."""
    lo, hi, step0 = slider.valmin, slider.valmax, slider.valstep
    natural = int(round((hi - lo) / step0)) + 1
    factor = max(1, -(-natural // max_points))  # ceil division
    step = step0 * factor
    n = int(round((hi - lo) / step)) + 1
    grid = [round(lo + i * step, 10) for i in range(n)]
    if grid[-1] < hi - 1e-9:
        grid.append(hi)
    return grid, step


def _solve_for_sweep(key, param, v):
    """Solve `key` with `param` swept to raw slider value `v`, holding the
    other three solve-relevant parameters at their current slider values."""
    if param == "n":
        returns = state["train_bank"][:int(v)]
        R, eps, gamma = target_return(), epsilon(), loss_limit()
    elif param == "target":
        returns, R, eps, gamma = training_returns(), v / 100.0, epsilon(), loss_limit()
    elif param == "eps":
        returns, R, eps, gamma = training_returns(), target_return(), v / 100.0, loss_limit()
    else:  # "gamma"
        returns, R, eps, gamma = training_returns(), target_return(), epsilon(), v / 100.0
    return solve_portfolio(key, market, returns, R, eps, gamma)


def _fmt_sweep_val(param, v):
    if param == "n":
        return f"{int(v)}"
    return f"{v:.2f}%" if param == "target" else f"{v:.1f}%"


_disable_overlays = {}


def _disable_widget(ax):
    if ax not in _disable_overlays:
        _disable_overlays[ax] = ax.add_patch(plt.Rectangle(
            (0, 0), 1, 1, transform=ax.transAxes, facecolor="white",
            edgecolor="none", alpha=0.6, zorder=10))


def _enable_widget(ax):
    rect = _disable_overlays.pop(ax, None)
    if rect is not None:
        rect.remove()


def _sweep_other_widgets(param):
    return ([s for k, s in SWEEP_SLIDERS.items() if k != param]
            + [s_trials, s_speed] + list(model_buttons.values())
            + list(action_buttons.values())
            + [b for k, b in SWEEP_BUTTONS.items() if k != param])


def _set_sweep_mode(active, param):
    SWEEP_BUTTONS[param].label.set_text("■" if active else "▶")
    for w in _sweep_other_widgets(param):
        w.set_active(not active)
        (_disable_widget if active else _enable_widget)(w.ax)


def _start_sweep(param):
    slider = SWEEP_SLIDERS[param]
    key = state["focus"]
    _cancel_sim()
    show_progress(f"Solving…\nsweeping {SWEEP_LABELS[param]}")
    grid, step = _sweep_grid_and_step(slider)
    results = []
    try:
        for v in grid:
            try:
                sol = _solve_for_sweep(key, param, v)
                results.append((v, sol.weights, None))
            except Exception as exc:
                results.append((v, None, str(exc)))
    finally:
        hide_progress()
    state["sweep"] = {"param": param, "key": key, "grid": grid, "step": step,
                      "orig_valstep": slider.valstep, "results": results}
    slider.valstep = step
    nearest = min(grid, key=lambda g: abs(g - slider.val))
    slider.eventson = False
    slider.set_val(nearest)
    slider.eventson = True
    state["mode"] = "solution"
    _set_sweep_mode(True, param)
    redraw()


def _exit_sweep():
    param = state["sweep"]["param"]
    SWEEP_SLIDERS[param].valstep = state["sweep"]["orig_valstep"]
    state["sweep"] = None
    _set_sweep_mode(False, param)
    redraw()


def on_sweep_click(param):
    def cb(_):
        if state["sweep"] is not None and state["sweep"]["param"] == param:
            _exit_sweep()
        elif state["sweep"] is None:
            _start_sweep(param)
    return cb


def _param_summary(key):
    """Short caption distinguishing two snapshots of the same (or different)
    model -- only the parameters that actually affect that model's solve
    (matches current_config(), which is the source of truth for this).
    Mean-variance solves on population μ, Σ directly, so N is irrelevant --
    only R matters."""
    if key == "mean_variance":
        return f"R={s_target.val:.2f}%"
    if key == "chance":
        return f"γ={s_gamma.val:.1f}%, ε={int(s_eps.val)}%, N={int(s_n.val)}"
    return f"ε={int(s_eps.val)}%, N={int(s_n.val)}"


def _snapshot_label():
    n_filled = sum(s is not None for s in state["snapshots"])
    action_buttons["Snapshot"].label.set_text(f"Snapshot ({n_filled}/2)")


def on_snapshot(_):
    """Pin the focused model's current solved + simulated result into slot A,
    then slot B; a third press starts over (replaces A, clears B). Needs
    Solve *and* Run trials to have completed for the current slider values."""
    key = state["focus"]
    sol = state["solutions"].get(key)
    tm = state["test_metrics"]
    if sol is None or tm is None or is_stale(key):
        _snapshot_label()
        return
    snap = {
        "key": key,
        "label": _param_summary(key),
        "tm": {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in tm.items()},
        "eps_pct": 100 * epsilon(),
        "gamma_pct": 100 * loss_limit() if key == "chance" else None,
    }
    slots = state["snapshots"]
    if slots[0] is None:
        slots[0] = snap
    elif slots[1] is None:
        slots[1] = snap
    else:
        slots[0], slots[1] = snap, None
    _snapshot_label()
    redraw()


def on_compare(_):
    if state["snapshots"][0] is None or state["snapshots"][1] is None:
        return
    _cancel_sim()
    state["mode"] = "compare2"
    redraw()


# ── animated OOS reveal (armed -> running <-> paused -> finished) ────────────
def _timer():
    if state["_timer"] is None:
        t = fig.canvas.new_timer(interval=60)
        t.add_callback(sim_tick)
        state["_timer"] = t
    return state["_timer"]


def _cancel_sim():
    """Any action that invalidates an in-progress reveal (re-solving, moving
    a slider, switching models, ...) cancels it back to the idle button."""
    if state["trial_stage"] == "idle":
        return
    if state["_timer"] is not None:
        state["_timer"].stop()
    state["sim"] = None
    state["trial_stage"] = "idle"
    action_buttons["Run trials"].label.set_text("Run trials")
    if state["mode"] == "sim":
        state["mode"] = "solution"


def sim_tick():
    sim = state["sim"]
    if sim is None:
        return
    sim["i"] = min(sim["i"] + int(s_speed.val), sim["total"])
    redraw()
    if sim["i"] >= sim["total"]:
        _finish_sim()


def _finish_sim():
    if state["_timer"] is not None:
        state["_timer"].stop()
    sim = state["sim"]
    sol = state["solutions"][sim["key"]]
    state["test_metrics"] = evaluate_portfolio(market, sol.weights,
                                               state["test_bank"][:sim["total"]], epsilon())
    state["sim"] = None
    state["trial_stage"] = "idle"
    action_buttons["Run trials"].label.set_text("Run trials")
    state["mode"] = "distribution"
    redraw()


def on_trials(_):
    key = state["focus"]
    stage = state["trial_stage"]
    if stage == "idle":
        sol = state["solutions"].get(key)
        if sol is None or is_stale(key):
            state["message"] = "Solve this model first."
            state["mode"] = "solution"; redraw(); return
        state["sim"] = {"key": key, "i": 0, "total": int(s_trials.val)}
        state["trial_stage"] = "armed"
        state["mode"] = "sim"
        action_buttons["Run trials"].label.set_text("▶ Simulate")
        redraw()
    elif stage in ("armed", "paused"):
        state["trial_stage"] = "running"
        action_buttons["Run trials"].label.set_text("Pause")
        redraw()
        _timer().start()
    else:  # running
        _timer().stop()
        state["trial_stage"] = "paused"
        action_buttons["Run trials"].label.set_text("▶ Simulate")
        redraw()


def on_learning(_):
    _cancel_sim()
    key = state["focus"]
    if key == "mean_variance":
        state["message"] = "Mean–variance uses μ, Σ directly — N does not change it."
        state["mode"] = "solution"; redraw(); return
    Ns = [20, 50, 100, 200, 500, 1000]
    test = state["test_bank"][:2000]
    mean, lo, hi = [], [], []
    try:
        for i, n in enumerate(Ns, 1):
            show_progress(f"Performance vs N\nstep {i}/{len(Ns)}   N={n}")
            row = []
            for _ in range(3 if n <= 200 else 2):
                tr = market.sample(n, _new_seed())
                try:
                    sol = solve_portfolio(key, market, tr, target_return(),
                                          epsilon(), loss_limit())
                except Exception:
                    continue
                mt = evaluate_portfolio(market, sol.weights, test, epsilon())
                if key == "cvar":
                    row.append(100 * float(mt["cvar"]))
                elif key == "var":
                    row.append(100 * float(mt["var"]))
                else:
                    row.append(100 * float(np.mean(mt["losses"] > loss_limit())))
            if row:
                mean.append(np.mean(row)); lo.append(np.min(row)); hi.append(np.max(row))
            else:
                mean.append(np.nan); lo.append(np.nan); hi.append(np.nan)
    finally:
        hide_progress()
    if key == "cvar":
        metric, ylabel, ref, rl = "mean OOS CVaR", "out-of-sample CVaR (%)", None, ""
    elif key == "var":
        metric, ylabel, ref, rl = "mean OOS VaR", "out-of-sample VaR (%)", None, ""
    else:
        metric, ylabel, ref, rl = ("violation rate", "P(loss > γ) out-of-sample (%)",
                                   100 * epsilon(), f"target ε = {epsilon():.0%}")
    state["learning"] = {"Ns": Ns, "mean": mean, "lo": lo, "hi": hi, "key": key,
                         "metric": metric, "ylabel": ylabel, "ref": ref, "ref_label": rl}
    state["mode"] = "vsN"; redraw()


def on_data(_):
    _cancel_sim()
    state["mode"] = "data"; redraw()


def on_new_sample(_):
    _cancel_sim()
    seed = _new_seed()
    state["train_seed"] = seed
    state["train_bank"] = market.sample(1000, seed)
    state["solutions"] = {k: None for k in MODEL_KEYS}
    state["errors"] = {k: None for k in MODEL_KEYS}
    state["solved_config"] = {k: None for k in MODEL_KEYS}
    state["test_metrics"] = None
    state["learning"] = None
    state["mode"] = "solution"
    state["message"] = "New training sample — press  Solve."
    redraw()


def on_param(_):
    if state["sweep"] is not None:
        # only the swept slider is active right now; just redraw from the
        # cached sweep results instead of the normal stale/needs-Solve path
        redraw()
        return
    _cancel_sim()
    key = state["focus"]
    if is_stale(key) and state["solutions"].get(key) is not None:
        state["message"] = f"Inputs changed — press  Solve  to update {MODEL_LABEL[key]}."
        if state["mode"] in ("distribution", "vsN"):
            state["mode"] = "solution"
    redraw()


# ── wiring ───────────────────────────────────────────────────────────────────
for key, b in model_buttons.items():
    b.on_clicked(on_focus(key))
for s in (s_n, s_target, s_eps, s_gamma):
    s.on_changed(on_param)
s_trials.on_changed(lambda _: (_cancel_sim(), redraw()))
action_buttons["Solve"].on_clicked(on_solve_focused)
action_buttons["Run trials"].on_clicked(on_trials)
action_buttons["Perf vs N"].on_clicked(on_learning)
action_buttons["View data"].on_clicked(on_data)
action_buttons["New sample"].on_clicked(on_new_sample)
action_buttons["Snapshot"].on_clicked(on_snapshot)
action_buttons["Compare"].on_clicked(on_compare)
for _pk, _b in SWEEP_BUTTONS.items():
    _b.on_clicked(on_sweep_click(_pk))

fig.text(0.315, 0.018,
         "Six-sector synthetic monthly market  ·  long-only, fully invested  ·  "
         "instructional use only", ha="center", fontsize=8, color="#777")

_snapshot_label()
redraw()

if __name__ == "__main__":
    plt.show()
