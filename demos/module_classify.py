"""
CLASSIFICATION MODULE — draw your own separating line vs. the MILP optimum
==========================================================================
IP-course exhibit (same dashboard style as the LAP modules).  Students are given a
2-D labelled dataset and DRAW a linear classifier by clicking two points; the panel
reports how many points their line misclassifies.  Pressing "Optimal (MILP)" solves
the exact minimum-misclassification problem and overlays the optimal line to compare.

The 0-1 (indicator) loss is modelled as a MILP — one binary z_i per point + big-M:
    min  sum_i z_i   s.t.  y_i (w.x_i + b) >= 1 - M z_i ,  z_i in {0,1}.

Run:  python viz/module_classify.py
"""
import os
import sys
import time
import contextlib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from matplotlib.lines import Line2D
from matplotlib.offsetbox import TextArea, HPacker, VPacker, AnnotationBbox
from scipy.optimize import milp, LinearConstraint, Bounds, minimize


@contextlib.contextmanager
def _silence_stdio():
    """Mute the C-level stdout/stderr (HiGHS prints internal MIP logging there)."""
    sys.stdout.flush(); sys.stderr.flush()
    saved = (os.dup(1), os.dup(2))
    devnull = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(devnull, 1); os.dup2(devnull, 2)
        yield
    finally:
        os.dup2(saved[0], 1); os.dup2(saved[1], 2)
        os.close(devnull); os.close(saved[0]); os.close(saved[1])

POS, NEG, YOU, OPT, WB = "#1f5fd0", "#c0392b", "#111111", "#1a8a3a", "#e8820c"
SVM_C, LOG_C, NN_C = "#7e3ff2", "#0e8a8a", "#d6336c"   # SVM purple / logistic teal / NN pink
SIG_C = "#b5179e"   # sigmoid (nonconvex smooth 0-1 loss) — magenta
DEC, DATA, PAR = "#1f5fd0", "#c0392b", "#1a1a1a"   # decision (blue) / data (red) / params
W_BND, B_BND = 20.0, 8.0    # w-bound generous so the margin ε is a pure scale normalization
seed_rng = np.random.default_rng(0)
state = {"X": None, "y": None, "clicks": [], "you": None, "opt": None,
         "wb": None, "wb_on": False, "bigm": None,
         "svm": None, "logistic": None, "sigmoid": None, "sig_seed": 0,
         "nn": None, "nn_size": "small", "algo": "milp", "show_algos": False}


def _minop(sub, fs):
    return VPacker(align="center", pad=0, sep=0, children=[
        TextArea(r"$\min$", textprops=dict(color=PAR, fontsize=fs)),
        TextArea(sub, textprops=dict(color=DEC, fontsize=fs * 0.6))])


def _eqline(items, fs):
    ch = [it if not isinstance(it, tuple)
          else TextArea(it[0], textprops=dict(color=it[1], fontsize=fs)) for it in items]
    return HPacker(align="baseline", pad=0, sep=0, children=ch)


def gen_data(N, noise=1.15):
    rng = np.random.default_rng(int(seed_rng.integers(1, 99999)))
    n1 = N // 2
    X = np.vstack([rng.normal([1.25, 1.25], noise, (n1, 2)),
                   rng.normal([-1.25, -1.25], noise, (N - n1, 2))])
    y = np.r_[np.ones(n1), -np.ones(N - n1)]
    state["X"], state["y"] = X, y
    state["clicks"], state["you"], state["opt"] = [], None, None
    state["svm"] = state["logistic"] = state["sigmoid"] = state["nn"] = None  # stale on new data


def evaluate(w, b):
    """Misclassified count + mask for a line, taking the better of the 2 orientations."""
    X, y = state["X"], state["y"]
    s = X @ w + b
    e1 = (np.where(s >= 0, 1, -1) != y)
    e2 = (np.where(-s >= 0, 1, -1) != y)
    if e1.sum() <= e2.sum():
        return int(e1.sum()), e1, (np.asarray(w), b)
    return int(e2.sum()), e2, (-np.asarray(w), -b)


def line_from_clicks(p1, p2):
    d = np.array(p2) - np.array(p1)
    w = np.array([-d[1], d[0]])                 # normal to the drawn segment
    return w, -w @ np.array(p1)


def solve_milp():
    X, y = state["X"], state["y"]; N = len(y)
    gamma = float(s_margin.val)
    M = 10 ** float(s_bigm.val)                         # from the big-M slider (log scale)
    A = np.zeros((N, 3 + N))
    A[:, 0:2] = y[:, None] * X
    A[:, 2] = y
    A[np.arange(N), 3 + np.arange(N)] = M
    con = LinearConstraint(A, lb=np.full(N, gamma), ub=np.inf)
    bounds = Bounds(np.r_[-W_BND, -W_BND, -B_BND, np.zeros(N)],
                    np.r_[W_BND, W_BND, B_BND, np.ones(N)])
    t = time.perf_counter()
    # presolve OFF so a loose big-M is NOT auto-tightened — the M really affects the time
    with _silence_stdio():
        res = milp(c=np.r_[0, 0, 0, np.ones(N)], constraints=con,
                   integrality=np.r_[0, 0, 0, np.ones(N)], bounds=bounds,
                   options={"presolve": False, "time_limit": 20})
    dt = time.perf_counter() - t
    if res.x is None:                                   # too-small M → can be infeasible
        state["opt"] = {"infeasible": True, "M": M, "time": dt, "miss": None}
    else:
        miss, mask, (w, b) = evaluate(res.x[0:2], res.x[2])
        state["opt"] = {"w": w, "b": b, "miss": miss, "mask": mask, "time": dt, "M": M}


def _solve_M(M, relax=False, tl=10):
    """Solve the MILP (or its LP relaxation) for a GIVEN big-M.  presolve is turned
    off so a loose big-M is not silently tightened away by the solver."""
    X, y = state["X"], state["y"]; N = len(y); eps = float(s_margin.val)
    A = np.zeros((N, 3 + N)); A[:, 0:2] = y[:, None] * X; A[:, 2] = y
    A[np.arange(N), 3 + np.arange(N)] = M
    con = LinearConstraint(A, lb=np.full(N, eps), ub=np.inf)
    bnd = Bounds(np.r_[-W_BND, -W_BND, -B_BND, np.zeros(N)],
                 np.r_[W_BND, W_BND, B_BND, np.ones(N)])
    integ = np.r_[0, 0, 0, (np.zeros(N) if relax else np.ones(N))]
    t = time.perf_counter()
    with _silence_stdio():
        r = milp(c=np.r_[0, 0, 0, np.ones(N)], constraints=con, integrality=integ,
                 bounds=bnd, options={"presolve": False, "time_limit": tl})
    return r, time.perf_counter() - t


def compute_bigm():
    """Solve at several big-M values to show the cost of choosing M badly."""
    X = state["X"]; eps = float(s_margin.val)
    Mt = eps + W_BND * np.abs(X).sum(1).max() + B_BND     # smallest valid big-M
    rows = []
    for M in (3.0, Mt, 30 * Mt, 300 * Mt):
        lp, _ = _solve_M(M, relax=True, tl=8)
        ip, t = _solve_M(M, tl=10)
        lpb = lp.fun if (lp.success and lp.fun is not None) else None
        miss = evaluate(ip.x[0:2], ip.x[2])[0] if ip.success else None
        rows.append((M, t, miss, lpb))
    state["bigm"] = {"Mt": Mt, "rows": rows}


# ── surrogate-loss classifiers (convex, NO binary variables → fast) ────────────
def solve_svm(C=1.0):
    """Soft-margin SVM (hinge loss): min ½‖w‖² + C·Σ max(0, 1 − yᵢ(wᵀxᵢ+b))."""
    X, y = state["X"], state["y"]

    def f(th):
        w, b = th[:2], th[2]
        return 0.5 * (w @ w) + C * np.maximum(0.0, 1.0 - y * (X @ w + b)).sum()

    def g(th):
        w, b = th[:2], th[2]
        a = (y * (X @ w + b) < 1.0).astype(float)        # active (margin-violating) points
        return np.r_[w - C * (X.T @ (a * y)), -C * (a * y).sum()]

    t = time.perf_counter()
    r = minimize(f, np.zeros(3), jac=g, method="L-BFGS-B")
    dt = time.perf_counter() - t
    miss, mask, (w, b) = evaluate(r.x[:2], r.x[2])
    return {"w": w, "b": b, "miss": miss, "mask": mask, "time": dt}


def solve_logistic(lam=0.01):
    """Logistic regression (log loss): min Σ log(1 + exp(−yᵢ(wᵀxᵢ+b)))."""
    X, y = state["X"], state["y"]

    def f(th):
        w, b = th[:2], th[2]
        return np.logaddexp(0.0, -y * (X @ w + b)).sum() + 0.5 * lam * (w @ w)

    def g(th):
        w, b = th[:2], th[2]
        p = 1.0 / (1.0 + np.exp(np.clip(y * (X @ w + b), -50, 50)))   # σ(−margin)
        return np.r_[X.T @ (-y * p) + lam * w, (-y * p).sum()]

    t = time.perf_counter()
    r = minimize(f, np.zeros(3), jac=g, method="L-BFGS-B")
    dt = time.perf_counter() - t
    miss, mask, (w, b) = evaluate(r.x[:2], r.x[2])
    return {"w": w, "b": b, "miss": miss, "mask": mask, "time": dt}


def solve_sigmoid(C=4.0, seed=0, lam=1e-3):
    """Smooth (NONCONVEX) approximation to the 0-1 loss itself:
        min_{w,b}  Σᵢ 1 / (1 + exp(C·yᵢ(wᵀxᵢ+b)))  (+ tiny ridge to pin the scale).
    Each term is a sigmoid in the margin: ≈1 when misclassified, ≈0 when correct, and
    as C→∞ it → the exact indicator.  But it is NONCONVEX, so gradient descent only
    reaches a LOCAL optimum that depends on the random start — change the seed and the
    line (and the error count) can jump.  This is exactly what the MILP avoids."""
    X, y = state["X"], state["y"]
    rng = np.random.default_rng(int(seed))

    def f(th):
        w, b = th[:2], th[2]
        s = 1.0 / (1.0 + np.exp(np.clip(C * y * (X @ w + b), -50, 50)))   # loss per point
        return s.sum() + 0.5 * lam * (w @ w)

    def g(th):
        w, b = th[:2], th[2]
        s = 1.0 / (1.0 + np.exp(np.clip(C * y * (X @ w + b), -50, 50)))
        d = (-C * s * (1.0 - s)) * y                 # dℓ/d(score) per point
        return np.r_[X.T @ d + lam * w, d.sum()]

    th0 = np.r_[4.0 * rng.standard_normal(2), 2.0 * rng.standard_normal()]   # spread-out random start
    t = time.perf_counter()
    r = minimize(f, th0, jac=g, method="L-BFGS-B")
    dt = time.perf_counter() - t
    miss, mask, (w, b) = evaluate(r.x[:2], r.x[2])
    return {"w": w, "b": b, "miss": miss, "mask": mask, "time": dt}


# ── neural-net classifier (nonlinear boundary; capacity → accuracy → overfit) ──
# small/slim → ~linear;  medium → much better;  large → memorises the data (~100%).
#                hidden units,   max-epochs, weight-decay, stop-when-train-miss≤
NN_SPECS = {"small":  ([6],          2500,  1e-3, None),  # slim → barely beyond a line
            "medium": ([24, 24],     3000,  1e-4, None),  # bends the boundary → much better
            "large":  ([80, 80, 40], 15000, 0.0,  0)}     # high capacity → memorises (~100%)
NN_ORDER = ["small", "medium", "large"]


def _mlp_predict(params, X):
    a = X
    for i, (W, b) in enumerate(params):
        a = a @ W + b
        if i < len(params) - 1:
            a = np.maximum(0.0, a)              # ReLU on hidden layers
    return a[:, 0]                              # linear output logit


def train_nn(size="small"):
    """Train an MLP by full-batch Adam on the logistic (cross-entropy) loss.  Bigger
    nets + more epochs + less weight-decay → they carve a wigglier nonlinear boundary
    and eventually MEMORISE the training points (≈100% train accuracy = overfitting)."""
    hidden, epochs, l2, stop = NN_SPECS[size]
    X, y = state["X"], state["y"]
    mu, sd = X.mean(0), X.std(0) + 1e-9
    Xn = (X - mu) / sd                          # standardise inputs (helps training)
    t = (y + 1) / 2.0                           # labels → {0,1}
    n = len(y)
    rng = np.random.default_rng(0)              # fixed seed → reproducible
    sizes = [2] + hidden + [1]
    params = [[rng.standard_normal((a, b)) * np.sqrt(2.0 / a), np.zeros(b)]
              for a, b in zip(sizes[:-1], sizes[1:])]
    mAd = [[np.zeros_like(W), np.zeros_like(bb)] for W, bb in params]
    vAd = [[np.zeros_like(W), np.zeros_like(bb)] for W, bb in params]
    b1, b2, eps, lr = 0.9, 0.999, 1e-8, 0.02
    t0 = time.perf_counter()
    for ep in range(1, epochs + 1):
        acts = [Xn]                             # forward, caching activations
        for i, (W, b) in enumerate(params):
            z = acts[-1] @ W + b
            acts.append(np.maximum(0.0, z) if i < len(params) - 1 else z)
        p = 1.0 / (1.0 + np.exp(-np.clip(acts[-1][:, 0], -50, 50)))
        g = ((p - t) / n)[:, None]              # dL/d(logit)
        grads = [None] * len(params)
        for i in reversed(range(len(params))):
            gW = acts[i].T @ g + l2 * params[i][0]
            grads[i] = [gW, g.sum(0)]
            if i > 0:
                g = (g @ params[i][0].T) * (acts[i] > 0)   # back-prop through ReLU
        for i in range(len(params)):            # Adam update
            for j in range(2):
                mAd[i][j] = b1 * mAd[i][j] + (1 - b1) * grads[i][j]
                vAd[i][j] = b2 * vAd[i][j] + (1 - b2) * grads[i][j] ** 2
                mh = mAd[i][j] / (1 - b1 ** ep); vh = vAd[i][j] / (1 - b2 ** ep)
                params[i][j] -= lr * mh / (np.sqrt(vh) + eps)
        if stop is not None and ep % 250 == 0:        # early stop once it memorises
            if int((np.where(_mlp_predict(params, Xn) >= 0, 1, -1) != y).sum()) <= stop:
                break
    dt = time.perf_counter() - t0
    predict = lambda Q: _mlp_predict(params, (Q - mu) / sd)   # logit on raw coords
    yhat = np.where(predict(X) >= 0, 1, -1)
    mask = yhat != y
    return {"miss": int(mask.sum()), "mask": mask, "time": dt,
            "predict": predict, "hidden": hidden}


gen_data(40)

# ── figure ────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(14, 7.6))
fig.canvas.manager.set_window_title("Classification — your line vs the MILP optimum")
ax = fig.add_axes([0.055, 0.30, 0.52, 0.60])
ax_model = fig.add_axes([0.635, 0.42, 0.34, 0.51])
ax_results = fig.add_axes([0.635, 0.14, 0.34, 0.25])

s_n = Slider(plt.axes([0.115, 0.255, 0.15, 0.02]), "N points", 20, 400, valinit=40, valstep=4)
s_noise = Slider(plt.axes([0.115, 0.215, 0.15, 0.02]), "noise", 0.6, 2.0, valinit=1.15, valstep=0.05)
s_margin = Slider(plt.axes([0.115, 0.175, 0.15, 0.02]), "margin ε", 0.0, 10.0, valinit=1.0, valstep=0.5)
# big-M slider is LOG-scale (M spans 3 … 500000); valtext shows the actual M
s_bigm = Slider(plt.axes([0.115, 0.135, 0.15, 0.02]), "big-M", np.log10(3), np.log10(5e5),
                valinit=3.0, valstep=0.02)
s_bigm.valtext.set_text("%.0f" % (10 ** s_bigm.val))
s_w1 = Slider(plt.axes([0.43, 0.255, 0.15, 0.02]), "w1", -3.0, 3.0, valinit=1.0, valstep=0.1)
s_w2 = Slider(plt.axes([0.43, 0.215, 0.15, 0.02]), "w2", -3.0, 3.0, valinit=1.0, valstep=0.1)
s_b = Slider(plt.axes([0.43, 0.175, 0.15, 0.02]), "b", -5.0, 5.0, valinit=0.0, valstep=0.1)
# sigmoid steepness C (only relevant once "Sigmoid" is used) — larger C ⇒ closer to the 0-1 loss
s_sigC = Slider(plt.axes([0.43, 0.135, 0.15, 0.02]), "sigmoid C", 0.5, 30.0, valinit=4.0, valstep=0.5)
# row A — the CLASSIFIERS (SVM/Logistic/Sigmoid stay hidden until "Other classifiers" is pressed)
b_opt = Button(plt.axes([0.045, 0.082, 0.123, 0.046]), "Optimal (MILP)", color="#cfeed8", hovercolor="#a9dfb9")
b_svm = Button(plt.axes([0.172, 0.082, 0.058, 0.046]), "SVM", color="#e3d7fb", hovercolor="#cdb6f4")
b_log = Button(plt.axes([0.234, 0.082, 0.083, 0.046]), "Logistic", color="#cce9e6", hovercolor="#a4d8d2")
b_sig = Button(plt.axes([0.321, 0.082, 0.085, 0.046]), "Sigmoid", color="#f7cdec", hovercolor="#eda6da")
b_nn = Button(plt.axes([0.410, 0.082, 0.135, 0.046]), "Neural net", color="#f6d3e2", hovercolor="#eeacc8")
b_svm.ax.set_visible(False); b_log.ax.set_visible(False)
b_sig.ax.set_visible(False); b_nn.ax.set_visible(False)
# row B — actions
b_more = Button(plt.axes([0.045, 0.032, 0.135, 0.046]), "Other classifiers", color="#e3e7ee", hovercolor="#cfd6e0")
b_wb = Button(plt.axes([0.188, 0.032, 0.10, 0.046]), "Tune w, b", color="#f3ddca", hovercolor="#e8b894")
b_bigm = Button(plt.axes([0.296, 0.032, 0.115, 0.046]), "Big-M table", color="#efe0b0", hovercolor="#e3cd86")
b_reset = Button(plt.axes([0.419, 0.032, 0.10, 0.046]), "New data", color="#dde2ea")


def _box(a, fc, ec):
    a.clear(); a.set_facecolor(fc); a.set_xticks([]); a.set_yticks([]); a.set_xlim(0, 1); a.set_ylim(0, 1)
    for sp in a.spines.values():
        sp.set_visible(True); sp.set_edgecolor(ec); sp.set_linewidth(1.4)


def _plot_line(w, b, color, style, label):
    x0, x1 = ax.get_xlim()
    if abs(w[1]) > 1e-9:
        xs = np.array([x0, x1]); ys = -(w[0] * xs + b) / w[1]
    else:
        xs = np.array([-b / w[0], -b / w[0]]); ys = np.array(ax.get_ylim())
    ax.plot(xs, ys, color=color, ls=style, lw=2.3, label=label, zorder=4)


def draw():
    X, y = state["X"], state["y"]
    ax.clear()
    # the W,B classifier (hidden until "Tune w, b" is pressed, so the first attempt
    # — the student's own drawn line — is not biased by seeing it)
    if state["wb_on"]:
        wm, wmask, (ww, wbo) = evaluate(np.array([s_w1.val, s_w2.val]), float(s_b.val))
        state["wb"] = {"w": ww, "b": wbo, "miss": wm, "mask": wmask}
    else:
        state["wb"] = None
    ax.scatter(X[y > 0, 0], X[y > 0, 1], c=POS, s=34, label="class +1", zorder=3)
    ax.scatter(X[y < 0, 0], X[y < 0, 1], c=NEG, s=34, label="class −1", zorder=3)
    lim = np.abs(X).max() * 1.15
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    for p in state["clicks"]:                    # pending clicked points
        ax.plot(p[0], p[1], "x", color=YOU, ms=11, mew=2.4, zorder=6)
    wb, you, opt = state["wb"], state["you"], state["opt"]
    nn_mode = state["algo"] == "nn" and state["nn"]
    if nn_mode:                                  # NONLINEAR boundary: shade regions + curve
        res = state["nn"][state["nn_size"]]
        gx = np.linspace(-lim, lim, 170)
        XX, YY = np.meshgrid(gx, gx)
        ZZ = res["predict"](np.c_[XX.ravel(), YY.ravel()]).reshape(XX.shape)
        ax.contourf(XX, YY, ZZ, levels=[-1e9, 0, 1e9],
                    colors=["#ffd9d2", "#d3e2ff"], alpha=0.55, zorder=0)
        ax.contour(XX, YY, ZZ, levels=[0], colors=[NN_C], linewidths=2.4, zorder=4)
        ax.plot([], [], color=NN_C, lw=2.4, label="NN boundary (%s)" % state["nn_size"])
        ax.scatter(X[res["mask"], 0], X[res["mask"], 1], s=150, facecolors="none",
                   edgecolors=NN_C, linewidths=1.9, zorder=5)
    if opt and not opt.get("infeasible"):        # optimal line (green) — always a reference
        _plot_line(opt["w"], opt["b"], OPT, "-", "MILP optimum")
        if not nn_mode:
            ax.scatter(X[opt["mask"], 0], X[opt["mask"], 1], s=300, facecolors="none",
                       edgecolors=OPT, linewidths=1.9, zorder=4)
    if not nn_mode:                              # the straight-line classifiers
        sv, lg = state["svm"], state["logistic"]
        if sv:                                    # SVM (purple); rings only when in focus
            _plot_line(sv["w"], sv["b"], SVM_C, "-", "SVM (hinge)")
            if state["algo"] == "svm":
                ax.scatter(X[sv["mask"], 0], X[sv["mask"], 1], s=210, facecolors="none",
                           edgecolors=SVM_C, linewidths=1.8, zorder=4.3)
        if lg:                                    # logistic (teal); rings only when in focus
            _plot_line(lg["w"], lg["b"], LOG_C, "-", "logistic")
            if state["algo"] == "logistic":
                ax.scatter(X[lg["mask"], 0], X[lg["mask"], 1], s=150, facecolors="none",
                           edgecolors=LOG_C, linewidths=1.8, zorder=4.4)
        sg = state["sigmoid"]
        if sg:                                    # sigmoid (magenta); rings only when in focus
            _plot_line(sg["w"], sg["b"], SIG_C, "-", "sigmoid (nonconvex)")
            if state["algo"] == "sigmoid":
                ax.scatter(X[sg["mask"], 0], X[sg["mask"], 1], s=170, facecolors="none",
                           edgecolors=SIG_C, linewidths=1.8, zorder=4.35)
        if wb:                                   # only after "Tune w, b" is pressed
            if np.hypot(*wb["w"]) > 1e-6:
                _plot_line(wb["w"], wb["b"], WB, "-", "W,B line (sliders)")
            ax.scatter(X[wb["mask"], 0], X[wb["mask"], 1], s=185, facecolors="none",
                       edgecolors=WB, linewidths=1.8, zorder=4.5)
        if you:                                  # clicked line (black) — innermost rings
            _plot_line(you["w"], you["b"], YOU, "--", "your line (clicked)")
            ax.scatter(X[you["mask"], 0], X[you["mask"], 1], s=90, facecolors="none",
                       edgecolors=YOU, linewidths=1.6, zorder=5)
    if nn_mode:
        ax.set_title("Neural net: a NONLINEAR boundary — capacity lets it bend "
                     "(the large net wraps every point).", fontsize=10.5)
    else:
        ax.set_title("Click two points to draw a separating line "
                     "(it splits the plane in two).", fontsize=11)
    ax.legend(loc="upper left", fontsize=7.3, framealpha=0.95, ncol=2, columnspacing=0.8)
    draw_model()
    if state["bigm"]:
        draw_bigm()
    elif nn_mode:
        draw_results_nn()
    else:
        draw_results()
    fig.canvas.draw_idle()


def draw_bigm():
    _box(ax_results, "white", "#9bb8de")
    ax_results.set_xticks([]); ax_results.set_yticks([])
    bm = state["bigm"]; Mt = bm["Mt"]
    ax_results.text(0.035, 0.96, "BIG-M:  too small breaks it · too large slows it",
                    fontsize=8.8, fontweight="bold", va="top", color="#1f3a5f")
    cell = [["%.0f" % M, "%.2f" % t, "infeasible" if miss is None else "%d" % miss,
             "—" if lpb is None else "%.1f" % lpb] for (M, t, miss, lpb) in bm["rows"]]
    tbl = ax_results.table(cellText=cell,
        colLabels=["big-M", "time (s)", "misclassified", "LP bound"],
        cellLoc="center", colLoc="center", colWidths=[0.26, 0.18, 0.34, 0.22],
        bbox=[0.02, 0.40, 0.96, 0.48])
    tbl.auto_set_font_size(False); tbl.set_fontsize(7.0)
    ref = next((mi for (M, t, mi, lp) in bm["rows"]
                if abs(M - Mt) < 1e-6 and mi is not None), None)   # the correct count
    for (rr, cc), c_ in tbl.get_celld().items():
        c_.set_edgecolor("#bcccdd"); c_.set_linewidth(0.8)
        if rr == 0:
            c_.set_facecolor("#dbe7f3"); c_.get_text().set_fontweight("bold")
        else:
            M, t, miss, lpb = bm["rows"][rr - 1]
            if miss is None or (ref is not None and miss > ref):
                c_.set_facecolor("#fcebea")                 # infeasible / wrong → red tint
            elif abs(M - Mt) < 1e-6:
                c_.set_facecolor("#e3f3e9")                 # the tight valid M → green
    ax_results.text(0.04, 0.32,
                    "small M → INFEASIBLE / wrong (it forbids valid classifiers).",
                    fontsize=7.3, va="top", color="#b03030", style="italic")
    ax_results.text(0.04, 0.205,
                    "large M → LP bound (the solver's head-start) → 0, so it must search harder.",
                    fontsize=7.3, va="top", color="#333", style="italic")
    ax_results.text(0.04, 0.09, "best = the smallest VALID big-M  (≈ %.0f here)." % Mt,
                    fontsize=7.5, va="top", color="#1a5f33", style="italic", fontweight="bold")


def draw_model():
    _box(ax_model, "#fff7d6", "#d9c179")
    algo = state.get("algo", "milp")
    if algo == "svm":
        _model_svm()
    elif algo == "logistic":
        _model_logistic()
    elif algo == "sigmoid":
        _model_sigmoid()
    elif algo == "nn":
        _model_nn()
    else:
        _model_milp()


def _model_milp():
    fs = 10
    rows = [
        TextArea("Min-misclassification classifier (MILP)",
                 textprops=dict(fontsize=10, fontweight="bold", color="#222")),
        TextArea(" ", textprops=dict(fontsize=3)),
        _eqline([_minop(r"$\mathbf{w},b,\mathbf{z}$", fs),
                 (r"$\ \sum_{i=1}^{N}$", PAR), (r"$z_i$", DEC)], fs),
        TextArea(" ", textprops=dict(fontsize=2)),
        _eqline([(r"$\mathrm{s.t.}\ \ $", PAR), (r"$y_i$", DATA), (r"$\,($", PAR),
                 (r"$\mathbf{w}^{\!\top}$", DEC), (r"$\mathbf{x}_i$", DATA),
                 (r"$+$", PAR), (r"$b$", DEC), (r"$)\,\geq\,\epsilon - M$", PAR),
                 (r"$z_i$", DEC)], fs),
        _eqline([(r"$\qquad\ $", PAR), (r"$z_i$", DEC),
                 (r"$\in\{0,1\},\ \ i=1,\dots,N$", PAR)], fs),
    ]
    ax_model.add_artist(AnnotationBbox(VPacker(align="left", pad=0, sep=3, children=rows),
                        (0.04, 0.975), xycoords="axes fraction", box_alignment=(0, 1),
                        frameon=False))
    ax_model.text(0.04, 0.55, "DECISION VARIABLES  (what we choose — the classifier)",
                  fontsize=8.3, fontweight="bold", va="top", color=DEC)
    ax_model.text(0.05, 0.485,
                  r"$\mathbf{w},b$: the line $\mathbf{w}^\top\mathbf{x}+b=0$  ($\mathbf{w}$"
                  r" = tilt/orientation, $b$ = offset)." "\n"
                  r"$z_i\in\{0,1\}$: 1 if point $i$ is misclassified — the 0-1 loss, as a binary.",
                  fontsize=7.5, va="top", color=DEC)
    ax_model.text(0.04, 0.345, "DATA  (given & observed — we do NOT choose it)",
                  fontsize=8.3, fontweight="bold", va="top", color=DATA)
    ax_model.text(0.05, 0.28,
                  r"$\mathbf{x}_i\in\mathbb{R}^2$: the features (the dot's coordinates)." "\n"
                  r"$y_i\in\{+1,-1\}$: its true class label  (blue dot $=+1$, red $=-1$).",
                  fontsize=7.5, va="top", color=DATA)
    ax_model.text(0.04, 0.145, "PARAMETERS", fontsize=8.3, fontweight="bold",
                  va="top", color=PAR)
    ax_model.text(0.05, 0.085,
                  r"$M$: big-$M$.   $\epsilon=%.1f$: the margin — fixes the scale (rules out "
                  r"$\mathbf{w}=b=0$)." "\n"
                  r"moderate $\epsilon$ → count is stable;  $\epsilon=0$ → degenerate (no line);"
                  r"  huge $\epsilon$ → distorts." % float(s_margin.val),
                  fontsize=7.5, va="top", color="#444")


def _model_svm():
    fs = 10
    rows = [
        TextArea("SVM — soft-margin (hinge loss)",
                 textprops=dict(fontsize=10, fontweight="bold", color="#222")),
        TextArea(" ", textprops=dict(fontsize=3)),
        _eqline([_minop(r"$\mathbf{w},b$", fs),
                 (r"$\ \frac{1}{2}\|$", PAR), (r"$\mathbf{w}$", DEC),
                 (r"$\|^2 + C\sum_i \max(0,\ 1-$", PAR), (r"$y_i$", DATA),
                 (r"$($", PAR), (r"$\mathbf{w}^{\!\top}$", DEC), (r"$\mathbf{x}_i$", DATA),
                 (r"$+$", PAR), (r"$b$", DEC), (r"$))$", PAR)], fs),
    ]
    ax_model.add_artist(AnnotationBbox(VPacker(align="left", pad=0, sep=3, children=rows),
                        (0.04, 0.975), xycoords="axes fraction", box_alignment=(0, 1),
                        frameon=False))
    ax_model.text(0.04, 0.66, "DECISION VARIABLES  (the classifier — the line)",
                  fontsize=8.3, fontweight="bold", va="top", color=DEC)
    ax_model.text(0.05, 0.595,
                  r"$\mathbf{w},b$: the line $\mathbf{w}^\top\mathbf{x}+b=0$." "\n"
                  r"NO binary $z_i$ — so the problem is convex & continuous (no integers).",
                  fontsize=7.5, va="top", color=DEC)
    ax_model.text(0.04, 0.46, "DATA  (given — we do NOT choose it)",
                  fontsize=8.3, fontweight="bold", va="top", color=DATA)
    ax_model.text(0.05, 0.395,
                  r"$\mathbf{x}_i\in\mathbb{R}^2$: features;   $y_i\in\{+1,-1\}$: true label.",
                  fontsize=7.5, va="top", color=DATA)
    ax_model.text(0.04, 0.30, "PARAMETERS", fontsize=8.3, fontweight="bold",
                  va="top", color=PAR)
    ax_model.text(0.05, 0.235,
                  r"$C>0$: how hard to penalise margin violations.",
                  fontsize=7.5, va="top", color="#444")
    ax_model.text(0.04, 0.135,
                  "hinge = a CONVEX upper bound on the 0-1 loss.  No integers → solves in\n"
                  "milliseconds, but minimises a proxy, so usually a few more errors than MILP.",
                  fontsize=7.6, va="top", color="#6a3fb0", style="italic", fontweight="bold")


def _model_logistic():
    fs = 10
    rows = [
        TextArea("Logistic regression (log loss)",
                 textprops=dict(fontsize=10, fontweight="bold", color="#222")),
        TextArea(" ", textprops=dict(fontsize=3)),
        _eqline([_minop(r"$\mathbf{w},b$", fs),
                 (r"$\ \sum_i \log(1+\exp(-$", PAR), (r"$y_i$", DATA),
                 (r"$($", PAR), (r"$\mathbf{w}^{\!\top}$", DEC), (r"$\mathbf{x}_i$", DATA),
                 (r"$+$", PAR), (r"$b$", DEC), (r"$)))$", PAR)], fs),
    ]
    ax_model.add_artist(AnnotationBbox(VPacker(align="left", pad=0, sep=3, children=rows),
                        (0.04, 0.975), xycoords="axes fraction", box_alignment=(0, 1),
                        frameon=False))
    ax_model.text(0.04, 0.66, "DECISION VARIABLES  (the classifier — the line)",
                  fontsize=8.3, fontweight="bold", va="top", color=DEC)
    ax_model.text(0.05, 0.595,
                  r"$\mathbf{w},b$: the line $\mathbf{w}^\top\mathbf{x}+b=0$." "\n"
                  r"NO binary $z_i$ — convex & continuous; $\sigma(\mathbf{w}^\top\mathbf{x}+b)$"
                  r" also gives a class PROBABILITY.",
                  fontsize=7.5, va="top", color=DEC)
    ax_model.text(0.04, 0.44, "DATA  (given — we do NOT choose it)",
                  fontsize=8.3, fontweight="bold", va="top", color=DATA)
    ax_model.text(0.05, 0.375,
                  r"$\mathbf{x}_i\in\mathbb{R}^2$: features;   $y_i\in\{+1,-1\}$: true label.",
                  fontsize=7.5, va="top", color=DATA)
    ax_model.text(0.04, 0.275, "PARAMETERS", fontsize=8.3, fontweight="bold",
                  va="top", color=PAR)
    ax_model.text(0.05, 0.21, "(none needed — a tiny ridge keeps $\\mathbf{w}$ finite.)",
                  fontsize=7.5, va="top", color="#444")
    ax_model.text(0.04, 0.12,
                  "log loss = another CONVEX surrogate for the 0-1 loss.  No integers →\n"
                  "milliseconds; smooth everywhere, and outputs a probability, not just a side.",
                  fontsize=7.6, va="top", color="#0a6e6a", style="italic", fontweight="bold")


def _model_sigmoid():
    fs = 10
    rows = [
        TextArea("Sigmoid loss — a smooth (NONCONVEX) 0-1 surrogate",
                 textprops=dict(fontsize=10, fontweight="bold", color="#222")),
        TextArea(" ", textprops=dict(fontsize=3)),
        _eqline([_minop(r"$\mathbf{w},b$", fs),
                 (r"$\ \sum_i (1+\exp(C\,$", PAR), (r"$y_i$", DATA),
                 (r"$($", PAR), (r"$\mathbf{w}^{\!\top}$", DEC), (r"$\mathbf{x}_i$", DATA),
                 (r"$+$", PAR), (r"$b$", DEC), (r"$)))^{-1}$", PAR)], fs),
    ]
    ax_model.add_artist(AnnotationBbox(VPacker(align="left", pad=0, sep=3, children=rows),
                        (0.04, 0.975), xycoords="axes fraction", box_alignment=(0, 1),
                        frameon=False))
    ax_model.text(0.04, 0.66, "DECISION VARIABLES  (the classifier — the line)",
                  fontsize=8.3, fontweight="bold", va="top", color=DEC)
    ax_model.text(0.05, 0.595,
                  r"$\mathbf{w},b$: the line $\mathbf{w}^\top\mathbf{x}+b=0$." "\n"
                  r"No binary $z_i$ — but each term is an S-curve, so the loss is NONCONVEX.",
                  fontsize=7.5, va="top", color=DEC)
    ax_model.text(0.04, 0.46, "DATA  (given — we do NOT choose it)",
                  fontsize=8.3, fontweight="bold", va="top", color=DATA)
    ax_model.text(0.05, 0.395,
                  r"$\mathbf{x}_i\in\mathbb{R}^2$: features;   $y_i\in\{+1,-1\}$: true label.",
                  fontsize=7.5, va="top", color=DATA)
    ax_model.text(0.04, 0.30, "PARAMETERS", fontsize=8.3, fontweight="bold",
                  va="top", color=PAR)
    ax_model.text(0.05, 0.235,
                  r"$C=%.1f$: steepness.  Larger $C$ $\to$ each term $\to$ the exact $1\{$miss$\}$,"
                  "\nbut the gradient goes flat away from the line, so it is HARDER to optimise."
                  % float(s_sigC.val),
                  fontsize=7.5, va="top", color="#444")
    ax_model.text(0.04, 0.135,
                  "≈ the 0-1 loss itself (bounded, robust to outliers) — but NONCONVEX.\n"
                  "Gradient descent finds only a LOCAL optimum: press  Sigmoid  again for a new\n"
                  "random start and the line can JUMP.  This is exactly what the MILP avoids.",
                  fontsize=7.5, va="top", color=SIG_C, style="italic", fontweight="bold")


def _model_nn():
    fs = 10
    hidden = NN_SPECS[state["nn_size"]][0]
    rows = [
        TextArea("Neural network — nonlinear classifier  (%s: %s)"
                 % (state["nn_size"], "→".join(["2"] + [str(h) for h in hidden] + ["1"])),
                 textprops=dict(fontsize=9.5, fontweight="bold", color="#222")),
        TextArea(" ", textprops=dict(fontsize=3)),
        _eqline([(r"$\hat{y}=\mathrm{sign}($", PAR), (r"$W_2$", DEC),
                 (r"$\,\max(0,\,$", PAR), (r"$W_1$", DEC), (r"$\mathbf{x}$", DATA),
                 (r"$+$", PAR), (r"$b_1$", DEC), (r"$)+$", PAR), (r"$b_2$", DEC),
                 (r"$)$", PAR)], fs),
    ]
    ax_model.add_artist(AnnotationBbox(VPacker(align="left", pad=0, sep=3, children=rows),
                        (0.04, 0.975), xycoords="axes fraction", box_alignment=(0, 1),
                        frameon=False))
    ax_model.text(0.04, 0.67, "DECISION VARIABLES  (the network's weights)",
                  fontsize=8.3, fontweight="bold", va="top", color=DEC)
    ax_model.text(0.05, 0.605,
                  r"$W_1,b_1,W_2,b_2,\dots$: ALL the weights — slim net = a few, large net" "\n"
                  r"= thousands.  Stacked $\max(0,\cdot)$ layers BEND the boundary into any shape.",
                  fontsize=7.5, va="top", color=DEC)
    ax_model.text(0.04, 0.45, "DATA  (given — we do NOT choose it)",
                  fontsize=8.3, fontweight="bold", va="top", color=DATA)
    ax_model.text(0.05, 0.385,
                  r"$\mathbf{x}_i\in\mathbb{R}^2$: features;   $y_i\in\{+1,-1\}$: true label.",
                  fontsize=7.5, va="top", color=DATA)
    ax_model.text(0.04, 0.29, "TRAINING", fontsize=8.3, fontweight="bold", va="top", color=PAR)
    ax_model.text(0.05, 0.225,
                  "fit by gradient descent (not a solver).  Loss is NONCONVEX — no\n"
                  "global guarantee, no binaries; just keep nudging the weights downhill.",
                  fontsize=7.5, va="top", color="#444")
    ax_model.text(0.04, 0.115,
                  "Capacity ↑ → fits anything.  The LARGE net MEMORISES the data: ≈100%\n"
                  "in-sample.  That is OVERFITTING — superb on training, not on new points.",
                  fontsize=7.6, va="top", color="#b01b56", style="italic", fontweight="bold")


def fmt_time(t):
    return "%.0f ms" % (t * 1000) if t < 1 else "%.2f s" % t


def draw_results():
    _box(ax_results, "#eef4fb", "#9bb8de")
    ax_results.text(0.035, 0.93, "RESULTS — misclassified  ·  solve time", fontsize=9.5,
                    fontweight="bold", va="top", color="#1f3a5f")
    N = len(state["y"])
    wb, you, opt = state["wb"], state["you"], state["opt"]
    sv, lg, sg = state["svm"], state["logistic"], state["sigmoid"]
    rows = []
    if you:
        rows.append(("your line (clicked)", YOU, "%d / %d wrong" % (you["miss"], N)))
    if wb:
        rows.append(("W,B line (sliders)", WB, "%d / %d wrong" % (wb["miss"], N)))
    if opt:
        rt = ("INFEASIBLE · %s" % fmt_time(opt["time"]) if opt.get("infeasible")
              else "%d / %d wrong · %s" % (opt["miss"], N, fmt_time(opt["time"])))
        rows.append(("MILP optimum  (M=%.0f)" % opt["M"], OPT, rt))
    if sv:
        rows.append(("SVM (hinge)", SVM_C, "%d / %d wrong · %s" % (sv["miss"], N, fmt_time(sv["time"]))))
    if lg:
        rows.append(("logistic", LOG_C, "%d / %d wrong · %s" % (lg["miss"], N, fmt_time(lg["time"]))))
    if sg:
        rows.append(("sigmoid (C=%.1f, start #%d)" % (float(s_sigC.val), state["sig_seed"]),
                     SIG_C, "%d / %d wrong · %s" % (sg["miss"], N, fmt_time(sg["time"]))))
    if not rows:
        ax_results.text(0.05, 0.55, "Click two points on the plot to draw your line.",
                        fontsize=9.5, va="top", color="#566", style="italic")
        return
    yv, step = 0.81, min(0.135, 0.70 / len(rows))
    for label, col, rt in rows:
        ax_results.text(0.05, yv, label, fontsize=8.5, fontweight="bold", va="top", color=col)
        ax_results.text(0.965, yv, rt, fontsize=8.6, fontweight="bold", va="top",
                        ha="right", color="#16324f")
        yv -= step
    manual = [m for m in (you, wb) if m]
    surro = [s for s in (sv, lg) if s]
    opt_ok = opt and not opt.get("infeasible")
    if state["algo"] == "sigmoid" and sg:
        tail = (" — vs MILP %+d" % (sg["miss"] - opt["miss"])) if opt_ok else ""
        ax_results.text(0.05, yv + 0.01,
                        "NONCONVEX: this is start #%d's LOCAL optimum%s.\n"
                        "Press  Sigmoid  again (new start) — the count can jump." %
                        (state["sig_seed"], tail),
                        fontsize=8.0, fontweight="bold", va="top", color=SIG_C)
    elif opt_ok and surro:
        extra = min(s["miss"] for s in surro) - opt["miss"]
        speed = opt["time"] / max(min(s["time"] for s in surro), 1e-6)
        ax_results.text(0.05, yv + 0.01,
                        "MILP: fewest errors but slowest.\n"
                        "Surrogates ≈ %.0f× faster, %+d error(s) — a convex proxy." % (speed, extra),
                        fontsize=8.0, fontweight="bold", va="top", color="#5a2d91")
    elif opt_ok and manual:
        gap = min(m["miss"] for m in manual) - opt["miss"]
        if gap > 0:
            msg, c = "MILP beats your best by %d — tune w, b to close it." % gap, "#b03030"
        elif gap == 0:
            msg, c = "You matched the MILP optimum! 🎯", "#1a7a3a"
        else:
            msg, c = "'Optimum' is worse — ε too small (degenerate). Raise ε.", "#a8330a"
        ax_results.text(0.05, yv - 0.02, msg, fontsize=8.4, fontweight="bold", va="top", color=c)
    elif opt and opt.get("infeasible"):
        ax_results.text(0.05, yv - 0.02,
                        "big-M too small → the MILP is INFEASIBLE. Raise big-M.",
                        fontsize=8.4, fontweight="bold", va="top", color="#b03030")
    elif not opt:
        ax_results.text(0.05, yv - 0.02,
                        "Set big-M, then press  Optimal (MILP)  (it shows the solve time).",
                        fontsize=8.1, va="top", color="#566", style="italic")
    else:
        ax_results.text(0.05, yv - 0.02,
                        "Draw a line or press  Tune w, b  to compare with the optimum.",
                        fontsize=8.1, va="top", color="#566", style="italic")


def draw_results_nn():
    _box(ax_results, "#fdeef4", "#e3a7c2")
    N = len(state["y"])
    ax_results.text(0.035, 0.93, "NEURAL NET — in-sample misclassified · train time",
                    fontsize=9.3, fontweight="bold", va="top", color="#8a1c50")
    # best straight-line classifier, as the reference to beat
    lin = [(m["miss"], nm) for m, nm in ((state["opt"], "MILP"), (state["svm"], "SVM"),
                                         (state["logistic"], "logistic"))
           if m and not m.get("infeasible")]
    yv = 0.77
    if lin:
        bm, bn = min(lin)
        ax_results.text(0.05, yv, "best straight line (%s)" % bn, fontsize=8.4,
                        fontweight="bold", va="top", color=OPT)
        ax_results.text(0.965, yv, "%d / %d wrong" % (bm, N), fontsize=8.5,
                        fontweight="bold", va="top", ha="right", color="#16324f")
        yv -= 0.145
    for s in NN_ORDER:
        r = state["nn"][s]
        active = s == state["nn_size"]
        units = "→".join(str(h) for h in NN_SPECS[s][0])
        ax_results.text(0.05, yv, ("▶ " if active else "   ") + "NN %s  (%s)" % (s, units),
                        fontsize=8.4, fontweight="bold" if active else "normal", va="top",
                        color=NN_C if active else "#7a4a60")
        ax_results.text(0.965, yv, "%d / %d wrong · %s" % (r["miss"], N, fmt_time(r["time"])),
                        fontsize=8.5, fontweight="bold" if active else "normal", va="top",
                        ha="right", color="#16324f")
        yv -= 0.145
    ax_results.text(0.05, yv - 0.01,
                    "Bigger net → fewer in-sample errors; the large one ≈ 100%.\n"
                    "But that is MEMORISING the data (overfitting), not real skill.",
                    fontsize=7.9, fontweight="bold", va="top", color="#b01b56")


# ── interaction ───────────────────────────────────────────────────────────────
_prog = {"box": None}


def show_progress(msg):
    """A force-rendered overlay so a slow solve does not look frozen."""
    hide_progress()
    _prog["box"] = fig.text(0.315, 0.62, msg, ha="center", va="center", fontsize=13,
                            fontweight="bold", color="#7a2e12", zorder=100,
                            bbox=dict(boxstyle="round,pad=0.7", fc="#ffe7c4",
                                      ec="#e0992f", lw=2.2))
    fig.canvas.draw(); fig.canvas.flush_events()


def hide_progress():
    if _prog["box"] is not None:
        _prog["box"].remove(); _prog["box"] = None


def on_click(event):
    if event.inaxes is not ax or event.xdata is None:
        return
    state["bigm"] = None
    if len(state["clicks"]) >= 2:
        state["clicks"] = []
    state["clicks"].append((event.xdata, event.ydata))
    if len(state["clicks"]) == 2:
        w, b = line_from_clicks(*state["clicks"])
        miss, mask, (w, b) = evaluate(w, b)
        state["you"] = {"w": w, "b": b, "miss": miss, "mask": mask}
    draw()


def on_opt(_):
    state["bigm"] = None
    state["algo"] = "milp"              # model panel shows the MILP formulation
    show_progress("solving the MILP…")
    solve_milp()
    hide_progress()
    draw()


def on_more(_):
    state["show_algos"] = True          # reveal the other classifiers
    b_svm.ax.set_visible(True); b_log.ax.set_visible(True)
    b_sig.ax.set_visible(True); b_nn.ax.set_visible(True)
    draw()


def on_nn(_):
    if not state["show_algos"]:
        return
    state["bigm"] = None; state["algo"] = "nn"
    if state["nn"] is None:             # train all three sizes once (cached)
        show_progress("training neural nets…\nsmall → medium → large")
        state["nn"] = {s: train_nn(s) for s in NN_ORDER}
        hide_progress()
        state["nn_size"] = "small"
    else:                              # subsequent clicks cycle small → medium → large
        i = NN_ORDER.index(state["nn_size"])
        state["nn_size"] = NN_ORDER[(i + 1) % len(NN_ORDER)]
    b_nn.label.set_text("Neural net: " + state["nn_size"])
    draw()


def on_svm(_):
    if not state["show_algos"]:
        return
    state["bigm"] = None; state["algo"] = "svm"
    if state["svm"] is None:
        state["svm"] = solve_svm()
    draw()


def on_log(_):
    if not state["show_algos"]:
        return
    state["bigm"] = None; state["algo"] = "logistic"
    if state["logistic"] is None:
        state["logistic"] = solve_logistic()
    draw()


def on_sig(_):
    if not state["show_algos"]:
        return
    state["bigm"] = None; state["algo"] = "sigmoid"
    # each press is a NEW random start → a (possibly) different local optimum
    state["sig_seed"] = 0 if state["sigmoid"] is None else state["sig_seed"] + 1
    state["sigmoid"] = solve_sigmoid(C=float(s_sigC.val), seed=state["sig_seed"])
    draw()


def on_sigC(_):
    # re-solve from the SAME start so the C slider's effect is isolated
    if state["sigmoid"] is not None:
        state["bigm"] = None
        state["sigmoid"] = solve_sigmoid(C=float(s_sigC.val), seed=state["sig_seed"])
        draw()


def on_data(_):
    state["bigm"] = None
    gen_data(int(s_n.val), float(s_noise.val))
    state["sig_seed"] = 0                                            # sigmoid restarts reset
    state["nn_size"] = "small"; b_nn.label.set_text("Neural net")    # nets reset on new data
    draw()


def on_margin(_):
    state["bigm"] = None
    if state["opt"] is not None:        # re-solve so students see errors stay the same
        solve_milp()
    draw()


def on_wb(_):
    state["bigm"] = None
    draw()                              # only shows the W,B line once it's been activated


def on_wb_btn(_):
    state["bigm"] = None
    state["wb_on"] = True               # reveal the W,B classifier (after the manual try)
    draw()


def on_bigm(_):
    state["algo"] = "milp"             # big-M is a MILP lesson
    show_progress("solving at several big-M values…\n(this can take a moment)")
    compute_bigm()
    hide_progress()
    draw()


def on_bigm_slider(_):
    s_bigm.valtext.set_text("%.0f" % (10 ** s_bigm.val))   # show the actual M, not log
    state["opt"] = None        # the green line is stale until you re-solve with the new M
    state["bigm"] = None
    draw()


fig.canvas.mpl_connect("button_press_event", on_click)
s_n.on_changed(on_data)
s_noise.on_changed(on_data)
s_margin.on_changed(on_margin)
s_bigm.on_changed(on_bigm_slider)
s_w1.on_changed(on_wb)
s_w2.on_changed(on_wb)
s_b.on_changed(on_wb)
b_wb.on_clicked(on_wb_btn)
b_opt.on_clicked(on_opt)
b_svm.on_clicked(on_svm)
b_log.on_clicked(on_log)
b_sig.on_clicked(on_sig)
s_sigC.on_changed(on_sigC)
b_nn.on_clicked(on_nn)
b_more.on_clicked(on_more)
b_bigm.on_clicked(on_bigm)
b_reset.on_clicked(on_data)

draw()

if __name__ == "__main__":
    plt.show()
