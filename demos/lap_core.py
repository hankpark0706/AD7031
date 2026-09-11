"""
lap_core.py  —  Self-contained, *parameterized* two-stage LAP + SAA solver.

No cvxpy.  Pure numpy / scipy (HiGHS).  The model is the same basic two-stage
SAA problem as SAA_ResAlloc.py, but the INSTANCE (network size, node positions,
arc transport costs, demand scenarios) is generated synthetically so it can be
driven by sliders, and transport cost = Euclidean distance (intuitive to show
on the map).

Two teaching levers the demo exposes:
  • network size  n  (number of candidate nodes / demand regions)
  • facilities    p  (how many locations you are allowed to open)  -> sum(x)=p

Model
-----
Stage 1 (here-and-now):
    x_i in {0,1}   open a facility at node i      (sum_i x_i = p, if p given)
    r_i >= 0       reserve pre-positioned at i     (sum_i r_i <= budget,
                                                     r_i <= budget * x_i)
    cost: c.x + f.r
Stage 2 (recourse, after demand zeta is revealed):
    y_arc in [0, y_max]   flow along each arc      (cost = dist)
    q_i  in [0, zeta_i]   shortage (unmet demand)  (cost = penalty)
    s_i  >= 0             surplus                   (cost = small)
    balance:  inflow_i - outflow_i + q_i - s_i  >=  zeta_i - r_i
SAA objective:
    min  c.x + f.r + (1/|train|) sum_m [ stage-2 cost(m) ]
"""
import time
import numpy as np
import scipy.sparse as sp
from scipy.optimize import milp, linprog, LinearConstraint, Bounds

# Baseline mean demand per node — fixed (absolute), INDEPENDENT of the budget B,
# so that B can be varied as a free knob against a fixed demand distribution.
DEMAND_MU = 10.0
DEFAULT_BUDGET_PER_NODE = 20.0      # default B = 20 * n_nodes (≈ 2× baseline demand)


# ── instance ──────────────────────────────────────────────────────────────────
class Instance:
    """A generated two-stage LAP instance."""

    def __init__(self, n_nodes=10, n_scenarios=200, seed=1,
                 k_neighbors=3, budget=None,
                 trans_scale=12.0, shortage_penalty=60.0, surplus_cost=1.0,
                 contextual=False, forecast_noise=0.12, surge_radius=0.25,
                 ctx_base=2.0, ctx_surge=12.0):
        rng = np.random.default_rng(seed)
        self.seed = seed
        self.I = int(n_nodes)
        self.budget = float(budget) if budget else DEFAULT_BUDGET_PER_NODE * self.I
        self.shortage_penalty = shortage_penalty
        self.surplus_cost = surplus_cost
        self.trans_scale = trans_scale

        # node positions (the map) ------------------------------------------------
        pos = rng.uniform(0.06, 0.94, size=(self.I, 2))
        self.pos = pos

        # build an undirected graph: k-nearest neighbours + a spanning tree so the
        # network is always connected, then make every edge bidirectional --------
        D = np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=2)
        edges = set()
        for i in range(self.I):
            order = np.argsort(D[i])
            for j in order[1:k_neighbors + 1]:
                edges.add((min(i, int(j)), max(i, int(j))))
        # spanning tree (Prim) on distance to guarantee connectivity
        tree_edges = set()
        in_tree = {0}
        while len(in_tree) < self.I:
            best = None
            for a in in_tree:
                for b in range(self.I):
                    if b not in in_tree and (best is None or D[a, b] < best[0]):
                        best = (D[a, b], a, b)
            e = (min(best[1], best[2]), max(best[1], best[2]))
            tree_edges.add(e); edges.add(e)
            in_tree.add(best[2])

        # Directionality: spanning-tree edges stay TWO-WAY (so the network is
        # always strongly connected — no node is permanently unreachable).
        # Every *extra* edge gets a random orientation: two-way / one-way each way.
        line = []
        for (a, b) in sorted(edges):
            if (a, b) in tree_edges:
                line += [(a, b), (b, a)]
            else:
                roll = rng.random()
                if roll < 0.34:    line += [(a, b), (b, a)]
                elif roll < 0.67:  line.append((a, b))
                else:              line.append((b, a))
        self.line_mat = np.array(line, dtype=int)
        self.A = self.line_mat.shape[0]
        self.N2 = self.A + 2 * self.I

        # arc transport cost = distance * scale ----------------------------------
        self.trans = np.array([D[a, b] * trans_scale for a, b in self.line_mat])
        # arc capacity is generous and INDEPENDENT of B, so that the binding
        # supply constraint is the reserve budget, not the pipes.
        self.y_max = 3.0 * DEMAND_MU * self.I
        self.M = self.budget                       # a facility may hold up to B

        # first-stage costs -------------------------------------------------------
        self.c = np.round(rng.uniform(60, 140, size=self.I), 0)     # open cost
        self.f = np.round(rng.uniform(1.0, 3.0, size=self.I), 2)    # reserve unit cost

        # second-stage costs ------------------------------------------------------
        self.v_under = np.full(self.I, shortage_penalty)
        self.v_over = np.full(self.I, surplus_cost)
        self.a = np.concatenate([self.trans, self.v_under, self.v_over])

        # incidence ---------------------------------------------------------------
        self.to_mat = np.zeros((self.I, self.A))
        self.from_mat = np.zeros((self.I, self.A))
        for arc, (frm, to) in enumerate(self.line_mat):
            self.from_mat[frm, arc] = 1.0
            self.to_mat[to, arc] = 1.0

        # demand scenarios: baseline + localized "disaster" surges ----------------
        # NOTE: mu is fixed (absolute), so demand does NOT move with the budget B.
        self.num_data = int(n_scenarios)
        self.contextual = bool(contextual)
        mu = DEMAND_MU                           # baseline mean per node (fixed)
        demand = []
        if self.contextual:
            # CONTEXTUAL DGP (Module 3): each disaster has a continuous true epicenter
            # e on the map; demand surges around e (decaying with distance); the
            # forecast gamma is a NOISY observation of e.  The history is {(gamma, xi)}.
            self.forecast_noise = float(forecast_noise)
            self.surge_radius = float(surge_radius)
            self.ctx_base = float(ctx_base)
            self.ctx_surge = float(ctx_surge)
            self.epicenter = np.zeros((self.num_data, 2))
            self.forecast = np.zeros((self.num_data, 2))
            self.severity = np.zeros(self.num_data)
            for n in range(self.num_data):
                # SMALL baseline so the localized (forecastable) surge dominates
                base = np.clip(rng.normal(ctx_base, 0.4 * ctx_base, self.I), 0, None)
                e = rng.uniform(0.06, 0.94, size=2)               # true epicenter
                S = (rng.uniform(3.5, 6.5) if rng.random() < 0.35  # severity (not forecast)
                     else rng.uniform(1.0, 2.5)) * ctx_surge
                dist = np.linalg.norm(pos - e, axis=1)
                base += S * np.exp(-(dist / self.surge_radius) ** 2)
                demand.append(np.round(base, 1))
                self.epicenter[n] = e
                self.severity[n] = S
                self.forecast[n] = np.clip(e + rng.normal(0, self.forecast_noise, 2),
                                           0.0, 1.0)
        else:
            for _ in range(self.num_data):
                base = np.clip(rng.normal(mu, 0.3 * mu, self.I), 0, None)
                # a disaster hits a random node + its neighbours; ~35% are "major"
                center = rng.integers(self.I)
                major = rng.random() < 0.35
                if major:
                    surge = rng.uniform(3.5, 6.5) * mu
                    near = np.argsort(D[center])[:max(3, self.I // 3)]
                else:
                    surge = rng.uniform(1.0, 2.5) * mu
                    near = np.argsort(D[center])[:max(2, self.I // 5)]
                base[near] += surge * rng.uniform(0.6, 1.0, size=len(near))
                demand.append(np.round(base, 1))
        self.demand = demand

        self.layout = pos.tolist()

    def set_economics(self, budget=None, shortage_penalty=None, surplus_cost=None,
                      facility_cost=None, reserve_cost=None):
        """Update cost / budget knobs in place (network and demand unchanged).
        facility_cost / reserve_cost set a UNIFORM c_j / f_j across all nodes."""
        if budget is not None:
            self.budget = float(budget)
            self.M = float(budget)
        if shortage_penalty is not None:
            self.shortage_penalty = float(shortage_penalty)
            self.v_under = np.full(self.I, float(shortage_penalty))
        if surplus_cost is not None:
            self.surplus_cost = float(surplus_cost)
            self.v_over = np.full(self.I, float(surplus_cost))
        if facility_cost is not None:
            self.c = np.full(self.I, float(facility_cost))
        if reserve_cost is not None:
            self.f = np.full(self.I, float(reserve_cost))
        self.a = np.concatenate([self.trans, self.v_under, self.v_over])

    # second-stage variable indexing within one scenario block of length N2
    def arc(self, k):      return k
    def short(self, i):    return self.A + i
    def surplus(self, i):  return self.A + self.I + i


# ── stage 1: SAA optimisation ─────────────────────────────────────────────────
def solve_saa(data: Instance, train, n_facilities=None, demands=None, relax=False,
              min_stock=True, fix_open=None, weights=None,
              mip_rel_gap=0.005, time_limit=15):
    """Solve SAA over `train` scenario indices, OR over an explicit list of demand
    vectors `demands` (e.g. a single mean vector → the deterministic/EV model).
    n_facilities: if given, cap the number of open facilities (sum x <= p).
    relax=True drops the integrality on x (x_j in [0,1]) → the LP relaxation.
    min_stock=True forces each opened facility to hold >= a floor (used by Module 2
    to spread reserve); set False for the TRUE optimum (Module 1).
    fix_open: if given (an iterable of node indices), FIX x to that open set
    (x_j = 1 for j in fix_open, else 0) and optimise only r + the recourse — a pure
    LP. This is the 'fix the rounded open decision, re-solve for the reserves' step
    used by Module 1's LP-relaxation-vs-IP lesson."""
    I, A, N2 = data.I, data.A, data.N2
    train = list(train)
    scen = demands if demands is not None else [data.demand[m] for m in train]
    nT = len(scen)
    nvar = 2 * I + nT * N2

    def Rc(i):     return I + i
    def Y(k, j):   return 2 * I + k * N2 + j

    if weights is not None:                    # contextual / weighted SAA (Module 3)
        w = np.asarray(weights, float)
        w = w / w.sum() if w.sum() > 0 else np.full(nT, 1.0 / nT)
    else:
        w = np.full(nT, 1.0 / nT)              # plain SAA: equal weights

    cobj = np.zeros(nvar)
    cobj[0:I] = data.c
    cobj[I:2 * I] = data.f
    for k in range(nT):
        cobj[2 * I + k * N2: 2 * I + (k + 1) * N2] = data.a * w[k]

    lb = np.zeros(nvar)
    ub = np.full(nvar, np.inf)
    ub[0:I] = 1.0
    for k in range(nT):
        zeta = scen[k]
        for arc in range(A):
            ub[Y(k, data.arc(arc))] = data.y_max
        for i in range(I):
            ub[Y(k, data.short(i))] = zeta[i]
    if fix_open is not None:                   # pin x to a chosen open set -> pure LP
        openset = set(int(i) for i in fix_open)
        for i in range(I):
            v = 1.0 if i in openset else 0.0
            lb[i] = v; ub[i] = v
        relax = True
    integrality = np.zeros(nvar)
    integrality[0:I] = 0 if relax else 1       # relax=True -> LP relaxation of x

    rows, cols, vals, lo, hi = [], [], [], [], []
    rc = 0

    # budget: sum r <= budget
    for i in range(I):
        rows.append(rc); cols.append(Rc(i)); vals.append(1.0)
    lo.append(-np.inf); hi.append(data.budget); rc += 1

    # linking: r_i - M x_i <= 0
    for i in range(I):
        rows += [rc, rc]; cols += [Rc(i), i]; vals += [1.0, -data.M]
        lo.append(-np.inf); hi.append(0.0); rc += 1

    # cardinality: sum x <= p  (open AT MOST p facilities — the optimizer may open
    # fewer, or none, if opening is not worth the cost c_j)
    if n_facilities is not None:
        for i in range(I):
            rows.append(rc); cols.append(i); vals.append(1.0)
        lo.append(-np.inf); hi.append(float(n_facilities)); rc += 1
        # min stock if open: r_i >= rmin * x_i  -> rmin*x_i - r_i <= 0
        if min_stock:
            rmin = 0.35 * data.budget / max(1, n_facilities)
            for i in range(I):
                rows += [rc, rc]; cols += [i, Rc(i)]; vals += [rmin, -1.0]
                lo.append(-np.inf); hi.append(0.0); rc += 1

    # balance per scenario per node
    for k in range(nT):
        zeta = scen[k]
        for i in range(I):
            for arc in np.where(data.to_mat[i])[0]:
                rows.append(rc); cols.append(Y(k, data.arc(arc))); vals.append(1.0)
            for arc in np.where(data.from_mat[i])[0]:
                rows.append(rc); cols.append(Y(k, data.arc(arc))); vals.append(-1.0)
            rows.append(rc); cols.append(Y(k, data.short(i))); vals.append(1.0)
            rows.append(rc); cols.append(Y(k, data.surplus(i))); vals.append(-1.0)
            rows.append(rc); cols.append(Rc(i)); vals.append(1.0)
            # flow conservation (EQUALITY): in - out + q - s + r = demand
            lo.append(zeta[i]); hi.append(zeta[i]); rc += 1

    Amat = sp.coo_matrix((vals, (rows, cols)), shape=(rc, nvar)).tocsr()
    # A small optimality-gap tolerance + time limit keep big instances (large N
    # and/or many nodes) interactive; 0.5% is negligible for teaching.
    _t0 = time.perf_counter()
    res = milp(c=cobj,
               constraints=LinearConstraint(Amat, np.array(lo), np.array(hi)),
               integrality=integrality, bounds=Bounds(lb, ub),
               options={'mip_rel_gap': mip_rel_gap, 'time_limit': time_limit})
    solve_time = time.perf_counter() - _t0
    if res.x is None:
        raise RuntimeError(f"SAA solve failed: {res.message}")

    z = res.x
    xr = z[0:I]
    x = xr.tolist() if relax else np.round(xr).astype(int).tolist()   # keep fractional if relaxed
    r = z[I:2 * I]
    r[r < 1e-6] = 0.0
    fs = float(data.c @ np.asarray(x) + data.f @ r)
    return {'x': x, 'r': [round(v, 1) for v in r],
            'obj': float(res.fun), 'first_stage_cost': fs,
            'expected_recourse': float(res.fun) - fs,
            'time': solve_time, 'relaxed': relax}


def solve_ev(data: Instance, train, n_facilities=None):
    """Deterministic 'expected-value' model: average the training demand vectors
    into ONE mean scenario and solve as if that mean is certain. This is the
    classic naive baseline that SAA is meant to beat out-of-sample."""
    mean_d = np.mean([np.asarray(data.demand[m], float) for m in train], axis=0)
    return solve_saa(data, [0], n_facilities=n_facilities, demands=[mean_d])


# ── score a FIXED plan (x, r) against ONE demand vector ───────────────────────
def score_plan(data: Instance, x, r, demand):
    """Given fixed first-stage decisions (x, r), ship optimally to meet `demand`
    and return the cost breakdown. Used for out-of-sample eval AND for Module 1
    (scoring a human's hand-made plan, or a rounded LP-relaxation strategy)."""
    I, A, N2 = data.I, data.A, data.N2
    x = np.asarray(x, float); r = np.asarray(r, float)
    zeta = np.asarray(demand, float)

    cobj = data.a.copy()
    lb = np.zeros(N2); ub = np.full(N2, np.inf)
    ub[0:A] = data.y_max
    for i in range(I):
        ub[data.short(i)] = zeta[i]

    rows, cols, vals, beq = [], [], [], []
    for i in range(I):
        for arc in np.where(data.to_mat[i])[0]:
            rows.append(i); cols.append(data.arc(arc)); vals.append(1.0)
        for arc in np.where(data.from_mat[i])[0]:
            rows.append(i); cols.append(data.arc(arc)); vals.append(-1.0)
        rows.append(i); cols.append(data.short(i)); vals.append(1.0)
        rows.append(i); cols.append(data.surplus(i)); vals.append(-1.0)
        beq.append(zeta[i] - r[i])               # in - out + q - s = demand - r

    Amat = sp.coo_matrix((vals, (rows, cols)), shape=(I, N2)).tocsr()
    res = linprog(cobj, A_eq=Amat, b_eq=np.array(beq),
                  bounds=list(zip(lb, ub)), method='highs')
    if not res.success:
        raise RuntimeError(f"OOS LP failed: {res.message}")

    y = res.x
    flows = y[0:A]; short = y[A:A + I]; surplus = y[A + I:A + 2 * I]
    short[short < 1e-6] = 0.0; flows[flows < 1e-6] = 0.0
    second = float(data.a @ y)
    first = float(data.c @ x + data.f @ r)
    return {'demand': [round(v, 1) for v in zeta],
            'shortages': [round(v, 1) for v in short],
            'surplus': [round(v, 1) for v in surplus],
            'flows': [round(v, 1) for v in flows],
            'second_stage_cost': second, 'first_stage_cost': first,
            'total_cost': first + second,
            'total_shortage': float(short.sum()),
            'total_demand': float(zeta.sum())}


def eval_oos(data: Instance, x, r, scenario_idx):
    """Out-of-sample evaluation against stored scenario `scenario_idx`."""
    return score_plan(data, x, r, data.demand[scenario_idx])


def oos_cost_distribution(data: Instance, x, r, scenario_ids):
    """Evaluate the recourse LP over many scenarios (the incidence matrix is
    built once).  Returns (costs, shortages) as numpy arrays, one per scenario."""
    I, A, N2 = data.I, data.A, data.N2
    x = np.asarray(x, float); r = np.asarray(r, float)

    rows, cols, vals = [], [], []
    for i in range(I):
        for arc in np.where(data.to_mat[i])[0]:
            rows.append(i); cols.append(data.arc(arc)); vals.append(1.0)
        for arc in np.where(data.from_mat[i])[0]:
            rows.append(i); cols.append(data.arc(arc)); vals.append(-1.0)
        rows.append(i); cols.append(data.short(i)); vals.append(1.0)
        rows.append(i); cols.append(data.surplus(i)); vals.append(-1.0)
    A_eq = sp.coo_matrix((vals, (rows, cols)), shape=(I, N2)).tocsr()

    cobj = data.a
    first = float(data.c @ x + data.f @ r)
    lb = np.zeros(N2)
    costs = np.empty(len(scenario_ids))
    shorts = np.empty(len(scenario_ids))
    for t, sid in enumerate(scenario_ids):
        zeta = data.demand[int(sid)]
        ub = np.full(N2, np.inf); ub[0:A] = data.y_max
        ub[A:A + I] = zeta                       # shortage q_i <= demand_i
        b_eq = zeta - r                          # in - out + q - s = demand - r
        res = linprog(cobj, A_eq=A_eq, b_eq=b_eq,
                      bounds=list(zip(lb, ub)), method='highs')
        y = res.x
        costs[t] = first + float(cobj @ y)
        shorts[t] = float(y[A:A + I].sum())
    return costs, shorts


# ── contextual prediction (Module 3) ──────────────────────────────────────────
def context_weights(forecasts, epicenters, gamma_star, method,
                    k=None, h=None, h_oracle=None):
    """Weights over historical scenarios given the query forecast `gamma_star`.

    forecasts / epicenters : (N,2) arrays (gamma^n and the true e^n).
    method : 'saa' (uniform), 'knn', 'kernel' (Nadaraya-Watson on forecast distance),
             or 'oracle' (kernel on the TRUE epicenter distance — a perfect-forecast
             upper bound).  Returns a length-N weight vector summing to 1.
    """
    forecasts = np.asarray(forecasts, float)
    N = len(forecasts)
    g = np.asarray(gamma_star, float)
    if method == "saa":
        return np.full(N, 1.0 / N)
    if method == "oracle":
        d = np.linalg.norm(np.asarray(epicenters, float) - g, axis=1)
        hh = h_oracle if h_oracle is not None else (np.median(d) + 1e-9)
        w = np.exp(-0.5 * (d / hh) ** 2)
        s = w.sum()
        return w / s if s > 0 else np.full(N, 1.0 / N)
    d = np.linalg.norm(forecasts - g, axis=1)
    if method == "knn":
        kk = k if k else max(1, int(round(np.sqrt(N))))
        idx = np.argsort(d)[:kk]
        w = np.zeros(N); w[idx] = 1.0 / len(idx)
        return w
    # kernel / Nadaraya-Watson
    hh = h if h is not None else max(np.median(d), 1e-6)
    w = np.exp(-0.5 * (d / hh) ** 2)
    s = w.sum()
    return w / s if s > 0 else np.full(N, 1.0 / N)


def sample_conditional(data: Instance, gamma_star, m, rng, sigma=None):
    """Draw `m` demand vectors from the TRUE conditional f(xi | gamma_star): the
    posterior epicenter e ~ N(gamma_star, sigma^2) (uniform prior), then severity +
    the surge.  Used as the honest out-of-sample test set for a query forecast."""
    sig = data.forecast_noise if sigma is None else float(sigma)
    rho = data.surge_radius
    b_mu, s_mu = data.ctx_base, data.ctx_surge      # MUST match the training DGP
    g = np.asarray(gamma_star, float)
    out = []
    for _ in range(m):
        e = np.clip(g + rng.normal(0, sig, 2), 0.0, 1.0)
        base = np.clip(rng.normal(b_mu, 0.4 * b_mu, data.I), 0, None)
        S = (rng.uniform(3.5, 6.5) if rng.random() < 0.35 else rng.uniform(1.0, 2.5)) * s_mu
        base += S * np.exp(-(np.linalg.norm(data.pos - e, axis=1) / rho) ** 2)
        out.append(np.round(base, 1))
    return out


def forecasts_at(data: Instance, sigma, rng):
    """Re-derive forecasts gamma = e + N(0, sigma^2) from the stored true epicenters
    (demand is NOT resampled) — used to build the fixed forecast-accuracy presets."""
    g = data.epicenter + rng.normal(0, float(sigma), size=data.epicenter.shape)
    return np.clip(g, 0.0, 1.0)


def score_costs(data: Instance, x, r, demands):
    """Realized total cost + shortage of a fixed plan (x, r) over an explicit list of
    demand vectors (incidence built once).  Returns (costs, shorts) numpy arrays."""
    I, A, N2 = data.I, data.A, data.N2
    x = np.asarray(x, float); r = np.asarray(r, float)
    rows, cols, vals = [], [], []
    for i in range(I):
        for arc in np.where(data.to_mat[i])[0]:
            rows.append(i); cols.append(data.arc(arc)); vals.append(1.0)
        for arc in np.where(data.from_mat[i])[0]:
            rows.append(i); cols.append(data.arc(arc)); vals.append(-1.0)
        rows.append(i); cols.append(data.short(i)); vals.append(1.0)
        rows.append(i); cols.append(data.surplus(i)); vals.append(-1.0)
    A_eq = sp.coo_matrix((vals, (rows, cols)), shape=(I, N2)).tocsr()
    cobj = data.a
    first = float(data.c @ x + data.f @ r)
    lb = np.zeros(N2)
    costs = np.empty(len(demands)); shorts = np.empty(len(demands))
    for t, dem in enumerate(demands):
        zeta = np.asarray(dem, float)
        ub = np.full(N2, np.inf); ub[0:A] = data.y_max; ub[A:A + I] = zeta
        res = linprog(cobj, A_eq=A_eq, b_eq=zeta - r, bounds=list(zip(lb, ub)),
                      method='highs')
        y = res.x
        costs[t] = first + float(cobj @ y)
        shorts[t] = float(y[A:A + I].sum())
    return costs, shorts


if __name__ == '__main__':
    inst = Instance(n_nodes=12, n_scenarios=200, seed=3)
    print(f"I={inst.I} nodes, A={inst.A} arcs, budget={inst.budget:.0f}, "
          f"{inst.num_data} scenarios")
    for p in (1, 2, 3, 4):
        sol = solve_saa(inst, range(20), n_facilities=p)
        opened = [i for i, xi in enumerate(sol['x']) if xi]
        rr = [round(v, 1) for v in sol['r'] if v > 0]
        print(f"p={p}: open {opened}  reserves {rr}  obj={sol['obj']:.0f}")
    sol = solve_saa(inst, range(20), n_facilities=3)
    nshort = 0
    for s in range(20, inst.num_data):
        ev = eval_oos(inst, sol['x'], sol['r'], s)
        if ev['total_shortage'] > 0.05:
            nshort += 1
    print(f"\nOOS (p=3): {nshort}/{inst.num_data-20} test scenarios show shortage")
    for s in range(20, 25):
        ev = eval_oos(inst, sol['x'], sol['r'], s)
        print(f"  scn {s}: demand={ev['total_demand']:.0f} "
              f"short={ev['total_shortage']:.1f} cost={ev['total_cost']:.0f}")
