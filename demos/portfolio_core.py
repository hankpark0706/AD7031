"""Core models for the single-stage portfolio teaching dashboard.

The module deliberately uses a small, fixed six-sector market.  Returns are
monthly and synthetic, drawn from a transparent Gaussian regime mixture.  The
mixture is the dashboard's fixed "true" distribution: training samples and
out-of-sample tests can be regenerated indefinitely without changing the
underlying problem.

All portfolios are long-only and fully invested.  The optimization models are:

* Markowitz mean-variance (convex QP)
* sample-average CVaR (LP)
* sample-average VaR (big-M MIP)
* sample-average chance constraint (big-M MIP)

Only NumPy and SciPy/HiGHS are required.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, linprog, milp, minimize


SECTORS = (
    "Consumer",
    "Manufacturing",
    "Energy",
    "Technology",
    "Healthcare",
    "Utilities",
)

# Validated categorical palette (fixed hue order, adjacent-pair CVD-safe):
# blue, orange, aqua, yellow, magenta, green.
SECTOR_COLORS = (
    "#2a78d6",
    "#eb6834",
    "#1baf7a",
    "#eda100",
    "#e87ba4",
    "#008300",
)


@dataclass(frozen=True)
class Regime:
    name: str
    probability: float
    mean: np.ndarray
    covariance: np.ndarray


@dataclass
class PortfolioSolution:
    model: str
    weights: np.ndarray
    objective: float
    expected_return: float
    variance: float
    std_dev: float
    var: float
    cvar: float
    train_losses: np.ndarray
    solve_time: float
    status: str
    threshold: float | None = None
    breaches: int | None = None
    allowed_breaches: int | None = None


def _covariance(vol: Iterable[float], corr: np.ndarray, scale: float = 1.0) -> np.ndarray:
    vol = np.asarray(tuple(vol), dtype=float) * scale
    return np.outer(vol, vol) * corr


def default_regimes() -> tuple[Regime, ...]:
    """Return the fixed classroom distribution for monthly sector returns."""
    corr = np.array(
        [
            [1.00, .62, .34, .53, .48, .36],
            [.62, 1.00, .55, .67, .42, .40],
            [.34, .55, 1.00, .30, .25, .43],
            [.53, .67, .30, 1.00, .40, .22],
            [.48, .42, .25, .40, 1.00, .47],
            [.36, .40, .43, .22, .47, 1.00],
        ],
        dtype=float,
    )
    vol = np.array([.032, .047, .070, .073, .039, .026])

    return (
        Regime(
            "ordinary",
            .900,
            np.array([.008, .011, .013, .020, .009, .006]),
            _covariance(vol, corr),
        ),
        Regime(
            "broad downturn",
            .060,
            np.array([-.060, -.080, -.060, -.110, -.035, -.020]),
            _covariance(vol, corr, 1.35),
        ),
        Regime(
            "technology shock",
            .025,
            np.array([-.020, -.030, .025, -.180, -.005, .005]),
            _covariance(vol, corr, 1.15),
        ),
        Regime(
            "energy shock",
            .015,
            np.array([.005, -.020, -.180, .005, .006, .008]),
            _covariance(vol, corr, 1.10),
        ),
    )


class SectorMarket:
    """Fixed synthetic market with unlimited iid monthly scenarios."""

    def __init__(self, regimes: tuple[Regime, ...] | None = None):
        self.sectors = SECTORS
        self.colors = SECTOR_COLORS
        self.regimes = regimes or default_regimes()
        probs = np.array([r.probability for r in self.regimes], dtype=float)
        if not np.isclose(probs.sum(), 1.0):
            raise ValueError("regime probabilities must sum to one")
        self.probabilities = probs
        self.mean, self.covariance = self._population_moments()

    @property
    def n_assets(self) -> int:
        return len(self.sectors)

    def _population_moments(self) -> tuple[np.ndarray, np.ndarray]:
        mean = sum(r.probability * r.mean for r in self.regimes)
        second = np.zeros((self.n_assets, self.n_assets))
        for r in self.regimes:
            second += r.probability * (r.covariance + np.outer(r.mean, r.mean))
        covariance = second - np.outer(mean, mean)
        return np.asarray(mean), covariance

    def sample(self, n: int, seed: int | np.random.Generator) -> np.ndarray:
        """Draw n independent monthly sector-return vectors."""
        if n <= 0:
            raise ValueError("n must be positive")
        rng = seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)
        regime_ids = rng.choice(len(self.regimes), size=int(n), p=self.probabilities)
        out = np.empty((int(n), self.n_assets), dtype=float)
        for k, regime in enumerate(self.regimes):
            rows = np.flatnonzero(regime_ids == k)
            if rows.size:
                out[rows] = rng.multivariate_normal(regime.mean, regime.covariance, rows.size)
        # The chosen parameters make impossible returns extraordinarily unlikely;
        # clipping is a final numerical guard for a classroom wealth visualization.
        return np.clip(out, -0.80, 0.80)


def portfolio_losses(returns: np.ndarray, weights: np.ndarray) -> np.ndarray:
    return -np.asarray(returns, dtype=float) @ np.asarray(weights, dtype=float)


def empirical_var_cvar(losses: np.ndarray, epsilon: float) -> tuple[float, float]:
    losses = np.asarray(losses, dtype=float)
    if losses.size == 0:
        return float("nan"), float("nan")
    if not 0 < epsilon < 1:
        raise ValueError("epsilon must lie strictly between zero and one")
    beta = float(np.quantile(losses, 1.0 - epsilon, method="higher"))
    cvar = beta + float(np.maximum(losses - beta, 0.0).mean()) / epsilon
    return beta, cvar


def evaluate_portfolio(
    market: SectorMarket,
    weights: np.ndarray,
    returns: np.ndarray,
    epsilon: float,
) -> dict[str, float | np.ndarray]:
    weights = np.asarray(weights, dtype=float)
    losses = portfolio_losses(returns, weights)
    var, cvar = empirical_var_cvar(losses, epsilon)
    variance = float(weights @ market.covariance @ weights)
    return {
        "weights": weights,
        "expected_return": float(market.mean @ weights),
        "variance": variance,
        "std_dev": float(np.sqrt(max(variance, 0.0))),
        "var": var,
        "cvar": cvar,
        "losses": losses,
        "wealth": 100.0 * (1.0 - losses),
    }


def _check_target(market: SectorMarket, target_return: float) -> None:
    if target_return > float(market.mean.max()) + 1e-10:
        raise ValueError(
            f"target return {target_return:.4%} exceeds the largest sector mean "
            f"{market.mean.max():.4%}"
        )


def _solution(
    model: str,
    weights: np.ndarray,
    objective: float,
    market: SectorMarket,
    training_returns: np.ndarray,
    epsilon: float,
    solve_time: float,
    status: str,
    threshold: float | None = None,
    breaches: int | None = None,
    allowed_breaches: int | None = None,
) -> PortfolioSolution:
    metrics = evaluate_portfolio(market, weights, training_returns, epsilon)
    return PortfolioSolution(
        model=model,
        weights=np.asarray(weights),
        objective=float(objective),
        expected_return=float(metrics["expected_return"]),
        variance=float(metrics["variance"]),
        std_dev=float(metrics["std_dev"]),
        var=float(metrics["var"]),
        cvar=float(metrics["cvar"]),
        train_losses=np.asarray(metrics["losses"]),
        solve_time=float(solve_time),
        status=status,
        threshold=threshold,
        breaches=breaches,
        allowed_breaches=allowed_breaches,
    )


def solve_markowitz(
    market: SectorMarket,
    target_return: float,
    evaluation_returns: np.ndarray,
    epsilon: float = .05,
) -> PortfolioSolution:
    """Minimize population variance subject to the target expected return."""
    import time

    _check_target(market, target_return)
    n = market.n_assets
    x0 = np.full(n, 1.0 / n)
    if market.mean @ x0 < target_return:
        best = int(np.argmax(market.mean))
        x0 *= .25
        x0[best] += .75

    constraints = (
        {"type": "eq", "fun": lambda x: np.sum(x) - 1.0, "jac": lambda x: np.ones(n)},
        {
            "type": "ineq",
            "fun": lambda x: market.mean @ x - target_return,
            "jac": lambda x: market.mean,
        },
    )
    start = time.perf_counter()
    res = minimize(
        lambda x: float(x @ market.covariance @ x),
        x0,
        jac=lambda x: 2.0 * market.covariance @ x,
        bounds=[(0.0, 1.0)] * n,
        constraints=constraints,
        method="SLSQP",
        options={"ftol": 1e-12, "maxiter": 1000},
    )
    elapsed = time.perf_counter() - start
    if not res.success:
        raise RuntimeError(f"mean-variance solve failed: {res.message}")
    x = np.maximum(res.x, 0.0)
    x /= x.sum()
    return _solution(
        "Mean-variance", x, res.fun, market, evaluation_returns, epsilon,
        elapsed, str(res.message),
    )


def solve_cvar(
    market: SectorMarket,
    training_returns: np.ndarray,
    epsilon: float = .05,
) -> PortfolioSolution:
    """Minimize scenario CVaR as a linear program (pure risk minimization --
    no return target: budget and long-only are the only constraints)."""
    import time

    returns = np.asarray(training_returns, dtype=float)
    s, n = returns.shape
    # Variables: x[0:n], beta, u[0:s]
    c = np.r_[np.zeros(n), 1.0, np.full(s, 1.0 / (epsilon * s))]
    A_ub = np.zeros((s, n + 1 + s))
    b_ub = np.zeros(s)
    A_ub[:s, :n] = -returns
    A_ub[:s, n] = -1.0
    A_ub[np.arange(s), n + 1 + np.arange(s)] = -1.0
    A_eq = np.zeros((1, n + 1 + s))
    A_eq[0, :n] = 1.0
    bounds = [(0.0, 1.0)] * n + [(None, None)] + [(0.0, None)] * s

    start = time.perf_counter()
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=[1.0],
                  bounds=bounds, method="highs")
    elapsed = time.perf_counter() - start
    if not res.success:
        raise RuntimeError(f"CVaR solve failed: {res.message}")
    x = res.x[:n]
    beta = float(res.x[n])
    return _solution(
        "CVaR", x, res.fun, market, returns, epsilon, elapsed, res.message,
        threshold=beta,
    )


def solve_var(
    market: SectorMarket,
    training_returns: np.ndarray,
    epsilon: float = .05,
    time_limit: float = 15.0,
) -> PortfolioSolution:
    """Minimize scenario VaR using binary breach indicators (pure risk
    minimization -- no return target: budget and long-only are the only
    constraints)."""
    import time

    returns = np.asarray(training_returns, dtype=float)
    s, n = returns.shape
    losses_by_asset = -returns
    gamma_lb = float(losses_by_asset.min())
    gamma_ub = float(losses_by_asset.max())
    big_m = losses_by_asset.max(axis=1) - gamma_lb
    allowed = int(np.floor(epsilon * s + 1e-12))

    # Variables: x[0:n], gamma, z[0:s]
    dim = n + 1 + s
    c = np.zeros(dim)
    c[n] = 1.0
    A = np.zeros((s + 1, dim))
    upper = np.zeros(s + 1)
    lower = np.full(s + 1, -np.inf)
    A[:s, :n] = losses_by_asset
    A[:s, n] = -1.0
    A[np.arange(s), n + 1 + np.arange(s)] = -big_m
    A[s, n + 1:] = 1.0
    upper[s] = allowed

    lb = np.r_[np.zeros(n), gamma_lb, np.zeros(s)]
    ub = np.r_[np.ones(n), gamma_ub, np.ones(s)]
    integrality = np.r_[np.zeros(n + 1), np.ones(s)]
    equality = LinearConstraint(np.r_[np.ones(n), np.zeros(1 + s)][None, :], [1.0], [1.0])

    start = time.perf_counter()
    res = milp(
        c,
        integrality=integrality,
        bounds=Bounds(lb, ub),
        constraints=(LinearConstraint(A, lower, upper), equality),
        options={"time_limit": time_limit, "mip_rel_gap": 1e-6},
    )
    elapsed = time.perf_counter() - start
    if res.x is None:
        raise RuntimeError(res.message)
    x = res.x[:n]
    gamma = float(res.x[n])
    breaches = int(np.sum(portfolio_losses(returns, x) > gamma + 1e-7))
    if breaches > allowed:
        raise RuntimeError("No feasible solution found within the time limit.")
    status = res.message if res.success else f"Feasible incumbent; {res.message}"
    return _solution(
        "VaR", x, gamma, market, returns, epsilon, elapsed, status,
        threshold=gamma, breaches=breaches, allowed_breaches=allowed,
    )


def solve_chance(
    market: SectorMarket,
    training_returns: np.ndarray,
    loss_limit: float,
    epsilon: float = .05,
    time_limit: float = 15.0,
) -> PortfolioSolution:
    """Maximize EMPIRICAL expected return subject to a scenario chance
    constraint -- both the objective and the constraint are sample average
    approximations from the same `training_returns`, matching solve_cvar /
    solve_var (which never touch the population mean either). Using the
    population market.mean here instead would be a hybrid that has no oracle
    access for the constraint but somehow does for the objective."""
    import time

    returns = np.asarray(training_returns, dtype=float)
    s, n = returns.shape
    mu_hat = returns.mean(axis=0)
    losses_by_asset = -returns
    big_m = np.maximum(losses_by_asset.max(axis=1) - loss_limit, 0.0)
    allowed = int(np.floor(epsilon * s + 1e-12))

    # Variables: x[0:n], z[0:s]
    dim = n + s
    c = np.r_[-mu_hat, np.zeros(s)]
    A = np.zeros((s + 1, dim))
    lower = np.full(s + 1, -np.inf)
    upper = np.zeros(s + 1)
    A[:s, :n] = losses_by_asset
    A[np.arange(s), n + np.arange(s)] = -big_m
    upper[:s] = loss_limit
    A[s, n:] = 1.0
    upper[s] = allowed
    equality = LinearConstraint(np.r_[np.ones(n), np.zeros(s)][None, :], [1.0], [1.0])

    start = time.perf_counter()
    res = milp(
        c,
        integrality=np.r_[np.zeros(n), np.ones(s)],
        bounds=Bounds(np.zeros(dim), np.ones(dim)),
        constraints=(LinearConstraint(A, lower, upper), equality),
        options={"time_limit": time_limit, "mip_rel_gap": 1e-6},
    )
    elapsed = time.perf_counter() - start
    if res.x is None:
        raise RuntimeError(res.message)
    x = res.x[:n]
    losses = portfolio_losses(returns, x)
    breaches = int(np.sum(losses > loss_limit + 1e-7))
    if breaches > allowed:
        raise RuntimeError("No feasible solution found within the time limit.")
    status = res.message if res.success else f"Feasible incumbent; {res.message}"
    return _solution(
        "Chance constraint", x, -res.fun, market, returns, epsilon, elapsed,
        status, threshold=loss_limit, breaches=breaches,
        allowed_breaches=allowed,
    )


def solve_portfolio(
    model: str,
    market: SectorMarket,
    training_returns: np.ndarray,
    target_return: float,
    epsilon: float,
    loss_limit: float,
) -> PortfolioSolution:
    key = model.lower().replace("-", "_").replace(" ", "_")
    if key in {"mean_variance", "markowitz"}:
        return solve_markowitz(market, target_return, training_returns, epsilon)
    if key == "cvar":
        return solve_cvar(market, training_returns, epsilon)
    if key == "var":
        return solve_var(market, training_returns, epsilon)
    if key in {"chance", "chance_constraint"}:
        return solve_chance(market, training_returns, loss_limit, epsilon)
    raise ValueError(f"unknown model: {model}")
