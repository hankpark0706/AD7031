"""
markowitz_portfolio.py -- the Markowitz portfolio model from the week-2 notebook,
as a plain script you can run from VS Code.

    cd demos
    python markowitz_portfolio.py

Same model as the "Reference: the completed model" cell of
notebooks/week1_markowitz.ipynb -- written out by hand, no arrays, no loops, no
quicksum, so every term matches the math on the slide:

    min   x' S x
    s.t.  mu' x >= 0.10          expected return meets the 10% target
          x_bond + x_tech + x_energy = 1
          x >= 0

    asset    expected return mu_j    variance S_jj
    bond     6%                      0.010
    tech     10%                     0.040
    energy   15%                     0.090

    covariances:  S(bond,tech) = 0.001,  S(tech,energy) = 0.015,  S(bond,energy) = 0

Unlike the notebook there is no WLS licence block: the notebook needs one
because it runs on Colab, while here grb_frozen_support points gurobipy at the
restricted licence shipped inside its own wheel (2000 vars / 2000 constraints).
Nothing to register, and an expired ~/gurobi.lic on the machine is ignored.
This model is 3 variables wide, so the limit never bites.

Running it writes markowitz.lp next to this script -- that is the model read
back in Gurobi's own format, printed before the solve.

Set DEMO_SMOKE=1 to run headless (no window), for checking the install:

    DEMO_SMOKE=1 python markowitz_portfolio.py          # macOS / Linux
    $env:DEMO_SMOKE=1; python markowitz_portfolio.py    # Windows PowerShell
"""
import os
import sys

# sibling modules importable no matter which folder you launched from
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from grb_frozen_support import prepare_gurobi

prepare_gurobi()                    # MUST run before gurobipy is imported

import gurobipy as gp
from gurobipy import GRB

import matplotlib

_SMOKE = bool(os.environ.get("DEMO_SMOKE"))
if _SMOKE:
    _backend = "Agg"               # no window, for a headless check
elif sys.platform == "darwin":
    _backend = "macosx"            # native macOS GUI -- no tkinter needed
else:
    _backend = "TkAgg"             # Windows / Linux

matplotlib.use(_backend, force=True)

import matplotlib.pyplot as plt

ASSETS = ("bond", "tech", "energy")


def build():
    """The model, term by term, exactly as in the notebook."""
    pf = gp.Model("markowitz")

    # decision variables -- one weight per asset, continuous, lower bound 0 by default
    x_bond = pf.addVar(vtype=GRB.CONTINUOUS, name="x_bond")
    x_tech = pf.addVar(vtype=GRB.CONTINUOUS, name="x_tech")
    x_energy = pf.addVar(vtype=GRB.CONTINUOUS, name="x_energy")

    # objective -- portfolio variance, written out term by term
    pf.setObjective(
        0.010 * x_bond * x_bond
        + 0.040 * x_tech * x_tech
        + 0.090 * x_energy * x_energy
        + 2 * 0.001 * x_bond * x_tech
        + 2 * 0.015 * x_tech * x_energy,
        GRB.MINIMIZE,
    )

    # constraints -- expected return meets the 10% target; weights sum to 1
    pf.addConstr(
        0.06 * x_bond + 0.10 * x_tech + 0.15 * x_energy >= 0.10,
        name="target_return",
    )
    pf.addConstr(x_bond + x_tech + x_energy == 1, name="budget")

    return pf, (x_bond, x_tech, x_energy)


def main():
    pf, weights = build()

    # read the model back before solving -- the quadratic objective shows up under [ ... ]/2
    pf.update()
    lp_path = os.path.join(HERE, "markowitz.lp")
    pf.write(lp_path)
    print(open(lp_path).read())

    pf.optimize()

    x_bond, x_tech, x_energy = weights
    print(f"\nbond={x_bond.X:.2f}  tech={x_tech.X:.2f}  energy={x_energy.X:.2f}")
    print(f"risk (variance) = {pf.ObjVal:.4f}")

    shares = [v.X for v in weights]

    # one bar per asset, height = x_j -- the share of the budget it gets
    plt.figure(figsize=(5.5, 4.5))
    plt.bar(ASSETS, shares, color=["#4C72B0", "#DD8452", "#C44E52"])
    for name, share in zip(ASSETS, shares):
        plt.annotate(
            f"{share:.2f}",
            (name, share),
            ha="center",
            textcoords="offset points",
            xytext=(0, 5),
        )
    plt.ylim(0, 1)
    plt.ylabel("share of the budget  $x_j$")
    plt.title(f"Markowitz weights -- risk (variance) = {pf.ObjVal:.4f}")
    plt.tight_layout()

    if _SMOKE:
        plt.gcf().canvas.draw()
        # the optimum is interior, not at a corner: every asset gets a share
        assert abs(sum(shares) - 1) < 1e-6, shares
        assert all(s > 1e-6 for s in shares), shares
        print("SMOKE OK")
        return

    plt.show()


if __name__ == "__main__":
    main()
