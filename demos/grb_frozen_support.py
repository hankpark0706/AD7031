"""
grb_frozen_support.py -- everything gurobipy needs to survive being frozen into
a standalone .exe, plus immunity to whatever Gurobi license sits on the machine.

Call prepare_gurobi() BEFORE `import gurobipy`.  It does two things:

1. Restores the `help` builtin.  A PyInstaller app does not run the `site`
   module, so `builtins.help` never gets installed -- and gurobipy's compiled
   core touches it while initialising, so the frozen demo dies at startup with
   a bare `NameError: name 'help' is not defined`.

2. Points GRB_LICENSE_FILE at the restricted license shipped inside the pip
   `gurobipy` wheel (2000 vars / 2000 linear constraints, valid to 2027-11-29).
   The Lecture-1 demos are 2 variables / 2 constraints, so that limit never
   bites and a *machine* license (~/gurobi.lic) is never needed -- while an
   expired one would otherwise kill the very first `gp.Model()`:

       GurobiError: License 2824426 has expired

   gurobipy latches the license the first time an environment is created, so
   this cannot be repaired afterwards; it has to happen before the import.
"""
import builtins
import importlib.util
import os
import sys


def ensure_builtin_help():
    """Re-install the `help` builtin that a frozen app is missing."""
    if hasattr(builtins, "help"):
        return
    try:
        import _sitebuiltins
        builtins.help = _sitebuiltins._Helper()
    except Exception:                       # last resort: a harmless stand-in
        builtins.help = lambda *args, **kwargs: None


def use_bundled_license():
    """Force gurobipy to use the restricted license shipped in its own wheel.

    Returns the license path used, or None if the bundled file could not be
    found (in which case gurobipy falls back to its normal search, and the
    machine license, if any, applies).
    """
    candidates = []
    if getattr(sys, "frozen", False):
        # PyInstaller onefile: --collect-all gurobipy unpacks the package
        # (dll + gurobi.lic) under the temp _MEIPASS folder
        candidates.append(os.path.join(getattr(sys, "_MEIPASS", ""), "gurobipy"))
    spec = importlib.util.find_spec("gurobipy")   # does not execute the package
    if spec is not None:
        candidates += list(getattr(spec, "submodule_search_locations", None) or [])

    for folder in candidates:
        lic = os.path.join(folder, "gurobi.lic")
        if os.path.isfile(lic):
            os.environ["GRB_LICENSE_FILE"] = lic
            return lic
    return None


def prepare_gurobi():
    """Both fixes, in the order the frozen demos need them."""
    ensure_builtin_help()
    return use_bundled_license()
