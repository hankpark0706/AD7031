"""
run_demo.py -- a tiny text menu for the interactive demos in this folder.

Run it:  python run_demo.py
Type the number of a demo and press Enter.  Close the plot window to come back
to the menu.  Type q to quit.

Each demo is a plain Python script, so you can also start one directly, e.g.

    python demo1a_lp_app.py

Needs numpy, scipy, matplotlib and gurobipy -- see SETUP.md if an import fails.
"""
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

REQUIRED = ("numpy", "scipy", "matplotlib", "gurobipy")


def check_deps():
    """Fail with instructions, not a bare ModuleNotFoundError, if deps are missing."""
    import importlib.util
    missing = [m for m in REQUIRED if importlib.util.find_spec(m) is None]
    if not missing:
        return
    print()
    print("  Missing package(s): " + ", ".join(missing))
    print(f"  (running {sys.executable})")
    print()
    print("  Install them, either way:")
    print()
    print("      uv run --with " + ",".join(REQUIRED) + " run_demo.py")
    print("      pip install -r ../requirements.txt")
    print()
    print("  In VS Code, also check you selected the .venv interpreter.")
    print("  Full instructions: SETUP.md")
    print()
    sys.exit(1)


DEMOS = [
    ("demo1a_lp_app.py",      "LP, graphical method -- the objective line sweeps to the optimal vertex"),
    ("demo1_toy_ip_app.py",   "Toy IP -- why 'solve the LP and round' fails"),
    ("demo1b_lp_3d_app.py",   "3D LP geometry -- the feasible polytope and its optimal vertex"),
    ("demo1c_lp_slide_app.py", "3D LP animation -- press Play, the objective plane slides"),
    ("demo1d_qp_app.py",      "QP, graphical method -- circles instead of lines"),
    ("module_classify_app.py", "Classification -- draw your own line vs the min-error MILP"),
    ("lap_app.py",            "Two-stage stochastic LAP (SAA) -- solve, then simulate a disaster"),
]


def menu():
    print("\n" + "=" * 72)
    print("  AD7031 -- interactive demos")
    print("=" * 72)
    for i, (script, desc) in enumerate(DEMOS, 1):
        print(f"  {i:>2}.  {desc}")
        print(f"       ({script})")
    print("   q.  quit")
    print("-" * 72)


def main():
    check_deps()
    while True:
        menu()
        choice = input("  choose a demo: ").strip().lower()
        if choice in ("q", "quit", "exit"):
            break
        if choice.isdigit() and 1 <= int(choice) <= len(DEMOS):
            script = DEMOS[int(choice) - 1][0]
            print(f"\n  launching {script} ...  (close the window to return)\n")
            # a subprocess keeps each demo's matplotlib state to itself
            subprocess.run([sys.executable, os.path.join(HERE, script)], cwd=HERE)
        else:
            print(f"  ?? please type a number from 1 to {len(DEMOS)}, or q")


if __name__ == "__main__":
    main()
