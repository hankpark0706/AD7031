# Interactive demos

Plain Python scripts — they run on **macOS, Windows and Linux**.
(These replace the earlier `.exe` files, which only ran on Windows.)

Materials for this folder are added weekly as the course progresses.

## Setup (once)

Short version below. If anything fails -- especially `ModuleNotFoundError: No module named 'matplotlib'` -- see **[SETUP.md](SETUP.md)**, which covers picking the right interpreter in VS Code.

```bash
cd AD7031-optimization-under-uncertainty
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt   # numpy, scipy, matplotlib, gurobipy
```

No Gurobi license is needed: the `gurobipy` wheel ships a restricted license
(2000 variables / 2000 constraints, valid to 2027-11-29), and the demos point at
it explicitly, so an expired machine license on your laptop is ignored. The toy
models are 2 variables wide, so the size limit never bites.

## Run

```bash
cd demos
python run_demo.py          # menu of all demos
python demo1a_lp_app.py     # or start one directly
```

Close the plot window to end a demo (or to return to the menu).

## What's here

| Demo | Run | Was |
|---|---|---|
| LP, graphical method — the objective line sweeps to the optimal vertex | `demo1a_lp_app.py` | `LP-Graphical-Demo.exe` |
| Toy IP — why "solve the LP and round" fails | `demo1_toy_ip_app.py` | `Toy-IP-Demo.exe` |
| 3D LP geometry — the feasible polytope and its optimal vertex | `demo1b_lp_3d_app.py` | `LP-3D-Demo.exe` |
| 3D LP animation — press Play, the objective plane slides | `demo1c_lp_slide_app.py` | `LP-3D-Slide-Demo.exe` |
| QP, graphical method — circles instead of lines | `demo1d_qp_app.py` | `QP-Graphical-Demo.exe` |
| Classification — draw your own line vs the min-error MILP | `module_classify_app.py` | `Classify-Dashboard.exe` |
| Two-stage stochastic LAP (SAA) — solve, then simulate a disaster | `lap_app.py` | `LAP-Dashboard.exe` |

Each `*_app.py` is the entry point: it picks the right GUI backend for your OS
(`macosx` on a Mac, `TkAgg` elsewhere) and then imports the demo itself —
`demo1a_lp_graphical.py`, `demo1_toy_ip.py`, `demo1b_lp_3d.py`,
`demo1c_lp_slide_animation.py`, `demo1d_qp_graphical.py`, `module_classify.py`,
`lap_dashboard.py` (with `lap_core.py`). `grb_frozen_support.py` is the shared
Gurobi-license helper.

## Troubleshooting

- **No window opens on Linux, or `TkAgg` fails** — install Tk:
  `sudo apt install python3-tk` (macOS and Windows need nothing extra).
- **`ModuleNotFoundError`** — run the scripts from inside this `demos/` folder
  so the sibling modules are importable.
