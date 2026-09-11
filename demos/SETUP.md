# Setup — installing the dependencies

The demos need four packages: **numpy, scipy, matplotlib, gurobipy**.
Same steps on **macOS, Windows and Linux**.

If you see this, the packages are missing from whichever Python you ran:

```
ModuleNotFoundError: No module named 'matplotlib'
```

---

## The way that always works: a virtual environment

Do this **once**, in the **repo root** (not inside `demos/`).

### macOS / Linux

```bash
cd AD7031-optimization-under-uncertainty
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

### Windows (PowerShell)

```powershell
cd AD7031-optimization-under-uncertainty
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Takes about a minute. Then, **every time** you want a demo — the prompt must
show `(.venv)`:

```bash
cd demos
python run_demo.py          # menu of all demos
python demo1_toy_ip_app.py  # or start one directly
```

Once the venv exists you don't reinstall anything; you just activate it
(`source .venv/bin/activate`) in each new terminal.

---

## Running from VS Code

This is the usual cause of `ModuleNotFoundError`: VS Code's ▷ Run button uses the
interpreter *it* has selected, which is often **not** the `.venv` you installed into.

1. `Cmd+Shift+P` (macOS) / `Ctrl+Shift+P` (Windows) → **Python: Select Interpreter**
2. Pick the one whose path contains `.venv` — once the venv exists in the repo
   root, VS Code lists it as *(Recommended)*.
3. Open a **new** terminal afterwards, so it picks up the venv.

To see which Python is actually running:

```bash
python -c "import sys; print(sys.executable)"
```

That path must be inside `.venv`. If it prints something like
`AppData\Roaming\uv\python\cpython-3.14...\python.exe`, `/usr/bin/python3`, or a
Homebrew path, you are on the wrong interpreter — that Python has no matplotlib,
which is exactly the error above.

---

## Check it worked

```bash
cd demos
python -c "import numpy, scipy, matplotlib, gurobipy; print('all four OK')"
```

Or run a demo headless, without opening a window:

```bash
DEMO_SMOKE=1 python demo1_toy_ip_app.py         # should end with: SMOKE OK
```

```powershell
$env:DEMO_SMOKE=1; python demo1_toy_ip_app.py   # Windows
```

(`module_classify_app.py` uses `CLASSIFY_SMOKE=1`, `lap_app.py` uses `LAP_SMOKE=1`.)

`run_demo.py` also checks the four packages before showing the menu, and tells
you which one is missing instead of dying with a traceback.

---

## Alternative: uv, no environment to manage

If you use [`uv`](https://docs.astral.sh/uv/), this one line fetches the packages
into a cache and runs the demo — no venv, no activate:

```bash
cd demos
uv run --with numpy,scipy,matplotlib,gurobipy run_demo.py
```

Heads-up: on the Windows machine these demos were prepared on, `uv` currently
fails with

```
error: Failed to update Windows PE resources: ...uv-trampoline-NNNN.exe
```

which is `uv` being blocked from writing to `%TEMP%` (usually antivirus or a
locked temp folder), not a problem with the demos. `python -m venv` above is
unaffected — use that instead. The error is Windows-only; macOS has no such
trampolines.

---

## Notes

**Python version** — 3.10 or newer. 3.14 is fine; all four packages have wheels
for it. Versions these demos were tested against:

| Package | Version |
|---|---|
| numpy | 2.4.6 |
| scipy | 1.17.1 |
| matplotlib | 3.10.9 |
| gurobipy | 13.0.2 |

**Gurobi license** — nothing to install, nothing to register. The `gurobipy`
wheel ships a restricted license (2000 variables / 2000 constraints, valid to
2027-11-29), and the demos point at it explicitly, so an expired `~/gurobi.lic`
on your machine is ignored. The toy models are 2 variables wide, so the size
limit never bites. Only `demo1a_lp_app.py` and `demo1_toy_ip_app.py` call Gurobi
at all — the rest use numpy or scipy.

**macOS** — nothing extra to install. The demos select matplotlib's native
`macosx` backend automatically, so Tk is not needed.

**Linux** — if no window ever opens, install Tk: `sudo apt install python3-tk`.

**`ModuleNotFoundError` naming a *demo* module** (e.g. `demo1_toy_ip`) — run the
scripts from inside this `demos/` folder, so the sibling files are importable.
