# Isothermal Soil Theory — Interactive 2D Isothermal Superposition Tool

This repository contains a single interactive Python script,
`isothermal_superposition_ui.py`, for building simple buried cable/source
layouts and visualizing the resulting 2D temperature-rise field in soil using
an image-source superposition model.

## What the script does

The script opens **three Matplotlib windows**:

1. **Grid / Source Layout window**
   - Shows the source placement grid (depth positive downward).
   - Lets you add/remove heat sources on snapped 10 cm points.
   - Displays mirrored image-source markers above the ground line.

2. **Thermal Field window**
   - Shows the computed temperature-rise field `θ(x, depth)` as a contour plot.
   - Draws one or more target isotherms (epsilon levels, e.g. 5 K).
   - Displays summary metrics (closed contour count, enclosed area, etc.).

3. **Thermal Parameters window**
   - Lets you edit model/plot controls such as:
     - source heat rate `q` [W/m]
     - soil conductivity `k` [W/(m·K)]
     - ambient temperature `T_amb`
     - target isothermal temperature / epsilon levels
   - Includes bounding-box Select/Unselect controls to add/remove many
     snapped grid points at once.

---

## Physical model (plain-language summary)

Each cable/source is represented as a **2D line heat source** in a homogeneous
soil half-space. The ground surface is treated as an isothermal boundary using
a mirrored image source of opposite sign. The script then adds contributions
from all selected sources.

For each source, the temperature rise contribution is proportional to:

- source strength `q`
- inverse of soil conductivity `k`
- logarithm of distance ratio (image distance / real distance)

The total field is the sum over all sources.

### Important caveat

Because a 2D line source is mathematically singular at its center, the model
cannot assign a physically meaningful finite temperature exactly at the source
point unless a finite source radius model is added. This script therefore
focuses on **heat-rate input (`q`)** and contour behavior.

---

## Requirements

- Python 3.10+
- `numpy`
- `matplotlib`

Install dependencies (example):

```bash
pip install numpy matplotlib
```

---

## How to run

From the repository root:

```bash
python isothermal_superposition_ui.py
```

Optional CLI parameters:

```bash
python isothermal_superposition_ui.py \
  --epsilon 5 \
  --q 35 \
  --k 1.2 \
  --T-amb 20 \
  --x-min -4 --x-max 4 \
  --depth-max 8 \
  --res-x 500 --res-y 500
```

Run the built-in non-interactive check:

```bash
python isothermal_superposition_ui.py --test
```

The self-test writes output files into `out_selftest/` and reports pass/fail
based on contour closure, enclosure, and ground-plane residual checks.

---

## Mouse and keyboard controls

### In Grid / Source Layout window

- **Left click**: add source at nearest grid point
- **Right click**: remove nearest source

### Global key controls

- **R**: reset all sources
- **C** or **Enter**: force recompute/redraw
- **S**: save outputs
- **Esc**: close all windows

### Save output (`S`)

Files are written to `./out`:

- `selected_sources.csv` — all selected source coordinates and `q`
- `isothermal_contour_5K.csv` — contour path vertices
- `isothermal_superposition_plot.png` — thermal plot image

---

## Typical workflow

1. Launch script.
2. Add one or more sources in the grid window.
3. Adjust `q`, `k`, and epsilon/target temperature in controls window.
4. Inspect whether contour is closed and whether all sources are enclosed.
5. If contour touches domain edge, increase domain size (`--x-min/--x-max/--depth-max`) or lower `q`.
6. Press `S` to export CSV + PNG outputs.

---

## Notes on coordinates

- UI depth axis is positive downward (`depth >= 0` below ground).
- Internal model uses `y_model = -depth`.
- Sources are real points below ground; image points are mirrored above ground.

---

## Repository contents

- `isothermal_superposition_ui.py` — main interactive script
- `README.md` — this documentation
