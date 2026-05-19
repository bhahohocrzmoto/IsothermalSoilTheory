"""
isothermal_superposition_ui.py
==============================

Interactive Python tool for sketching 2D buried heat-source layouts and
visualising the resulting isothermal-superposition temperature-rise field.
Inspired by the `isothermal_superposition` mode of TB880Case0GeometryGeneration.py
in the FEMTB880Case-0 repository; generalised to arbitrary n sources and
exposed through a click-based matplotlib UI.

Physical model
--------------
2D steady-state conduction in homogeneous semi-infinite soil. Each cable is
idealised as a line heat source with prescribed heat rate q_i [W/m]. The
isothermal ground plane y = 0 is enforced by mirroring each real source at
(x_i, y_i)  (y_i < 0)  to an opposite-sign image at  (x_i, -y_i). The
temperature rise above ambient is

    theta(x, y) = sum_i  q_i / (2 pi k_soil) * ln( r_img_i / r_real_i )

with r_real_i and r_img_i the Euclidean distances from (x, y) to the real
and image source. On y = 0 the two are identical, so theta(x, 0) = 0
exactly — the ground-plane Dirichlet T = T_amb is satisfied analytically.

Important caveat
----------------
A 2D line source has a logarithmically singular temperature at its centre,
so prescribing a finite "source temperature" at a point is not physically
meaningful without introducing a finite cable radius (or an inverse problem
on a finite control surface). This tool therefore takes a single shared
heat rate q_source_Wpm [W/m] for every selected source. Per-source heat
rates and finite-radius temperature inversion are listed in the future-work
section but not implemented in this prototype.

Coordinate convention
---------------------
The UI uses an engineering depth axis: depth = 0 at the ground surface,
positive downward, drawn with the y-axis inverted so the surface is at the
top of the plot. The mathematical formula is evaluated in y_model with
y_model = -depth_ui (so real sources have y_model < 0 and images have
y_model > 0). Both quantities are reported in CSV output.

Controls
--------
    left-click    add a source at the nearest 10 cm grid point
    right-click   remove the source nearest the cursor
    R             reset (clear all sources)
    C / Enter     force a recompute (auto-updates on each click anyway)
    S             save sources, contour, and plot image to ./out
    Esc           close the window

Run
---
    python isothermal_superposition_ui.py
    python isothermal_superposition_ui.py --epsilon 2.0 --q 45.0
    python isothermal_superposition_ui.py --test     # non-interactive self-test
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Button


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULTS = dict(
    grid_x_min=-2.0,
    grid_x_max=2.0,
    depth_min=0.0,
    depth_max=2.0,
    grid_step=0.10,        # 10 cm snap
    k_soil=1.0,            # W/(m K)
    T_amb=20.0,            # °C
    epsilon_K=5.0,         # K
    q_source_Wpm=30.0,     # W/m, equal per source in v1
    result_nx=800,
    result_ny=500,
    r_cutoff_m=0.01,       # singularity cutoff
)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class Source2D:
    id: int
    x_m: float
    depth_m: float
    q_Wpm: float

    @property
    def y_model_m(self) -> float:
        """Mathematical y coordinate. y_model < 0 below ground."""
        return -self.depth_m


# ---------------------------------------------------------------------------
# Coordinate / snap helpers
# ---------------------------------------------------------------------------

def snap_to_grid(value: float, step: float) -> float:
    """Snap to the nearest integer multiple of `step`.

    A tiny epsilon avoids banker's-rounding artefacts at exact half-points.
    """
    return round(value / step + 1.0e-9) * step


def ui_to_model_y(depth_ui):
    """UI depth (positive downward) -> mathematical y (negative below ground)."""
    return -np.asarray(depth_ui)


# ---------------------------------------------------------------------------
# Field evaluation
# ---------------------------------------------------------------------------

def theta_superposition_ui(
    X_ui: np.ndarray,
    Depth_ui: np.ndarray,
    sources: Sequence[Source2D],
    k_soil: float,
    r_cutoff: float = 0.01,
) -> np.ndarray:
    """Evaluate theta on a (x, depth) UI grid.

    Internally converts to y_model = -depth and places one image source per
    real source at (x_i, -y_model_i) = (x_i, +depth_i). The log-ratio form
    encodes the opposite sign of the image automatically.
    """
    Y_model = ui_to_model_y(Depth_ui)
    theta = np.zeros_like(X_ui, dtype=float)
    coeff_base = 1.0 / (2.0 * math.pi * k_soil)

    for s in sources:
        dx = X_ui - s.x_m
        # Real source at (x_i, y_model_i)
        dy_real = Y_model - s.y_model_m
        r_real = np.sqrt(dx * dx + dy_real * dy_real)
        # Image source at (x_i, -y_model_i) = (x_i, +depth_i)
        dy_img = Y_model + s.y_model_m  # because image y = -y_model_i
        r_img = np.sqrt(dx * dx + dy_img * dy_img)
        r_real = np.maximum(r_real, r_cutoff)
        r_img = np.maximum(r_img, r_cutoff)
        theta = theta + s.q_Wpm * coeff_base * np.log(r_img / r_real)

    return theta


def ground_plane_residual(
    sources: Sequence[Source2D],
    k_soil: float,
    cfg: dict,
    n_samples: int = 201,
) -> float:
    """max |theta(x, depth=0)| over the visible x-range — should be ~0."""
    if not sources:
        return 0.0
    xs = np.linspace(cfg["grid_x_min"], cfg["grid_x_max"], n_samples)
    ds = np.zeros_like(xs)
    th = theta_superposition_ui(xs, ds, sources, k_soil, cfg["r_cutoff_m"])
    return float(np.max(np.abs(th)))


# ---------------------------------------------------------------------------
# Contour helpers
# ---------------------------------------------------------------------------

def get_paths_compat(contour_set) -> list:
    """matplotlib >= 3.8 exposes get_paths() on QuadContourSet directly;
    earlier versions need cs.collections[0].get_paths()."""
    try:
        return list(contour_set.get_paths())
    except AttributeError:
        paths = []
        for coll in contour_set.collections:
            paths.extend(coll.get_paths())
        return paths


def is_closed_path(verts: np.ndarray, gap_frac: float = 0.05) -> bool:
    """Treat a polyline as 'effectively closed' if the gap between its first
    and last vertices is small compared to the total polyline length.

    matplotlib's contour() returns single-piece paths that loop back near
    where the marching-squares trace started, but it does NOT duplicate the
    first vertex at the end or set a CLOSEPOLY code. The endpoint gap is
    typically a few times the local grid spacing, which is small relative
    to the perimeter of a normal closed contour. A genuinely open contour
    (one that exits the plot edge) has endpoints far apart along the
    polyline — usually the path length is short while the gap is large.
    The 5 % cut-off discriminates the two reliably for typical grids.
    """
    if verts.shape[0] < 3:
        return False
    gap = float(np.linalg.norm(verts[0] - verts[-1]))
    if gap == 0.0:
        return True
    seg_lens = np.linalg.norm(np.diff(verts, axis=0), axis=1)
    total = float(seg_lens.sum())
    if total <= 0.0:
        return False
    return gap < gap_frac * total


def polygon_area(verts: np.ndarray) -> float:
    if verts.shape[0] < 3:
        return 0.0
    x, y = verts[:, 0], verts[:, 1]
    return float(0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def point_in_polygon(x: float, y: float, verts: np.ndarray) -> bool:
    """Ray-casting test; verts is (N, 2)."""
    if verts.shape[0] < 3:
        return False
    inside = False
    n = verts.shape[0]
    j = n - 1
    for i in range(n):
        xi, yi = verts[i, 0], verts[i, 1]
        xj, yj = verts[j, 0], verts[j, 1]
        if (yi > y) != (yj > y):
            x_cross = (xj - xi) * (y - yi) / (yj - yi + 1e-30) + xi
            if x < x_cross:
                inside = not inside
        j = i
    return inside


# ---------------------------------------------------------------------------
# CSV save
# ---------------------------------------------------------------------------

def save_sources_csv(sources: Sequence[Source2D], path: Path) -> None:
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["source_id", "x_m", "depth_m", "y_model_m", "q_Wpm"])
        for s in sources:
            w.writerow([
                s.id,
                f"{s.x_m:.4f}",
                f"{s.depth_m:.4f}",
                f"{s.y_model_m:.4f}",
                f"{s.q_Wpm:.6f}",
            ])


def save_contours_csv(paths, path: Path) -> None:
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["contour_id", "point_index", "x_m", "depth_m", "y_model_m"])
        for cid, p in enumerate(paths):
            for idx, (x, d) in enumerate(p.vertices):
                w.writerow([cid, idx, f"{x:.6f}", f"{d:.6f}", f"{-d:.6f}"])


# ---------------------------------------------------------------------------
# Interactive UI
# ---------------------------------------------------------------------------

class IsothermalSuperpositionUI:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.sources: list[Source2D] = []
        self.next_id = 1
        self.result_fig: Optional[plt.Figure] = None
        self._last_contour_paths: list = []
        self._last_result_payload: Optional[dict] = None

        # Source-selection figure only.
        self.fig, self.ax = plt.subplots(figsize=(9.5, 7.5))
        self.fig.subplots_adjust(left=0.10, right=0.97, top=0.92, bottom=0.17)
        run_ax = self.fig.add_axes([0.80, 0.05, 0.14, 0.06])
        self.run_button = Button(run_ax, "Run")
        self.run_button.on_clicked(self._on_run_clicked)

        # Event bindings
        self.fig.canvas.mpl_connect("button_press_event", self._on_click)
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)

        self.redraw()

    # ------------------------------- input ----------------------------------

    def _result_grid(self):
        c = self.cfg
        x = np.linspace(c["result_x_min"], c["result_x_max"], c["result_nx"])
        y_plot = np.linspace(c["result_y_min"], c["result_y_max"], c["result_ny"])
        return np.meshgrid(x, y_plot)

    def _snap_in_grid(self, x: float, depth: float) -> Optional[tuple[float, float]]:
        c = self.cfg
        step = c["grid_step"]
        x_s = snap_to_grid(x, step)
        d_s = snap_to_grid(depth, step)
        if not (c["grid_x_min"] - 1e-9 <= x_s <= c["grid_x_max"] + 1e-9):
            return None
        # Reject sources at or above the ground plane (singular: source coincides with image)
        if d_s < step - 1e-9:
            return None
        if d_s > c["depth_max"] + 1e-9:
            return None
        return x_s, d_s

    def _find_nearest_source(self, x: float, depth: float) -> Optional[Source2D]:
        tol = 0.6 * self.cfg["grid_step"]
        best = None
        best_d = float("inf")
        for s in self.sources:
            d = math.hypot(s.x_m - x, s.depth_m - depth)
            if d < best_d and d < tol:
                best_d = d
                best = s
        return best

    def _add_source(self, x_s: float, d_s: float) -> bool:
        for s in self.sources:
            if abs(s.x_m - x_s) < 1e-9 and abs(s.depth_m - d_s) < 1e-9:
                return False  # duplicate
        self.sources.append(Source2D(
            id=self.next_id, x_m=x_s, depth_m=d_s,
            q_Wpm=self.cfg["q_source_Wpm"],
        ))
        self.next_id += 1
        return True

    def _on_click(self, event):
        if event.inaxes != self.ax or event.xdata is None or event.ydata is None:
            return
        if event.button == 1:  # left
            snapped = self._snap_in_grid(event.xdata, event.ydata)
            if snapped is None:
                return
            if self._add_source(*snapped):
                self.redraw()
        elif event.button == 3:  # right
            victim = self._find_nearest_source(event.xdata, event.ydata)
            if victim is not None:
                self.sources.remove(victim)
                self.redraw()

    def _on_key(self, event):
        k = (event.key or "").lower()
        if k == "r":
            self.sources = []
            self.next_id = 1
            self.redraw()
        elif k == "s":
            self.save_all()
        elif k == "escape":
            plt.close(self.fig)
        elif k in ("c", "enter"):
            self._on_run_clicked(None)

    def _on_run_clicked(self, _event):
        if not self.sources:
            print("No sources selected. Add at least one source before Run.")
            return
        self._make_result_figure()

    # ------------------------------- drawing --------------------------------

    def redraw(self):
        self.ax.clear()
        self._init_axes()
        self._draw_grid_dots()
        self._draw_ground_line()
        self._draw_sources_only()
        self._draw_title_and_status()
        self._draw_controls_footer()
        self.fig.canvas.draw_idle()

    def _init_axes(self):
        c = self.cfg
        self.ax.set_xlim(c["grid_x_min"], c["grid_x_max"])
        self.ax.set_ylim(c["depth_max"], c["depth_min"])
        self.ax.set_xlabel("x [m]")
        self.ax.set_ylabel("depth [m]   (positive downward)")
        self.ax.set_aspect("equal", adjustable="box")
        self.ax.set_xticks(np.arange(c["grid_x_min"], c["grid_x_max"] + 0.5 * c["grid_step"], c["grid_step"]))
        self.ax.set_yticks(np.arange(c["depth_min"], c["depth_max"] + 0.5 * c["grid_step"], c["grid_step"]))
        self.ax.grid(True, color="0.82", linewidth=0.7)

    def _draw_grid_dots(self):
        c = self.cfg
        xs = np.arange(c["grid_x_min"], c["grid_x_max"] + 0.5 * c["grid_step"], c["grid_step"])
        ds = np.arange(c["depth_min"],
                       c["depth_max"] + 0.5 * c["grid_step"],
                       c["grid_step"])
        XX, DD = np.meshgrid(xs, ds)
        self.ax.plot(XX, DD, ".", color="0.75", markersize=1.4, zorder=1)

    def _draw_ground_line(self):
        c = self.cfg
        self.ax.axhline(0.0, color="green", linewidth=1.4, zorder=4)
        self.ax.text(c["grid_x_max"], 0.0, " ground (depth = 0)",
                     color="green", va="bottom", ha="right", fontsize=8, zorder=4)

    def _draw_sources_only(self):
        for s in self.sources:
            self.ax.plot(s.x_m, s.depth_m, "o",
                          markersize=7, color="tab:red", markeredgecolor="black",
                          zorder=7)

    def _draw_title_and_status(self):
        c = self.cfg
        n = len(self.sources)
        T_target = c["T_amb"] + c["epsilon_K"]
        title = f"Source selection (n = {n})"
        self.ax.set_title(title, fontsize=9, pad=14)
        status = f"q={c['q_source_Wpm']:.2f} W/m, k={c['k_soil']:.3g} W/(m·K), ε={c['epsilon_K']:.2f} K"
        self.ax.text(0.01, 0.98, status, transform=self.ax.transAxes,
                     fontsize=8, color="0.1", va="top", ha="left", zorder=9)

    def _count_sources_inside_contour(self) -> int:
        """How many real sources lie inside at least one closed contour."""
        closed = [p.vertices for p in self._last_contour_paths if is_closed_path(p.vertices)]
        if not closed:
            return 0
        n_inside = 0
        for s in self.sources:
            if any(point_in_polygon(s.x_m, s.depth_m, v) for v in closed):
                n_inside += 1
        return n_inside

    def _largest_closed_area(self) -> float:
        areas = [polygon_area(p.vertices) for p in self._last_contour_paths
                 if is_closed_path(p.vertices)]
        return max(areas) if areas else 0.0

    def _draw_controls_footer(self):
        controls = (
            "left-click: add  ·  right-click: remove nearest  ·  "
            "R: reset  ·  S: save  ·  Run button: solve field  ·  Esc: close"
        )
        self.fig.text(0.5, 0.015, controls, fontsize=8, color="0.25",
                       ha="center", va="bottom")

    def _make_result_figure(self):
        c = self.cfg
        X_plot, Y_plot = self._result_grid()
        theta = theta_superposition_ui(X_plot, Y_plot, self.sources, c["k_soil"], c["r_cutoff_m"])

        fig, ax = plt.subplots(figsize=(11.5, 7.5))
        finite = theta[np.isfinite(theta)]
        vmax_clip = float(np.percentile(finite, 99.0))
        vmax_show = max(vmax_clip, 1.5 * c["epsilon_K"])
        theta_disp = np.clip(theta, 0.0, vmax_show)

        cf = ax.contourf(X_plot, Y_plot, theta_disp, levels=24, cmap="inferno")
        cbar = fig.colorbar(cf, ax=ax)
        cbar.set_label(r"Temperature rise $\theta$ [K]")
        cs = ax.contour(X_plot, Y_plot, theta, levels=[c["epsilon_K"]], colors="cyan", linewidths=2.0)
        self._last_contour_paths = get_paths_compat(cs)
        try:
            ax.clabel(cs, fmt={c["epsilon_K"]: f"{c['epsilon_K']:g} K"}, fontsize=9)
        except Exception:
            pass

        for s in self.sources:
            ax.plot(s.x_m, s.depth_m, "o", color="white", markeredgecolor="black", markersize=8, label="_nolegend_")
            ax.plot(s.x_m, -s.depth_m, "x", color="cyan", markeredgewidth=1.5, markersize=8, label="_nolegend_")
        ax.plot([], [], "o", color="white", markeredgecolor="black", label="real sources")
        ax.plot([], [], "x", color="cyan", markeredgewidth=1.5, label="image sources")
        ax.axhline(0.0, color="green", linewidth=1.4, label="ground plane")
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y_plot / depth [m] (positive downward)")
        ax.set_title("Temperature-rise field with 5 K isotherm")
        ax.set_aspect("equal", adjustable="box")
        ax.legend(loc="upper right")
        ax.set_xlim(c["grid_x_min"], c["grid_x_max"])
        ax.set_ylim(c["depth_max"], c["depth_min"])
        fig.canvas.draw_idle()

        self.result_fig = fig
        self._last_result_payload = {"X": X_plot, "Y": Y_plot, "theta": theta}

    # ------------------------------- save -----------------------------------

    def save_all(self, outdir: str = "out") -> dict:
        outdir = Path(outdir)
        outdir.mkdir(parents=True, exist_ok=True)
        srcs_csv = outdir / "selected_sources.csv"
        cnt_csv = outdir / "isothermal_contour_5K.csv"
        png_path = outdir / "isothermal_superposition_plot.png"

        save_sources_csv(self.sources, srcs_csv)
        save_contours_csv(self._last_contour_paths, cnt_csv)
        target_fig = self.result_fig if self.result_fig is not None else self.fig
        target_fig.savefig(png_path, dpi=200, bbox_inches="tight")

        print(f"Saved →\n  {srcs_csv}\n  {cnt_csv}\n  {png_path}")
        return {"sources_csv": srcs_csv, "contour_csv": cnt_csv, "png": png_path}

    def run(self):
        plt.show()


# ---------------------------------------------------------------------------
# CLI / self-test
# ---------------------------------------------------------------------------

def parse_cli(argv=None) -> dict:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--k", type=float, default=DEFAULTS["k_soil"], help="soil k [W/(m·K)]")
    p.add_argument("--T-amb", dest="T_amb", type=float, default=DEFAULTS["T_amb"],
                   help="ambient temperature [°C]")
    p.add_argument("--epsilon", type=float, default=DEFAULTS["epsilon_K"],
                   help="target contour level [K] (default 5)")
    p.add_argument("--q", type=float, default=DEFAULTS["q_source_Wpm"],
                   help="per-source heat rate [W/m] (equal for all sources in v1)")
    p.add_argument("--res-x", type=int, default=DEFAULTS["result_nx"])
    p.add_argument("--res-y", type=int, default=DEFAULTS["result_ny"])
    p.add_argument("--x-min", type=float, default=DEFAULTS["grid_x_min"])
    p.add_argument("--x-max", type=float, default=DEFAULTS["grid_x_max"])
    p.add_argument("--depth-max", type=float, default=DEFAULTS["depth_max"])
    p.add_argument("--step", type=float, default=DEFAULTS["grid_step"])
    p.add_argument("--test", action="store_true",
                   help="run a non-interactive self-test instead of opening the UI")
    args = p.parse_args(argv)

    if not args.k > 0:
        raise SystemExit("--k must be > 0")
    if not args.epsilon > 0:
        raise SystemExit("--epsilon must be > 0")
    if not math.isfinite(args.q):
        raise SystemExit("--q must be finite")
    if not args.x_max > args.x_min:
        raise SystemExit("--x-max must be > --x-min")
    if not args.depth_max > 0:
        raise SystemExit("--depth-max must be > 0")
    if args.res_x < 64 or args.res_y < 64:
        raise SystemExit("--res-x and --res-y must be >= 64")

    cfg = dict(DEFAULTS)
    cfg.update(
        k_soil=args.k,
        T_amb=args.T_amb,
        epsilon_K=args.epsilon,
        q_source_Wpm=args.q,
        result_nx=args.res_x,
        result_ny=args.res_y,
        result_x_min=-100.0,
        result_x_max=100.0,
        result_y_min=-2.0,
        result_y_max=100.0,
        grid_x_min=args.x_min,
        grid_x_max=args.x_max,
        depth_max=args.depth_max,
        grid_step=args.step,
    )
    return cfg, args


def self_test(cfg: dict) -> int:
    """Non-interactive self-test: pre-populate sources, render, save, verify.

    For TB880-style heat rates (~35 W/m per cable, 3 cables) the 5 K contour
    reaches down to depth ~5.8 m, well outside the default UI domain of 2 m.
    The self-test enlarges the domain so the contour closes and we can
    exercise the area / containment / ground-plane checks. The interactive
    UI still uses the user-specified defaults; this only affects the test.
    """
    print("[self-test] starting non-interactive run")
    test_cfg = dict(cfg)
    test_cfg.update(
        grid_x_min=-4.0,
        grid_x_max=4.0,
        depth_max=8.0,
        q_source_Wpm=104.7564 / 3.0,   # TB880 Case 0 per-cable heat rate
    )
    app = IsothermalSuperpositionUI(test_cfg)
    # Three sources reminiscent of TB880 Case 0 (touching trefoil ~depth 1 m).
    # Snap to the 10 cm grid: depth 1.0, lateral spread ±0.1 m.
    for x_s, d_s in [(0.0, 1.0), (0.1, 1.0), (-0.1, 1.0)]:
        app._add_source(x_s, d_s)
    app.redraw()

    out = app.save_all(outdir="out_selftest")

    n_closed = sum(1 for p in app._last_contour_paths if is_closed_path(p.vertices))
    largest_area = app._largest_closed_area()
    gp_max = ground_plane_residual(app.sources, cfg["k_soil"], cfg)
    inside = app._count_sources_inside_contour()

    print(f"[self-test] contour segments    : {len(app._last_contour_paths)}")
    print(f"[self-test] closed segments     : {n_closed}")
    print(f"[self-test] largest closed area : {largest_area:.4f} m^2")
    print(f"[self-test] sources inside      : {inside} / {len(app.sources)}")
    print(f"[self-test] max |theta(x, 0)|   : {gp_max:.3e} K   (expect ~0)")

    ok = (
        n_closed >= 1
        and largest_area > 0.0
        and inside == len(app.sources)
        and gp_max < 1.0e-9
    )
    print(f"[self-test] result              : {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


def main():
    cfg, args = parse_cli()
    if args.test:
        sys.exit(self_test(cfg))
    app = IsothermalSuperpositionUI(cfg)
    app.run()


if __name__ == "__main__":
    main()
