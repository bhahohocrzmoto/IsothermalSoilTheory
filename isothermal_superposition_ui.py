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
from matplotlib.gridspec import GridSpec
from matplotlib.widgets import Button, TextBox


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULTS = dict(
    grid_x_min=-100.0,
    grid_x_max=100.0,
    depth_min=0.0,
    depth_max=100.0,
    grid_step=0.10,        # 10 cm snap
    grid_view_x_min=-2.0,
    grid_view_x_max=2.0,
    grid_view_depth_max=2.0,
    k_soil=1.0,            # W/(m K)
    T_amb=20.0,            # °C
    epsilon_K=5.0,         # K
    q_source_Wpm=30.0,     # W/m, equal per source in v1
    field_res_x=400,
    field_res_y=400,
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

        # Two figures with matched styling:
        #   - grid figure: geometry/grid/source placement
        #   - thermal figure: contourf + isotherm
        self.fig_grid = plt.figure(figsize=(10.5, 7.5))
        self.ax_grid = self.fig_grid.add_subplot(111)

        self.fig_thermal = plt.figure(figsize=(11.5, 7.5))
        gs = GridSpec(1, 2, width_ratios=[40, 1], wspace=0.08,
                      left=0.07, right=0.95, top=0.92, bottom=0.10)
        self.ax_thermal = self.fig_thermal.add_subplot(gs[0])
        self.cax_thermal = self.fig_thermal.add_subplot(gs[1])
        self.cax_thermal.set_visible(False)
        self.epsilon_levels: list[float] = [self.cfg["epsilon_K"]]
        self.fig_controls = plt.figure(figsize=(6.6, 5.0))
        self.fig_controls.suptitle("Thermal Parameters", fontsize=11)
        self._build_controls_panel()

        # Cached evaluation grid (allocated once)
        self._eval_grid: Optional[tuple[np.ndarray, np.ndarray]] = None

        # Cached last contour paths (for save_all)
        self._last_contour_paths: list = []

        # Event bindings (both windows)
        self.fig_grid.canvas.mpl_connect("button_press_event", self._on_click)
        self.fig_grid.canvas.mpl_connect("key_press_event", self._on_key)
        self.fig_thermal.canvas.mpl_connect("button_press_event", self._on_click)
        self.fig_thermal.canvas.mpl_connect("key_press_event", self._on_key)
        self.fig_controls.canvas.mpl_connect("key_press_event", self._on_key)

        # Initial draw (empty)
        self.redraw()

    def _build_controls_panel(self):
        """Technical-parameter input panel in a dedicated controls window."""
        self._controls: dict[str, object] = {}
        x0 = 0.10
        w = 0.80
        h = 0.08
        gap = 0.025
        y = 0.84

        # Parameters
        fields = [
            ("q_source_Wpm", "q [W/m]", f"{self.cfg['q_source_Wpm']:.2f}"),
            ("k_soil", "k [W/(m K)]", f"{self.cfg['k_soil']:.3g}"),
            ("T_amb", "T_amb [°C]", f"{self.cfg['T_amb']:.2f}"),
            ("T_isothermal", "T_isothermal [°C]", f"{self.cfg['T_amb'] + self.cfg['epsilon_K']:.2f}"),
        ]
        for key, label, initial in fields:
            ax = self.fig_controls.add_axes([x0, y, w, h])
            box = TextBox(ax, label=label, initial=initial)
            box.on_submit(self._on_params_submit)
            self._controls[key] = box
            y -= (h + gap)

        # Epsilon boxes (dynamic) + add button
        self._epsilon_axes: list = []
        self._epsilon_boxes: list[TextBox] = []
        self._eps_y_start = y - 0.04
        self._eps_box_h = h
        self._eps_gap = gap
        self._eps_x0 = x0
        self._eps_w = w * 0.68
        ax_add = self.fig_controls.add_axes([x0 + w * 0.72, self._eps_y_start, w * 0.26, h])
        btn_add = Button(ax_add, "+")
        btn_add.on_clicked(self._on_add_epsilon)
        self._controls["add_eps_btn"] = btn_add
        self._controls["add_eps_ax"] = ax_add
        self._rebuild_epsilon_boxes()

        # Batch select/unselect region controls
        y_bounds = 0.07
        bw = 0.24
        bh = 0.07
        bgap = 0.03

        bound_fields = [
            ("bound_x_min", "x lower [m]", f"{self.cfg['grid_view_x_min']:.2f}", [x0, y_bounds + bh + 0.01, w * 0.48, bh]),
            ("bound_x_max", "x upper [m]", f"{self.cfg['grid_view_x_max']:.2f}", [x0 + w * 0.52, y_bounds + bh + 0.01, w * 0.48, bh]),
            ("bound_d_min", "depth lower [m]", f"{self.cfg['grid_step']:.2f}", [x0, y_bounds, w * 0.48, bh]),
            ("bound_d_max", "depth upper [m]", f"{self.cfg['grid_view_depth_max']:.2f}", [x0 + w * 0.52, y_bounds, w * 0.48, bh]),
        ]
        for key, label, initial, pos in bound_fields:
            ax = self.fig_controls.add_axes(pos)
            box = TextBox(ax, label=label, initial=initial)
            self._controls[key] = box

        ax_sel = self.fig_controls.add_axes([x0, 0.01, bw, bh])
        ax_unsel = self.fig_controls.add_axes([x0 + bw + bgap, 0.01, bw, bh])
        btn_sel = Button(ax_sel, "Select")
        btn_unsel = Button(ax_unsel, "Unselect")
        btn_sel.on_clicked(self._on_select_bounds)
        btn_unsel.on_clicked(self._on_unselect_bounds)
        self._controls["select_btn"] = btn_sel
        self._controls["unselect_btn"] = btn_unsel

    def _rebuild_epsilon_boxes(self):
        if not hasattr(self, "epsilon_levels") or not self.epsilon_levels:
            self.epsilon_levels = [float(self.cfg.get("epsilon_K", 5.0))]
        for ax in self._epsilon_axes:
            ax.remove()
        self._epsilon_axes = []
        self._epsilon_boxes = []
        y = self._eps_y_start
        for i, eps in enumerate(self.epsilon_levels):
            y_i = y - i * (self._eps_box_h + self._eps_gap)
            ax = self.fig_controls.add_axes([self._eps_x0, y_i, self._eps_w, self._eps_box_h])
            box = TextBox(ax, label=f"ε{i+1} [K]", initial=f"{eps:.2f}")
            box.on_submit(self._on_epsilon_submit)
            self._epsilon_axes.append(ax)
            self._epsilon_boxes.append(box)
        add_ax = self._controls.get("add_eps_ax")
        if add_ax is not None:
            add_ax.set_position([self._eps_x0 + self._eps_w + 0.01, y, 0.05, self._eps_box_h])

    def _on_add_epsilon(self, _event):
        self.epsilon_levels.append(self.epsilon_levels[-1] if self.epsilon_levels else 5.0)
        self._rebuild_epsilon_boxes()
        self.redraw()

    def _on_epsilon_submit(self, _text):
        vals = []
        for box in self._epsilon_boxes:
            try:
                v = float(box.text)
                if v > 0.0 and math.isfinite(v):
                    vals.append(v)
            except ValueError:
                continue
        if vals:
            self.epsilon_levels = vals
            self.cfg["epsilon_K"] = vals[0]
            t_iso_box = self._controls.get("T_isothermal")
            if t_iso_box is not None:
                t_iso_box.set_val(f"{self.cfg['T_amb'] + vals[0]:.2f}")
            self.redraw()

    def _on_params_submit(self, _text):
        try:
            q = float(self._controls["q_source_Wpm"].text)
            k = float(self._controls["k_soil"].text)
            t_amb = float(self._controls["T_amb"].text)
            t_iso = float(self._controls["T_isothermal"].text)
            if not (k > 0 and math.isfinite(q) and math.isfinite(t_amb) and math.isfinite(t_iso)):
                return
            self.cfg["q_source_Wpm"] = q
            self.cfg["k_soil"] = k
            self.cfg["T_amb"] = t_amb
            eps_from_t = t_iso - t_amb
            if eps_from_t > 0:
                self.cfg["epsilon_K"] = eps_from_t
                if self.epsilon_levels:
                    self.epsilon_levels[0] = eps_from_t
                    self._epsilon_boxes[0].set_val(f"{eps_from_t:.2f}")
            for s in self.sources:
                s.q_Wpm = q
            self.redraw()
        except Exception:
            return

    # ------------------------------- input ----------------------------------

    def _ensure_eval_grid(self):
        if self._eval_grid is None:
            c = self.cfg
            x = np.linspace(c["grid_x_min"], c["grid_x_max"], c["field_res_x"])
            d = np.linspace(c["depth_min"], c["depth_max"], c["field_res_y"])
            self._eval_grid = np.meshgrid(x, d)
        return self._eval_grid

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

    def _parse_bounds(self) -> Optional[tuple[float, float, float, float]]:
        try:
            x_lo = float(self._controls["bound_x_min"].text)
            x_hi = float(self._controls["bound_x_max"].text)
            d_lo = float(self._controls["bound_d_min"].text)
            d_hi = float(self._controls["bound_d_max"].text)
        except (ValueError, KeyError):
            return None
        if not (math.isfinite(x_lo) and math.isfinite(x_hi) and math.isfinite(d_lo) and math.isfinite(d_hi)):
            return None
        if x_lo > x_hi:
            x_lo, x_hi = x_hi, x_lo
        if d_lo > d_hi:
            d_lo, d_hi = d_hi, d_lo
        return x_lo, x_hi, d_lo, d_hi

    def _iter_boundary_points(self, bounds: tuple[float, float, float, float]):
        x_lo, x_hi, d_lo, d_hi = bounds
        step = self.cfg["grid_step"]
        x0 = snap_to_grid(x_lo, step)
        x1 = snap_to_grid(x_hi, step)
        d0 = snap_to_grid(d_lo, step)
        d1 = snap_to_grid(d_hi, step)
        xs = np.arange(min(x0, x1), max(x0, x1) + 0.5 * step, step)
        ds = np.arange(min(d0, d1), max(d0, d1) + 0.5 * step, step)
        for x in xs:
            for d in ds:
                snapped = self._snap_in_grid(float(x), float(d))
                if snapped is not None:
                    yield snapped

    def _on_select_bounds(self, _event):
        bounds = self._parse_bounds()
        if bounds is None:
            return
        changed = False
        for x_s, d_s in self._iter_boundary_points(bounds):
            changed = self._add_source(x_s, d_s) or changed
        if changed:
            self.redraw()

    def _on_unselect_bounds(self, _event):
        bounds = self._parse_bounds()
        if bounds is None:
            return
        wanted = set(self._iter_boundary_points(bounds))
        if not wanted:
            return
        before = len(self.sources)
        self.sources = [
            s for s in self.sources
            if (s.x_m, s.depth_m) not in wanted
        ]
        if len(self.sources) != before:
            self.redraw()

    def _on_click(self, event):
        # Figure 2 (thermal field) is intentionally read-only: users may pan/zoom
        # to inspect artefacts without accidentally mutating the source layout.
        if event.inaxes != self.ax_grid or event.xdata is None or event.ydata is None:
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
        elif k in ("c", "enter"):
            self.redraw()
        elif k == "s":
            self.save_all()
        elif k == "escape":
            plt.close(self.fig_grid)
            plt.close(self.fig_thermal)
            plt.close(self.fig_controls)

    # ------------------------------- drawing --------------------------------

    def redraw(self):
        self.ax_grid.clear()
        self.ax_thermal.clear()
        self.cax_thermal.clear()
        self.cax_thermal.set_visible(False)

        self._init_axes(self.ax_grid, is_grid_view=True)
        self._init_axes(self.ax_thermal, is_grid_view=False)

        self._draw_grid_dots(self.ax_grid)
        self._draw_grid_dots(self.ax_thermal)
        self._draw_field()           # thermal-only; also draws epsilon contours, populates cax
        self._draw_ground_line(self.ax_grid)
        self._draw_ground_line(self.ax_thermal)
        self._draw_sources_and_images(self.ax_grid)
        self._draw_sources_and_images(self.ax_thermal)
        self._draw_title_and_status()
        self._draw_controls_footer()
        self.fig_grid.canvas.draw_idle()
        self.fig_thermal.canvas.draw_idle()

    def _init_axes(self, ax, is_grid_view: bool):
        c = self.cfg
        # Above-ground strip: just enough to show every image marker, with a
        # 0.3 m floor for a visible reference strip when nothing is placed
        # yet, and a cap at 50 % of depth_max to avoid wasting plot real
        # estate for deep-source / large-domain cases.
        if self.sources:
            max_src_depth = max(s.depth_m for s in self.sources)
        else:
            max_src_depth = 0.0
        strip = max(0.3, min(max_src_depth + 0.2, 0.5 * c["depth_max"]))
        if is_grid_view:
            x_min = c["grid_view_x_min"]
            x_max = c["grid_view_x_max"]
            depth_max = c["grid_view_depth_max"]
        else:
            x_min = c["grid_x_min"]
            x_max = c["grid_x_max"]
            depth_max = c["depth_max"]

        ax.set_xlim(x_min - 0.05, x_max + 0.05)
        ax.set_ylim(depth_max + 0.05, -strip)
        ax.set_xlabel("x [m]")
        ax.set_ylabel("depth [m]   (positive downward)")
        ax.set_aspect("equal", adjustable="box")
        ax.grid(False)

    def _draw_grid_dots(self, ax):
        c = self.cfg
        xs = np.arange(c["grid_x_min"], c["grid_x_max"] + 0.5 * c["grid_step"], c["grid_step"])
        ds = np.arange(c["depth_min"] + c["grid_step"],
                       c["depth_max"] + 0.5 * c["grid_step"],
                       c["grid_step"])
        XX, DD = np.meshgrid(xs, ds)
        ax.plot(XX, DD, ".", color="0.85", markersize=1.5, zorder=1)

    def _draw_ground_line(self, ax):
        c = self.cfg
        ax.axhline(0.0, color="green", linewidth=1.4, zorder=4)
        ax.text(c["grid_x_max"], -0.04, " ground (depth = 0)",
                     color="green", va="bottom", ha="right", fontsize=8, zorder=4)

    def _draw_field(self):
        c = self.cfg
        if not self.sources:
            self.cax_thermal.set_visible(False)
            self._last_contour_paths = []
            return

        X, D = self._ensure_eval_grid()
        theta = theta_superposition_ui(X, D, self.sources, c["k_soil"], c["r_cutoff_m"])

        # Cap colourbar so source singularities don't dominate the scale
        finite = theta[np.isfinite(theta)]
        vmax_clip = float(np.percentile(finite, 99.0))
        vmax_show = max(vmax_clip, 1.5 * c["epsilon_K"])
        theta_disp = np.clip(theta, 0.0, vmax_show)

        cf = self.ax_thermal.contourf(X, D, theta_disp, levels=20, cmap="inferno", zorder=2)
        self.cax_thermal.set_visible(True)
        cbar = self.fig_thermal.colorbar(cf, cax=self.cax_thermal)
        cbar.set_label(r"Temperature rise $\theta$ [K]")

        # Target isotherms (one or more epsilon levels)
        eps_levels = sorted({float(v) for v in self.epsilon_levels if v > 0})
        cs = self.ax_thermal.contour(
            X, D, theta, levels=eps_levels, colors="cyan", linewidths=2.0, zorder=5
        )
        try:
            self.ax_thermal.clabel(cs, fmt={v: f"{v:g} K" for v in eps_levels}, fontsize=9)
        except Exception:
            pass
        self._last_contour_paths = get_paths_compat(cs)

        # Domain-too-small warning
        if any(self._contour_touches_plot_edge(p.vertices) for p in self._last_contour_paths):
            self.ax_thermal.text(
                c["grid_x_min"] + 0.05, c["depth_max"] - 0.08,
                "⚠  ε-contour reaches plot edge — domain may be too small.\n"
                "    Increase --depth-max / --x-min / --x-max or lower --q.",
                color="yellow", fontsize=9,
                bbox=dict(facecolor="black", edgecolor="yellow", alpha=0.85, pad=4),
                zorder=6,
            )

    def _contour_touches_plot_edge(self, verts: np.ndarray) -> bool:
        c = self.cfg
        # tolerance = one pixel
        tol_x = (c["grid_x_max"] - c["grid_x_min"]) / c["field_res_x"]
        tol_d = (c["depth_max"] - c["depth_min"]) / c["field_res_y"]
        x0, x1 = c["grid_x_min"], c["grid_x_max"]
        d1 = c["depth_max"]
        return bool(
            np.any(np.abs(verts[:, 0] - x0) < tol_x) or
            np.any(np.abs(verts[:, 0] - x1) < tol_x) or
            np.any(np.abs(verts[:, 1] - d1) < tol_d)
        )

    def _draw_sources_and_images(self, ax):
        c = self.cfg
        for s in self.sources:
            # Real source
            ax.plot(s.x_m, s.depth_m, "o",
                          markersize=9, color="white", markeredgecolor="black",
                          zorder=7)
            ax.text(s.x_m + 0.04, s.depth_m, f"#{s.id}",
                          fontsize=7, color="white",
                          bbox=dict(facecolor="black", edgecolor="none", alpha=0.6, pad=1),
                          va="center", zorder=8)
            # Image (drawn at depth_ui = -depth_source, i.e. above ground;
            # clipped naturally if outside view limits)
            ax.plot(s.x_m, -s.depth_m, "x",
                          markersize=9, color="cyan", markeredgewidth=1.5,
                          zorder=7)
        if self.sources:
            ax.plot([], [], "x", color="cyan", markeredgewidth=1.5,
                          markersize=9, label="image sources (above ground)")

    def _draw_title_and_status(self):
        c = self.cfg
        n = len(self.sources)
        eps_txt = ", ".join(f"{v:g}" for v in self.epsilon_levels)
        T_target = c["T_amb"] + c["epsilon_K"]
        title = (
            f"n = {n}    q = {c['q_source_Wpm']:.2f} W/m    "
            f"k = {c['k_soil']:.3g} W/(m·K)    "
            f"T_amb = {c['T_amb']:.1f} °C    "
            f"ε = [{eps_txt}] K  →  primary isotherm at T = {T_target:.2f} °C"
        )
        self.ax_grid.set_title("Grid / source layout", fontsize=10, pad=14)
        self.ax_thermal.set_title(title, fontsize=9, pad=14)

        # Summary text in plot corner
        if self.sources:
            n_paths = len(self._last_contour_paths)
            n_closed = sum(1 for p in self._last_contour_paths if is_closed_path(p.vertices))
            gp_max = ground_plane_residual(self.sources, c["k_soil"], c)
            inside = self._count_sources_inside_contour()
            largest_area = self._largest_closed_area()
            summary = (
                f"contour segments: {n_paths}    closed: {n_closed}\n"
                f"largest closed area: {largest_area:.3f} m²\n"
                f"sources inside contour: {inside} / {len(self.sources)}\n"
                f"max |θ(x, depth=0)| : {gp_max:.2e} K  (image-method residual)"
            )
            self.ax_thermal.text(
                0.01, 0.98, summary, transform=self.ax_thermal.transAxes,
                fontsize=8, color="white", va="top", ha="left",
                bbox=dict(facecolor="black", edgecolor="0.4", alpha=0.7, pad=4),
                zorder=9,
            )

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
        controls_grid = (
            "left-click: add  ·  right-click: remove nearest  ·  "
            "box-select/unselect: controls window  ·  "
            "R: reset  ·  C/Enter: recompute  ·  S: save  ·  Esc: close"
        )
        controls_thermal = (
            "Figure is read-only (no add/remove on click)  ·  "
            "R: reset  ·  C/Enter: recompute  ·  S: save  ·  Esc: close"
        )
        self.fig_grid.text(0.5, 0.015, controls_grid, fontsize=8, color="0.25",
                            ha="center", va="bottom")
        self.fig_thermal.text(0.5, 0.015, controls_thermal, fontsize=8, color="0.25",
                               ha="center", va="bottom")

    # ------------------------------- save -----------------------------------

    def save_all(self, outdir: str = "out") -> dict:
        outdir = Path(outdir)
        outdir.mkdir(parents=True, exist_ok=True)
        srcs_csv = outdir / "selected_sources.csv"
        cnt_csv = outdir / "isothermal_contour_5K.csv"
        png_path = outdir / "isothermal_superposition_plot.png"

        save_sources_csv(self.sources, srcs_csv)
        save_contours_csv(self._last_contour_paths, cnt_csv)
        self.fig_thermal.savefig(png_path, dpi=200, bbox_inches="tight")

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
    p.add_argument("--res-x", type=int, default=DEFAULTS["field_res_x"])
    p.add_argument("--res-y", type=int, default=DEFAULTS["field_res_y"])
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
        field_res_x=args.res_x,
        field_res_y=args.res_y,
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
