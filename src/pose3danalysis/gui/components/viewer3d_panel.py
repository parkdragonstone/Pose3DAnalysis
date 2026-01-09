# -*- coding: utf-8 -*-
"""3D marker viewer panel (Matplotlib inside Tkinter).

Extracted from gui.motionanalysis_tab to keep that file smaller and easier to maintain.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional, List, Tuple

import tkinter as tk
from tkinter import ttk, filedialog
import re

import numpy as np

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import proj3d

from pose3danalysis.gui.utils.motion_settings import DEFAULT_SKELETON_EDGES


class Viewer3DPanel(ttk.Frame):
    """
    3D Viewer requirements:
    - Ground grid only (±5m around origin)
    - Global coordinate triad should be visible
    - Y-up default:
        * Y is vertical
        * To keep right-handed, flip Z sign when Y-up (x, y, z) -> (x, -z, y)
    - Z-up option:
        * Z is vertical (x, y, z) -> (x, y, z)
    - Mouse:
        * wheel = zoom
        * left drag = rotate
        * right drag = translation (pixel-accurate feeling: dx pixels -> proportional delta based on axis bbox)
    - During playback, keep the user's camera view & limits (no autoscale, no cla)
    - C3D upload supported if ezc3d installed
    """

    GRID_HALF_RANGE_M = 5.0
    GRID_MINOR_M = 0.1
    GRID_MAJOR_M = 1.0
    PAN_SENSITIVITY = 0.45  # smaller => less translation per pixel

    def __init__(
        self,
        parent,
        log: Callable[[str], None],
        z_up_var: tk.BooleanVar,
        on_loaded: Optional[Callable[[], None]] = None,
        *args,
        **kwargs,
    ):
        super().__init__(parent, *args, **kwargs)
        self._external_log = log
        self._pending_logs = []  # buffer until log_text exists
        self.z_up_var = z_up_var
        self._on_loaded = on_loaded

        self.marker_names: List[str] = []
        self.frames_xyz: Optional[List[List[Tuple[float, float, float]]]] = None  # original (x,y,z)
        self.fps: float = 60.0

        self._frame_idx: int = 0

        # Figure/Axes
        self.fig = Figure(figsize=(10, 6), dpi=100)
        self.fig.patch.set_facecolor("black")
        # Use tight layout with minimal padding to prevent clipping while ensuring full visibility
        self.ax = self.fig.add_subplot(111, projection="3d")
        self.ax.set_facecolor("black")

        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(fill="both", expand=True)
        
        # top controls (below canvas, original position)
        ctrl = ttk.Frame(self)
        ctrl.pack(fill="x", pady=(6, 0))

        ttk.Button(ctrl, text="Load Markers", command=self._on_load_motion_file).pack(side="left")
        ttk.Checkbutton(ctrl, text="Z-up", variable=self.z_up_var, command=self._on_toggle_zup).pack(side="left", padx=(12, 0))

        # interaction state
        self._drag = {
            "pressed": False,
            "button": None,
            "x": 0,
            "y": 0,
            "azim": -60.0,
            "elev": 20.0,
            "bbox_w": 600,
            "bbox_h": 400,
            "xlim": (-5, 5),
            "ylim": (-5, 5),
        }

        # mpl events
        self.canvas.mpl_connect("scroll_event", self._on_scroll_zoom)
        self.canvas.mpl_connect("button_press_event", self._on_press)
        self.canvas.mpl_connect("button_release_event", self._on_release)
        self.canvas.mpl_connect("motion_notify_event", self._on_motion)

        # artists
        self._grid_artists: List = []
        self._triad_artists: List = []
        self._scat = None
        self._skeleton_lines: List = []
        self._skeleton_enabled: bool = False
        
        # scene init
        self._init_scene()
        self.z_up_var.trace_add("write", lambda *_: self._update_frame_artists(force=True))

    def has_data(self) -> bool:
        return bool(self.frames_xyz and self.marker_names)


    def _log(self, msg: str):
        """Safe logger for Viewer3DPanel."""
        try:
            ext = getattr(self, "_external_log", None)
            if callable(ext):
                ext(str(msg))
            else:
                self._pending_logs.append(str(msg))
        except Exception:
            try:
                self._pending_logs.append(str(msg))
            except Exception:
                pass

    def attach_logger(self, log):
        """Attach logger later and flush buffered logs."""
        self._external_log = log
        try:
            buf = list(getattr(self, "_pending_logs", []))
            self._pending_logs = []
        except Exception:
            buf = []
        if callable(log):
            for m in buf:
                try:
                    log(m)
                except Exception:
                    break
    def clear_marker_data(self, keep_view: bool = True):
        """Clear loaded TRC/C3D marker data and remove artists."""
        # keep current camera/limits if requested
        if keep_view:
            try:
                az, el = float(self.ax.azim), float(self.ax.elev)
                xlim = self.ax.get_xlim3d()
                ylim = self.ax.get_ylim3d()
                zlim = self.ax.get_zlim3d()
            except Exception:
                az = el = None
                xlim = ylim = zlim = None
        else:
            az = el = None
            xlim = ylim = zlim = None

        # clear stored data
        self.marker_names = []
        self.frames_xyz = None
        self._frame_idx = 0

        # clear scatter
        try:
            if self._scat is not None:
                self._scat._offsets3d = ([], [], [])
        except Exception:
            pass

        # remove skeleton lines
        for ln in list(getattr(self, "_skeleton_lines", [])):
            try:
                ln.remove()
            except Exception:
                pass
        self._skeleton_lines = []
        self._skeleton_enabled = False

        # reset camera/zoom/pan to defaults
        if not keep_view:
            try:
                self._init_scene()
                # default view
                self.ax.view_init(elev=self._drag.get('elev', 20.0), azim=self._drag.get('azim', -60.0))
            except Exception:
                pass

        # restore view/limits
        if keep_view and (az is not None):
            try:
                self.ax.view_init(elev=el, azim=az)
                if xlim:
                    self.ax.set_xlim3d(*xlim)
                if ylim:
                    self.ax.set_ylim3d(*ylim)
                if zlim:
                    self.ax.set_zlim3d(*zlim)
            except Exception:
                pass

        try:
            self.canvas.draw_idle()
        except Exception:
            pass


    def n_frames(self) -> int:
        return int(len(self.frames_xyz) if self.frames_xyz else 0)

    # ---------------- scene / grid / triad ----------------
    def _init_scene(self):
        self.ax.cla()
        self.ax.set_facecolor("black")

        self._draw_ground_grid()
        self._draw_global_triad()

        # fixed limits around origin for ground (±5m)
        # keep equal range so the vertical axis doesn't look "squashed"
        r = self.GRID_HALF_RANGE_M
        self.ax.set_xlim(-r, r)
        self.ax.set_ylim(-r, r)
        self.ax.set_zlim(-r, r)

        # fill canvas and keep 1:1:1 box aspect if available
        # Use _update_figure_layout to prevent clipping
        try:
            self._update_figure_layout()
        except Exception:
            pass
        try:
            self.ax.set_position([0, 0, 1, 1])
        except Exception:
            pass
        try:
            self.ax.set_box_aspect((1, 1, 1))
        except Exception:
            pass

        self._hide_axes_box()

        self._scat = self.ax.scatter([], [], [], s=16, c="dodgerblue", depthshade=False)
        self.canvas.draw_idle()

    def _hide_axes_box(self):
        # hide ticks/labels/box, keep our custom grid+triad
        try:
            self.ax.set_axis_off()
        except Exception:
            pass
        for axis in (self.ax.xaxis, self.ax.yaxis, self.ax.zaxis):
            try:
                axis.pane.set_visible(False)
            except Exception:
                pass
            try:
                axis._axinfo["grid"]["linewidth"] = 0
            except Exception:
                pass

    def _draw_ground_grid(self):
        # remove old
        for a in self._grid_artists:
            try:
                a.remove()
            except Exception:
                pass
        self._grid_artists = []

        r = self.GRID_HALF_RANGE_M
        minor = self.GRID_MINOR_M
        major = self.GRID_MAJOR_M

        def add_line(x0, y0, x1, y1, lw, alpha):
            ln, = self.ax.plot([x0, x1], [y0, y1], [0, 0], color=(1, 1, 1, alpha), linewidth=lw)
            self._grid_artists.append(ln)

        # minor 0.1m (very light)
        n = int((2 * r) / minor)
        for i in range(n + 1):
            v = -r + i * minor
            if abs(v) < 1e-9 or abs((v / major) - round(v / major)) < 1e-9:
                continue
            add_line(v, -r, v, r, lw=0.6, alpha=0.08)
            add_line(-r, v, r, v, lw=0.6, alpha=0.08)

        # major 1m
        nM = int((2 * r) / major)
        for i in range(nM + 1):
            v = -r + i * major
            if abs(v) < 1e-9:
                continue
            add_line(v, -r, v, r, lw=1.0, alpha=0.22)
            add_line(-r, v, r, v, lw=1.0, alpha=0.22)

        # origin
        add_line(0, -r, 0, r, lw=2.4, alpha=0.55)
        add_line(-r, 0, r, 0, lw=2.4, alpha=0.55)

    def _draw_global_triad(self):
        # remove old
        for a in self._triad_artists:
            try:
                a.remove()
            except Exception:
                pass
        self._triad_artists = []

        # Keep small triad at origin
        L = 0.8

        if self.z_up_var.get():
            # Z-up mode: Y and Z swapped from Y-up, Y reversed
            # X (red) along mpl -X (left, reversed, same as Y-up)
            qx = self.ax.quiver(0, 0, 0, -L, 0, 0, color=(1, 0, 0, 1), linewidth=2)
            # Z (blue) vertical along mpl Z (was original Z, now vertical)
            qz = self.ax.quiver(0, 0, 0, 0, 0, L, color=(0, 0, 1, 1), linewidth=2)
            # Y (green) along mpl -Y (reversed, was forward in Y-up)
            qy = self.ax.quiver(0, 0, 0, 0, -L, 0, color=(0, 1, 0, 1), linewidth=2)
            
            tx = self.ax.text(-L, 0, 0, "X", color="white")
            tz = self.ax.text(0, 0, L, "Z", color="white")
            ty = self.ax.text(0, -L, 0, "Y", color="white")
        else:
            # Y-up mode: X left (reversed), Z forward, Y vertical
            # X (red) along mpl -X (left, reversed)
            qx = self.ax.quiver(0, 0, 0, -L, 0, 0, color=(1, 0, 0, 1), linewidth=2)
            # Y (blue) vertical along mpl Z (up)
            qy = self.ax.quiver(0, 0, 0, 0, 0, L, color=(0, 0, 1, 1), linewidth=2)
            # Z (green) along mpl Y (forward, positive direction)
            qz = self.ax.quiver(0, 0, 0, 0, L, 0, color=(0, 1, 0, 1), linewidth=2)
            
            tx = self.ax.text(-L, 0, 0, "X", color="white")
            ty = self.ax.text(0, 0, L, "Y", color="white")
            tz = self.ax.text(0, L, 0, "Z", color="white")

        self._triad_artists.extend([qx, qy, qz, tx, ty, tz])

    def _on_toggle_zup(self):
        self._log(f"[3D] Z-up = {self.z_up_var.get()}")
        self._draw_global_triad()
        self._update_frame_artists(force=True)

    # ---------------- external control ----------------
    def set_frame(self, idx: int):
        if not self.frames_xyz:
            return
        idx = int(max(0, min(idx, len(self.frames_xyz) - 1)))
        self._frame_idx = idx
        self._update_frame_artists()

    def get_fps(self) -> float:
        return float(self.fps or 60.0)

    # ---------------- mapping / artists update ----------------
    def _map_points(self, pts: List[Tuple[float, float, float]]):
        """
        Map 3D points to matplotlib coordinate system.
        
        Data is stored in Y-up coordinate system (x, y, z) where Y is vertical.
        - Y-up mode: (x, y, z) -> mpl(-x, z, y) [X left (reversed), Z forward, Y vertical in mpl Z]
        - Z-up mode: Swap Y and Z from Y-up, reverse Y: (x, y, z) -> mpl(-x, -y, z) [X left, Y reversed, Z vertical in mpl Z]
        
        Note: Data itself is not modified, only the display mapping changes.
        """
        if not pts:
            return [], [], []
        if self.z_up_var.get():
            # Z-up: Swap Y and Z axes from Y-up mode, and reverse Y
            # Y-up mode: (x, y, z) -> mpl(-x, z, y)
            # Z-up mode: Swap Y and Z, reverse Y
            # Original: (x, y, z)
            # Swap Y and Z: (x, z, y)
            # Reverse Y: (x, -y, z)
            # Keep X reversed: (-x, -y, z)
            # Result: mpl(-x, -y, z) where mpl Z (vertical) = original Z
            xs = [-p[0] for p in pts]  # X: left (reversed, same as Y-up)
            ys = [-p[1] for p in pts]  # Y: reversed (was forward in Y-up)
            zs = [p[2] for p in pts]   # Z: vertical (was Y in Y-up)
        else:
            # Y-up: X left (reversed), Z forward, Y vertical
            # (x, y, z) -> mpl(-x, z, y)
            # X: left (mpl X, reversed) - negative direction
            # Z: forward (mpl Y) - positive Z direction
            # Y: vertical (mpl Z) - unchanged
            xs = [-p[0] for p in pts]  # X: left (reversed)
            ys = [p[2] for p in pts]   # Z: forward
            zs = [p[1] for p in pts]   # Y: vertical
        return xs, ys, zs

    def _skeleton_available(self) -> bool:
        if not self.marker_names:
            return False
        name_set = set(self.marker_names)
        for a, b in DEFAULT_SKELETON_EDGES:
            if a not in name_set or b not in name_set:
                return False
        return True

    def _ensure_skeleton_artists(self):
        for ln in self._skeleton_lines:
            try:
                ln.remove()
            except Exception:
                pass
        self._skeleton_lines = []

        self._skeleton_enabled = self._skeleton_available()
        if not self._skeleton_enabled:
            return

        for _ in DEFAULT_SKELETON_EDGES:
            ln, = self.ax.plot([], [], [], linewidth=2)
            self._skeleton_lines.append(ln)

    def _update_frame_artists(self, force: bool = False):
        if not self.frames_xyz or not self.marker_names:
            self.canvas.draw_idle()
            return

        if force:
            self._ensure_skeleton_artists()
            self._draw_global_triad()

        pts = self.frames_xyz[self._frame_idx]
        xs, ys, zs = self._map_points(pts)

        if self._scat is not None:
            self._scat._offsets3d = (xs, ys, zs)

        if self._skeleton_enabled and self._skeleton_lines:
            idx_map = {nm: k for k, nm in enumerate(self.marker_names)}
            for k, (a, b) in enumerate(DEFAULT_SKELETON_EDGES):
                ia, ib = idx_map[a], idx_map[b]
                xa, ya, za = pts[ia]
                xb, yb, zb = pts[ib]

                if self.z_up_var.get():
                    # Z-up: Swap Y and Z from Y-up, reverse Y
                    # (x, y, z) -> mpl(-x, -y, z) [X left, Y reversed, Z vertical]
                    x1, y1, z1 = -xa, -ya, za
                    x2, y2, z2 = -xb, -yb, zb
                else:
                    # Y-up: (x, y, z) -> mpl(-x, z, y) [X left (reversed), Z forward, Y vertical]
                    x1, y1, z1 = -xa, za, ya
                    x2, y2, z2 = -xb, zb, yb

                ln = self._skeleton_lines[k]
                ln.set_data([x1, x2], [y1, y2])
                ln.set_3d_properties([z1, z2])

        # preserve camera view & limits
        self.canvas.draw_idle()

    # ---------------- interaction ----------------
    def _on_scroll_zoom(self, event):
        step = 0.9 if getattr(event, "button", None) == "up" else 1.1

        x0, x1 = self.ax.get_xlim()
        y0, y1 = self.ax.get_ylim()
        z0, z1 = self.ax.get_zlim()

        cx = (x0 + x1) / 2
        cy = (y0 + y1) / 2
        cz = (z0 + z1) / 2

        self.ax.set_xlim(cx + (x0 - cx) * step, cx + (x1 - cx) * step)
        self.ax.set_ylim(cy + (y0 - cy) * step, cy + (y1 - cy) * step)
        self.ax.set_zlim(cz + (z0 - cz) * step, cz + (z1 - cz) * step)

        self.canvas.draw_idle()

    def _on_press(self, event):
        if event.x is None or event.y is None:
            return
        self._drag["pressed"] = True
        self._drag["button"] = event.button  # 1 left, 3 right
        self._drag["x"] = event.x
        self._drag["y"] = event.y
        self._drag["azim"] = self.ax.azim
        self._drag["elev"] = self.ax.elev

        # store bbox and limits for pixel-accurate panning feel
        try:
            renderer = self.canvas.get_renderer()
            bb = self.ax.get_window_extent(renderer=renderer)
            self._drag["bbox_w"] = max(1.0, float(bb.width))
            self._drag["bbox_h"] = max(1.0, float(bb.height))
        except Exception:
            self._drag["bbox_w"] = 600.0
            self._drag["bbox_h"] = 400.0
        self._drag["xlim"] = self.ax.get_xlim()
        self._drag["ylim"] = self.ax.get_ylim()

        try:
            self._drag["zlim"] = self.ax.get_zlim()
        except Exception:
            self._drag["zlim"] = (-5.0, 5.0)

        # screen-aligned basis vectors for right-drag panning on ground plane (no Z scaling)
        self._drag["pan_basis"] = None
        try:
            x0, x1 = self._drag["xlim"]
            y0, y1 = self._drag["ylim"]
            z0, z1 = self._drag["zlim"]
            cx = (x0 + x1) / 2.0
            cy = (y0 + y1) / 2.0
            cz = (z0 + z1) / 2.0

            M = self.ax.get_proj()
            def _to_px(x, y, z):
                x2, y2, z2 = proj3d.proj_transform(x, y, z, M)
                return self.ax.transData.transform((x2, y2))

            p0 = _to_px(cx, cy, cz)
            px = _to_px(cx + 1.0, cy, cz)
            py = _to_px(cx, cy + 1.0, cz)
            vx = (px - p0)
            vy = (py - p0)
            self._drag["pan_basis"] = (vx, vy)
        except Exception:
            self._drag["pan_basis"] = None


    def _on_release(self, _event):
        self._drag["pressed"] = False
        self._drag["button"] = None

    def _on_motion(self, event):
        if not self._drag["pressed"]:
            return
        if event.x is None or event.y is None:
            return

        dx = float(event.x - self._drag["x"])
        dy = float(event.y - self._drag["y"])

        if self._drag["button"] == 1:
            az = self._drag["azim"] + dx * 0.5
            el = self._drag["elev"] - dy * 0.5
            self.ax.view_init(elev=el, azim=az)
            self.canvas.draw_idle()

        elif self._drag["button"] == 3:
            # translation / pan (screen-aligned; no Z scaling changes)
            x0, x1 = self._drag["xlim"]
            y0, y1 = self._drag["ylim"]
            z0, z1 = self._drag.get("zlim", self.ax.get_zlim())
            bw = float(self._drag.get("bbox_w", 600.0))
            bh = float(self._drag.get("bbox_h", 400.0))

            dx_data = None
            dy_data = None

            basis = self._drag.get("pan_basis", None)
            if basis is not None:
                try:
                    vx, vy = basis  # pixel vectors for +1 data-unit along X and Y
                    A = np.array([[vx[0], vy[0]],
                                  [vx[1], vy[1]]], dtype=float)
                    b = np.array([dx, dy], dtype=float)
                    sol = np.linalg.solve(A, b)
                    dx_data = float(sol[0])
                    dy_data = float(sol[1])
                except Exception:
                    dx_data = None
                    dy_data = None

            # fallback: axis-range mapping
            if dx_data is None or dy_data is None:
                dx_data = (dx / bw) * (x1 - x0)
                dy_data = (dy / bh) * (y1 - y0)

            # move the scene with the mouse (limits shift opposite)
            shift_x = -dx_data * self.PAN_SENSITIVITY
            shift_y = -dy_data * self.PAN_SENSITIVITY

            self.ax.set_xlim(x0 + shift_x, x1 + shift_x)
            self.ax.set_ylim(y0 + shift_y, y1 + shift_y)
            # keep vertical axis scale unchanged
            try:
                self.ax.set_zlim(z0, z1)
            except Exception:
                pass
            self.canvas.draw_idle()

    # ---------------- data loading ----------------
    def _on_load_motion_file(self):
        path = filedialog.askopenfilename(
            title="Select TRC or C3D file",
            filetypes=[("Motion files", "*.trc *.c3d"), ("TRC", "*.trc"), ("C3D", "*.c3d"), ("All files", "*.*")],
        )
        if not path:
            return
        ext = Path(path).suffix.lower()
        if ext == ".trc":
            self._load_trc_path(path)
        elif ext == ".c3d":
            self._load_c3d_path(path)
        else:
            self._log(f"[3D][WARN] Unsupported file type: {path}")


    def load_motion_path(self, path: str):
        """Programmatic loader (used by Video-folder auto TRC load)."""
        if not path:
            return
        ext = Path(path).suffix.lower()
        if ext == ".trc":
            self._load_trc_path(path)
        elif ext == ".c3d":
            self._load_c3d_path(path)
        else:
            self._log(f"[3D][WARN] Unsupported file type: {path}")

    def _load_trc_path(self, path: str):
        try:
            names, frames, fps = self._read_trc_simple(path)
            self.marker_names = names
            self.frames_xyz = frames
            self.fps = fps if fps and fps > 0 else 60.0
            self._frame_idx = 0

            self._ensure_skeleton_artists()
            self._update_frame_artists(force=True)

            self._log(f"[3D] Loaded TRC: {Path(path).name} | frames={len(frames)} | markers={len(names)} | fps={self.fps:.2f}")
            if self._on_loaded:
                try:
                    self._on_loaded()
                except Exception:
                    pass
        except Exception as e:
            self._log(f"[3D] TRC load failed: {e}")

    def _load_c3d_path(self, path: str):
        try:
            names, frames, fps = self._read_c3d(path)
            self.marker_names = names
            self.frames_xyz = frames
            self.fps = fps if fps and fps > 0 else 60.0
            self._frame_idx = 0

            self._ensure_skeleton_artists()
            self._update_frame_artists(force=True)

            self._log(f"[3D] Loaded C3D: {Path(path).name} | frames={len(frames)} | markers={len(names)} | fps={self.fps:.2f}")
            if self._on_loaded:
                try:
                    self._on_loaded()
                except Exception:
                    pass
        except ImportError:
            self._log("[3D] C3D load requires 'ezc3d'. Install: pip install ezc3d")
        except Exception as e:
            self._log(f"[3D] C3D load failed: {e}")
    def _on_load_trc(self):
        path = filedialog.askopenfilename(
            title="Select TRC file",
            filetypes=[("TRC", "*.trc"), ("All files", "*.*")],
        )
        if not path:
            return
        try:
            names, frames, fps = self._read_trc_simple(path)
            self.marker_names = names
            self.frames_xyz = frames
            self.fps = fps if fps and fps > 0 else 60.0
            self._frame_idx = 0

            self._ensure_skeleton_artists()

            self._update_frame_artists(force=True)

            self._log(f"[3D] Loaded TRC: {Path(path).name} | frames={len(frames)} | markers={len(names)} | fps={self.fps:.2f}")
            if self._on_loaded:
                try:
                    self._on_loaded()
                except Exception:
                    pass
        except Exception as e:
            self._log(f"[3D][ERROR] TRC load failed: {e}")

    def _on_load_c3d(self):
        path = filedialog.askopenfilename(
            title="Select C3D file",
            filetypes=[("C3D", "*.c3d"), ("All files", "*.*")],
        )
        if not path:
            return
        try:
            names, frames, fps = self._read_c3d(path)
            self.marker_names = names
            self.frames_xyz = frames
            self.fps = fps if fps and fps > 0 else 60.0
            self._frame_idx = 0

            self._ensure_skeleton_artists()
            self._update_frame_artists(force=True)

            self._log(f"[3D] Loaded C3D: {Path(path).name} | frames={len(frames)} | markers={len(names)} | fps={self.fps:.2f}")
            if self._on_loaded:
                try:
                    self._on_loaded()
                except Exception:
                    pass
        except ImportError:
            self._log("[3D] C3D load requires 'ezc3d'. Install: pip install ezc3d")
        except Exception as e:
            self._log(f"[3D][ERROR] C3D load failed: {e}")

    # ---------------- minimal loaders ----------------
    def _read_trc_simple(
        self, trc_path: str
    ) -> Tuple[List[str], List[List[Tuple[float, float, float]]], float]:
        with open(trc_path, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()
        if len(lines) < 6:
            raise ValueError("TRC file is too short.")        # fps from header if possible (robust TRC parsing)
        fps = 60.0
        try:
            # Typical TRC:
            # line 0: PathFileType ...
            # line 1: DataRate    CameraRate ...
            # line 2: numeric values (DataRate etc.)
            dr_idx = None
            for i in range(0, min(20, len(lines) - 1)):
                if ("DataRate" in lines[i]) and ("CameraRate" in lines[i]):
                    dr_idx = i
                    break

            if dr_idx is not None:
                header = re.split(r"[	 ,]+", lines[dr_idx].strip())
                j = dr_idx + 1
                while j < len(lines) and not lines[j].strip():
                    j += 1
                if j < len(lines):
                    values = re.split(r"[	 ,]+", lines[j].strip())
                    if "DataRate" in header:
                        k = header.index("DataRate")
                        if k < len(values):
                            fps = float(values[k])

            if not fps or fps <= 0:
                raise ValueError("Invalid fps in header")
        except Exception:
            # Fallback: infer fps from first two numeric rows' time delta
            try:
                data_start = None
                for i in range(0, len(lines)):
                    toks = lines[i].strip().split("	")
                    if len(toks) >= 2:
                        try:
                            float(toks[0]); float(toks[1])
                            data_start = i
                            break
                        except Exception:
                            pass
                if data_start is not None and (data_start + 1) < len(lines):
                    t0 = float(lines[data_start].strip().split("	")[1])
                    t1 = float(lines[data_start + 1].strip().split("	")[1])
                    dt = (t1 - t0)
                    if dt > 0:
                        fps = 1.0 / dt
            except Exception:
                fps = 60.0

        # find marker name row
        name_row = None
        for i in range(0, min(10, len(lines))):
            if "Frame" in lines[i] and "Time" in lines[i]:
                name_row = i
                break
        if name_row is None:
            name_row = 3

        name_tokens = lines[name_row].strip("\n").split("\t")
        raw = name_tokens[2:]
        marker_names = []
        for i in range(0, len(raw), 3):
            nm = raw[i].strip()
            if nm:
                marker_names.append(nm)

        # find first numeric data row
        data_start = None
        for i in range(name_row + 1, len(lines)):
            toks = lines[i].strip().split("\t")
            if len(toks) >= 2:
                try:
                    float(toks[0])
                    float(toks[1])
                    data_start = i
                    break
                except Exception:
                    pass
        if data_start is None:
            raise ValueError("No numeric data lines found in TRC.")

        frames_xyz: List[List[Tuple[float, float, float]]] = []
        m = len(marker_names)
        expected_cols = 2 + 3 * m

        for i in range(data_start, len(lines)):
            toks = lines[i].strip().split("\t")
            if len(toks) < expected_cols:
                continue
            coords = toks[2:2 + 3 * m]
            pts: List[Tuple[float, float, float]] = []
            for j in range(m):
                x = float(coords[3*j + 0])
                y = float(coords[3*j + 1])
                z = float(coords[3*j + 2])
                pts.append((x, y, z))
            frames_xyz.append(pts)

        return marker_names, frames_xyz, fps

    def _read_c3d(self, c3d_path: str):
        """Read C3D markers with the lightweight `c3d` package.

        Returns labels, frames_xyz (meters), fps.
        """
        try:
            import c3d  # type: ignore
        except Exception as e:
            raise ImportError(str(e))

        with open(c3d_path, 'rb') as fh:
            reader = c3d.Reader(fh)
            fps = float(getattr(reader.header, 'frame_rate', 60.0) or 60.0)

            # Labels
            labels = []
            try:
                labels = [str(x).strip() for x in reader.point_labels]
            except Exception:
                labels = []

            frames_xyz: List[List[Tuple[float, float, float]]] = []
            for _, points, _ in reader.read_frames():
                # points: (n_points, 5) -> x,y,z, residual, camera_mask; typically mm
                frame_pts: List[Tuple[float, float, float]] = []
                for p in points:
                    x = float(p[0]) / 1000.0
                    y = float(p[1]) / 1000.0
                    z = float(p[2]) / 1000.0
                    frame_pts.append((x, y, z))
                frames_xyz.append(frame_pts)

        # If labels were missing, derive placeholder names
        if not labels and frames_xyz:
            labels = [f"M{i+1}" for i in range(len(frames_xyz[0]))]

        return labels, frames_xyz, fps


# ---------------------- Side settings panel (collapsible sideways) ----------------------


class ScrollableFrame(ttk.Frame):
    """A vertical scrollable frame (Canvas + inner frame) with mouse-wheel support."""

    def __init__(self, parent, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)

        self.canvas = tk.Canvas(self, highlightthickness=0, borderwidth=0)
        self.vsb = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=self.vsb.set)

        self.vsb.pack(side="right", fill="y")
        self.canvas.pack(side="left", fill="both", expand=True)

        self.inner = ttk.Frame(self.canvas)
        self._win = self.canvas.create_window((0, 0), window=self.inner, anchor="nw")

        def _on_inner_configure(event=None):
            self.canvas.configure(scrollregion=self.canvas.bbox("all"))

        def _on_canvas_configure(event):
            # keep inner width synced to canvas width
            try:
                self.canvas.itemconfig(self._win, width=event.width)
            except Exception:
                pass

        self.inner.bind("<Configure>", _on_inner_configure)
        self.canvas.bind("<Configure>", _on_canvas_configure)

        self._bind_mousewheel(self.canvas)

    def _bind_mousewheel(self, widget):
        # Windows / macOS
        widget.bind_all("<MouseWheel>", self._on_mousewheel, add="+")
        # Linux
        widget.bind_all("<Button-4>", self._on_mousewheel_linux, add="+")
        widget.bind_all("<Button-5>", self._on_mousewheel_linux, add="+")

    def _on_mousewheel(self, event):
        # Only scroll if the mouse is over this widget tree
        try:
            x, y = self.winfo_pointerxy()
            w = self.winfo_containing(x, y)
            if w is None:
                return
            if not (w == self.canvas or str(w).startswith(str(self.canvas)) or str(w).startswith(str(self.inner))):
                return
        except Exception:
            pass
        delta = int(-1 * (event.delta / 120))
        self.canvas.yview_scroll(delta, "units")

    def _on_mousewheel_linux(self, event):
        try:
            x, y = self.winfo_pointerxy()
            w = self.winfo_containing(x, y)
            if w is None:
                return
            if not (w == self.canvas or str(w).startswith(str(self.canvas)) or str(w).startswith(str(self.inner))):
                return
        except Exception:
            pass
        if event.num == 4:
            self.canvas.yview_scroll(-1, "units")
        elif event.num == 5:
            self.canvas.yview_scroll(1, "units")

class SideSettingsPanel(ttk.Frame):
    """Right settings panel that can be collapsed sideways (toggle button always visible)."""

    def __init__(self, parent, on_toggle: Callable[[bool], None], *args, **kwargs):
        super().__init__(parent, *args, **kwargs)
        self._on_toggle = on_toggle
        self._open = True

        header = ttk.Frame(self)
        header.pack(fill="x", padx=6, pady=(6, 0))
        self.btn = ttk.Button(header, text="◀  Settings", command=self.toggle)
        self.btn.pack(side="left", fill="x", expand=True)

        self._scroll = ScrollableFrame(self)
        self._scroll.pack(fill="both", expand=True, padx=6, pady=6)

        self.body = self._scroll.inner

    def toggle(self):
        self._open = not self._open
        self._on_toggle(self._open)

    def set_header_text(self, open_: bool):
        self.btn.configure(text=("◀  Settings" if open_ else "▶  Settings"))

    def show_body(self):
        # avoid duplicate packing
        if not self._scroll.winfo_ismapped():
            self._scroll.pack(fill="both", expand=True, padx=6, pady=6)

    def hide_body(self):
        if self._scroll.winfo_ismapped():
            self._scroll.pack_forget()
