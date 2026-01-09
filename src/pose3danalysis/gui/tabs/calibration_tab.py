# -*- coding: utf-8 -*-
"""Calibration Tab for multi-camera intrinsic/extrinsic calibration.

Extracted from gui.main_app to improve code organization.
"""

from __future__ import annotations

import threading
import time
import traceback
from pathlib import Path
from typing import Optional, List, Callable

import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

from pose3danalysis.core.frame_extract import extract_frames_every_n_sec
from pose3danalysis.core.utils import ensure_dir
from pose3danalysis.core.point_picker import PointPicker
from pose3danalysis.core.zoom_preview import ZoomPreview

from pose3danalysis.gui.utils.helpers import cam_id
from pose3danalysis.gui.utils.calibration_utils import (
    parse_obj_points_mm,
    estimate_depth_m,
    draw_scene_overlay,
    annotate_chessboard_files,
)
from pose3danalysis.gui.calibration.session_manager import SessionManager
from pose3danalysis.gui.calibration.intrinsic_processor import process_intrinsic_calibration
from pose3danalysis.gui.calibration.extrinsic_processor import process_extrinsic_calibration


class CalibrationTab(ttk.Frame):
    """Calibration Tab for multi-camera (2~8) intrinsic/extrinsic calibration."""

    def __init__(
        self,
        parent,
        log: Callable[[str], None],
        shared_log_text=None,
        app_dir: Optional[Path] = None,
        *args,
        **kwargs
    ):
        super().__init__(parent, *args, **kwargs)
        self._external_log = log
        self.shared_log_text = shared_log_text
        self._pending_logs = []

        # session/temp (preview cache + Pose2Sim workspace)
        if app_dir is None:
            # Go up from tabs/calibration_tab.py -> gui -> pose3danalysis -> src -> project root
            self.app_dir = Path(__file__).resolve().parent.parent.parent.parent.parent
        else:
            self.app_dir = app_dir

        self.session_manager = SessionManager(self.app_dir, lambda: self.num_cams.get())

        # cameras
        self.num_cams = tk.IntVar(value=2)
        self._init_camera_state(self.num_cams.get())

        # output folder (path only)
        self.output_dir = tk.StringVar(value=str(Path.cwd() / "output"))

        # intrinsic inputs
        self.square_size_mm = tk.DoubleVar(value=60.0)
        self.inner_cols = tk.IntVar(value=3)
        self.inner_rows = tk.IntVar(value=5)
        self.extract_every_sec = tk.IntVar(value=1)
        self.show_corners_preview = tk.BooleanVar(value=True)
        # Show clicked/projected overlay on preview (scene method). Default ON.
        self.show_scene_overlay_preview = tk.BooleanVar(value=True)

        # extrinsic inputs
        self.lens_toml_path = tk.StringVar(value="")
        self.extrinsic_method = tk.StringVar(value="board")  # board | scene
        self.board_position = tk.StringVar(value="horizontal")

        # scene points (object in mm, image in px)
        self.scene_obj_points_mm: Optional[np.ndarray] = None  # (N,3) mm
        self.scene_img_points_px: List[Optional[np.ndarray]] = []  # per cam (N,2)

        # widgets that should wrap text within the right panel
        self._wrap_widgets: list[tk.Widget] = []

        # Last extrinsic solution for preview overlay
        self._last_extr_R = None
        self._last_extr_T = None
        self._last_extr_K = None
        self._last_extr_D = None

        self._build_ui()

        # Flush pending logs
        try:
            for _m in getattr(self, '_pending_logs', []):
                self._append_log(_m)
            if hasattr(self, '_pending_logs'):
                self._pending_logs.clear()
        except Exception:
            pass

    def _log(self, msg: str, level: str = "INFO"):
        """Log message to shared log (avoid duplication)."""
        msg = str(msg)
        # If external_log is provided, it already handles shared_log_text
        # So we only append if external_log is not available
        if self._external_log:
            try:
                self._external_log(msg, level=level)
            except Exception:
                pass
            # Don't append again if external_log is used (it already appends to shared_log_text)
            return

        if self.shared_log_text is not None:
            self._append_log(msg)
        elif self._pending_logs is not None:
            self._pending_logs.append(msg)

    def _append_log(self, msg: str):
        """Append message to shared log widget."""
        if self.shared_log_text is None:
            return
        try:
            self.shared_log_text.configure(state="normal")
            self.shared_log_text.insert("end", msg + "\n")
            self.shared_log_text.see("end")
            self.shared_log_text.configure(state="disabled")
        except Exception:
            pass

    def _popup_info(self, title: str, msg: str):
        try:
            messagebox.showinfo(title, msg)
        except Exception:
            pass

    def _popup_error(self, title: str, msg: str):
        try:
            messagebox.showerror(title, msg)
        except Exception:
            pass

    def _init_camera_state(self, n: int):
        n = int(n)
        self.video_paths: List[str] = [""] * n

        self.playing: List[bool] = [False] * n
        self.pos: List[int] = [0] * n

        self.preview_files: List[List[Path]] = [[] for _ in range(n)]
        self.corner_overlay_files: List[List[Path]] = [[] for _ in range(n)]

        self._last_rgb: List[Optional[np.ndarray]] = [None] * n

        self.scene_img_points_px = [None] * n

        # UI handles (created in _build_preview_grid)
        self.preview_widgets: List[ZoomPreview] = []
        self.scale_widgets: List[ttk.Scale] = []
        self.frame_labels: List[ttk.Label] = []

    def _build_ui(self):
        # top bar
        topbar = ttk.Frame(self)
        topbar.pack(fill="x", padx=10, pady=(10, 0))

        ttk.Label(topbar, text="Number of Cameras:").pack(side="left")
        self.cmb_num_cams = ttk.Combobox(
            topbar,
            values=[str(i) for i in range(2, 9)],
            state="readonly",
            width=5
        )
        self.cmb_num_cams.set(str(self.num_cams.get()))
        self.cmb_num_cams.pack(side="left", padx=8)
        self.cmb_num_cams.bind("<<ComboboxSelected>>", self.on_num_cams_changed)
        
        # Load Videos and Clear buttons next to Number of Cameras
        ttk.Button(topbar, text="Load Videos", command=self.on_load_videos).pack(side="left", padx=(12, 6))
        ttk.Button(topbar, text="Clear", command=self.on_clear_videos).pack(side="left", padx=(0, 6))

        lbl_note = ttk.Label(topbar, text="")
        lbl_note.pack(side="left", padx=8)
        self._register_wrap(lbl_note)

        paned = ttk.PanedWindow(self, orient="horizontal")
        paned.pack(fill="both", expand=True, padx=10, pady=10)

        left = ttk.Frame(paned)
        right = ttk.Frame(paned)
        # auto-wrap long labels within right panel
        right.bind("<Configure>", lambda e: self._update_wraplength(e.width))
        paned.add(left, weight=7)
        paned.add(right, weight=3)

        # preview grid container
        self.preview_container = ttk.Frame(left)
        self.preview_container.pack(fill="both", expand=True, padx=6, pady=6)

        self._build_preview_grid()

        # ---- right panel ----
        out = ttk.LabelFrame(right, text="Output / Workspace")
        out.pack(fill="x", padx=6, pady=6)

        row = ttk.Frame(out)
        row.pack(fill="x", padx=10, pady=8)
        ttk.Label(row, text="Output folder:").pack(side="left")
        ttk.Entry(row, textvariable=self.output_dir).pack(side="left", fill="x", expand=True, padx=6)
        ttk.Button(row, text="Browse", command=self.on_choose_output).pack(side="left")

        # ---- Intrinsic ----
        intr = ttk.LabelFrame(right, text="Intrinsic Calibration (Checkerboard)")
        intr.pack(fill="x", padx=6, pady=6)

        grid = ttk.Frame(intr)
        grid.pack(fill="x", padx=10, pady=10)
        self._entry(grid, "Square size (mm):", self.square_size_mm, 0)
        self._entry(grid, "Inner corners (cols):", self.inner_cols, 1)
        self._entry(grid, "Inner corners (rows):", self.inner_rows, 2)
        self._entry(grid, "Extract every N sec:", self.extract_every_sec, 3)

        self.btn_run_intrinsic = ttk.Button(
            intr,
            text="Run Intrinsic -> lens_calibration.toml",
            command=self.on_run_intrinsic
        )
        self.btn_run_intrinsic.pack(fill="x", padx=10, pady=(0, 10))

        chkrow = ttk.Frame(intr)
        chkrow.pack(fill="x", padx=10, pady=(0, 8))
        ttk.Checkbutton(
            chkrow,
            text="Show chessboard corners on preview",
            variable=self.show_corners_preview,
            command=self._render_all
        ).pack(side="left")

        # ---- Extrinsic ----
        extr = ttk.LabelFrame(right, text="Extrinsic Calibration")
        extr.pack(fill="x", padx=6, pady=6)

        row3 = ttk.Frame(extr)
        row3.pack(fill="x", padx=10, pady=8)
        ttk.Label(row3, text="lens_calibration.toml:").pack(side="left")
        ttk.Entry(row3, textvariable=self.lens_toml_path).pack(side="left", fill="x", expand=True, padx=6)
        ttk.Button(row3, text="Browse", command=self.on_choose_lens_toml).pack(side="left")

        mrow = ttk.Frame(extr)
        mrow.pack(fill="x", padx=10, pady=(0, 8))
        ttk.Label(mrow, text="Method:").pack(side="left")
        ttk.Radiobutton(
            mrow, text="Chessboard (board)", value="board",
            variable=self.extrinsic_method, command=self._update_extrinsic_ui
        ).pack(side="left", padx=6)
        ttk.Radiobutton(
            mrow, text="Object/Scene (scene)", value="scene",
            variable=self.extrinsic_method, command=self._update_extrinsic_ui
        ).pack(side="left", padx=6)

        # board-only UI
        self.board_box = ttk.Frame(extr)
        self.board_box.pack(fill="x", padx=10, pady=(0, 8))

        brow = ttk.Frame(self.board_box)
        brow.pack(fill="x")
        ttk.Label(brow, text="Board position:").pack(side="left")
        ttk.Combobox(
            brow, values=["horizontal", "vertical"],
            state="readonly", textvariable=self.board_position, width=12
        ).pack(side="left", padx=6)

        bgrid = ttk.Frame(self.board_box)
        bgrid.pack(fill="x", pady=(6, 0))
        self._entry(bgrid, "Square size (mm):", self.square_size_mm, 0)
        self._entry(bgrid, "Inner corners (cols):", self.inner_cols, 1)
        self._entry(bgrid, "Inner corners (rows):", self.inner_rows, 2)

        # scene-only UI
        self.scene_box = ttk.LabelFrame(extr, text="Scene/Object points (manual click)")
        self.scene_box.pack(fill="x", padx=10, pady=(0, 8))

        lbl_inst1 = ttk.Label(self.scene_box, text="1) Paste object 3D points in m or mm: x,y,z per line")
        lbl_inst1.pack(anchor="w")
        self._register_wrap(lbl_inst1)
        self.txt_obj = tk.Text(self.scene_box, height=4)
        self.txt_obj.pack(fill="x", padx=6, pady=4)

        btn_row = ttk.Frame(self.scene_box)
        btn_row.pack(fill="x", padx=6, pady=(0, 6))
        ttk.Button(
            btn_row,
            text="Pick 2D points by clicking (CURRENT preview frames)",
            command=self.on_pick_scene_points
        ).pack(side="left")

        self.lbl_pick_status = ttk.Label(btn_row, text="picked: 0 pts")
        self.lbl_pick_status.pack(side="left", padx=10)

        cb_overlay = ttk.Checkbutton(
            self.scene_box,
            text="Show click/projected overlay on preview",
            variable=self.show_scene_overlay_preview,
            command=self._render_all
        )
        cb_overlay.pack(anchor="w", padx=6, pady=(0, 4))
        self._register_wrap(cb_overlay)

        lbl_tip = ttk.Label(
            self.scene_box,
            text="Tip: For N cameras, you will click the same N points in each Cam window (same order)."
        )
        lbl_tip.pack(anchor="w", padx=6, pady=(0, 6))
        self._register_wrap(lbl_tip)

        ttk.Button(extr, text="Run Extrinsic -> camera_calibration.toml", command=self.on_run_extrinsic)\
            .pack(fill="x", padx=10, pady=(0, 10))

        self._update_extrinsic_ui()

    def _entry(self, parent, label, var, row):
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", pady=3)
        e = ttk.Entry(parent, textvariable=var, width=18)
        e.grid(row=row, column=1, sticky="ew", pady=3)
        parent.grid_columnconfigure(1, weight=1)

    def _build_preview_grid(self):
        # clear old
        for child in self.preview_container.winfo_children():
            child.destroy()

        self.preview_widgets = []
        self.scale_widgets = []
        self.frame_labels = []

        n = self.num_cams.get()

        # Layout rule (as requested):
        # 2 cams  -> 1 col x 2 rows
        # 3-4     -> 2 cols x 2 rows
        # 5-6     -> 2 cols x 3 rows
        # 7-8     -> 2 cols x 4 rows
        if n <= 2:
            cols, rows = 1, 2
        elif n <= 4:
            cols, rows = 2, 2
        elif n <= 6:
            cols, rows = 3, 2
        else:
            cols, rows = 4, 2

        # grid config
        for c in range(cols):
            self.preview_container.grid_columnconfigure(c, weight=1)
        for r in range(rows):
            self.preview_container.grid_rowconfigure(r, weight=1)

        for i in range(n):
            r = i // cols
            c = i % cols
            # UI title: use Cam01..Cam08 only (do not show internal cam_id).
            title = f"Cam{i+1:02d}"
            box = ttk.LabelFrame(self.preview_container, text=title)
            box.grid(row=r, column=c, sticky="nsew", padx=6, pady=6)

            prev = ZoomPreview(box)
            prev.pack(fill="both", expand=True, padx=6, pady=6)

            ctrl = ttk.Frame(box)
            ctrl.pack(fill="x", padx=10, pady=(0, 10))

            ttk.Button(ctrl, text="Play/Pause", command=lambda idx=i: self.on_toggle_play(idx)).pack(side="left")
            scale = ttk.Scale(ctrl, from_=0, to=0, orient="horizontal",
                              command=lambda val, idx=i: self.on_seek(idx, val))
            scale.pack(side="left", fill="x", expand=True, padx=8)

            lbl = ttk.Label(ctrl, text="extracted frame 0/0")
            lbl.pack(side="right")

            self.preview_widgets.append(prev)
            self.scale_widgets.append(scale)
            self.frame_labels.append(lbl)

    def _update_extrinsic_ui(self):
        if self.extrinsic_method.get() == "board":
            self.board_box.pack(fill="x", padx=10, pady=(0, 8))
            self.scene_box.pack_forget()
        else:
            self.board_box.pack_forget()
            self.scene_box.pack(fill="x", padx=10, pady=(0, 8))

        # Ensure preview reflects current overlay settings immediately.
        try:
            self._render_all()
        except Exception:
            pass

    # ---------------- callbacks ----------------
    def on_num_cams_changed(self, _evt=None):
        try:
            n = int(self.cmb_num_cams.get())
        except Exception:
            return

        if n == self.num_cams.get():
            return

        self.num_cams.set(n)
        self._log(f"Number of Cameras changed -> {n}. Resetting state...")

        # reset state
        self._init_camera_state(n)
        # IMPORTANT: Pose2Sim's calibration utilities scan the workspace folders.
        # When camera count changes, delete the whole workspace and recreate lazily.
        try:
            self.session_manager.reset_workspace()
        except Exception:
            pass

        # rebuild UI grid
        self._build_preview_grid()

        # reset sliders
        for scale in self.scale_widgets:
            try:
                scale.configure(to=0)
                scale.set(0)
            except Exception:
                pass
        for lbl in self.frame_labels:
            try:
                lbl.configure(text="extracted frame 0/0")
            except Exception:
                pass

        self._popup_info("Number of Cameras", "Camera count changed.\nPlease reload videos.")

    def on_choose_output(self):
        d = filedialog.askdirectory(title="Choose output folder")
        if not d:
            return
        self.output_dir.set(d)
        self._log(f"Output folder selected: {d}")

    def on_load_videos(self):
        n = self.num_cams.get()
        self._log(f"Selecting {n} video files...")
        paths = filedialog.askopenfilenames(
            title=f"Select exactly {n} video files (Cam01..Cam{n:02d})",
            filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv"), ("All files", "*.*")]
        )
        if not paths or len(paths) != n:
            return

        self.video_paths = list(paths)

        # When the user loads videos again, wipe the current session workspace
        try:
            # stop playback
            for i in range(n):
                self.playing[i] = False

            # clear scene picks / last extrinsic overlays
            self.scene_obj_points_mm = None
            self.scene_img_points_px = [None] * n
            self._last_extr_R = None
            self._last_extr_T = None
            self._last_extr_K = None
            self._last_extr_D = None

            # wipe workspace folder
            self.session_manager.reset_workspace()
        except Exception:
            pass

        # workspace
        ws = self.session_manager.ensure_workspace()

        try:
            self._build_preview_cache(ws)
            for i in range(n):
                self.pos[i] = 0
                self.scale_widgets[i].configure(to=max(len(self.preview_files[i]) - 1, 0))
                self.scale_widgets[i].set(0)
            self._render_all()
            self._log("Videos loaded:\n" + "\n".join([f"- {cam_id(i)}: {Path(p).name}" for i, p in enumerate(self.video_paths)]))
        except Exception as e:
            self._popup_error(
                "Video decode / preview error",
                f"{e}\n\n코덱 문제(codec_id=61 등)면:\n"
                f"- 비디오를 H.264(mp4)로 변환하거나\n"
                f"- conda-forge ffmpeg/opencv 설치를 확인하세요."
            )
    
    def on_clear_videos(self):
        """Clear loaded videos and reset preview."""
        n = self.num_cams.get()
        self._log("Clearing videos...")
        
        try:
            # Stop playback
            for i in range(n):
                if i < len(self.playing):
                    self.playing[i] = False
            
            # Clear preview widgets first (before resetting state)
            for widget in self.preview_widgets:
                # ZoomPreview is a Canvas, so we can delete all items
                widget.delete("all")
                # Reset internal state
                widget._rgb = None
                widget._img = None
                widget._photo = None
                widget._img_id = None
            
            # Reset camera state
            self._init_camera_state(n)
            
            # Clear scene picks / last extrinsic overlays
            self.scene_obj_points_mm = None
            self._last_extr_R = None
            self._last_extr_T = None
            self._last_extr_K = None
            self._last_extr_D = None
            
            # Reset scale widgets and frame labels
            for i in range(n):
                if i < len(self.scale_widgets):
                    self.scale_widgets[i].configure(to=0)
                    self.scale_widgets[i].set(0)
                if i < len(self.frame_labels):
                    self.frame_labels[i].configure(text="Frame: 0/0")
            
            # Wipe workspace folder
            self.session_manager.reset_workspace()
            
            # Force canvas update to show black background
            for widget in self.preview_widgets:
                widget.update_idletasks()
            
            self._log("Videos cleared.")
        except Exception as e:
            self._log(f"Error clearing videos: {e}")
            self._popup_error("Clear Error", f"Failed to clear videos: {e}")

    def _build_preview_cache(self, ws: Path):
        n = self.num_cams.get()
        every = int(self.extract_every_sec.get())

        for i in range(n):
            self.playing[i] = False
            prev_dir = ws / "_preview_cache" / cam_id(i)
            ensure_dir(prev_dir)
            for p in prev_dir.glob("*.png"):
                p.unlink(missing_ok=True)

            self.preview_files[i] = extract_frames_every_n_sec(self.video_paths[i], prev_dir, every)

            # reset overlays
            self.corner_overlay_files[i] = []

    # ---- playback/seek/render ----
    def on_toggle_play(self, idx: int):
        if not self.preview_files[idx]:
            return
        self.playing[idx] = not self.playing[idx]
        if self.playing[idx]:
            self._play_loop(idx)

    def _play_loop(self, idx: int):
        if not self.playing[idx] or not self.preview_files[idx]:
            return
        if self.pos[idx] >= len(self.preview_files[idx]) - 1:
            self.playing[idx] = False
            return
        self.pos[idx] += 1
        try:
            self.scale_widgets[idx].set(self.pos[idx])
        except Exception:
            pass
        self._render_cam(idx)
        self.after(60, lambda: self._play_loop(idx))

    def on_seek(self, idx: int, val: str):
        if not self.preview_files[idx]:
            return
        try:
            self.pos[idx] = int(float(val))
        except Exception:
            return
        self.pos[idx] = max(0, min(self.pos[idx], len(self.preview_files[idx]) - 1))
        self._render_cam(idx)

    def _choose_preview_image(self, idx: int) -> Path:
        i = idx
        j = max(0, min(self.pos[i], len(self.preview_files[i]) - 1))
        if self.show_corners_preview.get() and self.corner_overlay_files[i] and j < len(self.corner_overlay_files[i]):
            return self.corner_overlay_files[i][j]
        return self.preview_files[i][j]

    def _render_cam(self, idx: int):
        if not self.preview_files[idx]:
            return
        j = max(0, min(self.pos[idx], len(self.preview_files[idx]) - 1))
        path = self._choose_preview_image(idx)
        bgr = cv2.imread(str(path))
        if bgr is None:
            return
        # Optional overlay: clicked (red) and projected (green) points (scene extrinsic)
        if self.show_scene_overlay_preview.get() and self.extrinsic_method.get() == "scene":
            try:
                if self.scene_obj_points_mm is not None and self.scene_img_points_px and idx < len(self.scene_img_points_px):
                    clicked = self.scene_img_points_px[idx]
                    if clicked is not None:
                        clicked = np.asarray(clicked, dtype=np.float32).reshape(-1, 2)
                        proj = None
                        if self._last_extr_R is not None and self._last_extr_T is not None:
                            if idx < len(self._last_extr_R) and idx < len(self._last_extr_T):
                                obj_m = (np.asarray(self.scene_obj_points_mm, dtype=np.float64) / 1000.0).reshape(-1, 3)
                                # Rodrigues vector
                                R = np.asarray(self._last_extr_R[idx], dtype=np.float64)
                                if R.shape == (3, 3):
                                    rvec, _ = cv2.Rodrigues(R)
                                else:
                                    rvec = R.reshape(3, 1)
                                tvec = np.asarray(self._last_extr_T[idx], dtype=np.float64).reshape(3, 1)
                                K = np.asarray(self._last_extr_K[idx], dtype=np.float64) if self._last_extr_K is not None else None
                                D = np.asarray(self._last_extr_D[idx], dtype=np.float64) if self._last_extr_D is not None else None
                                if K is not None:
                                    proj, _ = cv2.projectPoints(obj_m, rvec, tvec, K, D)
                                    proj = proj.reshape(-1, 2)
                        if proj is not None and proj.shape[0] == clicked.shape[0]:
                            bgr = draw_scene_overlay(bgr, clicked, proj)
                        else:
                            # draw clicked only
                            for (x, y) in clicked:
                                cv2.circle(bgr, (int(round(x)), int(round(y))), 6, (0, 0, 255), 2)
            except Exception:
                pass

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        self._last_rgb[idx] = rgb
        self.preview_widgets[idx].set_rgb(rgb)
        self.frame_labels[idx].configure(text=f"extracted frame {j+1}/{len(self.preview_files[idx])}")

    def _render_all(self):
        for i in range(self.num_cams.get()):
            self._render_cam(i)

    # ---------------- Intrinsic ----------------
    def on_run_intrinsic(self):
        self._log("Run Intrinsic...")
        self.btn_run_intrinsic.config(state="disabled")

        if any(not p for p in self.video_paths):
            self._popup_error("Intrinsic Calibration Error", "Load videos first.")
            self.btn_run_intrinsic.config(state="normal")
            return

        out_base = Path(self.output_dir.get()).expanduser().resolve()
        ensure_dir(out_base)
        result_dir = out_base / f"Intrinsic_calibration_{time.strftime('%Y%m%d_%H%M%S')}"
        ensure_dir(result_dir)

        ws = self.session_manager.ensure_workspace()

        params = dict(
            ws=ws,
            result_dir=result_dir,
            videos=list(self.video_paths),
            every=int(self.extract_every_sec.get()),
            cols=int(self.inner_cols.get()),
            rows=int(self.inner_rows.get()),
            square=float(self.square_size_mm.get()),
        )
        threading.Thread(target=self._intrinsic_worker, args=(params,), daemon=True).start()

    def _intrinsic_worker(self, params: dict):
        """Worker thread for intrinsic calibration using separated processor."""
        process_intrinsic_calibration(
            workspace=params["ws"],
            result_dir=params["result_dir"],
            videos=params["videos"],
            num_cams=self.num_cams.get(),
            extract_every_sec=int(params["every"]),
            cols=int(params["cols"]),
            rows=int(params["rows"]),
            square_size_mm=float(params["square"]),
            log_callback=self._log,
            popup_info_callback=self._popup_info,
            popup_error_callback=self._popup_error,
            after_callback=self.after,
            render_all_callback=self._render_all,
            lens_toml_path_setter=lambda p: self.lens_toml_path.set(p),
            btn_run_intrinsic_setter=lambda s: self.btn_run_intrinsic.config(state=s),
            build_corner_overlays_callback=self._build_corner_overlays,
        )

    def _build_corner_overlays(self, pattern: tuple[int, int]):
        ws = self.session_manager.workspace_dir
        if ws is None:
            return
        n = self.num_cams.get()
        for i in range(n):
            if not self.preview_files[i]:
                continue
            out_dir = ws / "_overlay" / "corners" / cam_id(i)
            ensure_dir(out_dir)
            self.corner_overlay_files[i] = annotate_chessboard_files(self.preview_files[i], out_dir, pattern)

    # ---------------- Scene pick ----------------
    def on_pick_scene_points(self):
        try:
            n_cams = self.num_cams.get()
            if any(not files for files in self.preview_files):
                raise RuntimeError("Preview frames are empty. Load videos first.")

            obj_mm = parse_obj_points_mm(self.txt_obj.get("1.0", "end"))
            if obj_mm.shape[0] < 6:
                raise RuntimeError("Need at least 6 object points.")
            self.scene_obj_points_mm = obj_mm
            n_pts = obj_mm.shape[0]

            picked: List[np.ndarray] = []
            for i in range(n_cams):
                j = max(0, min(self.pos[i], len(self.preview_files[i]) - 1))
                img_path = self.preview_files[i][j]
                bgr = cv2.imread(str(img_path))
                if bgr is None:
                    raise RuntimeError(f"Failed to read preview image: {img_path}")
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                pts = PointPicker(f"Pick points - Cam{i+1:02d}", rgb, n_pts).pick()
                if pts is None:
                    return
                picked.append(pts)

            self.scene_img_points_px = picked
            self.lbl_pick_status.configure(text=f"picked: {n_pts} pts (all cams)")
            self._log(f"Picked {n_pts} scene points for {n_cams} cameras.")
            self._render_all()
            self._popup_info("Point Picking", f"Picked {n_pts} points for {n_cams} cameras.\nNow run Extrinsic.")

        except Exception as e:
            self._popup_error("Point Picking Error", f"{e}\n\n{traceback.format_exc()}")

    # ---------------- Extrinsic ----------------
    def on_choose_lens_toml(self):
        p = filedialog.askopenfilename(
            title="Select lens_calibration.toml",
            filetypes=[("TOML files", "*.toml"), ("All files", "*.*")]
        )
        if p:
            self.lens_toml_path.set(p)
            self._log(f"lens_calibration.toml selected: {p}")

    def on_run_extrinsic(self):
        self._log("Run Extrinsic...")
        if any(not p for p in self.video_paths):
            self._popup_error("Extrinsic Calibration Error", "Load videos first.")
            return
        if not self.lens_toml_path.get():
            self._popup_error("Extrinsic Calibration Error", "Select lens_calibration.toml first.")
            return

        out_base = Path(self.output_dir.get()).expanduser().resolve()
        ensure_dir(out_base)
        result_dir = out_base / f"Extrinsic_calibration_{time.strftime('%Y%m%d_%H%M%S')}"
        ensure_dir(result_dir)

        ws = self.session_manager.ensure_workspace()

        params = dict(
            ws=ws,
            result_dir=result_dir,
            videos=list(self.video_paths),
            lens=self.lens_toml_path.get(),
            method=self.extrinsic_method.get(),
            board_position=self.board_position.get(),
            cols=int(self.inner_cols.get()),
            rows=int(self.inner_rows.get()),
            square=float(self.square_size_mm.get()),
        )
        threading.Thread(target=self._extrinsic_worker, args=(params,), daemon=True).start()

    def _extrinsic_worker(self, params: dict):
        """Worker thread for extrinsic calibration using separated processor."""
        out_cam, lines, last_extr = process_extrinsic_calibration(
            workspace=params["ws"],
            result_dir=params["result_dir"],
            videos=params["videos"],
            lens_toml_path=params["lens"],
            method=params["method"],
            num_cams=self.num_cams.get(),
            board_position=params["board_position"],
            cols=int(params["cols"]),
            rows=int(params["rows"]),
            square_size_mm=float(params["square"]),
            scene_obj_points_mm=self.scene_obj_points_mm,
            scene_img_points_px=self.scene_img_points_px,
            preview_files_getter=lambda: self.preview_files,
            pos_getter=lambda: self.pos,
            log_callback=self._log,
            popup_info_callback=self._popup_info,
            popup_error_callback=self._popup_error,
            after_callback=self.after,
            render_all_callback=self._render_all,
            last_extr_setters={
                'R': lambda v: setattr(self, '_last_extr_R', v),
                'T': lambda v: setattr(self, '_last_extr_T', v),
                'K': lambda v: setattr(self, '_last_extr_K', v),
                'D': lambda v: setattr(self, '_last_extr_D', v),
            },
        )


    # ---------------- Wrapping helpers ----------------
    def _register_wrap(self, w: tk.Widget):
        """Register a widget (Label/Checkbutton) that should wrap within the right panel."""
        self._wrap_widgets.append(w)

    def _update_wraplength(self, width_px: int):
        """Update wraplength for registered widgets."""
        w = max(120, int(width_px) - 30)
        for widget in getattr(self, "_wrap_widgets", []):
            try:
                widget.configure(wraplength=w)
            except Exception:
                pass
