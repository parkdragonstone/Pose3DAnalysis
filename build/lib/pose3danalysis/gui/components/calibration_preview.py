# -*- coding: utf-8 -*-
"""Preview-related components for calibration tab.

Extracted from gui.tabs.calibration_tab to improve code organization.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, List, Callable

import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk

from pose3danalysis.core.frame_extract import extract_frames_every_n_sec
from pose3danalysis.core.utils import ensure_dir
from pose3danalysis.core.zoom_preview import ZoomPreview

from pose3danalysis.gui.utils.helpers import cam_id
from pose3danalysis.gui.utils.calibration_utils import draw_scene_overlay


class CalibrationPreviewManager:
    """Manages preview grid, playback, and rendering for calibration tab."""

    def __init__(
        self,
        parent: ttk.Frame,
        num_cams_getter: Callable[[], int],
        preview_files_getter: Callable[[], List[List[Path]]],
        corner_overlay_files_getter: Callable[[], List[List[Path]]],
        pos_getter: Callable[[], List[int]],
        playing_getter: Callable[[], List[bool]],
        show_corners_preview_getter: Callable[[], bool],
        show_scene_overlay_preview_getter: Callable[[], bool],
        extrinsic_method_getter: Callable[[], str],
        scene_obj_points_mm_getter: Callable[[], Optional[np.ndarray]],
        scene_img_points_px_getter: Callable[[], List[Optional[np.ndarray]]],
        last_extr_getters: dict,
        after_callback: Callable[[int, Callable], None],
    ):
        """
        Initialize preview manager.

        Args:
            parent: Parent frame for preview container
            num_cams_getter: Callable that returns current number of cameras
            preview_files_getter: Callable that returns list of preview file lists
            corner_overlay_files_getter: Callable that returns list of corner overlay file lists
            pos_getter: Callable that returns list of current frame positions
            playing_getter: Callable that returns list of playing states
            show_corners_preview_getter: Callable that returns show corners preview flag
            show_scene_overlay_preview_getter: Callable that returns show scene overlay preview flag
            extrinsic_method_getter: Callable that returns extrinsic method ("board" or "scene")
            scene_obj_points_mm_getter: Callable that returns scene object points in mm
            scene_img_points_px_getter: Callable that returns scene image points in px
            last_extr_getters: Dict with keys 'R', 'T', 'K', 'D' for last extrinsic solution getters
            after_callback: Tkinter after() callback: after_callback(ms, callback)
        """
        self.parent = parent
        self.num_cams_getter = num_cams_getter
        self.preview_files_getter = preview_files_getter
        self.corner_overlay_files_getter = corner_overlay_files_getter
        self.pos_getter = pos_getter
        self.playing_getter = playing_getter
        self.show_corners_preview_getter = show_corners_preview_getter
        self.show_scene_overlay_preview_getter = show_scene_overlay_preview_getter
        self.extrinsic_method_getter = extrinsic_method_getter
        self.scene_obj_points_mm_getter = scene_obj_points_mm_getter
        self.scene_img_points_px_getter = scene_img_points_px_getter
        self.last_extr_getters = last_extr_getters
        self.after_callback = after_callback

        # Preview container
        self.preview_container = ttk.Frame(parent)
        self.preview_container.pack(fill="both", expand=True, padx=6, pady=6)

        # UI handles
        self.preview_widgets: List[ZoomPreview] = []
        self.scale_widgets: List[ttk.Scale] = []
        self.frame_labels: List[ttk.Label] = []

        # Last rendered RGB images
        self._last_rgb: List[Optional[np.ndarray]] = [None] * self.num_cams_getter()

    def build_preview_grid(self):
        """Build preview grid with camera previews."""
        # clear old
        for child in self.preview_container.winfo_children():
            child.destroy()

        self.preview_widgets = []
        self.scale_widgets = []
        self.frame_labels = []

        n = self.num_cams_getter()

        # Layout rule:
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
            # UI title: use Cam01..Cam08 only
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

        # Update last_rgb list size
        self._last_rgb = [None] * n

    def build_preview_cache(self, ws: Path, video_paths: List[str], extract_every_sec: int, 
                            preview_files_setter: Callable[[int, List[Path]], None],
                            corner_overlay_files_setter: Callable[[int, List[Path]], None],
                            playing_setter: Callable[[int, bool], None]):
        """Build preview cache by extracting frames from videos."""
        n = self.num_cams_getter()

        for i in range(n):
            playing_setter(i, False)
            prev_dir = ws / "_preview_cache" / cam_id(i)
            ensure_dir(prev_dir)
            for p in prev_dir.glob("*.png"):
                p.unlink(missing_ok=True)

            files = extract_frames_every_n_sec(video_paths[i], prev_dir, extract_every_sec)
            preview_files_setter(i, files)
            corner_overlay_files_setter(i, [])

    def on_toggle_play(self, idx: int, pos_setter: Callable[[int, int], None]):
        """Toggle play/pause for camera at index."""
        preview_files = self.preview_files_getter()
        if not preview_files[idx]:
            return
        playing = self.playing_getter()
        playing[idx] = not playing[idx]
        if playing[idx]:
            self._play_loop(idx, pos_setter)

    def _play_loop(self, idx: int, pos_setter: Callable[[int, int], None]):
        """Playback loop for camera at index."""
        preview_files = self.preview_files_getter()
        playing = self.playing_getter()
        pos = self.pos_getter()
        
        if not playing[idx] or not preview_files[idx]:
            return
        if pos[idx] >= len(preview_files[idx]) - 1:
            playing[idx] = False
            return
        pos_setter(idx, pos[idx] + 1)
        try:
            self.scale_widgets[idx].set(pos[idx])
        except Exception:
            pass
        self.render_cam(idx)
        self.after_callback(60, lambda: self._play_loop(idx, pos_setter))

    def on_seek(self, idx: int, val: str, pos_setter: Callable[[int, int], None]):
        """Seek to frame for camera at index."""
        preview_files = self.preview_files_getter()
        if not preview_files[idx]:
            return
        try:
            pos_val = int(float(val))
        except Exception:
            return
        pos_val = max(0, min(pos_val, len(preview_files[idx]) - 1))
        pos_setter(idx, pos_val)
        self.render_cam(idx)

    def _choose_preview_image(self, idx: int) -> Path:
        """Choose which preview image to display (with or without corner overlay)."""
        preview_files = self.preview_files_getter()
        corner_overlay_files = self.corner_overlay_files_getter()
        pos = self.pos_getter()
        
        j = max(0, min(pos[idx], len(preview_files[idx]) - 1))
        if (self.show_corners_preview_getter() and 
            corner_overlay_files[idx] and 
            j < len(corner_overlay_files[idx])):
            return corner_overlay_files[idx][j]
        return preview_files[idx][j]

    def render_cam(self, idx: int):
        """Render camera preview at index with optional overlays."""
        preview_files = self.preview_files_getter()
        if not preview_files[idx]:
            return
        
        pos = self.pos_getter()
        j = max(0, min(pos[idx], len(preview_files[idx]) - 1))
        path = self._choose_preview_image(idx)
        bgr = cv2.imread(str(path))
        if bgr is None:
            return

        # Optional overlay: clicked (red) and projected (green) points (scene extrinsic)
        if (self.show_scene_overlay_preview_getter() and 
            self.extrinsic_method_getter() == "scene"):
            try:
                scene_obj_points_mm = self.scene_obj_points_mm_getter()
                scene_img_points_px = self.scene_img_points_px_getter()
                
                if (scene_obj_points_mm is not None and 
                    scene_img_points_px and 
                    idx < len(scene_img_points_px)):
                    clicked = scene_img_points_px[idx]
                    if clicked is not None:
                        clicked = np.asarray(clicked, dtype=np.float32).reshape(-1, 2)
                        proj = None
                        
                        last_R = self.last_extr_getters.get('R')
                        last_T = self.last_extr_getters.get('T')
                        last_K = self.last_extr_getters.get('K')
                        last_D = self.last_extr_getters.get('D')
                        
                        if last_R is not None and last_T is not None:
                            R_list = last_R() if callable(last_R) else last_R
                            T_list = last_T() if callable(last_T) else last_T
                            
                            if idx < len(R_list) and idx < len(T_list):
                                obj_m = (np.asarray(scene_obj_points_mm, dtype=np.float64) / 1000.0).reshape(-1, 3)
                                # Rodrigues vector
                                R = np.asarray(R_list[idx], dtype=np.float64)
                                if R.shape == (3, 3):
                                    rvec, _ = cv2.Rodrigues(R)
                                else:
                                    rvec = R.reshape(3, 1)
                                tvec = np.asarray(T_list[idx], dtype=np.float64).reshape(3, 1)
                                
                                K_list = last_K() if callable(last_K) else last_K if last_K is not None else None
                                D_list = last_D() if callable(last_D) else last_D if last_D is not None else None
                                
                                K = np.asarray(K_list[idx], dtype=np.float64) if K_list is not None and idx < len(K_list) else None
                                D = np.asarray(D_list[idx], dtype=np.float64) if D_list is not None and idx < len(D_list) else None
                                
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
        self.frame_labels[idx].configure(text=f"extracted frame {j+1}/{len(preview_files[idx])}")

    def render_all(self):
        """Render all camera previews."""
        for i in range(self.num_cams_getter()):
            self.render_cam(i)

    def update_scale_range(self, idx: int, max_val: int):
        """Update scale widget range for camera at index."""
        try:
            self.scale_widgets[idx].configure(to=max(max_val - 1, 0))
        except Exception:
            pass

    def update_scale_value(self, idx: int, value: int):
        """Update scale widget value for camera at index."""
        try:
            self.scale_widgets[idx].set(value)
        except Exception:
            pass

    def reset_scales(self):
        """Reset all scale widgets to 0."""
        for scale in self.scale_widgets:
            try:
                scale.configure(to=0)
                scale.set(0)
            except Exception:
                pass

    def reset_labels(self):
        """Reset all frame labels."""
        for lbl in self.frame_labels:
            try:
                lbl.configure(text="extracted frame 0/0")
            except Exception:
                pass

