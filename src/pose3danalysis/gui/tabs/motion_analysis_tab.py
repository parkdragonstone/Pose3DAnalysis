# gui/motionanalysis_tab.py
# 2025 Pose3DAnalysis

from __future__ import annotations

import tkinter as tk
from tkinter import ttk, filedialog
from tkinter.scrolledtext import ScrolledText
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, List, Tuple

import copy
import ast
import json
import threading
import re
from collections import OrderedDict

import logging
import time
import cv2
import numpy as np

from pose3danalysis.core.zoom_preview import ZoomPreview

# Matplotlib (Tk)
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from mpl_toolkits.mplot3d import proj3d

from pose3danalysis.runner.pose2sim_runner import run_pose2sim as runner_run_pose2sim

# Import separated modules
from pose3danalysis.gui.components.scrollable_frame import ScrollableFrame
from pose3danalysis.gui.components.side_settings_panel import SideSettingsPanel
from pose3danalysis.gui.utils.motion_settings import (
    MotionSettings,
    DEFAULT_POSE2SIM_CONFIG,
    DEFAULT_SKELETON_EDGES,
    build_pose2sim_config,
    parse_csv_list,
)
from pose3danalysis.gui.utils.pose_utils import (
    infer_project_dir_from_video_folder,
    resolve_pose_json_dir,
    find_openpose_json_file,
    extract_frame_index_from_name,
    list_pose_json_sorted,
    build_pose_json_index_map,
    skeleton_pairs_for_kpts,
    parse_people_openpose,
    kp_side_halpe26,
    draw_openpose_overlay,
)
from pose3danalysis.gui.utils.video_utils import CamStream, bgr_to_rgb
from pose3danalysis.gui.utils.logging_utils import UILogHandler
from pose3danalysis.gui.components.multicam_preview import MultiCamPreview
from pose3danalysis.gui.components.viewer3d_panel import Viewer3DPanel

DEFAULT_DEEPSORT_PARAMS = (
    """{'max_age':30, 'n_init':3, 'nms_max_overlap':0.8, """
    """'max_cosine_distance':0.3, 'nn_budget':200, 'max_iou_distance':0.8}"""
)


class MotionAnalysisTab(ttk.Frame):
    """Motion Analysis > Batch Processing tab UI (synced video + 3D viewer)"""
    def __init__(self, parent, log: Callable[[str], None], shared_log_text=None, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)
        self._external_log = log
        self.shared_log_text = shared_log_text  # Shared log widget from main app
        self._pending_logs = []  # buffer until log_text exists

        # shared playback state
        self.play_frame = tk.IntVar(value=0)
        self.play_speed = tk.DoubleVar(value=1.0)  # Max speed: 4.0x
        self.z_up = tk.BooleanVar(value=False)
        
        # Speed sync flag to prevent circular updates
        self._syncing_speed = False

        # settings state
        self.settings = MotionSettings()

        # layout
        self.paned = ttk.PanedWindow(self, orient="horizontal")
        self.paned.pack(fill="both", expand=True, padx=10, pady=10)

        self.left = ttk.Frame(self.paned)
        self.right = ttk.Frame(self.paned)

        # Initial weights: favor left panel, but will be adjusted when settings panel is toggled
        self.paned.add(self.left, weight=3)
        self.paned.add(self.right, weight=1)
        
        # Center panel removed (moved to left panel)
        self.center = None

        self._build_left()
        self._build_right()
        
        # Initialize log text (shared from main app)
        if self.shared_log_text is not None:
            self.log_text = self.shared_log_text
        else:
            self.log_text = None

        # bridge Python logging -> Motion Analysis Logs
        self._install_logging_bridge()

        # sync: when frame changes, update viewer and playback UI
        self._frame_trace = self.play_frame.trace_add("write", lambda *_: self._on_frame_changed())
        self._speed_trace = self.play_speed.trace_add("write", lambda *_: self._on_speed_changed())
        
        # Sync speed spinbox when play_speed changes (but avoid circular updates)
        self.play_speed.trace_add("write", lambda *_: self._sync_speed_to_spinbox())
        
        # Initialize speed spinbox
        self._sync_speed_to_spinbox()

    # -------- left --------
    def _build_left(self):
        # Top row: Upload Video Folder, Clear, and Run buttons (aligned with Settings Apply button)
        top_buttons = ttk.Frame(self.left)
        top_buttons.pack(fill="x", padx=6, pady=(0, 6))
        
        self.btn_upload = ttk.Button(top_buttons, text="Upload Video Folder…", command=self._on_upload_video_folder)
        self.btn_upload.pack(side="left")
        self.btn_clear = ttk.Button(top_buttons, text="Clear", command=self._on_clear_all)
        self.btn_clear.pack(side="left", padx=6)
        ttk.Button(top_buttons, text="Run", command=self._run_pose2sim).pack(side="left", padx=6)
        
        # Playback row: Overlay, Speed, Play, Playback slider (all in one row)
        playback_row = ttk.LabelFrame(self.left, text="Playback")
        playback_row.pack(fill="x", padx=6, pady=(0, 6))
        
        # Single row with all controls
        controls_row = ttk.Frame(playback_row)
        controls_row.pack(fill="x", padx=6, pady=6)
        
        # Overlay checkbox
        self.overlay_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(controls_row, text="Overlay", variable=self.overlay_var, command=self._on_overlay_toggled).pack(side="left", padx=(0, 10))
        
        # Speed control using Spinbox (best UX for numeric input)
        speed_frame = ttk.Frame(controls_row)
        speed_frame.pack(side="left", padx=(0, 10))
        ttk.Label(speed_frame, text="Speed").pack(side="left", padx=(0, 4))
        
        # Use Spinbox for better numeric input control
        self.var_speed = tk.DoubleVar(value=1.0)
        self.speed_spinbox = ttk.Spinbox(
            speed_frame,
            from_=0.0,
            to=4.0,
            increment=0.1,
            textvariable=self.var_speed,
            width=6,
            format="%.2f",
            command=self._on_speed_spinbox_change
        )
        self.speed_spinbox.pack(side="left")
        # Bind events for direct input and mouse wheel
        self.speed_spinbox.bind("<Return>", self._on_speed_spinbox_return)
        self.speed_spinbox.bind("<FocusOut>", self._on_speed_spinbox_focusout)
        self.speed_spinbox.bind("<MouseWheel>", self._on_speed_spinbox_wheel)
        self.speed_spinbox.bind("<Button-4>", lambda e: self._adjust_speed_spinbox(0.1))
        self.speed_spinbox.bind("<Button-5>", lambda e: self._adjust_speed_spinbox(-0.1))
        # Sync with play_speed variable (with circular update prevention)
        self.var_speed.trace_add("write", lambda *_: self._sync_speed_from_spinbox())
        
        # "x" label next to speed spinbox
        ttk.Label(speed_frame, text="x").pack(side="left", padx=(4, 0))
        
        # Play button
        self.btn_play = ttk.Button(controls_row, text="Play", command=self._toggle_play)
        self.btn_play.pack(side="left", padx=(0, 10))
        
        # Playback slider (playbar)
        ttk.Label(controls_row, text="Frame").pack(side="left", padx=(0, 6))
        self.seek_scale = ttk.Scale(
            controls_row,
            from_=0,
            to=0,
            orient="horizontal",
            command=self._on_seek_scale,
        )
        self.seek_scale.pack(side="left", fill="x", expand=True, padx=6)
        self.seek_scale.bind("<Enter>", lambda e: self.seek_scale.focus_set())
        self.seek_scale.bind("<MouseWheel>", self._on_seek_wheel)
        self.seek_scale.bind("<Button-4>", lambda e: self._step_frame(+1))
        self.seek_scale.bind("<Button-5>", lambda e: self._step_frame(-1))
        self.lbl_frame = ttk.Label(controls_row, text="0/0", width=10, anchor="e")
        self.lbl_frame.pack(side="right")
        
        # Bottom: Inputs / Preview and 3D Viewer side by side
        bottom_paned = ttk.PanedWindow(self.left, orient="horizontal")
        bottom_paned.pack(fill="both", expand=True, padx=6, pady=(0, 6))
        
        # Left: Inputs / Preview
        preview_frame = ttk.LabelFrame(bottom_paned, text="Inputs / Preview")
        bottom_paned.add(preview_frame, weight=1)
        
        self.preview = MultiCamPreview(
            preview_frame,
            log=self._log,
            frame_var=self.play_frame,
            speed_var=self.play_speed,
            on_loaded=self._on_video_loaded,
            on_clear=self._on_clear_all,
            show_controls=False,  # Hide internal controls since we moved them out
        )
        self.preview.pack(fill="both", expand=True, padx=6, pady=6)
        
        # Right: 3D Viewer (50:50 ratio with Preview)
        viewer_frame = ttk.LabelFrame(bottom_paned, text="3D Viewer")
        bottom_paned.add(viewer_frame, weight=1)
        
        # Store reference for later use
        self.bottom_paned = bottom_paned
        
        # Set initial sash position to 50:50
        def set_50_50():
            try:
                # Force update to get actual width
                bottom_paned.update_idletasks()
                paned_w = bottom_paned.winfo_width()
                if paned_w > 0:
                    # Set sash to middle position (50:50)
                    bottom_paned.sashpos(0, paned_w // 2)
                    return True
                return False
            except Exception:
                return False
        
        # Bind to configure event to set 50:50 when paned window is first configured
        def on_paned_configure(event):
            if event.widget == bottom_paned:
                paned_w = bottom_paned.winfo_width()
                if paned_w > 0:
                    # Only set if sash hasn't been moved by user (check if it's at default position)
                    try:
                        current_pos = bottom_paned.sashpos(0)
                        # If sash is at 0 or very close to edges, set to 50:50
                        if current_pos is None or current_pos < 10 or current_pos > paned_w - 10:
                            bottom_paned.sashpos(0, paned_w // 2)
                    except Exception:
                        bottom_paned.sashpos(0, paned_w // 2)
        
        bottom_paned.bind("<Configure>", on_paned_configure)
        
        # Set 50:50 after layout is complete (multiple attempts for reliability)
        self.after(10, set_50_50)
        self.after(100, set_50_50)
        self.after(300, set_50_50)
        self.after(500, set_50_50)
        
        self.viewer = Viewer3DPanel(viewer_frame, log=self._log, z_up_var=self.z_up, on_loaded=self._on_3d_loaded)
        self.viewer.pack(fill="both", expand=True, padx=0, pady=0)
    
    def _on_upload_video_folder(self):
        """Handle Upload Video Folder button click."""
        if hasattr(self, 'preview') and hasattr(self.preview, '_upload_video_folder'):
            self.preview._upload_video_folder()
    
    def _on_overlay_toggled(self):
        """Handle Overlay checkbox toggle."""
        if hasattr(self, 'preview') and hasattr(self.preview, '_overlay_enabled'):
            self.preview._overlay_enabled.set(self.overlay_var.get())
            self.preview._on_overlay_toggled()
    
    def _adjust_speed(self, delta: float):
        """Adjust speed by delta (legacy method for compatibility)."""
        current = float(self.play_speed.get())
        new_speed = max(0.0, min(4.0, current + delta))
        self.play_speed.set(new_speed)
        self._sync_speed_to_spinbox()
    
    def _adjust_speed_spinbox(self, delta: float):
        """Adjust speed in spinbox by delta."""
        try:
            current = float(self.var_speed.get())
            new_speed = max(0.0, min(4.0, current + delta))
            self.var_speed.set(new_speed)
        except (ValueError, TypeError):
            pass
    
    def _on_speed_spinbox_change(self):
        """Handle spinbox value change (from buttons)."""
        self._sync_speed_from_spinbox()
    
    def _on_speed_spinbox_return(self, event):
        """Handle Enter key in speed spinbox."""
        self._on_speed_spinbox_focusout(event)
        return "break"
    
    def _on_speed_spinbox_focusout(self, event):
        """Handle focus out from speed spinbox - validate and update."""
        try:
            value = float(self.var_speed.get())
            new_speed = max(0.0, min(4.0, value))
            self.var_speed.set(new_speed)
            self._sync_speed_from_spinbox()
        except (ValueError, TypeError):
            # Invalid input, restore current speed
            self._sync_speed_to_spinbox()
    
    def _on_speed_spinbox_wheel(self, event):
        """Handle mouse wheel on speed spinbox."""
        delta = 0.1 if event.delta > 0 else -0.1
        self._adjust_speed_spinbox(delta)
        return "break"
    
    def _sync_speed_from_spinbox(self):
        """Sync play_speed from spinbox value (avoid circular updates)."""
        if self._syncing_speed:
            return
        try:
            self._syncing_speed = True
            speed = float(self.var_speed.get())
            speed = max(0.0, min(4.0, speed))
            self.play_speed.set(speed)
        except (ValueError, TypeError):
            pass
        finally:
            self._syncing_speed = False
    
    def _sync_speed_to_spinbox(self):
        """Sync spinbox value from play_speed (avoid circular updates)."""
        if self._syncing_speed:
            return
        try:
            self._syncing_speed = True
            speed = float(self.play_speed.get())
            self.var_speed.set(speed)
        except (ValueError, TypeError):
            pass
        finally:
            self._syncing_speed = False
    
    def _toggle_play(self):
        """Toggle play/pause."""
        if hasattr(self, 'preview') and hasattr(self.preview, 'toggle_play'):
            self.preview.toggle_play()
            # Update button text
            try:
                if self.preview._playing.get():
                    self.btn_play.configure(text="Pause")
                else:
                    self.btn_play.configure(text="Play")
            except Exception:
                pass


    # -------- right (side settings, collapsible sideways) --------
    def _build_right(self):
        # Collapsible settings panel (right side)
        self.settings_panel = SideSettingsPanel(self.right, on_toggle=self._toggle_settings_panel)
        self.settings_panel.pack(fill="both", expand=True)

        body = self.settings_panel.body  # scrollable inner frame

        # ---------- helpers ----------
        def _row(parent):
            fr = ttk.Frame(parent)
            fr.pack(fill="x", padx=6, pady=3)
            return fr

        def _add_entry(parent, label, var, width=12, help_text=None):
            r = _row(parent)
            ttk.Label(r, text=label, width=26).pack(side="left")
            e = ttk.Entry(r, textvariable=var, width=width)
            e.pack(side="left", fill="x", expand=True)
            if help_text:
                ttk.Label(r, text=help_text, foreground="#666").pack(side="left", padx=6)
            return e

        def _add_check(parent, label, var):
            r = _row(parent)
            cb = ttk.Checkbutton(r, text=label, variable=var)
            cb.pack(side="left", anchor="w")
            return cb

        def _add_combo(parent, label, var, values):
            r = _row(parent)
            ttk.Label(r, text=label, width=26).pack(side="left")
            cb = ttk.Combobox(r, textvariable=var, values=list(values), state="readonly", width=20)
            cb.pack(side="left", fill="x", expand=True)
            return cb

        def _add_multiline(parent, label, textvar, height=5):
            r = ttk.Frame(parent)
            r.pack(fill="both", expand=True, padx=6, pady=3)
            ttk.Label(r, text=label).pack(anchor="w")
            txt = tk.Text(r, height=height, wrap="none")
            txt.pack(fill="both", expand=True)
            # set initial
            txt.insert("1.0", textvar.get())
            # keep synced (on apply we read from widget)
            return txt

        # ---------- Output ----------
        out_box = ttk.LabelFrame(body, text="Output")
        out_box.pack(fill="x", pady=(0, 8))

        self.var_output_dir = tk.StringVar(value=self.settings.output_dir)
        _add_entry(out_box, "Output Folder", self.var_output_dir, width=36)
        r = _row(out_box)
        ttk.Button(r, text="Browse", command=self._browse_output_dir).pack(side="right")

        # ---------- Camera Calibration ----------
        calib_box = ttk.LabelFrame(body, text="Camera Calibration")
        calib_box.pack(fill="x", pady=(0, 8))

        self.var_calib = tk.StringVar(value=self.settings.camera_calibration_path)
        _add_entry(calib_box, "Calibration TOML", self.var_calib, width=36)
        r = _row(calib_box)
        ttk.Button(r, text="Browse", command=self._browse_calib_file).pack(side="right")

        # ---------- Pipeline (checkbox only) ----------
        pipe_box = ttk.LabelFrame(body, text="Pipeline (enable/disable steps)")
        pipe_box.pack(fill="x", pady=(0, 8))

        self.var_sync = tk.BooleanVar(value=bool(self.settings.do_synchronization))
        self.var_assoc = tk.BooleanVar(value=bool(self.settings.do_person_association))
        self.var_triang = tk.BooleanVar(value=bool(self.settings.do_triangulation))
        self.var_filter = tk.BooleanVar(value=bool(self.settings.do_filtering))
        self.var_marker = tk.BooleanVar(value=bool(self.settings.do_marker_augmentation))
        self.var_kin = tk.BooleanVar(value=bool(self.settings.do_kinematic))

        _add_check(pipe_box, "Synchronization", self.var_sync)
        _add_check(pipe_box, "Person Association", self.var_assoc)
        _add_check(pipe_box, "Triangulation", self.var_triang)
        _add_check(pipe_box, "Filtering", self.var_filter)
        _add_check(pipe_box, "Marker Augmentation", self.var_marker)
        _add_check(pipe_box, "Kinematics", self.var_kin)

        # ---------- Project ----------
        proj_box = ttk.LabelFrame(body, text="Project")
        proj_box.pack(fill="x", pady=(0, 8))

        self.var_multi_person = tk.BooleanVar(value=bool(getattr(self.settings, "multi_person", DEFAULT_POSE2SIM_CONFIG["project"]["multi_person"])))
        self.var_frame_rate = tk.StringVar(value=str(getattr(self.settings, "frame_rate", DEFAULT_POSE2SIM_CONFIG["project"]["frame_rate"])))
        self.var_frame_range = tk.StringVar(value=str(getattr(self.settings, "frame_range", DEFAULT_POSE2SIM_CONFIG["project"]["frame_range"])))
        self.var_exclude_from_batch = tk.StringVar(value=str(getattr(self.settings, "exclude_from_batch", DEFAULT_POSE2SIM_CONFIG["project"].get("exclude_from_batch", []))))

        _add_check(proj_box, "multi_person", self.var_multi_person)
        _add_entry(proj_box, "frame_rate", self.var_frame_rate, help_text="auto or number")
        _add_entry(proj_box, "frame_range", self.var_frame_range, help_text="auto or [start,end]")
        _add_entry(proj_box, "exclude_from_batch", self.var_exclude_from_batch, help_text="e.g. ['trial1','trial2']")

        # ---------- Subject ----------
        subj_box = ttk.LabelFrame(body, text="Subject")
        subj_box.pack(fill="x", pady=(0, 8))

        self.var_height = tk.StringVar(value=str(getattr(self.settings, "participant_height", DEFAULT_POSE2SIM_CONFIG["project"]["participant_height"])))
        self.var_mass = tk.StringVar(value=str(getattr(self.settings, "participant_mass_kg", DEFAULT_POSE2SIM_CONFIG["project"]["participant_mass"])))
        _add_entry(subj_box, "participant_height (m)", self.var_height, help_text="'auto' or meters")
        _add_entry(subj_box, "participant_mass (kg)", self.var_mass)

        # ---------- Pose Estimation ----------
        pose_box = ttk.LabelFrame(body, text="Pose Estimation")
        pose_box.pack(fill="x", pady=(0, 8))

        self.var_vid_ext = tk.StringVar(value=str(getattr(self.settings, "vid_img_extension", DEFAULT_POSE2SIM_CONFIG["pose"]["vid_img_extension"])))
        self.var_pose_mode = tk.StringVar(value=str(getattr(self.settings, "mode", "balanced")))
        self.var_det_frequency = tk.StringVar(value=str(getattr(self.settings, "det_frequency", DEFAULT_POSE2SIM_CONFIG["pose"]["det_frequency"])))
        self.var_device = tk.StringVar(value=str(getattr(self.settings, "device", DEFAULT_POSE2SIM_CONFIG["pose"]["device"])))
        self.var_backend = tk.StringVar(value=str(getattr(self.settings, "backend", DEFAULT_POSE2SIM_CONFIG["pose"]["backend"])))
        self.var_tracking = tk.StringVar(value=str(getattr(self.settings, "tracking_mode", DEFAULT_POSE2SIM_CONFIG["pose"]["tracking_mode"])))
        self.var_maxdist = tk.StringVar(value=str(getattr(self.settings, "max_distance_px", DEFAULT_POSE2SIM_CONFIG["pose"]["max_distance_px"])))
        self.var_deepsort = tk.StringVar(value=str(getattr(self.settings, "deepsort_params", DEFAULT_POSE2SIM_CONFIG["pose"]["deepsort_params"])))
        self.var_display_det = tk.BooleanVar(value=bool(getattr(self.settings, "display_detection", DEFAULT_POSE2SIM_CONFIG["pose"]["display_detection"])))
        self.var_overwrite_pose = tk.BooleanVar(value=bool(getattr(self.settings, "overwrite_pose", DEFAULT_POSE2SIM_CONFIG["pose"]["overwrite_pose"])))
        self.var_save_video = tk.StringVar(value=str(getattr(self.settings, "save_video", DEFAULT_POSE2SIM_CONFIG["pose"]["save_video"])))
        self.var_output_format = tk.StringVar(value=str(getattr(self.settings, "output_format", DEFAULT_POSE2SIM_CONFIG["pose"]["output_format"])))

        _add_entry(pose_box, "vid_img_extension", self.var_vid_ext)
        _add_combo(pose_box, "mode", self.var_pose_mode, ["balanced", "performance"])
        _add_entry(pose_box, "det_frequency", self.var_det_frequency)
        _add_combo(pose_box, "device", self.var_device, ["auto", "CPU", "CUDA", "MPS", "ROCM"])
        _add_combo(pose_box, "backend", self.var_backend, ["auto", "openvino", "onnxruntime", "opencv"])
        _add_combo(pose_box, "tracking_mode", self.var_tracking, ["none", "sports2d", "deepsort"])
        _add_entry(pose_box, "max_distance_px", self.var_maxdist)
        _add_entry(pose_box, "deepsort_params", self.var_deepsort, width=60)
        _add_check(pose_box, "display_detection", self.var_display_det)
        _add_check(pose_box, "overwrite_pose", self.var_overwrite_pose)
        _add_combo(pose_box, "save_video", self.var_save_video, ["none", "to_video", "to_images", "to_video+to_images"])
        _add_combo(pose_box, "output_format", self.var_output_format, ["openpose", "mmpose", "none"])

        # CUSTOM skeleton dict (advanced)
        self.var_pose_custom = tk.StringVar(value=str(getattr(self.settings, "pose_custom", DEFAULT_POSE2SIM_CONFIG["pose"].get("CUSTOM", {}))))
        self._pose_custom_text = _add_multiline(pose_box, "CUSTOM (advanced, dict)", self.var_pose_custom, height=7)

        # ---------- Synchronization ----------
        sync_box = ttk.LabelFrame(body, text="Synchronization")
        sync_box.pack(fill="x", pady=(0, 8))

        self.var_sync_gui = tk.BooleanVar(value=bool(getattr(self.settings, "synchronization_gui", DEFAULT_POSE2SIM_CONFIG["synchronization"]["synchronization_gui"])))
        self.var_disp_sync = tk.BooleanVar(value=bool(getattr(self.settings, "display_sync_plots", DEFAULT_POSE2SIM_CONFIG["synchronization"]["display_sync_plots"])))
        self.var_save_sync = tk.BooleanVar(value=bool(getattr(self.settings, "save_sync_plots", DEFAULT_POSE2SIM_CONFIG["synchronization"]["save_sync_plots"])))
        self.var_sync_kps = tk.StringVar(value=str(getattr(self.settings, "keypoints_to_consider", DEFAULT_POSE2SIM_CONFIG["synchronization"]["keypoints_to_consider"])))
        self.var_sync_approx = tk.StringVar(value=str(getattr(self.settings, "approx_time_maxspeed", DEFAULT_POSE2SIM_CONFIG["synchronization"]["approx_time_maxspeed"])))
        self.var_sync_timerange = tk.StringVar(value=str(getattr(self.settings, "time_range_around_maxspeed", DEFAULT_POSE2SIM_CONFIG["synchronization"]["time_range_around_maxspeed"])))
        self.var_sync_lh = tk.StringVar(value=str(getattr(self.settings, "sync_likelihood_threshold", DEFAULT_POSE2SIM_CONFIG["synchronization"]["likelihood_threshold"])))
        self.var_sync_cutoff = tk.StringVar(value=str(getattr(self.settings, "sync_filter_cutoff", DEFAULT_POSE2SIM_CONFIG["synchronization"]["filter_cutoff"])))
        self.var_sync_order = tk.StringVar(value=str(getattr(self.settings, "sync_filter_order", DEFAULT_POSE2SIM_CONFIG["synchronization"]["filter_order"])))

        _add_check(sync_box, "synchronization_gui", self.var_sync_gui)
        _add_check(sync_box, "display_sync_plots", self.var_disp_sync)
        _add_check(sync_box, "save_sync_plots", self.var_save_sync)
        _add_entry(sync_box, "keypoints_to_consider", self.var_sync_kps, help_text="e.g. ['RWrist']")
        _add_entry(sync_box, "approx_time_maxspeed", self.var_sync_approx, help_text="'auto' or seconds")
        _add_entry(sync_box, "time_range_around_maxspeed", self.var_sync_timerange)
        _add_entry(sync_box, "likelihood_threshold", self.var_sync_lh)
        _add_entry(sync_box, "filter_cutoff", self.var_sync_cutoff)
        _add_entry(sync_box, "filter_order", self.var_sync_order)

        # ---------- Person Association ----------
        assoc_box = ttk.LabelFrame(body, text="Person Association")
        assoc_box.pack(fill="x", pady=(0, 8))

        self.var_assoc_lh = tk.StringVar(value=str(getattr(self.settings, "likelihood_threshold_association", DEFAULT_POSE2SIM_CONFIG["personAssociation"]["likelihood_threshold_association"])))
        self.var_assoc_reproj = tk.StringVar(value=str(getattr(self.settings, "reproj_error_threshold_association", DEFAULT_POSE2SIM_CONFIG["personAssociation"]["single_person"]["reproj_error_threshold_association"])))
        self.var_tracked_kp = tk.StringVar(value=str(getattr(self.settings, "tracked_keypoint", DEFAULT_POSE2SIM_CONFIG["personAssociation"]["single_person"]["tracked_keypoint"])))
        self.var_multi_recon = tk.StringVar(value=str(getattr(self.settings, "reconstruction_error_threshold", DEFAULT_POSE2SIM_CONFIG["personAssociation"]["multi_person"]["reconstruction_error_threshold"])))
        self.var_multi_aff = tk.StringVar(value=str(getattr(self.settings, "min_affinity", DEFAULT_POSE2SIM_CONFIG["personAssociation"]["multi_person"]["min_affinity"])))

        _add_entry(assoc_box, "likelihood_threshold_association", self.var_assoc_lh)
        _add_entry(assoc_box, "single_person.reproj_error_threshold_association", self.var_assoc_reproj)
        _add_entry(assoc_box, "single_person.tracked_keypoint", self.var_tracked_kp)
        _add_entry(assoc_box, "multi_person.reconstruction_error_threshold", self.var_multi_recon)
        _add_entry(assoc_box, "multi_person.min_affinity", self.var_multi_aff)

        # ---------- Triangulation ----------
        tri_box = ttk.LabelFrame(body, text="Triangulation")
        tri_box.pack(fill="x", pady=(0, 8))

        self.var_tr_reproj = tk.StringVar(value=str(getattr(self.settings, "reproj_error_threshold_triangulation", DEFAULT_POSE2SIM_CONFIG["triangulation"]["reproj_error_threshold_triangulation"])))
        self.var_tr_lh = tk.StringVar(value=str(getattr(self.settings, "likelihood_threshold_triangulation", DEFAULT_POSE2SIM_CONFIG["triangulation"]["likelihood_threshold_triangulation"])))
        self.var_tr_min_cam = tk.StringVar(value=str(getattr(self.settings, "min_cameras_for_triangulation", DEFAULT_POSE2SIM_CONFIG["triangulation"]["min_cameras_for_triangulation"])))
        self.var_tr_interp_gap = tk.StringVar(value=str(getattr(self.settings, "interp_if_gap_smaller_than", DEFAULT_POSE2SIM_CONFIG["triangulation"]["interp_if_gap_smaller_than"])))
        self.var_tr_maxdist_m = tk.StringVar(value=str(getattr(self.settings, "max_distance_m", DEFAULT_POSE2SIM_CONFIG["triangulation"]["max_distance_m"])))
        self.var_tr_interp = tk.StringVar(value=str(getattr(self.settings, "interpolation", DEFAULT_POSE2SIM_CONFIG["triangulation"]["interpolation"])))
        self.var_tr_remove_incomplete = tk.BooleanVar(value=bool(getattr(self.settings, "remove_incomplete_frames", DEFAULT_POSE2SIM_CONFIG["triangulation"]["remove_incomplete_frames"])))
        self.var_tr_sections = tk.StringVar(value=str(getattr(self.settings, "sections_to_keep", DEFAULT_POSE2SIM_CONFIG["triangulation"]["sections_to_keep"])))
        self.var_tr_min_chunk = tk.StringVar(value=str(getattr(self.settings, "min_chunk_size", DEFAULT_POSE2SIM_CONFIG["triangulation"]["min_chunk_size"])))
        self.var_tr_fill = tk.StringVar(value=str(getattr(self.settings, "fill_large_gaps_with", DEFAULT_POSE2SIM_CONFIG["triangulation"]["fill_large_gaps_with"])))
        self.var_tr_show_idx = tk.BooleanVar(value=bool(getattr(self.settings, "show_interp_indices", DEFAULT_POSE2SIM_CONFIG["triangulation"]["show_interp_indices"])))
        self.var_tr_make_c3d = tk.BooleanVar(value=bool(getattr(self.settings, "tri_make_c3d", DEFAULT_POSE2SIM_CONFIG["triangulation"]["make_c3d"])))

        _add_entry(tri_box, "reproj_error_threshold_triangulation", self.var_tr_reproj)
        _add_entry(tri_box, "likelihood_threshold_triangulation", self.var_tr_lh)
        _add_entry(tri_box, "min_cameras_for_triangulation", self.var_tr_min_cam)
        _add_entry(tri_box, "interp_if_gap_smaller_than", self.var_tr_interp_gap)
        _add_entry(tri_box, "max_distance_m", self.var_tr_maxdist_m)
        _add_combo(tri_box, "interpolation", self.var_tr_interp, ["linear", "slinear", "quadratic", "cubic", "none"])
        _add_check(tri_box, "remove_incomplete_frames", self.var_tr_remove_incomplete)
        _add_combo(tri_box, "sections_to_keep", self.var_tr_sections, ['all','largest','first','last'])
        _add_entry(tri_box, "min_chunk_size", self.var_tr_min_chunk)
        _add_combo(tri_box, "fill_large_gaps_with", self.var_tr_fill, ['last_value','nan','zeros'])
        _add_check(tri_box, "show_interp_indices", self.var_tr_show_idx)
        _add_check(tri_box, "make_c3d", self.var_tr_make_c3d)

        # ---------- Filtering ----------
        filt_box = ttk.LabelFrame(body, text="Filtering")
        filt_box.pack(fill="x", pady=(0, 8))

        self.var_reject_outliers = tk.BooleanVar(value=bool(getattr(self.settings, "reject_outliers", DEFAULT_POSE2SIM_CONFIG["filtering"]["reject_outliers"])))
        self.var_filter_enable = tk.BooleanVar(value=bool(getattr(self.settings, "filter_on", DEFAULT_POSE2SIM_CONFIG["filtering"]["filter"])))
        self.var_filter_type = tk.StringVar(value=str(getattr(self.settings, "filter_type", DEFAULT_POSE2SIM_CONFIG["filtering"]["type"])))
        self.var_disp_filt = tk.BooleanVar(value=bool(getattr(self.settings, "display_figures", DEFAULT_POSE2SIM_CONFIG["filtering"]["display_figures"])))
        self.var_save_filt = tk.BooleanVar(value=bool(getattr(self.settings, "save_filt_plots", DEFAULT_POSE2SIM_CONFIG["filtering"]["save_filt_plots"])))
        self.var_filt_make_c3d = tk.BooleanVar(value=bool(getattr(self.settings, "filt_make_c3d", DEFAULT_POSE2SIM_CONFIG["filtering"]["make_c3d"])))

        _add_check(filt_box, "reject_outliers", self.var_reject_outliers)
        _add_check(filt_box, "filter", self.var_filter_enable)
        _add_combo(filt_box, "type", self.var_filter_type, ["butterworth", "kalman", "gcv_spline", "loess", "gaussian", "median", "butterworth_on_speed"])
        _add_check(filt_box, "display_figures", self.var_disp_filt)
        _add_check(filt_box, "save_filt_plots", self.var_save_filt)
        _add_check(filt_box, "make_c3d", self.var_filt_make_c3d)

        # sub: butterworth
        bw = ttk.LabelFrame(filt_box, text="butterworth")
        bw.pack(fill="x", padx=6, pady=(6, 6))
        self.var_bw_cutoff = tk.StringVar(value=str(getattr(self.settings, "butter_cutoff_hz", DEFAULT_POSE2SIM_CONFIG["filtering"]["butterworth"]["cut_off_frequency"])))
        self.var_bw_order = tk.StringVar(value=str(getattr(self.settings, "butter_order", DEFAULT_POSE2SIM_CONFIG["filtering"]["butterworth"]["order"])))
        _add_entry(bw, "cut_off_frequency", self.var_bw_cutoff)
        _add_entry(bw, "order", self.var_bw_order)

        # sub: kalman
        kf = ttk.LabelFrame(filt_box, text="kalman")
        kf.pack(fill="x", padx=6, pady=(0, 6))
        self.var_kalman_trust = tk.StringVar(value=str(getattr(self.settings, "kalman_trust_ratio", DEFAULT_POSE2SIM_CONFIG["filtering"]["kalman"]["trust_ratio"])))
        self.var_kalman_smooth = tk.BooleanVar(value=bool(getattr(self.settings, "kalman_smooth", DEFAULT_POSE2SIM_CONFIG["filtering"]["kalman"]["smooth"])))
        _add_entry(kf, "trust_ratio", self.var_kalman_trust)
        _add_check(kf, "smooth", self.var_kalman_smooth)

        # sub: gcv_spline
        gcv = ttk.LabelFrame(filt_box, text="gcv_spline")
        gcv.pack(fill="x", padx=6, pady=(0, 6))
        self.var_gcv_cutoff = tk.StringVar(value=str(getattr(self.settings, "gcv_cutoff", DEFAULT_POSE2SIM_CONFIG["filtering"]["gcv_spline"]["cut_off_frequency"])))
        self.var_gcv_smooth = tk.StringVar(value=str(getattr(self.settings, "gcv_smoothing_factor", DEFAULT_POSE2SIM_CONFIG["filtering"]["gcv_spline"]["smoothing_factor"])))
        _add_entry(gcv, "cut_off_frequency", self.var_gcv_cutoff)
        _add_entry(gcv, "smoothing_factor", self.var_gcv_smooth)

        # sub: loess
        lo = ttk.LabelFrame(filt_box, text="loess")
        lo.pack(fill="x", padx=6, pady=(0, 6))
        self.var_loess_nb = tk.StringVar(value=str(getattr(self.settings, "loess_nb_values_used", DEFAULT_POSE2SIM_CONFIG["filtering"]["loess"]["nb_values_used"])))
        _add_entry(lo, "nb_values_used", self.var_loess_nb)

        # sub: gaussian
        ga = ttk.LabelFrame(filt_box, text="gaussian")
        ga.pack(fill="x", padx=6, pady=(0, 6))
        self.var_gauss_sigma = tk.StringVar(value=str(getattr(self.settings, "gaussian_sigma_kernel", DEFAULT_POSE2SIM_CONFIG["filtering"]["gaussian"]["sigma_kernel"])))
        _add_entry(ga, "sigma_kernel", self.var_gauss_sigma)

        # sub: median
        md = ttk.LabelFrame(filt_box, text="median")
        md.pack(fill="x", padx=6, pady=(0, 6))
        self.var_med_kernel = tk.StringVar(value=str(getattr(self.settings, "median_kernel_size", DEFAULT_POSE2SIM_CONFIG["filtering"]["median"]["kernel_size"])))
        _add_entry(md, "kernel_size", self.var_med_kernel)

        # sub: butterworth_on_speed
        bws = ttk.LabelFrame(filt_box, text="butterworth_on_speed")
        bws.pack(fill="x", padx=6, pady=(0, 6))
        self.var_bws_cutoff = tk.StringVar(value=str(getattr(self.settings, "bws_cutoff_hz", DEFAULT_POSE2SIM_CONFIG["filtering"]["butterworth_on_speed"]["cut_off_frequency"])))
        self.var_bws_order = tk.StringVar(value=str(getattr(self.settings, "bws_order", DEFAULT_POSE2SIM_CONFIG["filtering"]["butterworth_on_speed"]["order"])))
        _add_entry(bws, "cut_off_frequency", self.var_bws_cutoff)
        _add_entry(bws, "order", self.var_bws_order)


        # Show only sub-settings for selected filtering type
        self._filter_subframes = {
            "butterworth": bw,
            "kalman": kf,
            "gcv_spline": gcv,
            "loess": lo,
            "gaussian": ga,
            "median": md,
            "butterworth_on_speed": bws,
        }
        self._filter_packinfo = {
            "butterworth": dict(fill="x", padx=6, pady=(6, 6)),
            "kalman": dict(fill="x", padx=6, pady=(0, 6)),
            "gcv_spline": dict(fill="x", padx=6, pady=(0, 6)),
            "loess": dict(fill="x", padx=6, pady=(0, 6)),
            "gaussian": dict(fill="x", padx=6, pady=(0, 6)),
            "median": dict(fill="x", padx=6, pady=(0, 6)),
            "butterworth_on_speed": dict(fill="x", padx=6, pady=(0, 6)),
        }

        def _update_filter_subframes(*_):
            t = (self.var_filter_type.get() or "").strip()
            # hide all
            for fr in self._filter_subframes.values():
                try:
                    fr.pack_forget()
                except Exception:
                    pass
            # show selected
            fr = self._filter_subframes.get(t)
            if fr is not None:
                try:
                    fr.pack(**self._filter_packinfo.get(t, dict(fill="x", padx=6, pady=(0, 6))))
                except Exception:
                    pass

        try:
            self.var_filter_type.trace_add("write", _update_filter_subframes)
        except Exception:
            try:
                self.var_filter_type.trace("w", _update_filter_subframes)
            except Exception:
                pass
        _update_filter_subframes()

        # ---------- MarkerAugmentation ----------
        ma_box = ttk.LabelFrame(body, text="MarkerAugmentation")
        ma_box.pack(fill="x", pady=(0, 8))

        self.var_feet_on_floor = tk.BooleanVar(value=bool(getattr(self.settings, "feet_on_floor", DEFAULT_POSE2SIM_CONFIG["markerAugmentation"]["feet_on_floor"])))
        self.var_ma_make_c3d = tk.BooleanVar(value=bool(getattr(self.settings, "ma_make_c3d", DEFAULT_POSE2SIM_CONFIG["markerAugmentation"]["make_c3d"])))
        _add_check(ma_box, "feet_on_floor", self.var_feet_on_floor)
        _add_check(ma_box, "make_c3d", self.var_ma_make_c3d)

        # ---------- Kinematics ----------
        kin_box = ttk.LabelFrame(body, text="Kinematics")
        kin_box.pack(fill="x", pady=(0, 8))

        self.var_use_aug = tk.BooleanVar(value=bool(getattr(self.settings, "use_augmentation", DEFAULT_POSE2SIM_CONFIG["kinematics"]["use_augmentation"])))
        self.var_simple_model = tk.BooleanVar(value=bool(getattr(self.settings, "use_simple_model", DEFAULT_POSE2SIM_CONFIG["kinematics"]["use_simple_model"])))
        self.var_sym = tk.BooleanVar(value=bool(getattr(self.settings, "right_left_symmetry", DEFAULT_POSE2SIM_CONFIG["kinematics"]["right_left_symmetry"])))
        self.var_def_height = tk.StringVar(value=str(getattr(self.settings, "default_height", DEFAULT_POSE2SIM_CONFIG["kinematics"]["default_height"])))
        self.var_rm_scaling = tk.BooleanVar(value=bool(getattr(self.settings, "remove_individual_scaling_setup", DEFAULT_POSE2SIM_CONFIG["kinematics"]["remove_individual_scaling_setup"])))
        self.var_rm_ik = tk.BooleanVar(value=bool(getattr(self.settings, "remove_individual_ik_setup", DEFAULT_POSE2SIM_CONFIG["kinematics"]["remove_individual_ik_setup"])))
        self.var_fast_rm = tk.StringVar(value=str(getattr(self.settings, "fastest_frames_to_remove_percent", DEFAULT_POSE2SIM_CONFIG["kinematics"]["fastest_frames_to_remove_percent"])))
        self.var_close0 = tk.StringVar(value=str(getattr(self.settings, "close_to_zero_speed_m", DEFAULT_POSE2SIM_CONFIG["kinematics"]["close_to_zero_speed_m"])))
        self.var_large_angles = tk.StringVar(value=str(getattr(self.settings, "large_hip_knee_angles", DEFAULT_POSE2SIM_CONFIG["kinematics"]["large_hip_knee_angles"])))
        self.var_trim = tk.StringVar(value=str(getattr(self.settings, "trimmed_extrema_percent", DEFAULT_POSE2SIM_CONFIG["kinematics"]["trimmed_extrema_percent"])))

        _add_check(kin_box, "use_augmentation", self.var_use_aug)
        _add_check(kin_box, "use_simple_model", self.var_simple_model)
        _add_check(kin_box, "right_left_symmetry", self.var_sym)
        _add_entry(kin_box, "default_height", self.var_def_height)
        _add_check(kin_box, "remove_individual_scaling_setup", self.var_rm_scaling)
        _add_check(kin_box, "remove_individual_ik_setup", self.var_rm_ik)
        _add_entry(kin_box, "fastest_frames_to_remove_percent", self.var_fast_rm)
        _add_entry(kin_box, "close_to_zero_speed_m", self.var_close0)
        _add_entry(kin_box, "large_hip_knee_angles", self.var_large_angles)
        _add_entry(kin_box, "trimmed_extrema_percent", self.var_trim)


    def _toggle_settings_panel(self, open_: bool):
        # Toggle panel: hide/show body content but keep panel width and button visible
        self.settings_panel.set_header_text(open_)
        
        # rightmost sash index (between left and right)
        sash_idx = 0
        if open_:
            try:
                self.settings_panel.show_body()
            except Exception:
                pass
            # restore previous sash position if available
            if hasattr(self, "_settings_sash_pos") and self._settings_sash_pos is not None:
                try:
                    self.paned.sashpos(sash_idx, int(self._settings_sash_pos))
                except Exception:
                    pass
            else:
                # If no saved position, set a reasonable default width (about 25% of window width)
                self.update_idletasks()
                # Get PanedWindow width (not the whole tab width)
                paned_w = max(0, int(self.paned.winfo_width()))
                default_width = max(300, int(paned_w * 0.25))  # At least 300px or 25% of paned width
                try:
                    self.paned.sashpos(sash_idx, max(0, paned_w - default_width))
                except Exception:
                    pass
            # Force update to adjust layout
            self.update_idletasks()
        else:
            try:
                # remember current sash position
                try:
                    self._settings_sash_pos = self.paned.sashpos(sash_idx)
                except Exception:
                    self._settings_sash_pos = None
                self.settings_panel.hide_body()
                # Collapse panel but keep minimal width for button only
                # Adjust sash position to give more space to left panel
                # Use after() to ensure layout is updated before adjusting sash
                def adjust_sash():
                    try:
                        self.update_idletasks()
                        # Get PanedWindow width (not the whole tab width)
                        paned_w = max(0, int(self.paned.winfo_width()))
                        # Keep minimal width for button visibility (about 65px)
                        # Move sash almost to the right edge so left panel takes most of the space
                        min_width = 50
                        new_pos = max(0, paned_w - min_width)
                        if paned_w > min_width:
                            self.paned.sashpos(sash_idx, new_pos)
                            self.update_idletasks()
                            # Verify and adjust again after layout update
                            paned_w_after = max(0, int(self.paned.winfo_width()))
                            current_pos = self.paned.sashpos(sash_idx)
                            # Ensure sash is at the right position
                            if paned_w_after > min_width and current_pos < paned_w_after - min_width:
                                self.paned.sashpos(sash_idx, paned_w_after - min_width)
                                self.update_idletasks()
                    except Exception:
                        pass
                # Schedule adjustment after current events are processed
                self.after(10, adjust_sash)
            except Exception:
                pass

# -------- playback bar handlers --------
    def _on_video_loaded(self):
        self._update_playback_range()
        # If user selected a video folder, default Output Folder to that path
        try:
            if getattr(self.preview, 'last_video_folder', None):
                self.var_output_dir.set(self.preview.last_video_folder)
                self.settings.output_dir = self.preview.last_video_folder
        except Exception:
            pass
        # If pose-3d already exists in the selected folder, auto-load markers.
        self._auto_load_markers_from_video_folder(reason="video loaded")

    def _on_clear_all(self):
        """Clear Preview + Marker data together."""
        try:
            self.preview.pause()
        except Exception:
            pass

        # clear preview UI
        try:
            self.preview.clear()
        except Exception:
            pass

        # clear marker data in 3D viewer
        try:
            self.viewer.clear_marker_data(keep_view=False)
        except Exception:
            pass

        # reset frame & playback range
        try:
            self.play_frame.set(0)
        except Exception:
            pass
        self._update_playback_range()


    def _auto_load_markers_from_video_folder(self, *, reason: str = ""):
        """Auto-load marker data (TRC/C3D) from ./pose-3d if present.

        This is used in two situations:
          1) Right after the user uploads a video folder (if pose-3d already exists)
          2) Right after a pipeline run finishes (so new TRC/C3D gets auto-loaded)

        Priority (TRC first, then C3D):
          - '*LSTM*' or '*aug*'  (augmentation output)
          - '*butterworth*' or '*filt*' (filtering output)
          - any other

        If none exists, do nothing.
        """
        try:
            paths = getattr(self.preview, "paths", None) or []
            if not paths:
                return

            # Determine the project root folder for marker auto-load.
            # Videos are typically under <project_dir>/videos/*.ext, while TRC/C3D live under <project_dir>/pose-3d/.
            p0 = Path(paths[0]).expanduser().resolve()
            folder = p0.parent  # where the video file sits
            root = folder
            try:
                if folder.name.lower() == "videos":
                    root = folder.parent
                elif (folder / "videos").is_dir():
                    root = folder
                elif (folder.parent / "videos").is_dir():
                    root = folder.parent
            except Exception:
                root = folder

            pose3d = root / "pose-3d"
            if not (pose3d.exists() and pose3d.is_dir()):
                # If user switched to another folder that doesn't have markers, clear the viewer
                # so it doesn't keep showing previous trial's markers.
                try:
                    self.viewer.clear_marker_data(keep_view=True)
                except Exception:
                    pass
                return

            # collect candidates
            trc_files = sorted([p for p in pose3d.rglob("*.trc") if p.is_file()])
            c3d_files = sorted([p for p in pose3d.rglob("*.c3d") if p.is_file()])
            if (not trc_files) and (not c3d_files):
                try:
                    self.viewer.clear_marker_data(keep_view=True)
                except Exception:
                    pass
                return


            def _priority(p: Path):
                name = p.name.lower()
                if ("lstm" in name) or ("aug" in name) or ("augment" in name):
                    return (0, name)
                if ("butterworth" in name) or ("filt" in name) or ("filter" in name):
                    return (1, name)
                return (2, name)

            # choose TRC first; if none, choose C3D
            if trc_files:
                sel = sorted(trc_files, key=_priority)[0]
                sel_kind = "trc"
            else:
                sel = sorted(c3d_files, key=_priority)[0]
                sel_kind = "c3d"

            why = f" ({reason})" if reason else ""
            self._log(f"[AutoLoad] pose-3d {sel_kind.upper()} detected{why} -> loading: {sel.name}")

            # load into 3D viewer
            try:
                self.viewer.load_motion_path(str(sel))
            except Exception as e:
                self._log(f"[AutoLoad][WARN] Failed to load markers: {e}")
                return

            # reflect in settings UI: LSTM/aug => enable augmentation by default
            try:
                if hasattr(self, "var_aug"):
                    name_l = sel.name.lower()
                    self.var_aug.set(("lstm" in name_l) or ("aug" in name_l) or ("augment" in name_l))
            except Exception:
                pass

        except Exception as e:
            self._log(f"[AutoLoad][WARN] Marker auto-load failed: {e}")

    def _on_3d_loaded(self):
        """Called after TRC/C3D loaded in 3D viewer."""
        # allow playback even if no video is loaded (3D-only)
        try:
            self.preview.set_external_fps(getattr(self.viewer, "fps", 60.0))
        except Exception:
            pass
        self._update_playback_range()

    def _on_frame_changed(self):
        # called when shared frame changes
        idx = int(self.play_frame.get())
        self.seek_scale.set(idx)  # keep UI in sync
        self.viewer.set_frame(idx)
        self._update_frame_label()

    def _on_speed_changed(self):
        # if speed 0 -> pause
        if float(self.play_speed.get()) <= 0.0:
            self.preview.pause()

    def _on_seek_scale(self, v):
        try:
            idx = int(float(v))
        except Exception:
            return
        self.play_frame.set(idx)

    def _on_seek_wheel(self, e):
        direction = 1 if e.delta > 0 else -1
        self._step_frame(direction)
        return "break"

    def _step_frame(self, direction: int):
        maxf = int(self.seek_scale.cget("to"))
        cur = int(self.play_frame.get())
        nxt = max(0, min(cur + direction, maxf))
        self.play_frame.set(nxt)

    def _update_playback_range(self):
        # video frames
        v_max = self.preview.get_max_frames()
        # 3d frames
        d_max = self.viewer.n_frames()
        if v_max and d_max:
            total = min(v_max, d_max)
        else:
            total = max(v_max, d_max)

        total = int(total or 0)
        try:
            self.preview.set_external_max_frames(total)
        except Exception:
            pass
        self.seek_scale.configure(from_=0, to=max(0, total - 1))
        if int(self.play_frame.get()) > max(0, total - 1):
            self.play_frame.set(max(0, total - 1))
        self._update_frame_label()

    def _update_frame_label(self):
        total = int(self.seek_scale.cget("to")) + 1 if int(self.seek_scale.cget("to")) >= 0 else 0
        cur = int(self.play_frame.get()) + 1 if total > 0 else 0
        self.lbl_frame.configure(text=f"{cur}/{total}" if total else "0/0")

    # -------- settings actions --------
    def _browse_output_dir(self):
        d = filedialog.askdirectory(title="Select output directory")
        if d:
            self.var_output_dir.set(d)

    def _browse_calib_file(self):
        p = filedialog.askopenfilename(
            title="Select camera_calibration file",
            filetypes=[("TOML", "*.toml"), ("All files", "*.*")],
        )
        if p:
            self.var_calib.set(p)

    def _apply_settings(self):
        import ast as _ast

        def _safe_eval(s, default):
            try:
                return _ast.literal_eval(s)
            except Exception:
                return default

        # Output / calibration
        self.settings.output_dir = (self.var_output_dir.get() or '').strip()
        self.settings.camera_calibration_path = (self.var_calib.get() or '').strip()

        # Pipeline toggles
        self.settings.do_synchronization = bool(self.var_sync.get())
        self.settings.do_person_association = bool(self.var_assoc.get())
        self.settings.do_triangulation = bool(self.var_triang.get())
        self.settings.do_filtering = bool(self.var_filter.get())
        self.settings.do_marker_augmentation = bool(self.var_marker.get())
        self.settings.do_kinematic = bool(self.var_kin.get())

        # Project
        self.settings.multi_person = bool(self.var_multi_person.get())
        self.settings.frame_rate = (self.var_frame_rate.get() or '').strip() or 'auto'
        self.settings.frame_range = (self.var_frame_range.get() or '').strip() or 'auto'
        self.settings.exclude_from_batch = _safe_eval(self.var_exclude_from_batch.get(), [])

        # Subject
        self.settings.participant_height = (self.var_height.get() or '').strip() or 'auto'
        try:
            self.settings.participant_mass_kg = float(self.var_mass.get())
        except Exception:
            self.settings.participant_mass_kg = float(DEFAULT_POSE2SIM_CONFIG["project"]["participant_mass"])

        # Pose
        self.settings.vid_img_extension = (self.var_vid_ext.get() or '').strip() or DEFAULT_POSE2SIM_CONFIG["pose"]["vid_img_extension"]
        self.settings.mode = (self.var_pose_mode.get() or '').strip() or 'balanced'
        self.settings.det_frequency = int(float(self.var_det_frequency.get() or DEFAULT_POSE2SIM_CONFIG["pose"]["det_frequency"]))
        self.settings.device = (self.var_device.get() or '').strip() or DEFAULT_POSE2SIM_CONFIG["pose"]["device"]
        self.settings.backend = (self.var_backend.get() or '').strip() or DEFAULT_POSE2SIM_CONFIG["pose"]["backend"]
        self.settings.tracking_mode = (self.var_tracking.get() or '').strip() or DEFAULT_POSE2SIM_CONFIG["pose"]["tracking_mode"]
        self.settings.max_distance_px = int(float(self.var_maxdist.get() or DEFAULT_POSE2SIM_CONFIG["pose"]["max_distance_px"]))
        self.settings.deepsort_params = (self.var_deepsort.get() or '').strip()
        self.settings.display_detection = bool(self.var_display_det.get())
        self.settings.overwrite_pose = bool(self.var_overwrite_pose.get())
        self.settings.save_video = (self.var_save_video.get() or '').strip() or DEFAULT_POSE2SIM_CONFIG["pose"]["save_video"]
        self.settings.output_format = (self.var_output_format.get() or '').strip() or DEFAULT_POSE2SIM_CONFIG["pose"]["output_format"]
        # CUSTOM dict from text widget
        try:
            custom_txt = self._pose_custom_text.get("1.0", "end").strip()
            self.settings.pose_custom = _safe_eval(custom_txt, DEFAULT_POSE2SIM_CONFIG["pose"].get("CUSTOM", {}))
        except Exception:
            self.settings.pose_custom = DEFAULT_POSE2SIM_CONFIG["pose"].get("CUSTOM", {})

        # Synchronization
        self.settings.synchronization_gui = bool(self.var_sync_gui.get())
        self.settings.display_sync_plots = bool(self.var_disp_sync.get())
        self.settings.save_sync_plots = bool(self.var_save_sync.get())
        self.settings.keypoints_to_consider = _safe_eval(self.var_sync_kps.get(), DEFAULT_POSE2SIM_CONFIG["synchronization"]["keypoints_to_consider"])
        self.settings.approx_time_maxspeed = (self.var_sync_approx.get() or '').strip() or DEFAULT_POSE2SIM_CONFIG["synchronization"]["approx_time_maxspeed"]
        self.settings.time_range_around_maxspeed = float(self.var_sync_timerange.get() or DEFAULT_POSE2SIM_CONFIG["synchronization"]["time_range_around_maxspeed"])
        self.settings.sync_likelihood_threshold = float(self.var_sync_lh.get() or DEFAULT_POSE2SIM_CONFIG["synchronization"]["likelihood_threshold"])
        self.settings.sync_filter_cutoff = float(self.var_sync_cutoff.get() or DEFAULT_POSE2SIM_CONFIG["synchronization"]["filter_cutoff"])
        self.settings.sync_filter_order = int(float(self.var_sync_order.get() or DEFAULT_POSE2SIM_CONFIG["synchronization"]["filter_order"]))

        # Person Association
        self.settings.likelihood_threshold_association = float(self.var_assoc_lh.get() or DEFAULT_POSE2SIM_CONFIG["personAssociation"]["likelihood_threshold_association"])
        self.settings.reproj_error_threshold_association = float(self.var_assoc_reproj.get() or DEFAULT_POSE2SIM_CONFIG["personAssociation"]["single_person"]["reproj_error_threshold_association"])
        self.settings.tracked_keypoint = (self.var_tracked_kp.get() or '').strip() or DEFAULT_POSE2SIM_CONFIG["personAssociation"]["single_person"]["tracked_keypoint"]
        self.settings.reconstruction_error_threshold = float(self.var_multi_recon.get() or DEFAULT_POSE2SIM_CONFIG["personAssociation"]["multi_person"]["reconstruction_error_threshold"])
        self.settings.min_affinity = float(self.var_multi_aff.get() or DEFAULT_POSE2SIM_CONFIG["personAssociation"]["multi_person"]["min_affinity"])

        # Triangulation
        self.settings.reproj_error_threshold_triangulation = float(self.var_tr_reproj.get() or DEFAULT_POSE2SIM_CONFIG["triangulation"]["reproj_error_threshold_triangulation"])
        self.settings.likelihood_threshold_triangulation = float(self.var_tr_lh.get() or DEFAULT_POSE2SIM_CONFIG["triangulation"]["likelihood_threshold_triangulation"])
        self.settings.min_cameras_for_triangulation = int(float(self.var_tr_min_cam.get() or DEFAULT_POSE2SIM_CONFIG["triangulation"]["min_cameras_for_triangulation"]))
        self.settings.interp_if_gap_smaller_than = int(float(self.var_tr_interp_gap.get() or DEFAULT_POSE2SIM_CONFIG["triangulation"]["interp_if_gap_smaller_than"]))
        self.settings.max_distance_m = float(self.var_tr_maxdist_m.get() or DEFAULT_POSE2SIM_CONFIG["triangulation"]["max_distance_m"])
        self.settings.interpolation = (self.var_tr_interp.get() or '').strip() or DEFAULT_POSE2SIM_CONFIG["triangulation"]["interpolation"]
        self.settings.remove_incomplete_frames = bool(self.var_tr_remove_incomplete.get())
        self.settings.sections_to_keep = (self.var_tr_sections.get() or '').strip() or DEFAULT_POSE2SIM_CONFIG["triangulation"]["sections_to_keep"]
        self.settings.min_chunk_size = int(float(self.var_tr_min_chunk.get() or DEFAULT_POSE2SIM_CONFIG["triangulation"]["min_chunk_size"]))
        self.settings.fill_large_gaps_with = (self.var_tr_fill.get() or '').strip() or DEFAULT_POSE2SIM_CONFIG["triangulation"]["fill_large_gaps_with"]
        self.settings.show_interp_indices = bool(self.var_tr_show_idx.get())
        self.settings.tri_make_c3d = bool(self.var_tr_make_c3d.get())

        # Filtering
        self.settings.reject_outliers = bool(self.var_reject_outliers.get())
        self.settings.filter_on = bool(self.var_filter_enable.get())
        self.settings.filter_type = (self.var_filter_type.get() or '').strip() or DEFAULT_POSE2SIM_CONFIG["filtering"]["type"]
        self.settings.display_figures = bool(self.var_disp_filt.get())
        self.settings.save_filt_plots = bool(self.var_save_filt.get())
        self.settings.filt_make_c3d = bool(self.var_filt_make_c3d.get())

        self.settings.butter_cutoff_hz = float(self.var_bw_cutoff.get() or DEFAULT_POSE2SIM_CONFIG["filtering"]["butterworth"]["cut_off_frequency"])
        self.settings.butter_order = int(float(self.var_bw_order.get() or DEFAULT_POSE2SIM_CONFIG["filtering"]["butterworth"]["order"]))
        self.settings.kalman_trust_ratio = float(self.var_kalman_trust.get() or DEFAULT_POSE2SIM_CONFIG["filtering"]["kalman"]["trust_ratio"])
        self.settings.kalman_smooth = bool(self.var_kalman_smooth.get())
        self.settings.gcv_cutoff = (self.var_gcv_cutoff.get() or '').strip() or DEFAULT_POSE2SIM_CONFIG["filtering"]["gcv_spline"]["cut_off_frequency"]
        self.settings.gcv_smoothing_factor = float(self.var_gcv_smooth.get() or DEFAULT_POSE2SIM_CONFIG["filtering"]["gcv_spline"]["smoothing_factor"])
        self.settings.loess_nb_values_used = int(float(self.var_loess_nb.get() or DEFAULT_POSE2SIM_CONFIG["filtering"]["loess"]["nb_values_used"]))
        self.settings.gaussian_sigma_kernel = float(self.var_gauss_sigma.get() or DEFAULT_POSE2SIM_CONFIG["filtering"]["gaussian"]["sigma_kernel"])
        self.settings.median_kernel_size = int(float(self.var_med_kernel.get() or DEFAULT_POSE2SIM_CONFIG["filtering"]["median"]["kernel_size"]))
        self.settings.bws_cutoff_hz = float(self.var_bws_cutoff.get() or DEFAULT_POSE2SIM_CONFIG["filtering"]["butterworth_on_speed"]["cut_off_frequency"])
        self.settings.bws_order = int(float(self.var_bws_order.get() or DEFAULT_POSE2SIM_CONFIG["filtering"]["butterworth_on_speed"]["order"]))

        # MarkerAugmentation
        self.settings.feet_on_floor = bool(self.var_feet_on_floor.get())
        self.settings.ma_make_c3d = bool(self.var_ma_make_c3d.get())

        # Kinematics
        self.settings.use_augmentation = bool(self.var_use_aug.get())
        self.settings.use_simple_model = bool(self.var_simple_model.get())
        self.settings.right_left_symmetry = bool(self.var_sym.get())
        self.settings.default_height = float(self.var_def_height.get() or DEFAULT_POSE2SIM_CONFIG["kinematics"]["default_height"])
        self.settings.remove_individual_scaling_setup = bool(self.var_rm_scaling.get())
        self.settings.remove_individual_ik_setup = bool(self.var_rm_ik.get())
        self.settings.fastest_frames_to_remove_percent = float(self.var_fast_rm.get() or DEFAULT_POSE2SIM_CONFIG["kinematics"]["fastest_frames_to_remove_percent"])
        self.settings.close_to_zero_speed_m = float(self.var_close0.get() or DEFAULT_POSE2SIM_CONFIG["kinematics"]["close_to_zero_speed_m"])
        self.settings.large_hip_knee_angles = float(self.var_large_angles.get() or DEFAULT_POSE2SIM_CONFIG["kinematics"]["large_hip_knee_angles"])
        self.settings.trimmed_extrema_percent = float(self.var_trim.get() or DEFAULT_POSE2SIM_CONFIG["kinematics"]["trimmed_extrema_percent"])

        self._log("[Settings] Applied.")

    def _validate(self) -> bool:
        self._apply_settings()
        ok = True
        if not self.settings.output_dir:
            self._log("[Validate] output_dir is empty.")
            ok = False
        if not self.settings.camera_calibration_path:
            self._log("[Validate] camera_calibration is empty.")
            ok = False
        if ok:
            self._log("[Validate] OK.")
        else:
            self._log("[Validate] FAILED. Fill required fields.")
        return ok

    def _run_stub(self):
        if not self._validate():
            return
        self._log("[Run] UI skeleton mode: processing pipeline is not implemented yet.")


    # -------- Run Pose 3D Analysis (RTMPOSE) --------
    def _run_pose2sim(self):
        """Run Pose2Sim pipeline using the loaded video folder + selected calibration file.

        Notes
        - Videos are loaded via the left preview panel ("Upload videos" -> folder).
        - This handler validates required fields, builds config dict, then calls runner.pose2sim_runner.
        """
        if not self._validate():
            return

        # Determine video directory from preview
        video_dir = ""
        try:
            if getattr(self.preview, "last_video_folder", ""):
                video_dir = str(Path(self.preview.last_video_folder).expanduser().resolve())
            elif getattr(self.preview, "paths", None):
                paths = list(getattr(self.preview, "paths") or [])
                if paths:
                    video_dir = str(Path(paths[0]).expanduser().resolve().parent)
        except Exception:
            video_dir = ""

        if not video_dir:
            self._log("[Validate] No videos loaded. Use 'Upload videos' first.")
            return

        calib_file = (self.settings.camera_calibration_path or "").strip()
        out_dir = (self.settings.output_dir or "").strip()

        # Build Pose2Sim-like config dict (Config.toml-free)
        try:
            # NOTE: build_pose2sim_config is a module-level helper (not a MotionSettings method)
            cfg = build_pose2sim_config(self.settings)
        except Exception as e:
            self._log(f"[Run] Failed to build config dict: {e}")
            cfg = {}

        self._log("[Run] Starting Pose 3D Analysis (Pose2Sim pipeline)...")

        def _worker():
            try:
                runner_run_pose2sim(
                    settings=cfg,
                    video_dir=video_dir,
                    calib_file=calib_file,
                    output_dir=out_dir,
                    log=self._log,
                )
                # After the run finishes (and filtering generates TRC/C3D under ./pose-3d),
                # auto-load the newest marker file into the 3D viewer.
                try:
                    self.after(0, lambda: self._auto_load_markers_from_video_folder(reason="after run"))
                except Exception:
                    pass
                self._log("[Run] Done.")
            except Exception as e:
                self._log(f"[Run][ERROR] {e}")

        threading.Thread(target=_worker, daemon=True).start()
    # Backward compatibility: some builds may call `self.run_pose2sim`
    def run_pose2sim(self):
        return self._run_pose2sim()



    def _install_logging_bridge(self) -> None:
        """Attach a logging handler so `logging.info/error` also appears in this tab."""
        try:
            if getattr(self, "_log_bridge_installed", False):
                return
            self._log_bridge_installed = True

            def _emit_to_ui(m: str):
                # If using shared log, use _log method which handles deduplication
                # Otherwise, append directly
                if self.shared_log_text is not None:
                    # Use _log to avoid duplication (it won't call external_log when shared_log_text exists)
                    try:
                        self.after(0, lambda mm=m: self._log(mm))
                    except Exception:
                        try:
                            self._pending_logs.append(m)
                        except Exception:
                            pass
                else:
                    # Direct append for local log
                    try:
                        self.after(0, lambda mm=m: self._append_log(mm))
                    except Exception:
                        try:
                            self._pending_logs.append(m)
                        except Exception:
                            pass

            h = UILogHandler(_emit_to_ui)
            h.setLevel(logging.INFO)
            h.setFormatter(logging.Formatter("%(levelname)s:%(name)s:%(message)s"))
            root = logging.getLogger()
            root.addHandler(h)
            # keep existing level; if unset, make it INFO so we actually see messages
            if root.level == logging.NOTSET:
                root.setLevel(logging.INFO)
            self._tk_log_handler = h
        except Exception:
            pass


    # -------- logging (Motion Analysis tab) --------
    def _append_log(self, msg: str) -> None:
        """Append a log line to the Motion Analysis tab log box (UI-safe)."""
        try:
            widget = getattr(self, "log_text", None)
            if widget is None:
                self._pending_logs.append(msg)
                return
            widget.configure(state="normal")
            widget.insert("end", msg.rstrip("\n") + "\n")
            widget.see("end")
            widget.configure(state="disabled")
        except Exception:
            pass

    def _log(self, msg: str) -> None:
        """Log to Motion Analysis tab and also forward to app-level logger."""
        # If using shared log, only append once to avoid duplication
        if self.shared_log_text is not None:
            # shared log: only append to shared log (don't call external_log to avoid duplication)
            try:
                self.after(0, lambda m=msg: self._append_log(m))
            except Exception:
                try:
                    self._pending_logs.append(msg)
                except Exception:
                    pass
        else:
            # local log: append to local log and forward to external logger
            try:
                self.after(0, lambda m=msg: self._append_log(m))
            except Exception:
                try:
                    self._pending_logs.append(msg)
                except Exception:
                    pass

            # forward to external logger (only if not using shared log)
            cb = getattr(self, "_external_log", None)
            if cb:
                try:
                    cb(msg)
                except Exception:
                    pass
