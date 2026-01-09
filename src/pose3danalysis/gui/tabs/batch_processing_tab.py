# gui/tabs/batch_processing_tab.py
# Batch Processing Tab for processing multiple folders sequentially
# 2025 Pose3DAnalysis

from __future__ import annotations

import tkinter as tk
from tkinter import ttk, filedialog
from pathlib import Path
from typing import Callable, Optional, List, Dict
import threading

from pose3danalysis.gui.utils.motion_settings import MotionSettings, build_pose2sim_config, DEFAULT_POSE2SIM_CONFIG
from pose3danalysis.gui.components.side_settings_panel import SideSettingsPanel
from pose3danalysis.gui.components.scrollable_frame import ScrollableFrame
from pose3danalysis.runner.pose2sim_runner import run_pose2sim as runner_run_pose2sim


class BatchProcessingTab(ttk.Frame):
    """Batch Processing Tab: Process multiple folders sequentially with shared settings."""
    
    def __init__(self, parent, log: Callable[[str], None], shared_log_text=None, motion_settings: Optional[MotionSettings] = None, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)
        self._external_log = log
        self.shared_log_text = shared_log_text
        self._pending_logs = []
        
        # Share settings from Motion Analysis Tab
        if motion_settings is not None:
            self.settings = motion_settings
        else:
            self.settings = MotionSettings()
        
        # Batch processing state
        self.root_folder: Optional[Path] = None
        self.folder_items: Dict[str, Dict] = {}  # folder_path -> {item_id, status, videos}
        self.processing = False
        self.stop_requested = False
        self.current_processing_folder: Optional[str] = None
        
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
            if self._external_log:
                try:
                    self._external_log(msg)
                except Exception:
                    pass
    
    def _append_log(self, msg: str) -> None:
        """Append log to shared log text widget."""
        try:
            widget = self.shared_log_text
            if widget is None:
                self._pending_logs.append(msg)
                return
            widget.configure(state="normal")
            widget.insert("end", msg.rstrip("\n") + "\n")
            widget.see("end")
            widget.configure(state="disabled")
        except Exception:
            pass
    
    def _build_ui(self):
        """Build the batch processing UI."""
        # Main paned window
        self.paned = ttk.PanedWindow(self, orient="horizontal")
        self.paned.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Left panel: Folder tree
        self.left = ttk.Frame(self.paned)
        # Initial weights: favor left panel, but will be adjusted when settings panel is toggled
        self.paned.add(self.left, weight=3)
        
        # Right panel: Settings (same as Motion Analysis Tab)
        self.right = ttk.Frame(self.paned)
        self.paned.add(self.right, weight=1)
        
        self._build_left(self.left)
        self._build_right(self.right)
    
    def _build_left(self, parent):
        """Build left panel with folder tree."""
        # Top controls
        controls = ttk.Frame(parent)
        controls.pack(fill="x", padx=6, pady=(6, 0))
        
        ttk.Button(controls, text="Upload Folder", command=self._on_upload_folder).pack(side="left", padx=(0, 6))
        ttk.Button(controls, text="Clear", command=self._on_clear).pack(side="left", padx=(0, 6))
        
        self.run_button = ttk.Button(controls, text="Run", command=self._on_run)
        self.run_button.pack(side="left", padx=(0, 6))
        
        self.stop_button = ttk.Button(controls, text="Stop", command=self._on_stop, state="disabled")
        self.stop_button.pack(side="left", padx=(0, 6))
        
        # Folder tree with scrollbar
        tree_frame = ttk.Frame(parent)
        tree_frame.pack(fill="both", expand=True, padx=6, pady=6)
        
        # Scrollbar
        scrollbar = ttk.Scrollbar(tree_frame)
        scrollbar.pack(side="right", fill="y")
        
        # Configure Treeview font size (larger font for better readability)
        style = ttk.Style()
        # Create custom style with larger font
        try:
            import tkinter.font as tkfont
            # Get default font size and increase it
            default_font = style.lookup("Treeview", "font") or "TkDefaultFont 9"
            # Parse and increase font size
            font_parts = default_font.split()
            if len(font_parts) > 1:
                try:
                    current_size = int(font_parts[-1])
                    new_size = max(13, current_size + 5)  # Increase by 2, minimum 11
                    font_parts[-1] = str(new_size)
                except ValueError:
                    font_parts.append("13")
            else:
                font_parts.append("13")
            larger_font = " ".join(font_parts)
        except Exception:
            larger_font = "TkDefaultFont 13"
        
        # Create custom style for Treeview with larger font
        style.configure("Custom.Treeview", font=larger_font)
        
        # Treeview
        self.tree = ttk.Treeview(tree_frame, yscrollcommand=scrollbar.set, show="tree", style="Custom.Treeview")
        scrollbar.config(command=self.tree.yview)
        self.tree.pack(side="left", fill="both", expand=True)
        
        # Bind double-click to expand/collapse folders
        self.tree.bind("<Double-Button-1>", self._on_folder_click)
        
        # Status column (for circles)
        self.tree.heading("#0", text="Folders / Videos")
        
        # Store folder paths
        self.folder_items = {}
    
    def _build_right(self, parent):
        """Build right panel with settings (same UI as Motion Analysis Tab)."""
        # Collapsible settings panel (right side) - same as Motion Analysis Tab
        self.settings_panel = SideSettingsPanel(parent, on_toggle=self._toggle_settings_panel)
        self.settings_panel.pack(fill="both", expand=True)

        body = self.settings_panel.body  # scrollable inner frame
        
        # Import the settings building code from Motion Analysis Tab
        # We'll reuse the same UI structure
        self._build_settings_ui(body)
    
    def _build_settings_ui(self, body):
        """Build settings UI (same as Motion Analysis Tab's _build_right)."""
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

        # ---------- Apply Settings Button (at top) ----------
        btns = ttk.Frame(body)
        btns.pack(fill="x", padx=6, pady=(0, 8))
        ttk.Button(btns, text="Apply Settings", command=self._apply_settings).pack(side="left")

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
        """Toggle settings panel (same as Motion Analysis Tab)."""
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
                        min_width = 65
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
    
    def _browse_calib_file(self):
        """Browse for calibration file."""
        file = filedialog.askopenfilename(
            title="Select Calibration TOML",
            filetypes=[("TOML files", "*.toml"), ("All files", "*.*")]
        )
        if file:
            self.var_calib.set(file)
    
    def _apply_settings(self):
        """Apply settings changes to shared settings object (same as Motion Analysis Tab)."""
        import ast as _ast

        def _safe_eval(s, default):
            try:
                return _ast.literal_eval(s)
            except Exception:
                return default

        # Note: output_dir is not used in batch processing (each folder is its own output)
        # Camera Calibration
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

        self._log("[Batch] Settings applied. (Shared with Motion Analysis Tab)")
    
    def _on_upload_folder(self):
        """Handle Upload Folder button click."""
        folder = filedialog.askdirectory(title="Select Root Folder")
        if not folder:
            return
        
        self.root_folder = Path(folder)
        self._log(f"[Batch] Selected root folder: {self.root_folder}")
        self._load_subfolders()
    
    def _on_clear(self):
        """Clear folder tree."""
        for item in self.tree.get_children():
            self.tree.delete(item)
        self.folder_items.clear()
        self.root_folder = None
        self._log("[Batch] Cleared folder list.")
    
    def _load_subfolders(self):
        """Load subfolders from root folder."""
        if not self.root_folder or not self.root_folder.exists():
            self._log("[Batch][ERROR] Root folder does not exist.")
            return
        
        # Clear existing
        for item in self.tree.get_children():
            self.tree.delete(item)
        self.folder_items.clear()
        
        # Find subfolders (direct children only)
        subfolders = [d for d in self.root_folder.iterdir() if d.is_dir()]
        subfolders.sort()
        
        if not subfolders:
            self._log("[Batch][WARN] No subfolders found in root folder.")
            return
        
        self._log(f"[Batch] Found {len(subfolders)} subfolder(s).")
        
        # Add each subfolder to tree with status circle and load videos immediately
        for subfolder in subfolders:
            folder_path = str(subfolder)
            # Add empty circle icon (○) for pending status
            item_id = self.tree.insert("", "end", text=f"○ {subfolder.name}", tags=("folder",))
            
            # Load videos immediately
            folder_path_obj = Path(folder_path)
            video_extensions = [".mp4", ".avi", ".mov", ".mkv", ".MP4", ".AVI", ".MOV", ".MKV"]
            videos = []
            seen_names = set()
            for ext in video_extensions:
                found = list(folder_path_obj.glob(f"*{ext}"))
                for v in found:
                    if v.name not in seen_names:
                        videos.append(v)
                        seen_names.add(v.name)
            
            videos.sort(key=lambda x: x.name)
            
            # Store folder info
            self.folder_items[folder_path] = {
                "item_id": item_id,
                "status": "pending",  # pending, processing, success, error
                "videos": [str(v) for v in videos],
                "path": folder_path
            }
            
            # Add videos to tree
            if videos:
                for video in videos:
                    self.tree.insert(item_id, "end", text=video.name, tags=("video",))
                self.tree.item(item_id, open=True)
            else:
                self.tree.insert(item_id, "end", text="(No videos found)", tags=("empty",))
    
    def _on_folder_click(self, event):
        """Handle folder double-click to toggle expand/collapse."""
        item = self.tree.selection()[0] if self.tree.selection() else None
        if not item:
            return
        
        tags = self.tree.item(item, "tags")
        if "folder" not in tags:
            return
        
        # Toggle expand/collapse
        if self.tree.item(item, "open"):
            self.tree.item(item, open=False)
        else:
            self.tree.item(item, open=True)
    
    def _on_run(self):
        """Handle Run button click."""
        if self.processing:
            self._log("[Batch][WARN] Already processing. Please wait.")
            return
        
        if not self.root_folder:
            self._log("[Batch][ERROR] Please select a root folder first.")
            return
        
        if not self.settings.camera_calibration_path:
            self._log("[Batch][WARN] Calibration file not set. Processing may fail.")
        
        # Get all folders to process
        folders_to_process = [info["path"] for info in self.folder_items.values()]
        
        if not folders_to_process:
            self._log("[Batch][ERROR] No folders to process.")
            return
        
        self._log(f"[Batch] Starting batch processing for {len(folders_to_process)} folder(s)...")
        self._log("[Batch] Note: Results will be saved in each subfolder.")
        
        # Start processing in background thread
        self.processing = True
        self.stop_requested = False
        self.run_button.configure(state="disabled", text="Processing...")
        self.stop_button.configure(state="normal")
        
        threading.Thread(target=self._process_folders, args=(folders_to_process,), daemon=True).start()
    
    def _on_stop(self):
        """Handle Stop button click."""
        if not self.processing:
            return
        
        self.stop_requested = True
        self._log("[Batch] Stop requested. Processing will stop at the next stage boundary.")
        self.stop_button.configure(state="disabled", text="Stopping...")
    
    def _process_folders(self, folders: List[str]):
        """Process folders sequentially."""
        try:
            for i, folder_path in enumerate(folders, 1):
                # Check if stop was requested
                if self.stop_requested:
                    self._log(f"[Batch] Processing stopped by user. {len(folders) - i + 1} folder(s) remaining.")
                    break
                
                folder_info = self.folder_items.get(folder_path)
                if not folder_info:
                    continue
                
                folder_name = Path(folder_path).name
                self.current_processing_folder = folder_path
                
                # Update status to processing
                self.after(0, lambda f=folder_path: self._update_folder_status(f, "processing"))
                self._log(f"[Batch] [{i}/{len(folders)}] Processing: {folder_name}")
                
                try:
                    # Build config
                    cfg = build_pose2sim_config(self.settings)
                    
                    # Use the folder itself as output directory (results saved in each subfolder)
                    output_dir = folder_path
                    
                    # Run pose2sim with stop flag
                    runner_run_pose2sim(
                        settings=cfg,
                        video_dir=folder_path,
                        calib_file=self.settings.camera_calibration_path or "",
                        output_dir=output_dir,
                        log=self._log,
                        stop_flag=lambda: self.stop_requested,
                    )
                    
                    # Check again if stop was requested after processing
                    if self.stop_requested:
                        self._log(f"[Batch] Processing stopped by user after completing: {folder_name}")
                        break
                    
                    # Success
                    self.after(0, lambda f=folder_path: self._update_folder_status(f, "success"))
                    self._log(f"[Batch] [{i}/{len(folders)}] ✓ Completed: {folder_name}")
                    
                except RuntimeError as e:
                    # Check if stop was requested (stopped by user)
                    if self.stop_requested and "stopped by user" in str(e):
                        self._log(f"[Batch] Processing stopped by user during: {folder_name}")
                        break
                    # Other runtime errors
                    self.after(0, lambda f=folder_path: self._update_folder_status(f, "error"))
                    self._log(f"[Batch] [{i}/{len(folders)}] ✗ Failed: {folder_name} - {e}")
                except Exception as e:
                    # Check if stop was requested (might have been stopped during processing)
                    if self.stop_requested:
                        self._log(f"[Batch] Processing stopped by user. Error occurred: {e}")
                        break
                    
                    # Error
                    self.after(0, lambda f=folder_path: self._update_folder_status(f, "error"))
                    self._log(f"[Batch] [{i}/{len(folders)}] ✗ Failed: {folder_name} - {e}")
            
            if self.stop_requested:
                self._log("[Batch] Batch processing stopped by user.")
            else:
                self._log("[Batch] Batch processing completed.")
            
        except Exception as e:
            self._log(f"[Batch][ERROR] Batch processing failed: {e}")
        finally:
            self.processing = False
            self.stop_requested = False
            self.current_processing_folder = None
            self.after(0, lambda: self.run_button.configure(state="normal", text="Run"))
            self.after(0, lambda: self.stop_button.configure(state="disabled", text="Stop"))
    
    def _update_folder_status(self, folder_path: str, status: str):
        """Update folder status and update circle icon."""
        folder_info = self.folder_items.get(folder_path)
        if not folder_info:
            return
        
        folder_info["status"] = status
        item_id = folder_info["item_id"]
        
        # Update circle icon based on status
        # Note: Tkinter Treeview doesn't support custom icons easily,
        # so we'll use text prefix for now
        current_text = self.tree.item(item_id, "text")
        folder_name = Path(folder_path).name
        
        # Remove existing status prefix (handle various circle icons)
        if current_text.startswith("○ ") or current_text.startswith("● ") or current_text.startswith("◐ "):
            folder_name = current_text.split(" ", 1)[-1] if " " in current_text else current_text
        else:
            # If no prefix, use current text as folder name
            folder_name = current_text
        
        # Add status prefix
        if status == "pending":
            status_icon = "○"
        elif status == "processing":
            status_icon = "◐"  # Half-filled
        elif status == "success":
            status_icon = "●"  # Filled (will be green)
        elif status == "error":
            status_icon = "●"  # Filled (will be red)
        else:
            status_icon = "○"
        
        new_text = f"{status_icon} {folder_name}"
        self.tree.item(item_id, text=new_text)
        
        # Set tag for coloring (we'll configure tags)
        if status == "success":
            self.tree.item(item_id, tags=("folder", "success"))
        elif status == "error":
            self.tree.item(item_id, tags=("folder", "error"))
        elif status == "processing":
            self.tree.item(item_id, tags=("folder", "processing"))
        else:
            self.tree.item(item_id, tags=("folder",))
        
        # Configure tag colors
        self.tree.tag_configure("success", foreground="green")
        self.tree.tag_configure("error", foreground="red")
        self.tree.tag_configure("processing", foreground="blue")

