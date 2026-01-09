# -*- coding: utf-8 -*-
"""
Pose3DAnalysis - GUI Application for Multi-camera Pose Estimation and Analysis

Desktop GUI for multi-camera (2~8) intrinsic/extrinsic calibration and
markerless kinematics analysis.
"""

from __future__ import annotations

## AUTHORSHIP INFORMATION
__author__ = "Yongseok Park"
__copyright__ = "Copyright 2026, Pose3DAnalysis"
__credits__ = ["Yongseok Park"]
__license__ = "BSD 3-Clause License"
__version__ = "1.0.0"
__maintainer__ = "Yongseok Park"
__email__ = "pys9610@gmail.com"
__status__ = "Development"

import time
from pathlib import Path

import tkinter as tk
from tkinter import ttk, messagebox
from tkinter import scrolledtext

from pose3danalysis.gui.tabs.motion_analysis_tab import MotionAnalysisTab
from pose3danalysis.gui.tabs.batch_processing_tab import BatchProcessingTab
from pose3danalysis.gui.tabs.calibration_tab import CalibrationTab


class Pose3DAnalysisApp(tk.Tk):
    """
    Desktop GUI for multi-camera (2~8) intrinsic/extrinsic calibration.

    Key policies (per your request):
    - Selecting Output folder does NOT create any calibration folders.
    - Run Intrinsic creates: Output/Intrinsic_calibration_YYYYMMDD_HHMMSS/
        - lens_calibration.toml
        - per-camera chessboard overlay images
    - Run Extrinsic creates: Output/Extrinsic_calibration_YYYYMMDD_HHMMSS/
        - camera_calibration.toml (Pose2Sim section style)
        - per-camera overlay result images
    - preview_cache is created under a per-session folder next to the code
      and removed when the window closes.
    """

    def __init__(self):
        super().__init__()
        self.title("Pose3DAnalysis")
        self.minsize(1100, 720)
        # Start in maximized/zoomed state (fullscreen)
        try:
            # Windows
            self.state('zoomed')
        except:
            try:
                # Linux
                self.attributes('-zoomed', True)
            except:
                # Fallback: set large geometry
                self.geometry("1320x860")

        # logging (file + UI)
        self._setup_logging()

        # session/temp (preview cache + Pose2Sim workspace)
        # Go up from gui/main_app.py -> pose3danalysis -> src -> project root
        self.app_dir = Path(__file__).resolve().parent.parent.parent.parent


        self._build_ui()
        self.protocol("WM_DELETE_WINDOW", self.on_close)

    # ---------------- logging ----------------
    def _setup_logging(self):
        import logging

        log_path = (Path(__file__).resolve().parent.parent / "app.log")
        self.log_path = log_path

        logger = logging.getLogger("TwoCameras3DLifting")
        logger.setLevel(logging.INFO)
        logger.propagate = False

        if not any(isinstance(h, logging.FileHandler) and getattr(h, "baseFilename", "") == str(log_path)
                   for h in logger.handlers):
            fh = logging.FileHandler(self.log_path, encoding="utf-8")
            fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
            fh.setFormatter(fmt)
            logger.addHandler(fh)

        self.logger = logger
        self._log(f"Logger initialized -> {self.log_path}")

    def _log(self, msg: str, level: str = "INFO"):
        msg = str(msg)
        try:
            if hasattr(self, "logger"):
                if level.upper() == "ERROR":
                    self.logger.error(msg)
                elif level.upper() == "WARNING":
                    self.logger.warning(msg)
                else:
                    self.logger.info(msg)
        except Exception:
            pass

        if not hasattr(self, "log_text") or self.log_text is None:
            return

        def _append():
            try:
                self.log_text.configure(state="normal")
                self.log_text.insert("end", msg + "\n")
                self.log_text.see("end")
                self.log_text.configure(state="disabled")
            except Exception:
                pass

        try:
            self.after(0, _append)
        except Exception:
            _append()

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


    # ---------------- UI ----------------
    def _build_ui(self):
        # Main container with vertical layout: tabs on top, logs at bottom
        main_container = ttk.Frame(self)
        main_container.pack(fill="both", expand=True)
        
        nb = ttk.Notebook(main_container)
        nb.pack(fill="both", expand=True)

        tab_cal = ttk.Frame(nb)
        tab_motion = ttk.Frame(nb)
        tab_batch = ttk.Frame(nb)
        nb.add(tab_cal, text="Calibration")
        nb.add(tab_motion, text="Motion Analysis")
        nb.add(tab_batch, text="Batch Processing")
        
        # Shared log panel at the bottom
        logs_frame = ttk.LabelFrame(main_container, text="Logs")
        logs_frame.pack(fill="both", expand=False, padx=10, pady=(0, 10))
        
        self.log_text = scrolledtext.ScrolledText(logs_frame, height=12, wrap="word", state="disabled")
        self.log_text.pack(fill="both", expand=True, padx=6, pady=6)

        # ---- Calibration ----
        self.calibration_tab = CalibrationTab(
            tab_cal,
            log=self._log,
            shared_log_text=self.log_text,
            app_dir=self.app_dir
        )
        self.calibration_tab.pack(fill="both", expand=True)

        # ---- Motion Analysis ----
        self.motion_tab = MotionAnalysisTab(tab_motion, log=self._log, shared_log_text=self.log_text)
        self.motion_tab.pack(fill="both", expand=True)

        # ---- Batch Processing ----
        self.batch_tab = BatchProcessingTab(
            tab_batch, 
            log=self._log, 
            shared_log_text=self.log_text,
            motion_settings=self.motion_tab.settings
        )
        self.batch_tab.pack(fill="both", expand=True)

    # ---------------- Close ----------------
    def on_close(self):
        try:
            self._log("Closing app...")
            # CalibrationTab handles its own cleanup
            self.calibration_tab.session_manager.cleanup()
        finally:
            self.destroy()