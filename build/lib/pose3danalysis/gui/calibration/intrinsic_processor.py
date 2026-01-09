# -*- coding: utf-8 -*-
"""Intrinsic calibration processing logic.

Extracted from gui.tabs.calibration_tab to improve code organization.
"""

from __future__ import annotations

import time
import traceback
from pathlib import Path
from typing import List, Callable, Optional

from pose3danalysis.core.frame_extract import extract_frames_every_n_sec
from pose3danalysis.core.calib_io import write_lens_calibration_toml
from pose3danalysis.core.utils import ensure_dir

# Pose2Sim calibration (vendored under PoseAnalysis/Utilities)
from Pose2Sim import calibration as pose_calib

from pose3danalysis.gui.utils.helpers import cam_id
from pose3danalysis.gui.utils.calibration_utils import annotate_chessboard_files


def process_intrinsic_calibration(
    workspace: Path,
    result_dir: Path,
    videos: List[str],
    num_cams: int,
    extract_every_sec: int,
    cols: int,
    rows: int,
    square_size_mm: float,
    log_callback: Optional[Callable[[str, str], None]] = None,
    popup_info_callback: Optional[Callable[[str, str], None]] = None,
    popup_error_callback: Optional[Callable[[str, str], None]] = None,
    after_callback: Optional[Callable[[int, Callable], None]] = None,
    render_all_callback: Optional[Callable[[], None]] = None,
    lens_toml_path_setter: Optional[Callable[[str], None]] = None,
    btn_run_intrinsic_setter: Optional[Callable[[str], None]] = None,
    build_corner_overlays_callback: Optional[Callable[[tuple[int, int]], None]] = None,
) -> tuple[Optional[Path], List[str]]:
    """
    Process intrinsic calibration.

    Returns:
        Tuple of (output_lens_toml_path, result_lines)
    """
    try:
        # extract frames for each camera into ws/intrinsics/<cam_id>
        extracted_counts = []
        extracted_files_per_cam: List[List[Path]] = []
        for i in range(num_cams):
            cid = cam_id(i)
            cam_dir = workspace / "intrinsics" / cid
            ensure_dir(cam_dir)
            for p in cam_dir.glob("*.png"):
                p.unlink(missing_ok=True)
            files = extract_frames_every_n_sec(videos[i], cam_dir, extract_every_sec)
            extracted_files_per_cam.append(files)
            extracted_counts.append(len(files))

        cfg = dict(
            overwrite_intrinsics=True,
            intrinsics_extension="png",
            extract_every_N_sec=extract_every_sec,
            show_detection_intrinsics=False,
            intrinsics_corners_nb=[cols, rows],
            intrinsics_square_size=square_size_mm,
            intrinsics_marker_size=None,
            intrinsics_aruco_dict=None,
        )

        ret, C, S, D, K, _, _ = pose_calib.calibrate_intrinsics(str(workspace), cfg, save_debug_images=True)

        out_lens = result_dir / "lens_calibration.toml"
        write_lens_calibration_toml(out_lens, C, S, K, D)
        
        if lens_toml_path_setter:
            after_callback(0, lambda: lens_toml_path_setter(str(out_lens))) if after_callback else lens_toml_path_setter(str(out_lens))

        # save overlay images into result dir (per camera)
        for i in range(num_cams):
            cid = cam_id(i)
            ov = result_dir / f"{cid}_chessboard_overlay"
            ensure_dir(ov)
            annotate_chessboard_files(extracted_files_per_cam[i], ov, (cols, rows))

        # build preview corner overlays (session only)
        if build_corner_overlays_callback:
            build_corner_overlays_callback((cols, rows))

        lines = [
            "Intrinsic calibration finished.",
            f"Saved: {out_lens}",
            f"Extracted (every {extract_every_sec} sec): " + ", ".join([f"{cam_id(i)}={extracted_counts[i]}" for i in range(num_cams)]),
            "",
            "Reprojection error (px):"
        ]
        for cam, err in zip(C, ret):
            try:
                lines.append(f" - {cam}: {float(err):.4f} px")
            except Exception:
                lines.append(f" - {cam}: {err}")

        if log_callback:
            log_callback("Intrinsic calibration results:\n" + "\n".join(lines))
        
        if render_all_callback and after_callback:
            after_callback(0, render_all_callback)
        elif render_all_callback:
            render_all_callback()
        
        if popup_info_callback:
            popup_info_callback("Intrinsic Calibration", "\n".join(lines))

        return out_lens, lines

    except Exception as e:
        error_msg = f"Intrinsic calibration failed: {e}"
        if log_callback:
            log_callback(error_msg, level="error")
        if popup_error_callback:
            popup_error_callback("Intrinsic Calibration Error", f"{e}\n\n{traceback.format_exc()}")
        return None, []
    finally:
        if btn_run_intrinsic_setter and after_callback:
            after_callback(0, lambda: btn_run_intrinsic_setter("normal"))
        elif btn_run_intrinsic_setter:
            btn_run_intrinsic_setter("normal")

