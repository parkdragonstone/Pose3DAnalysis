# -*- coding: utf-8 -*-
"""Extrinsic calibration processing logic.

Extracted from gui.tabs.calibration_tab to improve code organization.
"""

from __future__ import annotations

import traceback
from pathlib import Path
from typing import List, Callable, Optional

import cv2
import numpy as np

from pose3danalysis.core.calib_io import load_lens_calibration_toml, write_camera_calibration_toml
from pose3danalysis.core.utils import ensure_dir, copy_file

# Pose2Sim calibration (vendored under PoseAnalysis/Utilities)
from Pose2Sim import calibration as pose_calib

from pose3danalysis.gui.utils.helpers import cam_id
from pose3danalysis.gui.utils.calibration_utils import (
    write_image_points_json_for_scene,
    px_to_mm,
    to_rvec,
    draw_scene_overlay,
)


def process_extrinsic_calibration(
    workspace: Path,
    result_dir: Path,
    videos: List[str],
    lens_toml_path: str,
    method: str,  # "board" | "scene"
    num_cams: int,
    board_position: str,
    cols: int,
    rows: int,
    square_size_mm: float,
    scene_obj_points_mm: Optional[np.ndarray],
    scene_img_points_px: List[Optional[np.ndarray]],
    preview_files_getter: Callable[[], List[List[Path]]],
    pos_getter: Callable[[], List[int]],
    log_callback: Optional[Callable[[str, str], None]] = None,
    popup_info_callback: Optional[Callable[[str, str], None]] = None,
    popup_error_callback: Optional[Callable[[str, str], None]] = None,
    after_callback: Optional[Callable[[int, Callable], None]] = None,
    render_all_callback: Optional[Callable[[], None]] = None,
    last_extr_setters: Optional[dict] = None,
) -> tuple[Optional[Path], List[str], Optional[dict]]:
    """
    Process extrinsic calibration.

    Returns:
        Tuple of (output_camera_toml_path, result_lines, last_extrinsic_dict)
        last_extrinsic_dict contains: {'R': R_list, 'T': T_list, 'K': K2, 'D': D2}
    """
    try:
        lens_path = Path(lens_toml_path).expanduser().resolve()
        if not lens_path.exists():
            raise RuntimeError("Invalid lens_calibration.toml path.")

        C, S, K, D = load_lens_calibration_toml(lens_path)

        # Copy extrinsic videos into ws/extrinsics/<cam_id>/
        exts = [Path(p).suffix.lower().lstrip(".") for p in videos]
        ext0 = exts[0] or "mp4"
        if any(e != ext0 for e in exts):
            if log_callback:
                log_callback(f"Warning: video extensions differ: {exts}. Using '{ext0}' for Pose2Sim search.", level="warning")

        for i in range(num_cams):
            cid = cam_id(i)
            dst = workspace / "extrinsics" / cid / Path(videos[i]).name
            copy_file(videos[i], dst)

        cfg = dict(
            calculate_extrinsics=True,
            extrinsics_method=method,
            extrinsics_extension=ext0,
            show_reprojection_error=False,
            board=dict(
                board_position=board_position,
                extrinsics_corners_nb=[cols, rows],
                extrinsics_square_size=square_size_mm,
            ),
            scene=dict(
                object_coords_3d=[],
                show_reprojection_error=False,
            )
        )

        if method == "scene":
            write_image_points_json_for_scene(
                workspace,
                num_cams,
                videos,
                scene_obj_points_mm,
                scene_img_points_px,
            )
            cfg["scene"]["object_coords_3d"] = (scene_obj_points_mm / 1000.0).tolist()

        ret, C2, S2, D2, K2, R_list, T_list = pose_calib.calibrate_extrinsics(
            str(workspace), cfg, C, S, K, D, save_debug_images=True
        )

        out_cam = result_dir / "camera_calibration.toml"
        write_camera_calibration_toml(out_cam, C2, S2, K2, D2, R_list, T_list)

        # Keep last extrinsic solution for preview overlay
        last_extr = {
            'R': R_list,
            'T': T_list,
            'K': K2,
            'D': D2,
        }
        if last_extr_setters:
            if 'R' in last_extr_setters:
                last_extr_setters['R'](R_list)
            if 'T' in last_extr_setters:
                last_extr_setters['T'](T_list)
            if 'K' in last_extr_setters:
                last_extr_setters['K'](K2)
            if 'D' in last_extr_setters:
                last_extr_setters['D'](D2)

        # If the overlay checkbox is already checked, refresh previews
        if render_all_callback and after_callback:
            after_callback(0, render_all_callback)
        elif render_all_callback:
            render_all_callback()

        # Save overlay result images into result folder
        _save_extrinsic_result_images(
            result_dir,
            method,
            C2,
            K2,
            D2,
            R_list,
            T_list,
            num_cams,
            cols,
            rows,
            scene_obj_points_mm,
            scene_img_points_px,
            preview_files_getter,
            pos_getter,
        )

        # Summarize
        lines = [
            "Extrinsic calibration finished.",
            f"Saved: {out_cam}",
            "",
            "Reprojection error (RMSE per point):"
        ]

        # Approximate conversion from px -> mm at the calibration object depth
        obj_m = None  # (N,3) meters
        if method == "scene" and scene_obj_points_mm is not None:
            obj_m = (np.asarray(scene_obj_points_mm, dtype=np.float64) / 1000.0).reshape(-1, 3)
        elif method == "board":
            sq_m = float(square_size_mm) / 1000.0
            obj_m = np.asarray(
                [[c * sq_m, r * sq_m, 0.0] for r in range(rows) for c in range(cols)],
                dtype=np.float64,
            )

        for i, (cam_name, err) in enumerate(zip(C2, ret)):
            try:
                err_px = float(err)
                err_mm = None
                if i < len(K2) and i < len(R_list) and i < len(T_list):
                    rvec = to_rvec(R_list[i])
                    tvec = np.asarray(T_list[i], dtype=np.float64).reshape(3, 1)
                    err_mm = px_to_mm(err_px, K2[i], rvec, tvec, obj_m)

                if err_mm is None:
                    lines.append(f" - {cam_name}: {err_px:.4f} px")
                else:
                    lines.append(f" - {cam_name}: {err_px:.4f} px | {err_mm:.2f} mm")
            except Exception:
                lines.append(f" - {cam_name}: {err}")

        if log_callback:
            log_callback("Extrinsic calibration results:\n" + "\n".join(lines))
        if popup_info_callback:
            popup_info_callback("Extrinsic Calibration", "\n".join(lines))

        return out_cam, lines, last_extr

    except Exception as e:
        error_msg = f"Extrinsic calibration failed: {e}"
        if log_callback:
            log_callback(error_msg, level="error")
        if popup_error_callback:
            popup_error_callback("Extrinsic Calibration Error", f"{e}\n\n{traceback.format_exc()}")
        return None, [], None


def _save_extrinsic_result_images(
    result_dir: Path,
    method: str,
    C2: List[str],
    K2: List,
    D2: List,
    R_list: List,
    T_list: List,
    num_cams: int,
    cols: int,
    rows: int,
    scene_obj_points_mm: Optional[np.ndarray],
    scene_img_points_px: List[Optional[np.ndarray]],
    preview_files_getter: Callable[[], List[List[Path]]],
    pos_getter: Callable[[], List[int]],
):
    """
    Save 'overlapped' visualization images into result_dir.

    - board: draw chessboard corners on the current preview frame for each cam
    - scene: draw clicked (red) vs projected (green) points on the current preview frame for each cam
    """
    preview_files = preview_files_getter()
    if any(not files for files in preview_files):
        return

    pat = (cols, rows)

    if method == "scene":
        if scene_obj_points_mm is None or not scene_img_points_px or any(p is None for p in scene_img_points_px):
            return
        obj_m = (np.asarray(scene_obj_points_mm, dtype=np.float64) / 1000.0).reshape(-1, 3)

    pos = pos_getter()
    for i in range(num_cams):
        j = max(0, min(pos[i], len(preview_files[i]) - 1))
        img_path = preview_files[i][j]
        bgr = cv2.imread(str(img_path))
        if bgr is None:
            continue

        if method == "scene":
            cam_name = C2[i] if i < len(C2) else cam_id(i)
            rvec = to_rvec(R_list[i])
            tvec = np.asarray(T_list[i], dtype=np.float64).reshape(3, 1)
            K = np.asarray(K2[i], dtype=np.float64)
            D = np.asarray(D2[i], dtype=np.float64).reshape(-1, 1) if np.asarray(D2[i]).ndim == 1 else np.asarray(D2[i], dtype=np.float64)
            proj, _ = cv2.projectPoints(obj_m, rvec, tvec, K, D)
            proj = proj.reshape(-1, 2)
            out = result_dir / f"{cam_id(i)}_scene_overlay.png"
            bgr = draw_scene_overlay(bgr, scene_img_points_px[i], proj)
            cv2.imwrite(str(out), bgr)
        else:
            # board: draw corners
            g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            ok, corners = cv2.findChessboardCorners(g, pat)
            if ok:
                crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                corners2 = cv2.cornerSubPix(g, corners, (11, 11), (-1, -1), crit)
                cv2.drawChessboardCorners(bgr, pat, corners2, ok)
            out = result_dir / f"{cam_id(i)}_board_overlay.png"
            cv2.imwrite(str(out), bgr)

