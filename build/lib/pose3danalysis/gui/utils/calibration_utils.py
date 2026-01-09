# -*- coding: utf-8 -*-
"""Calibration-related utility functions.

Extracted from gui.main_app to improve code organization.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, List, Tuple

import cv2
import numpy as np

from pose3danalysis.gui.utils.helpers import cam_id


def parse_obj_points_mm(text: str) -> np.ndarray:
    """Parse object 3D points from text (x,y,z per line).
    
    Heuristic: if values are small (<=20), assume meters and convert to mm.
    """
    pts = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = [p.strip() for p in line.replace("\t", ",").split(",") if p.strip() != ""]
        if len(parts) != 3:
            raise ValueError(f"Each line must have 3 values (x,y,z). Got: {line}")
        pts.append([float(parts[0]), float(parts[1]), float(parts[2])])
    arr = np.asarray(pts, dtype=np.float32).reshape(-1, 3)
    if arr.size == 0:
        return arr

    # heuristic: if values are small, likely meters -> convert to mm
    mx = float(np.max(np.abs(arr)))
    if mx <= 20.0:
        arr = arr * 1000.0
    return arr


def estimate_depth_m(
    method: str,
    scene_obj_points_mm: Optional[np.ndarray],
    rvec,
    tvec,
    cols: int,
    rows: int,
    square_mm: float,
    board_position: str,
) -> Optional[float]:
    """Estimate a representative depth (meters) of the 3D points used for reprojection.

    Using ||t|| can inflate depth if the board origin is off-axis or if the scale is wrong.
    A better approximation is the median Z of the 3D points transformed into the camera frame:

        X_cam = R * X_obj + t
        depth ~= median(X_cam.z)

    Returns None if depth cannot be estimated.
    """
    try:
        import numpy as _np
        import cv2
    except Exception:
        return None

    try:
        # rvec -> R
        rvec = _np.asarray(rvec, dtype=_np.float64).reshape(3, 1)
        R, _ = cv2.Rodrigues(rvec)
        t = _np.asarray(tvec, dtype=_np.float64).reshape(3, 1)

        if method == "scene":
            if scene_obj_points_mm is None:
                return None
            X = (_np.asarray(scene_obj_points_mm, dtype=_np.float64) / 1000.0).reshape(-1, 3)  # m

        else:  # "board"
            square_m = float(square_mm) / 1000.0
            # Pose2Sim's calibration.py generates board points this way.
            if board_position == "horizontal":
                obj = _np.zeros((cols * rows, 3), _np.float64)
                obj[:, :2] = _np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
                obj[:, :2] *= square_m
                X = obj
            else:  # vertical (default)
                obj = _np.zeros((cols * rows, 3), _np.float64)
                obj[:, [0, 2]] = _np.mgrid[0:cols, 0:rows][::-1].T.reshape(-1, 2)
                obj[:, [0, 2]] *= square_m
                X = obj

        Xc = (R @ X.T) + t  # (3,N)
        z = _np.asarray(Xc[2, :], dtype=_np.float64).reshape(-1)
        # Keep finite values
        z = z[_np.isfinite(z)]
        if z.size == 0:
            return None
        depth = float(_np.median(z))
        # Depth should be positive; take abs if sign convention differs.
        return abs(depth)
    except Exception:
        return None


def draw_scene_overlay(bgr: np.ndarray, clicked_xy: np.ndarray, projected_xy: np.ndarray) -> np.ndarray:
    """Draw clicked (red) and projected (green) points on BGR image."""
    clicked_xy = np.asarray(clicked_xy, dtype=np.float32).reshape(-1, 2)
    projected_xy = np.asarray(projected_xy, dtype=np.float32).reshape(-1, 2)
    for (x, y) in clicked_xy:
        cv2.circle(bgr, (int(round(x)), int(round(y))), 6, (0, 0, 255), 2)  # red
    for (x, y) in projected_xy:
        cv2.circle(bgr, (int(round(x)), int(round(y))), 4, (0, 255, 0), -1)  # green
    return bgr


def annotate_chessboard_files(src_files: List[Path], out_dir: Path, pattern_size: tuple[int, int]) -> List[Path]:
    """Annotate chessboard corners on images and save to output directory.
    
    Optimized with parallel processing for better performance.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    
    cols, rows = pattern_size
    
    def _process_single(p: Path) -> Path:
        """Process a single image file."""
        bgr = cv2.imread(str(p))
        if bgr is None:
            return p
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        ok, corners = cv2.findChessboardCorners(gray, (cols, rows))
        if ok:
            crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), crit)
            cv2.drawChessboardCorners(bgr, (cols, rows), corners2, ok)
        out_path = out_dir / p.name
        cv2.imwrite(str(out_path), bgr)
        return out_path
    
    # Use parallel processing for better performance
    outs: List[Path] = []
    max_workers = min(len(src_files), 4)  # Limit to 4 threads to avoid overwhelming the system
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_path = {executor.submit(_process_single, p): p for p in src_files}
        # Maintain order by using a dict to track original index
        path_to_index = {p: i for i, p in enumerate(src_files)}
        results = {}
        for future in as_completed(future_to_path):
            original_path = future_to_path[future]
            try:
                result_path = future.result()
                results[path_to_index[original_path]] = result_path
            except Exception:
                results[path_to_index[original_path]] = original_path
        
        # Reconstruct in original order
        outs = [results[i] for i in sorted(results.keys())]
    
    return outs


def write_image_points_json_for_scene(
    ws: Path,
    num_cams: int,
    video_paths: List[str],
    scene_obj_points_mm: np.ndarray,
    scene_img_points_px: List[np.ndarray],
) -> None:
    """Write Image_points.json for scene-based extrinsic calibration."""
    if scene_obj_points_mm is None:
        raise RuntimeError("Paste object points first.")
    if not scene_img_points_px or any(p is None for p in scene_img_points_px):
        raise RuntimeError("Pick scene points first.")

    obj_m = (scene_obj_points_mm / 1000.0).tolist()  # meters
    extr = []
    for i in range(num_cams):
        extr.append({
            "cam_name": Path(video_paths[i]).name,
            "image_points_2d": scene_img_points_px[i].tolist(),
            "object_points_3d": obj_m,
        })
    data = {"intrinsics": [], "extrinsics": extr}
    (ws / "Image_points.json").write_text(json.dumps(data, indent=2), encoding="utf-8")


def px_to_mm(err_px: float, K_i, rvec, tvec, obj_m: Optional[np.ndarray]) -> Optional[float]:
    """Convert reprojection error from pixels to millimeters.
    
    Uses Z (median depth of calibration points in camera coordinates) and fx:
        mm_per_px ≈ (Z * 1000) / fx
    """
    if obj_m is None or obj_m.size == 0:
        return None
    try:
        K_i = np.asarray(K_i, dtype=np.float64)
        fx = float(K_i[0, 0])
        if not np.isfinite(fx) or fx <= 0:
            return None
        Rm, _ = cv2.Rodrigues(rvec)
        Xc = (Rm @ obj_m.T + tvec).T  # (N,3)
        z = float(np.nanmedian(Xc[:, 2]))
        # Use absolute depth; depending on coordinate convention Z can be negative.
        z = abs(z) if np.isfinite(z) else z
        if (not np.isfinite(z)) or z <= 1e-9:
            z = float(np.linalg.norm(tvec))
        z = abs(z) if np.isfinite(z) else z
        if (not np.isfinite(z)) or z <= 1e-9:
            return None
        mm_per_px = (z * 1000.0) / fx
        return float(err_px) * mm_per_px
    except Exception:
        return None


def to_rvec(R) -> np.ndarray:
    """Convert rotation matrix to Rodrigues vector."""
    R = np.asarray(R, dtype=np.float64)
    if R.shape == (3, 3):
        rv, _ = cv2.Rodrigues(R)
        return rv.reshape(3, 1)
    return R.reshape(3, 1)

