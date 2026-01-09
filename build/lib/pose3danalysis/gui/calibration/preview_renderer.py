"""Preview rendering and playback for calibration."""

from pathlib import Path
from typing import List, Optional
import cv2
import numpy as np

from pose3danalysis.core.zoom_preview import ZoomPreview
from pose3danalysis.gui.utils.helpers import cam_id


class PreviewRenderer:
    """Handles preview rendering and playback."""

    def __init__(self, camera_state, show_corners_preview, show_scene_overlay_preview,
                 extrinsic_method, scene_obj_points_mm, scene_img_points_px):
        """
        Initialize preview renderer.

        Args:
            camera_state: CameraState instance
            show_corners_preview: BooleanVar for showing corners
            show_scene_overlay_preview: BooleanVar for showing scene overlay
            extrinsic_method: StringVar for extrinsic method
            scene_obj_points_mm: Optional numpy array of scene object points
            scene_img_points_px: List of optional numpy arrays for scene image points
        """
        self.camera_state = camera_state
        self.show_corners_preview = show_corners_preview
        self.show_scene_overlay_preview = show_scene_overlay_preview
        self.extrinsic_method = extrinsic_method
        self.scene_obj_points_mm = scene_obj_points_mm
        self.scene_img_points_px = scene_img_points_px
        self._last_extr_R = None
        self._last_extr_T = None
        self._last_extr_K = None
        self._last_extr_D = None

    def set_extrinsic_results(self, R_list, T_list, K_list, D_list):
        """Set extrinsic calibration results for overlay."""
        self._last_extr_R = R_list
        self._last_extr_T = T_list
        self._last_extr_K = K_list
        self._last_extr_D = D_list

    def choose_preview_image(self, idx: int) -> Path:
        """Choose which preview image to show (with or without overlay)."""
        i = idx
        j = max(0, min(self.camera_state.pos[i], len(self.camera_state.preview_files[i]) - 1))
        if (self.show_corners_preview.get() and 
            self.camera_state.corner_overlay_files[i] and 
            j < len(self.camera_state.corner_overlay_files[i])):
            return self.camera_state.corner_overlay_files[i][j]
        return self.camera_state.preview_files[i][j]

    def render_cam(self, idx: int, root):
        """Render a single camera preview."""
        if not self.camera_state.preview_files[idx]:
            return

        j = max(0, min(self.camera_state.pos[idx], len(self.camera_state.preview_files[idx]) - 1))
        path = self.choose_preview_image(idx)
        bgr = cv2.imread(str(path))
        if bgr is None:
            return

        # Optional overlay: clicked (red) and projected (green) points (scene extrinsic)
        if self.show_scene_overlay_preview.get() and self.extrinsic_method.get() == "scene":
            try:
                if (self.scene_obj_points_mm is not None and 
                    self.scene_img_points_px and 
                    idx < len(self.scene_img_points_px)):
                    clicked = self.scene_img_points_px[idx]
                    if clicked is not None:
                        clicked = np.asarray(clicked, dtype=np.float32).reshape(-1, 2)
                        proj = None
                        if (self._last_extr_R is not None and 
                            self._last_extr_T is not None and
                            idx < len(self._last_extr_R) and 
                            idx < len(self._last_extr_T)):
                            obj_m = (np.asarray(self.scene_obj_points_mm, dtype=np.float64) / 1000.0).reshape(-1, 3)
                            R = np.asarray(self._last_extr_R[idx], dtype=np.float64)
                            if R.shape == (3, 3):
                                rvec, _ = cv2.Rodrigues(R)
                            else:
                                rvec = R.reshape(3, 1)
                            tvec = np.asarray(self._last_extr_T[idx], dtype=np.float64).reshape(3, 1)
                            K = (np.asarray(self._last_extr_K[idx], dtype=np.float64) 
                                 if self._last_extr_K is not None else None)
                            D = (np.asarray(self._last_extr_D[idx], dtype=np.float64) 
                                 if self._last_extr_D is not None else None)
                            if K is not None:
                                proj, _ = cv2.projectPoints(obj_m, rvec, tvec, K, D)
                                proj = proj.reshape(-1, 2)
                        if proj is not None and proj.shape[0] == clicked.shape[0]:
                            bgr = self._draw_scene_overlay(bgr, clicked, proj)
                        else:
                            # draw clicked only
                            for (x, y) in clicked:
                                cv2.circle(bgr, (int(round(x)), int(round(y))), 6, (0, 0, 255), 2)
            except Exception:
                pass

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        self.camera_state._last_rgb[idx] = rgb
        self.camera_state.preview_widgets[idx].set_rgb(rgb)
        self.camera_state.frame_labels[idx].configure(
            text=f"extracted frame {j+1}/{len(self.camera_state.preview_files[idx])}"
        )

    def render_all(self, root, num_cams):
        """Render all camera previews."""
        for i in range(num_cams):
            self.render_cam(i, root)

    def _draw_scene_overlay(self, bgr: np.ndarray, clicked_xy: np.ndarray, 
                           projected_xy: np.ndarray) -> np.ndarray:
        """Draw clicked (red) and projected (green) points."""
        clicked_xy = np.asarray(clicked_xy, dtype=np.float32).reshape(-1, 2)
        projected_xy = np.asarray(projected_xy, dtype=np.float32).reshape(-1, 2)
        for (x, y) in clicked_xy:
            cv2.circle(bgr, (int(round(x)), int(round(y))), 6, (0, 0, 255), 2)  # red
        for (x, y) in projected_xy:
            cv2.circle(bgr, (int(round(x)), int(round(y))), 4, (0, 255, 0), -1)  # green
        return bgr

