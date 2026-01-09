"""Camera state management for calibration."""

from pathlib import Path
from typing import List, Optional
import numpy as np

from pose3danalysis.core.zoom_preview import ZoomPreview
import tkinter.ttk as ttk


class CameraState:
    """Manages state for multiple cameras."""

    def __init__(self, n: int):
        """
        Initialize camera state for n cameras.

        Args:
            n: Number of cameras
        """
        self.num_cams = n
        self.video_paths: List[str] = [""] * n
        self.playing: List[bool] = [False] * n
        self.pos: List[int] = [0] * n
        self.preview_files: List[List[Path]] = [[] for _ in range(n)]
        self.corner_overlay_files: List[List[Path]] = [[] for _ in range(n)]
        self._last_rgb: List[Optional[np.ndarray]] = [None] * n
        self.scene_img_points_px: List[Optional[np.ndarray]] = [None] * n

        # UI handles (created in _build_preview_grid)
        self.preview_widgets: List[ZoomPreview] = []
        self.scale_widgets: List[ttk.Scale] = []
        self.frame_labels: List[ttk.Label] = []

    def reset(self, n: int):
        """Reset state for new number of cameras."""
        self.__init__(n)

