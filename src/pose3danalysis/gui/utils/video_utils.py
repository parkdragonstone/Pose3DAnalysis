# -*- coding: utf-8 -*-
"""Video-related utilities.

Extracted from gui.tabs.motion_analysis_tab to improve code organization.
"""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np


class CamStream:
    """Video stream wrapper with FPS estimation."""

    def __init__(self, path: str):
        self.path = path
        self.cap = cv2.VideoCapture(path)
        self.ok = self.cap.isOpened()
        # OpenCV reported FPS can be wrong (e.g., VFR videos often show 25).
        # We try to estimate FPS from timestamps to better match actual playback.
        self.fps_reported = float(self.cap.get(cv2.CAP_PROP_FPS) or 0.0)
        self.n_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        self.fps_estimated = self._estimate_fps_from_pos_msec()

        self.fps = self._choose_fps(self.fps_reported, self.fps_estimated)
        if self.fps <= 0:
            self.fps = 30.0

    def _estimate_fps_from_pos_msec(self) -> float:
        """Best-effort FPS estimation using CAP_PROP_POS_MSEC at last frame."""
        if not self.ok or self.n_frames <= 5:
            return 0.0
        try:
            # Save current pos
            cur = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES) or 0)
            last = max(0, self.n_frames - 1)
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, last)
            ok, _ = self.cap.read()
            if not ok:
                return 0.0
            dur_ms = float(self.cap.get(cv2.CAP_PROP_POS_MSEC) or 0.0)
            # Restore
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, cur)
            if dur_ms <= 0.0:
                return 0.0
            fps = float(last) / (dur_ms / 1000.0)
            if 1.0 <= fps <= 240.0:
                return fps
            return 0.0
        except Exception:
            return 0.0

    @staticmethod
    def _choose_fps(fps_reported: float, fps_est: float) -> float:
        # Prefer a valid reported fps if it is close to estimated.
        if fps_reported and fps_reported > 1.0 and fps_est and fps_est > 1.0:
            if abs(fps_reported - fps_est) <= 3.0:
                return fps_reported
            # If OpenCV reports common wrong values (25/29.97/30) but estimate is far,
            # trust the estimate.
            if fps_reported in (24.0, 25.0, 29.97, 30.0) and abs(fps_reported - fps_est) > 5.0:
                return fps_est
            # Otherwise still prefer reported (safer)
            return fps_reported
        if fps_reported and fps_reported > 1.0:
            return fps_reported
        if fps_est and fps_est > 1.0:
            return fps_est
        return 0.0

    def release(self):
        try:
            self.cap.release()
        except Exception:
            pass

    def read_at(self, idx: int) -> Optional[np.ndarray]:
        if not self.ok:
            return None
        if self.n_frames > 0:
            idx = int(idx) % self.n_frames
        else:
            idx = max(0, int(idx))

        self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = self.cap.read()
        if not ok:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = self.cap.read()
        return frame if ok else None


def bgr_to_rgb(bgr):
    """Safely convert BGR image to RGB. Returns None if input is empty."""
    if bgr is None:
        return None
    try:
        if isinstance(bgr, np.ndarray) and bgr.size == 0:
            return None
    except Exception:
        pass

    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

