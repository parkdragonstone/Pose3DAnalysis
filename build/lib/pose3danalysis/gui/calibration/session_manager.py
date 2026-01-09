"""Session and workspace management for calibration."""

import time
import shutil
from pathlib import Path
from typing import Optional

from pose3danalysis.core.utils import ensure_dir
from pose3danalysis.gui.utils.helpers import cam_id


class SessionManager:
    """Manages session directories and workspace for calibration."""

    def __init__(self, app_dir: Path, num_cams_getter):
        """
        Initialize session manager.

        Args:
            app_dir: Application root directory
            num_cams_getter: Callable that returns current number of cameras
        """
        self.app_dir = app_dir
        self.num_cams_getter = num_cams_getter
        self.session_dir: Optional[Path] = None
        self.workspace_dir: Optional[Path] = None
        self._make_session_dir()

    def _make_session_dir(self) -> Path:
        """Create a new session directory."""
        ts = time.strftime("%Y%m%d_%H%M%S")
        session = (self.app_dir / f"_tmp_session_{ts}").resolve()
        ensure_dir(session)
        self.session_dir = session
        return session

    def cleanup(self):
        """Clean up session directory."""
        try:
            if self.session_dir and Path(self.session_dir).exists():
                shutil.rmtree(self.session_dir, ignore_errors=True)
        except Exception:
            pass

    def make_workspace(self) -> Path:
        """
        Create a per-session workspace under the session folder.
        Pose2Sim calibration expects:
          workspace/intrinsics/<cam_id>/
          workspace/extrinsics/<cam_id>/
        Also we store preview_cache and overlay caches here.
        """
        if self.session_dir is None:
            self._make_session_dir()

        ws = (self.session_dir / "workspace").resolve()
        n = self.num_cams_getter()
        for i in range(n):
            cid = cam_id(i)
            ensure_dir(ws / "intrinsics" / cid)
            ensure_dir(ws / "extrinsics" / cid)
            ensure_dir(ws / "_preview_cache" / cid)
            ensure_dir(ws / "_overlay" / "corners" / cid)
        self.workspace_dir = ws
        return ws

    def ensure_workspace(self) -> Path:
        """Ensure workspace exists, create if needed."""
        if self.workspace_dir is None:
            return self.make_workspace()
        return self.workspace_dir

    def reset_workspace(self):
        """Reset workspace (delete and recreate)."""
        try:
            if self.workspace_dir and self.workspace_dir.exists():
                shutil.rmtree(self.workspace_dir, ignore_errors=True)
        except Exception:
            pass
        self.workspace_dir = None

