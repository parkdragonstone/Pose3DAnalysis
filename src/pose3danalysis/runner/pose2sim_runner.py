"""Pose2Sim pipeline runner (called from GUI).

Currently implements the first stage: YOLO -> RTMPose 2D keypoint extraction.
Outputs OpenPose-like JSON frames under: <project_dir>/pose/<video_name>_json/

This runner is intentionally conservative and only orchestrates existing Pose2Sim modules.
"""

from __future__ import annotations

import os

# Prevent Qt/PyQt backends from initializing inside the Tkinter GUI process/thread.
# This avoids: "WARNING: QApplication was not created in the main() thread."
os.environ.setdefault("POSE2SIM_DISABLE_QT", "1")
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("QT_AUTO_SCREEN_SCALE_FACTOR", "0")
os.environ.setdefault("QT_ENABLE_HIGHDPI_SCALING", "0")

import shutil
from pathlib import Path
from typing import Callable, Optional

from pose3danalysis.runner.config_builder import build_pose2sim_config


def _log(log: Optional[Callable[[str], None]], msg: str) -> None:
    if log:
        try:
            log(msg)
        except Exception:
            pass


def _stage_videos_as_project(video_dir: Path, vid_ext: str, log: Optional[Callable[[str], None]]) -> Path:
    """Ensure <project_dir>/videos exists and contains input videos.

    Pose2Sim.poseEstimation.estimate_pose_all expects:
      project_dir/videos/*.<ext>
    We use symbolic links (or direct files) instead of copying to save space and time.
    """
    project_dir = video_dir
    videos_dir = project_dir / "videos"
    videos_dir.mkdir(parents=True, exist_ok=True)

    # Check if videos already exist in videos_dir
    existing = list(videos_dir.glob(f"*{vid_ext}"))
    if existing:
        _log(log, f"[Runner] Found {len(existing)} video(s) in: {videos_dir}")
        return project_dir

    # Find source videos in project_dir
    src_videos = list(project_dir.glob(f"*{vid_ext}"))

    # Also allow common video extensions if user selected e.g. .mkv but config says .mp4
    if not src_videos:
        for ext in [".mp4", ".avi", ".mov", ".mkv", ".MP4", ".AVI", ".MOV", ".MKV"]:
            src_videos = list(project_dir.glob(f"*{ext}"))
            if src_videos:
                vid_ext = ext
                break

    if not src_videos:
        raise FileNotFoundError(f"No video files found in: {project_dir}")

    # Create symbolic links instead of copying (faster, saves space)
    # On Windows, use symlinks; if that fails, fall back to hard links
    linked_count = 0
    for p in src_videos:
        dst = videos_dir / p.name
        if not dst.exists():
            try:
                # Try symbolic link first (works on Windows with admin rights or developer mode)
                dst.symlink_to(p)
                linked_count += 1
            except (OSError, NotImplementedError):
                try:
                    # Fall back to hard link (works on Windows without admin)
                    dst.hardlink_to(p)
                    linked_count += 1
                except (OSError, NotImplementedError):
                    # If both fail, skip (videos might already be accessible)
                    # This can happen on Windows without proper permissions
                    pass
    
    if linked_count > 0:
        _log(log, f"[Runner] Linked {linked_count} video(s) into: {videos_dir} (using symlinks/hardlinks, no copying)")
    else:
        # If no links were created, check if videos are already in videos_dir
        if existing:
            _log(log, f"[Runner] Using existing videos in: {videos_dir}")
        else:
            _log(log, f"[Runner] Warning: Could not create links. Videos should be in: {videos_dir}")
    
    return project_dir


def run_pose2sim(
    settings: dict,
    video_dir: str,
    calib_file: str,
    output_dir: str,
    log: Optional[Callable[[str], None]] = None,
    stop_flag: Optional[Callable[[], bool]] = None,
) -> None:
    """Run Pose2Sim (partial) pipeline from GUI.

    Parameters
    ----------
    settings: dict
        Config dict produced by GUI (Pose2Sim-like).
    video_dir: str
        User-selected folder that contains camera videos.
    calib_file: str
        Selected calibration toml path. (Not used in pose stage yet.)
    output_dir: str
        Output base directory. For now we use this as project_dir when different from video_dir.
    """
    import sys
    _log(log, f"[Runner] python: {sys.executable} ({sys.version.split()[0]})")
    cfg = build_pose2sim_config(settings)

    vdir = Path(video_dir).expanduser().resolve()
    out = Path(output_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)

    # Determine project_dir:
    # - Always use output_dir as project_dir to ensure results are saved in the correct location
    # - video_dir is used only for reading videos (via config['paths']['video_dir'])
    project_dir = out

    cfg.setdefault("project", {})
    cfg["project"]["project_dir"] = str(project_dir)

    # Keep some references for downstream stages
    cfg.setdefault("paths", {})
    cfg["paths"]["video_dir"] = str(vdir)
    cfg["paths"]["calibration_file"] = str(Path(calib_file).expanduser().resolve())
    cfg["paths"]["output_dir"] = str(out)

    _log(log, f"[Runner] video_dir: {cfg['paths']['video_dir']}")
    _log(log, f"[Runner] calibration: {cfg['paths']['calibration_file']}")
    _log(log, f"[Runner] output_dir: {cfg['paths']['output_dir']}")
    _log(log, f"[Runner] project_dir: {cfg['project']['project_dir']}")
    _log(log, "[Runner] Stage 1: YOLO -> RTMPose (2D pose estimation)")

    # Verify videos exist in video_dir
    vid_ext = str(cfg.get("pose", {}).get("vid_img_extension", ".mp4"))
    src_videos = list(vdir.glob(f"*{vid_ext}"))
    if not src_videos:
        # Try common extensions
        for ext in [".mp4", ".avi", ".mov", ".mkv", ".MP4", ".AVI", ".MOV", ".MKV"]:
            src_videos = list(vdir.glob(f"*{ext}"))
            if src_videos:
                break
    if not src_videos:
        raise FileNotFoundError(f"No video files found in: {vdir}")
    _log(log, f"[Runner] Found {len(src_videos)} video(s) in: {vdir}")

    # Helper to check stop flag
    def _check_stop():
        if stop_flag and stop_flag():
            raise RuntimeError("Processing stopped by user")
    
    # Run pose estimation
    _check_stop()
    from Pose2Sim.poseEstimation import estimate_pose_all
    estimate_pose_all(cfg)
    _log(log, "[Runner] Stage 1 complete: pose JSON written under <project_dir>/pose/")
    
    # ---------------- Pipeline stages (GUI toggles) ----------------
    ui = cfg.get("_ui", {}) or {}
    do_synchronization = bool(ui.get("do_synchronization", False))
    do_person_association = bool(ui.get("do_person_association", False))
    do_triangulation = bool(ui.get("do_triangulation", True))
    do_filtering = bool(ui.get("do_filtering", True))
    do_marker_augmentation = bool(ui.get("do_marker_augmentation", True))
    do_kinematic = bool(ui.get("do_kinematic", True))
    kinematic_source = str(ui.get("kinematic_source", "augmented")).lower().strip()

    _log(
        log,
        "[Runner] Pipeline: "
        f"sync={do_synchronization}, person_assoc={do_person_association}, "
        f"triangulation={do_triangulation}, filtering={do_filtering}, "
        f"marker_aug={do_marker_augmentation}, kinematics={do_kinematic} "
        f"(kinematic_source={kinematic_source})",
    )

    # Stage 2: Synchronization (optional)
    if do_synchronization:
        _check_stop()
        _log(log, "[Runner] Stage 2: Synchronization")
        try:
            from Pose2Sim.synchronization import synchronize_cams_all
            synchronize_cams_all(cfg)
        except RuntimeError as e:
            if "stopped by user" in str(e):
                raise
            raise RuntimeError(f"Synchronization failed: {e}") from e
        _log(log, "[Runner] Stage 2 complete.")

    # Stage 3: Person association (optional)
    if do_person_association:
        _check_stop()
        _log(log, "[Runner] Stage 3: Person association")
        try:
            from Pose2Sim.personAssociation import associate_all
            associate_all(cfg)
        except RuntimeError as e:
            if "stopped by user" in str(e):
                raise
            raise RuntimeError(f"Person association failed: {e}") from e
        _log(log, "[Runner] Stage 3 complete.")

    # Stage 4: Triangulation
    if do_triangulation:
        _check_stop()
        _log(log, "[Runner] Stage 4: Triangulation")
        try:
            from Pose2Sim.triangulation import triangulate_all
            triangulate_all(cfg)
        except RuntimeError as e:
            if "stopped by user" in str(e):
                raise
            raise RuntimeError(f"Triangulation failed: {e}") from e
        _log(log, "[Runner] Stage 4 complete. Outputs under <project_dir>/pose-3d/")

    # Stage 5: Filtering
    if do_filtering:
        _check_stop()
        _log(log, "[Runner] Stage 5: Filtering")
        try:
            from Pose2Sim.filtering import filter_all
            filter_all(cfg)
        except RuntimeError as e:
            if "stopped by user" in str(e):
                raise
            raise RuntimeError(f"Filtering failed: {e}") from e
        _log(log, "[Runner] Stage 5 complete. Outputs under <project_dir>/pose-3d/")

    # Stage 6: Marker augmentation
    if do_marker_augmentation:
        _check_stop()
        _log(log, "[Runner] Stage 6: Marker augmentation")
        try:
            from Pose2Sim.markerAugmentation import augment_markers_all
            augment_markers_all(cfg)
        except RuntimeError as e:
            if "stopped by user" in str(e):
                raise
            raise RuntimeError(f"Marker augmentation failed: {e}") from e
        _log(log, "[Runner] Stage 6 complete. Outputs under <project_dir>/pose-3d/")

    # Stage 7: Kinematics (OpenSim)
    if do_kinematic:
        _check_stop()
        cfg.setdefault("kinematics", {})
        # UI shortcut: whether to use augmented (LSTM) markers if available
        if kinematic_source in ("raw", "triangulated"):
            cfg["kinematics"]["use_augmentation"] = False
        else:
            cfg["kinematics"]["use_augmentation"] = True

        _log(log, "[Runner] Stage 7: Kinematics (OpenSim)")
        try:
            from Pose2Sim.kinematics import kinematics_all
            kinematics_all(cfg)
        except RuntimeError as e:
            if "stopped by user" in str(e):
                raise
            raise RuntimeError(f"Kinematics failed: {e}") from e
        _log(log, "[Runner] Stage 7 complete. Outputs under <project_dir>/kinematics/")

