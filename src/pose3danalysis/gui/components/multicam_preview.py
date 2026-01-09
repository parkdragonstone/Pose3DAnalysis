# -*- coding: utf-8 -*-
"""Multi-camera video preview widget.

Extracted from gui.motionanalysis_tab to keep that file smaller and easier to maintain.
"""

from __future__ import annotations

import os
import json
import time
from collections import OrderedDict
from pathlib import Path
from typing import Callable, Optional, List, Dict, Tuple

import tkinter as tk
from tkinter import ttk, filedialog

import cv2
import numpy as np

from pose3danalysis.core.zoom_preview import ZoomPreview
from pose3danalysis.gui.utils.video_utils import CamStream, bgr_to_rgb
from pose3danalysis.gui.utils.pose_utils import (
    infer_project_dir_from_video_folder,
    resolve_pose_json_dir,
    find_openpose_json_file,
    list_pose_json_sorted,
    build_pose_json_index_map,
    draw_openpose_overlay,
)


class MultiCamPreview(ttk.Frame):
    """
    - Upload videos (multi-select)
    - Preview grid shows all videos at once
    - Overlay is applied on the same preview (no separate window)
    - Playback uses each video FPS (based on min FPS among cams)
    - Speed control is shared with parent (0..2, 1.0 = normal)
    - Frame control is shared with parent (seek bar is placed above 3D viewer)
    """
    def __init__(
        self,
        parent,
        log: Callable[[str], None],
        frame_var: tk.IntVar,
        speed_var: tk.DoubleVar,
        on_loaded: Optional[Callable[[], None]] = None,
        on_clear: Optional[Callable[[], None]] = None,
        show_controls: bool = True,
        *args,
        **kwargs,
    ):
        # Extract show_controls from kwargs if provided (for backward compatibility)
        # This handles the case where show_controls is passed as a keyword argument
        if 'show_controls' in kwargs:
            show_controls = kwargs.pop('show_controls')
        
        super().__init__(parent, *args, **kwargs)
        self._external_log = log
        self._pending_logs = []  # buffer until log_text exists
        self._on_loaded = on_loaded
        self._on_clear = on_clear

        self.frame_var = frame_var
        self.speed_var = speed_var

        self.paths: List[str] = []
        self._raw_paths: List[str] = []
        self._overlay_video_paths: List[Optional[str]] = []
        self._using_overlay_video: bool = False
        self.streams: List[CamStream] = []
        self.previews: List[ZoomPreview] = []

        self._playing = tk.BooleanVar(value=False)
        self._overlay_enabled = tk.BooleanVar(value=False)

        self._max_frames = 0
        # external playback length (e.g., 3D-only)
        self._external_max_frames = 0
        self._external_fps = 30.0
        self._base_fps = 30.0



        # real-time playback timebase (prevents slowdown when decoding is heavy)
        self._play_t0 = None
        self._play_frame0 = 0
        self._play_speed = None
        self._play_fps = None
        # keep the original (raw) video FPS as the timebase even when switching to overlay video files
        self._raw_base_fps = None
        # remember last selected video folder (used to set Output Folder)
        self.last_video_folder: Optional[str] = None

        # pose overlay (OpenPose json under <project_dir>/pose/<stem>_json/)
        self._pose_project_dir: Optional[Path] = None
        self._pose_json_dirs: List[Optional[Path]] = []
        self._pose_json_cache: "OrderedDict[tuple[str, int], dict]" = OrderedDict()
        self._pose_cache_limit: int = 256
        self._overlay_warned_missing: bool = False

        self._pose_json_file_list_cache: dict[str, tuple[int, List[Path]]] = {}
        # Store show_controls value - ensure it's a boolean
        self._show_controls = bool(show_controls)
        
        # controls (only if show_controls is True)
        # IMPORTANT: When show_controls is False, do NOT create any control widgets
        if not self._show_controls:
            # Skip creating controls - go directly to preview grid
            pass
        else:
            top = ttk.Frame(self)
            top.pack(fill="x", pady=(0, 6))

            ttk.Button(top, text="Upload Video Folder…", command=self._upload_video_folder).pack(side="left")
            ttk.Button(top, text="Clear", command=self._on_clear_clicked).pack(side="left", padx=6)

            ttk.Separator(top, orient="vertical").pack(side="left", fill="y", padx=8)

            ttk.Checkbutton(
                top,
                text="Overlay",
                variable=self._overlay_enabled,
                command=self._on_overlay_toggled,
            ).pack(side="left")

            # Play button (요구사항: 왼쪽 영상의 Play 버튼으로 video+3D 동시 제어)
            self.btn_play = ttk.Button(top, text="Play", command=self.toggle_play)
            self.btn_play.pack(side="right")

            # Speed (shared)
            sp = ttk.Frame(top)
            sp.pack(side="right", padx=10)
            ttk.Label(sp, text="Speed").pack(side="left")
            self.speed_scale = ttk.Scale(
                sp,
                from_=0.0,
                to=2.0,
                variable=self.speed_var,
                orient="horizontal",
                length=120,
            )
            self.speed_scale.pack(side="left", padx=6)
            self.lbl_speed = ttk.Label(sp, text="1.00x", width=6)
            self.lbl_speed.pack(side="left")

            self.speed_var.trace_add("write", lambda *_: self._on_speed_changed())

        # preview grid
        self.grid_box = ttk.Frame(self)
        self.grid_box.pack(fill="both", expand=True)

        # when frame changes externally -> refresh
        self.frame_var.trace_add("write", lambda *_: self.refresh_current())

    def _log(self, msg: str) -> None:
        """Forward logs to parent tab/app logger.
        MultiCamPreview 자체에는 로그 박스가 없어서 외부 logger로만 보냅니다.
        """
        try:
            cb = getattr(self, "_external_log", None)
            if cb:
                cb(msg)
        except Exception:
            pass

    def _on_speed_changed(self):
        try:
            self.lbl_speed.configure(text=f"{float(self.speed_var.get()):.2f}x")
        except Exception:
            pass
        # if speed becomes 0 -> pause
        if float(self.speed_var.get()) <= 0.0:
            self.pause()


    def _on_overlay_toggled(self):
        # Switch preview source: raw video <-> pose overlay video (if exists)
        try:
            self._apply_overlay_toggle(keep_frame=True)
        except Exception as e:
            self._log(f"[Overlay][WARN] Toggle failed: {e}")
        self.refresh_current(fit=False)

    def _upload_video_folder(self):
        # folder-based upload (requested)
        self._upload_videos()

    def _upload_videos(self):
        folder = filedialog.askdirectory(title="Select a folder containing videos")
        if not folder:
            return
        folder_p = Path(folder).expanduser().resolve()
        self.last_video_folder = str(folder_p)
        # reset overlay cache (pose folder may change per trial)
        self._pose_project_dir = None
        self._pose_json_dirs = []
        self._pose_json_cache.clear()
        self._overlay_warned_missing = False
        exts = {".mp4", ".avi", ".mov", ".mkv", ".m4v", ".mpg", ".mpeg", ".wmv"}
        files = [str(p) for p in sorted(folder_p.iterdir()) if p.is_file() and p.suffix.lower() in exts]
        if not files:
            self._log(f"[Video] No video files found in: {folder_p}")
            return


        self.pause()
        self.clear(release_only=True)

        self.paths = list(files)
        self._raw_paths = list(self.paths)
        self.streams = [CamStream(p) for p in self.paths]
        self._using_overlay_video = False
        bad = [Path(self.paths[i]).name for i, s in enumerate(self.streams) if not s.ok]
        if bad:
            self._log(f"[Video][WARN] Failed to open: {bad}")

        # FPS diagnostics (OpenCV sometimes reports 25 even for 60fps sources)
        for i, s in enumerate(self.streams):
            if not s.ok:
                continue
            if s.fps_estimated and abs(s.fps_reported - s.fps_estimated) > 3.0:
                self._log(
                    f"[Video][FPS] Cam{i+1}: reported={s.fps_reported:.2f}, estimated={s.fps_estimated:.2f} -> using={s.fps:.2f}"
                )

        fps_list = [s.fps for s in self.streams if s.ok]
        self._base_fps = float((fps_list[0] if fps_list else 30.0))

        self._raw_base_fps = float(self._base_fps or 30.0)
        counts = [s.n_frames for s in self.streams if s.ok and s.n_frames > 0]
        self._max_frames = int(min(counts) if counts else 0)

        # build grid
        for w in self.grid_box.winfo_children():
            w.destroy()
        self.previews.clear()

        n = len(self.paths)
        cols = 2 if n > 1 else 1
        for i in range(n):
            lf = ttk.LabelFrame(self.grid_box, text=f"Cam {i+1}: {Path(self.paths[i]).name}")
            r = i // cols
            c = i % cols
            lf.grid(row=r, column=c, sticky="nsew", padx=4, pady=4)

            pv = ZoomPreview(lf, width=320, height=180)
            pv.pack(fill="both", expand=True)
            self.previews.append(pv)

        for c in range(cols):
            self.grid_box.columnconfigure(c, weight=1)
        for r in range((n + cols - 1) // cols):
            self.grid_box.rowconfigure(r, weight=1)

        # If Overlay is checked, try switching to pre-rendered overlay videos.
        try:
            self._apply_overlay_toggle(keep_frame=False)
        except Exception:
            pass

        self.frame_var.set(0)
        self.refresh_current(fit=True)

        self._log(f"[Video] Loaded {n} video(s). FPS≈{self._base_fps:.2f}, frames={self._max_frames or 'unknown'}")

        if self._on_loaded:
            self._on_loaded()

    def get_base_fps(self) -> float:
        return float(self._base_fps or 30.0)

    def get_max_frames(self) -> int:
        return int(self._max_frames or 0)


    def _infer_project_dir_from_video_folder(self, folder_p: Path) -> Path:
        # Heuristics: if selecting <project>/videos then project is parent
        try:
            if folder_p.name.lower() == "videos":
                return folder_p.parent
            if (folder_p / "pose").is_dir():
                return folder_p
            if (folder_p.parent / "pose").is_dir():
                return folder_p.parent
        except Exception:
            pass
        return folder_p.parent

    def _compute_overlay_video_paths(self) -> List[Optional[str]]:
        """Return overlay video path list aligned with self._raw_paths.
        Pose2Sim writes overlay videos as <project_dir>/pose/<stem>_pose.mp4
        """
        if not self._raw_paths:
            return []
        try:
            folder_p = Path(self.last_video_folder).expanduser().resolve() if self.last_video_folder else Path(self._raw_paths[0]).parent
        except Exception:
            folder_p = Path(self._raw_paths[0]).parent
        project_dir = self._infer_project_dir_from_video_folder(folder_p)
        pose_dir = project_dir / "pose"
        out = []
        for p in self._raw_paths:
            stem = Path(p).stem
            cand = pose_dir / f"{stem}_pose.mp4"
            out.append(str(cand) if cand.exists() else None)
        return out

    def _apply_overlay_toggle(self, keep_frame: bool = True):
        """If Overlay is ON and pose overlay videos exist, switch streams to those videos.
        If overlay videos are missing, fall back to raw streams and (optionally) json overlay drawing.
        """
        if not getattr(self, "streams", None):
            return
        cur_frame = int(self.frame_var.get()) if keep_frame else 0

        want_overlay = bool(self._overlay_enabled.get())
        # Always maintain raw list
        if not self._raw_paths:
            self._raw_paths = list(self.paths)

        self._overlay_video_paths = self._compute_overlay_video_paths()

        overlay_all_exist = want_overlay and self._overlay_video_paths and all(v is not None for v in self._overlay_video_paths)

        # Decide new paths
        new_paths = list(self._raw_paths)
        using_overlay_video = False
        if overlay_all_exist:
            new_paths = [v for v in self._overlay_video_paths if v is not None]
            using_overlay_video = True
        else:
            if want_overlay and (self._overlay_video_paths and any(v is not None for v in self._overlay_video_paths)) and not self._overlay_warned_missing:
                miss = [Path(self._raw_paths[i]).name for i, v in enumerate(self._overlay_video_paths) if v is None]
                self._log(f"[Overlay][WARN] Some overlay videos are missing under <project_dir>/pose/: {miss}. Falling back to raw video + JSON overlay.")
                self._overlay_warned_missing = True
            # raw video
            using_overlay_video = False

        # If no change, do nothing
        if (self.paths == new_paths) and (self._using_overlay_video == using_overlay_video):
            return

        # Release and reopen streams
        try:
            for st in self.streams:
                st.release()
        except Exception:
            pass

        self.paths = list(new_paths)
        self.streams = [CamStream(p) for p in self.paths]
        self._using_overlay_video = using_overlay_video        # FPS/max frames recompute
        # IMPORTANT: keep original raw FPS as the playback timebase so overlay videos play at the same speed as the source.
        if self._raw_base_fps is None:
            fps_list = [s.fps for s in self.streams if s.ok]
            self._raw_base_fps = float((fps_list[0] if fps_list else (self._base_fps or 30.0)))
        self._base_fps = float(self._raw_base_fps or 30.0)
        counts = [s.n_frames for s in self.streams if s.ok and s.n_frames > 0]
        self._max_frames = int(min(counts) if counts else 0)

        # Restore frame index
        if keep_frame:
            self.frame_var.set(max(0, min(cur_frame, max(0, self._max_frames - 1))))
        else:
            self.frame_var.set(0)

    def clear(self, release_only: bool = False):
        self.pause()
        for st in self.streams:
            st.release()
        self.paths = []
        self.streams = []
        self._max_frames = 0
        if release_only:
            return
        for w in self.grid_box.winfo_children():
            w.destroy()
        self.previews = []

    def _on_clear_clicked(self):
        """UI clear button: clear preview + notify parent to clear marker data too."""
        try:
            if callable(self._on_clear):
                self._on_clear()
                return
        except Exception:
            pass
        # fallback
        self.clear()

    def set_external_max_frames(self, n: int):
        """Set playback length when no video is loaded (e.g., TRC/C3D only)."""
        try:
            self._external_max_frames = int(n or 0)
        except Exception:
            self._external_max_frames = 0

    def set_external_fps(self, fps: float):
        """Set playback FPS when no video is loaded (e.g., TRC/C3D only)."""
        try:
            v = float(fps or 0.0)
            if v > 1.0:
                self._external_fps = v
        except Exception:
            pass

    def get_effective_max_frames(self) -> int:
        """Max frames used for wrap-around during playback."""
        v = int(self._max_frames or 0)
        e = int(self._external_max_frames or 0)
        return v if v > 0 else e

    def get_effective_fps(self) -> float:
        """FPS used for playback timing."""
        if self.streams:
            return float(self._base_fps or 30.0)
        return float(self._external_fps or self._base_fps or 30.0)

    def toggle_play(self):
        total = self.get_effective_max_frames()
        if total <= 0:
            return
        if not self._playing.get():
            self.play()
        else:
            self.pause()

    def play(self):
        total = self.get_effective_max_frames()
        if total <= 0:
            return
        sp = float(self.speed_var.get())
        if sp <= 0.0:
            return
        self._playing.set(True)
        if hasattr(self, "btn_play"):
            try:
                self.btn_play.configure(text="Pause")
            except Exception:
                pass

        # time-based scheduler: keeps real-time speed even when decoding/rendering is slow
        self._play_t0 = time.perf_counter()
        self._play_frame0 = int(self.frame_var.get())
        self._play_speed = sp
        self._play_fps = float(self.get_effective_fps() or 30.0)

        self._tick()

    def pause(self):
        self._playing.set(False)
        if hasattr(self, "btn_play"):
            try:
                self.btn_play.configure(text="Play")
            except Exception:
                pass

    def _tick(self):
        if not self._playing.get():
            return
        sp = float(self.speed_var.get())
        if sp <= 0.0:
            self.pause()
            return

        total = self.get_effective_max_frames()
        if total <= 0:
            self.pause()
            return

        fps = float(self.get_effective_fps() or 30.0)
        now = time.perf_counter()

        # Rebase timing if speed/fps changed or timebase missing
        if getattr(self, "_play_t0", None) is None:
            self._play_t0 = now
            self._play_frame0 = int(self.frame_var.get())
            self._play_speed = sp
            self._play_fps = fps
        else:
            if abs(fps - float(getattr(self, "_play_fps", fps))) > 1e-6 or abs(sp - float(getattr(self, "_play_speed", sp))) > 1e-6:
                self._play_t0 = now
                self._play_frame0 = int(self.frame_var.get())
                self._play_speed = sp
                self._play_fps = fps

        t0 = float(self._play_t0)
        frame0 = int(self._play_frame0)

        # number of frames that should have advanced since t0
        frames_elapsed = int(max(0.0, (now - t0) * fps * sp))
        nxt = (frame0 + frames_elapsed) % total

        # Update frame (triggers refresh + 3D sync via trace in parent)
        cur = int(self.frame_var.get())
        if nxt != cur:
            self.frame_var.set(nxt)

        # Schedule next tick aligned to the next frame boundary
        try:
            next_frame_time = (frames_elapsed + 1) / (fps * sp)
        except ZeroDivisionError:
            self.pause()
            return

        target = t0 + next_frame_time
        delay_ms = max(1, int((target - time.perf_counter()) * 1000))
        delay_ms = min(delay_ms, 200)  # keep UI responsive
        self.after(delay_ms, self._tick)


    def _ensure_pose_dirs(self):
        """Resolve pose JSON directories for each loaded video and cache json file lists.

        We only try this when Overlay is enabled. Pre-run, <project_dir>/pose
        won't exist, and we silently do nothing.
        """
        if self._pose_project_dir is None:
            self._pose_project_dir = infer_project_dir_from_video_folder(self.last_video_folder)

        if not self._pose_project_dir:
            self._pose_json_dirs = [None for _ in self.streams]
            return

        self._pose_json_dirs = []
        for st in self.streams:
            jd = resolve_pose_json_dir(self._pose_project_dir, st.path)
            self._pose_json_dirs.append(jd)
            # Prime/update file list cache for this dir
            if jd is not None and jd.exists():
                try:
                    mtime = int(jd.stat().st_mtime)
                except Exception:
                    mtime = 0
                key = str(jd)
                cached = self._pose_json_file_list_cache.get(key)
                if cached is None or cached[0] != mtime:
                    files = list_pose_json_sorted(jd)
                    idx_map = build_pose_json_index_map(files)
                    self._pose_json_file_list_cache[key] = (mtime, files, idx_map)


    def _load_pose_json_cached(self, json_path: Path) -> Optional[dict]:
        key = (str(json_path), int(json_path.stat().st_mtime))
        # Small cache keyed by path+mtime so updates after re-run are reflected
        cached = self._pose_json_cache.get(key)
        if cached is not None:
            self._pose_json_cache.move_to_end(key)
            return cached
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            return None
        self._pose_json_cache[key] = data
        self._pose_json_cache.move_to_end(key)
        while len(self._pose_json_cache) > int(self._pose_cache_limit):
            self._pose_json_cache.popitem(last=False)
        return data

    def refresh_current(self, fit: bool = False):
        if not self.streams:
            return
        idx = int(self.frame_var.get())

        do_overlay = bool(self._overlay_enabled.get()) and (not getattr(self, '_using_overlay_video', False))
        if do_overlay:
            self._ensure_pose_dirs()

        if do_overlay and self._pose_project_dir and not (self._pose_project_dir / "pose").exists():
            # pre-run: pose folder does not exist -> silent
            do_overlay = False

        for i, st in enumerate(self.streams):
            fr = st.read_at(idx)
            if fr is None:
                continue
            out = fr
            if do_overlay:
                json_dir = None
                if i < len(self._pose_json_dirs):
                    json_dir = self._pose_json_dirs[i]
                if json_dir is None or not json_dir.exists():
                    if not self._overlay_warned_missing:
                        # Warn only once to avoid log spam
                        self._log("[Overlay][INFO] pose JSON not found yet. Run pipeline first (results under <project_dir>/pose/*_json/).")
                        self._overlay_warned_missing = True
                else:
                    # Use cached sorted list when available (fast + robust to filename digit width)
                    key = str(json_dir)
                    files = None
                    cached = self._pose_json_file_list_cache.get(key)
                    idx_map = None
                    if cached is not None:
                        files = cached[1]
                        if len(cached) >= 3:
                            idx_map = cached[2]
                    json_path = None
                    if idx_map:
                        json_path = idx_map.get(idx) or idx_map.get(idx + 1)
                    if json_path is None and files:
                        # fallback to list position when indices are contiguous
                        if 0 <= idx < len(files):
                            json_path = files[idx]
                        elif 0 <= (idx + 1) < len(files):
                            json_path = files[idx + 1]
                    if json_path is None:
                        # fallback to pattern-based search
                        json_path = find_openpose_json_file(json_dir, idx)
                    if json_path is not None and json_path.exists():
                        data = self._load_pose_json_cached(json_path)
                        if isinstance(data, dict):
                            try:
                                out = draw_openpose_overlay(out, data, kpt_thr=0.2, draw_bbox=True, draw_skeleton=True, min_bbox_px=10)
                            except Exception as e:
                                # Do not break Tkinter callbacks on bad frames; just skip overlay for this frame
                                self._log(f"[Overlay][WARN] overlay draw failed at frame={idx}: {type(e).__name__}: {e}")


            rgb = bgr_to_rgb(out)
            if rgb is None:
                continue
            if i < len(self.previews):
                self.previews[i].set_rgb(rgb, fit_if_first=fit)


# ---------------------- 3D Viewer (Matplotlib) ----------------------
