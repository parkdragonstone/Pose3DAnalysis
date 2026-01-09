# -*- coding: utf-8 -*-
"""Pose-related utilities.

Extracted from gui.tabs.motion_analysis_tab to improve code organization.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional, List, Tuple

import numpy as np
import cv2


def infer_project_dir_from_video_folder(video_folder: Optional[str]) -> Optional[Path]:
    """Infer <project_dir> from the folder chosen in 'Upload Video Folder'.

    Expected layouts:
      - <project_dir>/videos/*.mp4  and results in <project_dir>/pose/
      - <project_dir>/*.mp4         and results in <project_dir>/pose/
    """
    if not video_folder:
        return None
    p = Path(video_folder).expanduser().resolve()
    # direct
    if (p / "pose").exists():
        return p
    # common: .../videos
    if p.name.lower() == "videos" and (p.parent / "pose").exists():
        return p.parent
    # best effort: search 2 levels up
    cur = p
    for _ in range(2):
        if (cur / "pose").exists():
            return cur
        cur = cur.parent
    return None


def resolve_pose_json_dir(project_dir: Path, video_path: str) -> Optional[Path]:
    """Resolve pose JSON directory from project directory and video path."""
    pose_root = project_dir / "pose"
    if not pose_root.exists():
        return None
    stem = Path(video_path).stem
    # Most common in Pose2Sim: <stem>_json
    cand = pose_root / f"{stem}_json"
    if cand.exists() and cand.is_dir():
        return cand
    # Fallback: glob
    matches = sorted([p for p in pose_root.glob(f"{stem}*_json") if p.is_dir()])
    if matches:
        return matches[0]
    # Some variants
    cand2 = pose_root / stem
    if cand2.exists() and cand2.is_dir():
        return cand2
    return None


def extract_frame_index_from_name(p: Path) -> int:
    """Extract frame index from a pose JSON filename.

    Accepts:
      - *_<digits>_keypoints.json
      - *_<digits>.json
    """
    name = p.name
    m = re.search(r"_(\d+)(?:_keypoints)?\.json$", name)
    if m:
        try:
            return int(m.group(1))
        except Exception:
            return 10**18
    return 10**18


def list_pose_json_sorted(json_dir: Path) -> List[Path]:
    """List frame-wise pose JSONs sorted by extracted frame index.

    We include:
      - *_keypoints.json
      - *.json that ends with _<digits>.json
    """
    hits = []
    for p in json_dir.glob("*.json"):
        # must look like a frame-wise json, not config/metadata
        if re.search(r"_(\d+)(?:_keypoints)?\.json$", p.name):
            hits.append(p)
    hits.sort(key=extract_frame_index_from_name)
    return hits


def find_openpose_json_file(json_dir: Path, frame_idx: int) -> Optional[Path]:
    """Find pose JSON for a given frame index.

    Supports multiple naming conventions that appear in our pipeline:
      - <stem>_000000000123_keypoints.json   (OpenPose default, 12 digits)
      - <stem>_00000001_keypoints.json       (8 digits)
      - <stem>_000000.json                   (6 digits, no suffix)
      - <stem>_000000000123.json             (12 digits, no suffix)
      - 1-based indexing (idx+1)
    """
    if not json_dir.exists():
        return None

    # Prefer exact match for current index, then 1-based.
    for idx in (frame_idx, frame_idx + 1):
        pats = [
            f"*_{idx:012d}_keypoints.json",
            f"*_{idx:08d}_keypoints.json",
            f"*_{idx:06d}_keypoints.json",
            f"*_{idx:012d}.json",
            f"*_{idx:08d}.json",
            f"*_{idx:06d}.json",
        ]
        for pat in pats:
            hits = sorted(json_dir.glob(pat))
            if hits:
                return hits[0]

    # Last resort: any json that looks like frame-wise output
    candidates = list_pose_json_sorted(json_dir)
    return candidates[frame_idx] if 0 <= frame_idx < len(candidates) else (candidates[0] if candidates else None)


def build_pose_json_index_map(files: List[Path]) -> dict:
    """Build {frame_index: path} map from a sorted list of files."""
    m = {}
    for p in files:
        idx = extract_frame_index_from_name(p)
        if idx != 10**18:
            # if duplicates, keep first (usually fine)
            m.setdefault(idx, p)
    return m


def skeleton_pairs_for_kpts(k: int) -> List[Tuple[int, int]]:
    """Return skeleton edge pairs for HALPE-26 keypoints.

    This UI uses RTMPose HALPE-26. COCO skeleton is intentionally not supported here.
    The pairs are derived from PoseAnalysis/skeleton(.py|skeletons.py) HALPE_26 definition:

        Hip(19)->RHip(12)->RKnee(14)->RAnkle(16)->RBigToe(21)->RSmallToe(23)
                                                  ->RHeel(25)
        Hip(19)->LHip(11)->LKnee(13)->LAnkle(15)->LBigToe(20)->LSmallToe(22)
                                                  ->LHeel(24)
        Hip(19)->Neck(18)->Head(17)->Nose(0)
        Neck(18)->RShoulder(6)->RElbow(8)->RWrist(10)
        Neck(18)->LShoulder(5)->LElbow(7)->LWrist(9)
    """
    if k < 26:
        return []
    return [
        # right leg/foot
        (19, 12), (12, 14), (14, 16), (16, 21), (21, 23), (16, 25),
        # left leg/foot
        (19, 11), (11, 13), (13, 15), (15, 20), (20, 22), (15, 24),
        # trunk/head
        (19, 18), (18, 17), (17, 0),
        # right arm
        (18, 6), (6, 8), (8, 10),
        # left arm
        (18, 5), (5, 7), (7, 9),
    ]


def parse_people_openpose(data: dict) -> List[Tuple[np.ndarray, np.ndarray, Optional[Tuple[float, float, float, float]]]]:
    """Return list of (kpts[K,2], scores[K], bbox(x1,y1,x2,y2)|None)."""
    out = []
    people = data.get("people") or []
    for person in people:
        flat = person.get("pose_keypoints_2d")
        if not flat:
            continue
        arr = np.asarray(flat, dtype=np.float32).reshape(-1, 3)
        kpts = arr[:, :2]
        scores = arr[:, 2]

        bbox = None
        # common custom keys we might have written
        for key in ("bbox", "bbox_xyxy", "bbox_xywh", "box", "person_bbox"):
            if key in person and person[key] is not None:
                v = person[key]
                if isinstance(v, dict):
                    x1 = float(v.get("x1", v.get("x", 0.0)))
                    y1 = float(v.get("y1", v.get("y", 0.0)))
                    x2 = float(v.get("x2", x1 + v.get("w", 0.0)))
                    y2 = float(v.get("y2", y1 + v.get("h", 0.0)))
                    bbox = (x1, y1, x2, y2)
                    break
                if isinstance(v, (list, tuple)) and len(v) == 4:
                    x1, y1, a, b = [float(x) for x in v]
                    # heuristic: xyxy if (a,b) looks like max corner
                    if a > x1 and b > y1:
                        bbox = (x1, y1, a, b)
                    else:
                        bbox = (x1, y1, x1 + a, y1 + b)
                    break

        out.append((kpts, scores, bbox))
    return out


def kp_side_halpe26(k: int) -> str:
    """Return 'L', 'R', or 'M' for left/right/mid keypoints (HALPE-26)."""
    left = {5, 7, 9, 11, 13, 15, 20, 22, 24}
    right = {6, 8, 10, 12, 14, 16, 21, 23, 25}
    mid = {0, 17, 18, 19, 1, 2, 3, 4}  # include face ids if present
    if k in left:
        return 'L'
    if k in right:
        return 'R'
    return 'M'


def draw_openpose_overlay(
    bgr: np.ndarray,
    data: dict,
    kpt_thr: float = 0.2,
    draw_bbox: bool = True,
    draw_skeleton: bool = True,
    min_bbox_px: int = 10,
) -> np.ndarray:
    """Overlay bbox + keypoints (+ skeleton) on the preview frame.

    Supports OpenPose-style json: data['people'][i]['pose_keypoints_2d'] = [x,y,score]*K
    Robustness:
    - Skips NaN/inf points.
    - If bbox is missing or invalid, recompute from valid keypoints.
    - Never raises exceptions to Tk loop; returns original frame on failure.
    """
    if bgr is None or getattr(bgr, "size", 0) == 0:
        return bgr
    h, w = bgr.shape[:2]
    vis = bgr.copy()

    try:
        people = parse_people_openpose(data)
    except Exception:
        return vis

    thr = float(kpt_thr)

    # --- overlay color scheme (BGR) ---
    COLOR_BBOX = (255, 0, 0)      # blue
    COLOR_LEFT = (255, 255, 0)    # cyan-ish (left)
    COLOR_RIGHT = (0, 255, 0)     # green (right)
    COLOR_MID = (0, 165, 255)     # orange (trunk/head)

    def edge_color(a: int, b: int) -> tuple:
        sa = kp_side_halpe26(a)
        sb = kp_side_halpe26(b)
        if sa == 'L' and sb == 'L':
            return COLOR_LEFT
        if sa == 'R' and sb == 'R':
            return COLOR_RIGHT
        return COLOR_MID

    def point_color(k: int) -> tuple:
        s = kp_side_halpe26(k)
        return COLOR_LEFT if s == 'L' else COLOR_RIGHT if s == 'R' else COLOR_MID

    for kpts, scores, bbox in people:
        try:
            if kpts is None or getattr(kpts, "size", 0) == 0:
                continue

            # normalize -> pixel if needed (0..1)
            kmax = float(np.nanmax(kpts)) if np.isfinite(kpts).any() else 0.0
            if 0.0 <= kmax <= 2.0:
                kpts = kpts.copy()
                kpts[:, 0] *= float(w)
                kpts[:, 1] *= float(h)

            finite_xy = np.isfinite(kpts[:, 0]) & np.isfinite(kpts[:, 1])
            good = finite_xy & (scores >= thr)
            if not np.any(good):
                continue

            # bbox sanitize / recompute
            bb = None
            if bbox is not None:
                try:
                    x1, y1, x2, y2 = map(float, bbox)
                    if np.isfinite([x1, y1, x2, y2]).all():
                        bb = (x1, y1, x2, y2)
                except Exception:
                    bb = None

            if bb is None:
                xs = kpts[good, 0]
                ys = kpts[good, 1]
                m2 = np.isfinite(xs) & np.isfinite(ys)
                if np.any(m2):
                    xs = xs[m2]
                    ys = ys[m2]
                    bb = (float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max()))

            if draw_bbox and bb is not None and np.isfinite(bb).all():
                x1, y1, x2, y2 = bb
                x1i = int(np.clip(x1, 0, w - 1))
                y1i = int(np.clip(y1, 0, h - 1))
                x2i = int(np.clip(x2, 0, w - 1))
                y2i = int(np.clip(y2, 0, h - 1))
                if (x2i - x1i) >= int(min_bbox_px) and (y2i - y1i) >= int(min_bbox_px):
                    cv2.rectangle(vis, (x1i, y1i), (x2i, y2i), COLOR_BBOX, 2)

            if draw_skeleton:
                pairs = skeleton_pairs_for_kpts(int(kpts.shape[0]))
                for a, b in pairs:
                    if a >= kpts.shape[0] or b >= kpts.shape[0]:
                        continue
                    if float(scores[a]) < thr or float(scores[b]) < thr:
                        continue
                    if not (finite_xy[a] and finite_xy[b]):
                        continue
                    x1, y1 = kpts[a]
                    x2, y2 = kpts[b]
                    if not np.isfinite([x1, y1, x2, y2]).all():
                        continue
                    p1 = (int(np.clip(x1, 0, w - 1)), int(np.clip(y1, 0, h - 1)))
                    p2 = (int(np.clip(x2, 0, w - 1)), int(np.clip(y2, 0, h - 1)))
                    cv2.line(vis, p1, p2, edge_color(a, b), 3)

            # keypoints
            for ki in range(kpts.shape[0]):
                if not (finite_xy[ki] and float(scores[ki]) >= thr):
                    continue
                x, y = kpts[ki]
                if not np.isfinite([x, y]).all():
                    continue
                cv2.circle(vis, (int(np.clip(x, 0, w - 1)), int(np.clip(y, 0, h - 1))), 4, point_color(ki), -1)

        except Exception:
            # never break Tk loop
            continue

    return vis

