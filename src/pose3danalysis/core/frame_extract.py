# core/frame_extract.py
from __future__ import annotations

from pathlib import Path
from typing import List

import cv2


def extract_frames_every_n_sec(video_path: str, out_dir: Path, every_sec: int) -> List[Path]:
    """
    Extract frames by timestamp, saving only frames at ~every_sec interval.
    Returns a sorted list of saved PNG paths.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    every_sec = int(every_sec)
    if every_sec <= 0:
        every_sec = 1

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(
            f"Failed to open video for extraction: {video_path}\n"
            f"(Codec issue: convert to H.264 mp4 or install ffmpeg-enabled opencv.)"
        )

    saved: List[Path] = []
    last_saved_ms = None
    idx = 0
    save_i = 0

    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break

        pos_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
        if last_saved_ms is None or (pos_ms - last_saved_ms) >= every_sec * 1000.0:
            save_i += 1
            out_path = out_dir / f"frame_{save_i:06d}.png"
            cv2.imwrite(str(out_path), frame)
            saved.append(out_path)
            last_saved_ms = pos_ms

        idx += 1

        # safety: prevent runaway if POS_MSEC is broken
        if idx > 10_000_000:
            break

    cap.release()
    return sorted(saved)