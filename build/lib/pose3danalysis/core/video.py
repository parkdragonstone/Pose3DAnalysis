from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import numpy as np
from PIL import Image, ImageTk

@dataclass
class VideoState:
    path: Optional[str] = None

def rgb_to_photo(rgb: np.ndarray, target_w: int, target_h: int) -> ImageTk.PhotoImage:
    """
    numpy RGB -> Tk PhotoImage, 타겟 영역에 맞게 비율 유지 resize.
    """
    if target_w <= 2:
        target_w = 640
    if target_h <= 2:
        target_h = 360

    img = Image.fromarray(rgb)
    w, h = img.size
    scale = min(target_w / w, target_h / h)
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    img = img.resize((new_w, new_h), Image.BILINEAR)
    return ImageTk.PhotoImage(img)
