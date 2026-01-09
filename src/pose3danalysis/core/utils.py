from __future__ import annotations
from pathlib import Path
from typing import Any, Optional

def ensure_dir(p: Path) -> Path:
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p

def copy_file(src: str | Path, dst: str | Path) -> None:
    """shutil 대신 직접 복사(혹시 프로젝트에 shutil.py 같은 shadowing 문제 대비)."""
    src = Path(src)
    dst = Path(dst)
    ensure_dir(dst.parent)
    with src.open("rb") as fsrc, dst.open("wb") as fdst:
        while True:
            buf = fsrc.read(1024 * 1024)
            if not buf:
                break
            fdst.write(buf)

def reprojection_px_to_mm(
    px_error: float,
    K: Any,
    depth_m: Optional[float] = None,
    square_mm: Optional[float] = None,
) -> float:
    """Approximate conversion from reprojection error in pixels to millimeters.

    Notes
    - **Physically meaningful** conversion needs a depth (distance from camera to the 3D points).
      Small-angle approximation:
        angle(rad) ≈ px / f(px)
        lateral_error(m) ≈ depth_m * angle
        => mm ≈ px * (depth_m*1000 / f)

    - Legacy fallback (NOT physical): if you pass square_mm (checkerboard square size),
      we compute mm_per_px ≈ square_mm / f. This is only a scale-like number and should
      not be interpreted as true metric error unless you know the scene scale corresponds.

    Parameters
    - px_error: error in pixels
    - K: camera intrinsic matrix (3x3)
    - depth_m: average depth (meters) of the points from the camera (recommended)
    - square_mm: checkerboard square size (mm) (legacy fallback)

    Returns
    - approx error in mm (float), or NaN if conversion is not possible.
    """
    try:
        fx = float(K[0][0])
        fy = float(K[1][1])
    except Exception:
        return float("nan")

    f = 0.5 * (fx + fy)
    if f <= 0:
        return float("nan")

    if depth_m is not None:
        mm_per_px = (float(depth_m) * 1000.0) / f
        return float(px_error) * mm_per_px

    if square_mm is not None:
        mm_per_px = float(square_mm) / f
        return float(px_error) * mm_per_px

    return float("nan")