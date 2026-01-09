from __future__ import annotations

from pathlib import Path
from typing import Any


def _toml_dumps_intrinsics(C, S, K, D) -> str:
    # C: camera names
    # S: image sizes
    # K: intrinsics matrices
    # D: distortion
    def _fmt_float_token(v: float) -> str:
        x = float(v)
        # Always emit TOML float token (avoid homogeneous array errors in toml parser)
        s = f"{x:.16f}".rstrip('0')
        if s.endswith('.'):
            s += '0'
        return s

    lines = []
    lines.append("# lens_calibration.toml (intrinsics only)")
    lines.append("")

    lines.append("cameras = [")
    for name in C:
        lines.append(f'  "{name}",')
    lines.append("]")
    lines.append("")

    lines.append("sizes = [")
    for s in S:
        # s could be [w,h] or tuple
        w, h = int(s[0]), int(s[1])
        lines.append(f"  [{w}, {h}],")
    lines.append("]")
    lines.append("")

    lines.append("K = [")
    for Ki in K:
        lines.append("  [")
        for r in Ki:
            lines.append("    [" + ", ".join(_fmt_float_token(x) for x in r) + "],")
        lines.append("  ],")
    lines.append("]")
    lines.append("")

    lines.append("D = [")
    for Di in D:
        lines.append("  [" + ", ".join(_fmt_float_token(x) for x in Di) + "],")
    lines.append("]")
    lines.append("")
    return "\n".join(lines)


def _to_Rmat(R) -> list[list[float]]:
    """Pose2Sim가 반환하는 R 형태(3x3 또는 rodrigues rvec)를 3x3 matrix로 통일."""
    import numpy as np
    import cv2

    R = np.asarray(R, dtype=np.float64)
    if R.shape == (3, 3):
        return R.tolist()
    if R.shape in [(3,), (3, 1), (1, 3)]:
        rvec = R.reshape(3, 1)
        Rm, _ = cv2.Rodrigues(rvec)
        return np.asarray(Rm, dtype=np.float64).tolist()
    raise ValueError(f"Unsupported R shape: {R.shape} (expected 3x3 or rvec(3,))")


def _to_Tvec(T) -> list[float]:
    """T를 길이 3의 벡터로 통일."""
    import numpy as np

    T = np.asarray(T, dtype=np.float64)
    if T.shape in [(3,), (3, 1), (1, 3)]:
        return T.reshape(3,).tolist()
    raise ValueError(f"Unsupported T shape: {T.shape} (expected 3,) ")


def _toml_dumps_camera_calibration(C, S, K, D, R_list, T_list) -> str:
    """Pose2Sim 호환 calibration TOML 생성 (intrinsics + extrinsics).

    Pose2Sim에서 생성되는 Calib_*.toml 예시 포맷을 따릅니다:
      [cam_00]
      name = "cam_00"
      size = [720.0, 1280.0]
      matrix = [[fx,0,cx],[0,fy,cy],[0,0,1]]
      distortions = [k1,k2,p1,p2,(k3...)]
      rotation = [rvec_x, rvec_y, rvec_z]
      translation = [tx, ty, tz]
      fisheye = false
      [metadata]
      adjusted = false
      error = 0.0

    - rotation은 기본적으로 Rodrigues(rvec) 3원소를 사용합니다.
      만약 R가 3x3 행렬로 들어오면 cv2.Rodrigues로 rvec로 변환합니다.
    """
    lines: list[str] = []
    lines.append("# Pose2Sim-compatible calibration file (intrinsics + extrinsics)")
    lines.append("")

    # local import to keep module import-light
    try:
        import numpy as _np
        import cv2  # type: ignore
    except Exception as e:
        raise RuntimeError("numpy and cv2 are required to write calibration TOML") from e

    def _as_list(x):
        if isinstance(x, (list, tuple)):
            return list(x)
        try:
            return x.tolist()
        except Exception:
            return [x]

    def _fmt_float(v: float) -> str:
        # Always emit TOML float token (avoid 'Not a homogeneous array' errors)
        x = float(v)
        s = f"{x:.16f}".rstrip('0')
        if s.endswith('.'):
            s += '0'
        return s

    def _fmt_vec(vec) -> str:
        vv = _np.asarray(vec, dtype=float).reshape(-1)
        return "[ " + ", ".join(_fmt_float(v) for v in vv.tolist()) + " ]"

    def _fmt_mat3(mat) -> str:
        M = _np.asarray(mat, dtype=float).reshape(3, 3)
        rows = []
        for r in range(3):
            rows.append("[ " + ", ".join(_fmt_float(v) for v in M[r].tolist()) + "]")
        return "[ " + ", ".join(rows) + "]"

    def _to_rvec(Ri):
        Ri_arr = _np.asarray(Ri, dtype=float)
        # Already rvec?
        if Ri_arr.shape in [(3,), (3, 1), (1, 3)]:
            return Ri_arr.reshape(3)
        # Rotation matrix -> rvec
        if Ri_arr.shape == (3, 3):
            try:
                import cv2  # type: ignore
                rvec, _ = cv2.Rodrigues(Ri_arr.astype(float))
                return _np.asarray(rvec, dtype=float).reshape(3)
            except Exception:
                # fallback: flatten first 3 (not ideal but avoids hard-crash)
                return Ri_arr.reshape(-1)[:3]
        return Ri_arr.reshape(-1)[:3]

    def _to_tvec(Ti):
        Ti_arr = _np.asarray(Ti, dtype=float)
        if Ti_arr.shape in [(3,), (3, 1), (1, 3)]:
            return Ti_arr.reshape(3)
        return Ti_arr.reshape(-1)[:3]

    def _transform_coordinate_system(R, T):
        """
        Transform coordinate system: swap X and Z axes (Y remains up).
        
        For rotation matrix R:
        - Swap first and third columns
        - Swap first and third rows
        
        For translation vector T:
        - Swap first and third elements with sign inversion
        """
        R_arr = _np.asarray(R, dtype=float)
        T_arr = _np.asarray(T, dtype=float).reshape(3)
        
        # If R is a rotation matrix (3x3), transform it
        if R_arr.shape == (3, 3):
            # Create transformation matrix to swap X and Z
            # [0, 0, 1]  -> X becomes -Z
            # [0, 1, 0]  -> Y stays Y
            # [1, 0, 0]  -> Z becomes -X
            swap_matrix = _np.array([
                [0, 0, -1],  # new X = -old Z
                [0, 1,  0],  # new Y = old Y
                [-1, 0, 0]   # new Z = -old X
            ], dtype=float)
            R_transformed = swap_matrix @ R_arr @ swap_matrix.T
            rvec, _ = cv2.Rodrigues(R_transformed)
            rvec = rvec.reshape(3)
        else:
            # If R is already a Rodrigues vector, convert to matrix first
            rvec_orig = R_arr.reshape(3)
            R_mat, _ = cv2.Rodrigues(rvec_orig)
            swap_matrix = _np.array([
                [0, 0, -1],
                [0, 1,  0],
                [-1, 0, 0]
            ], dtype=float)
            R_transformed = swap_matrix @ R_mat @ swap_matrix.T
            rvec, _ = cv2.Rodrigues(R_transformed)
            rvec = rvec.reshape(3)
        
        # Transform translation: swap X and Z with sign inversion
        T_transformed = _np.array([-T_arr[2], T_arr[1], -T_arr[0]], dtype=float)
        
        return rvec, T_transformed

    # cameras
    for i, cam_name in enumerate(C):
        lines.append(f"[{cam_name}]")
        lines.append(f'name = "{cam_name}"')

        # size: Pose2Sim은 float 표기를 사용 (w, h)
        w, h = float(S[i][0]), float(S[i][1])
        lines.append(f"size = [ {_fmt_float(w)}, {_fmt_float(h)} ]")

        lines.append(f"matrix = {_fmt_mat3(K[i])}")

        distort = _np.asarray(D[i], dtype=float).reshape(-1)
        lines.append("distortions = [ " + ", ".join(_fmt_float(v) for v in distort.tolist()) + " ]")

        # Keep original coordinate system (Y-up, no transformation)
        # Coordinate transformation is applied only in 3D viewer when Z-up option is enabled
        rvec = _to_rvec(R_list[i])
        tvec = _to_tvec(T_list[i])
        
        lines.append(f"rotation = {_fmt_vec(rvec)}")
        lines.append(f"translation = {_fmt_vec(tvec)}")

        # 기본값: 일반 렌즈로 처리
        lines.append("fisheye = false")
        lines.append("")

    # metadata block
    lines.append("[metadata]")
    lines.append("adjusted = false")
    lines.append("error = 0.0")
    lines.append("")
    return "\n".join(lines)


def write_lens_calibration_toml(path: str | Path, C, S, K, D) -> None:
    path = Path(path)
    path.write_text(_toml_dumps_intrinsics(C, S, K, D), encoding="utf-8")


def write_camera_calibration_toml(path: str | Path, C, S, K, D, R_list, T_list) -> None:
    """camera_calibration.toml (intrinsics + extrinsics) 저장."""
    path = Path(path)
    path.write_text(_toml_dumps_camera_calibration(C, S, K, D, R_list, T_list), encoding="utf-8")

def load_lens_calibration_toml(path: str | Path):
    """
    return: (C, S, K, D)
    가능한 파서 순서: tomllib(3.11+) -> tomli -> toml
    """
    path = Path(path)
    data: Any
    try:
        import tomllib  # py3.11+
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    except Exception:
        try:
            import toml
            data = toml.loads(path.read_text(encoding="utf-8"))
        except Exception:
            import toml
            data = toml.loads(path.read_text(encoding="utf-8"))

    C = data["cameras"]
    S = data["sizes"]
    K = data["K"]
    D = data["D"]
    return C, S, K, D
