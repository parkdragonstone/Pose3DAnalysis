# -*- coding: utf-8 -*-
"""Motion settings and Pose2Sim configuration utilities.

Extracted from gui.tabs.motion_analysis_tab to improve code organization.
"""

from __future__ import annotations

import copy
import ast
from dataclasses import dataclass
from typing import List, Tuple

# Default Pose2Sim configuration
DEFAULT_POSE2SIM_CONFIG = {
    'project': {
        'multi_person': False,
        'participant_height': 'auto',
        'participant_mass': 98.0,
        'frame_rate': 'auto',
        'frame_range': 'auto',
        'exclude_from_batch': []
    },
    'pose': {
        'vid_img_extension': 'avi',
        'pose_model': 'Body_with_feet',
        'mode': 'balanced',
        'det_frequency': 4,
        'device': 'auto',
        'backend': 'auto',
        'tracking_mode': 'sports2d',
        'max_distance_px': 100,
        'deepsort_params': "{'max_age':30, 'n_init':3, 'nms_max_overlap':0.8, "
                         "'max_cosine_distance':0.3, 'nn_budget':200, 'max_iou_distance':0.8}",
        'display_detection': False,
        'overwrite_pose': True,
        'save_video': 'to_video',
        'output_format': 'openpose',
        'CUSTOM': {
            'name': 'Hip',
            'id': 19,
            'children': [
                {
                    'name': 'RHip',
                    'id': 12,
                    'children': [
                        {
                            'name': 'RKnee',
                            'id': 14,
                            'children': [
                                {
                                    'name': 'RAnkle',
                                    'id': 16,
                                    'children': [
                                        {'name': 'RBigToe', 'id': 21, 'children': [{'name': 'RSmallToe', 'id': 23}]},
                                        {'name': 'RHeel', 'id': 25}
                                    ]
                                }
                            ]
                        }
                    ]
                },
                {
                    'name': 'LHip',
                    'id': 11,
                    'children': [
                        {
                            'name': 'LKnee',
                            'id': 13,
                            'children': [
                                {
                                    'name': 'LAnkle',
                                    'id': 15,
                                    'children': [
                                        {'name': 'LBigToe', 'id': 20, 'children': [{'name': 'LSmallToe', 'id': 22}]},
                                        {'name': 'LHeel', 'id': 24}
                                    ]
                                }
                            ]
                        }
                    ]
                },
                {
                    'name': 'Neck',
                    'id': 18,
                    'children': [
                        {'name': 'Head', 'id': 17, 'children': [{'name': 'Nose', 'id': 0}]},
                        {
                            'name': 'RShoulder',
                            'id': 6,
                            'children': [
                                {
                                    'name': 'RElbow',
                                    'id': 8,
                                    'children': [{'name': 'RWrist', 'id': 10}]
                                }
                            ]
                        },
                        {
                            'name': 'LShoulder',
                            'id': 5,
                            'children': [
                                {
                                    'name': 'LElbow',
                                    'id': 7,
                                    'children': [{'name': 'LWrist', 'id': 9}]
                                }
                            ]
                        }
                    ]
                }
            ]
        }
    },
    'synchronization': {
        'synchronization_gui': False,
        'display_sync_plots': False,
        'save_sync_plots': True,
        'keypoints_to_consider': ['RWrist'],
        'approx_time_maxspeed': 'auto',
        'time_range_around_maxspeed': 2.0,
        'likelihood_threshold': 0.3,
        'filter_cutoff': 6,
        'filter_order': 4
    },
    'calibration': {
        'calibration_type': 'calculate',
        'convert': {
            'convert_from': 'qualisys',
            'caliscope': {},
            'qualisys': {'binning_factor': 1},
            'optitrack': {},
            'vicon': {},
            'opencap': {},
            'easymocap': {},
            'biocv': {},
            'anipose': {},
            'freemocap': {}
        },
        'calculate': {
            'intrinsics': {
                'overwrite_intrinsics': False,
                'show_detection_intrinsics': True,
                'intrinsics_extension': 'avi',
                'extract_every_N_sec': 1,
                'intrinsics_corners_nb': [3, 5],
                'intrinsics_square_size': 75
            },
            'extrinsics': {
                'calculate_extrinsics': True,
                'extrinsics_method': 'scene',
                'moving_cameras': False,
                'board': {
                    'show_reprojection_error': True,
                    'board_position': 'horizontal',
                    'extrinsics_extension': 'avi',
                    'extrinsics_corners_nb': [5, 3],
                    'extrinsics_square_size': 75
                },
                'scene': {
                    'show_reprojection_error': True,
                    'extrinsics_extension': 'avi',
                    'object_coords_3d': [
                        [0.0, 0.0, 0.0],
                        [0.0, -0.492, 0.0],
                        [0.0, 0.0, 0.45],
                        [0.0, -0.492, 0.45],
                        [0.492, 0.0, 0.0],
                        [0.492, -0.492, 0.0],
                        [0.492, 0.0, 0.45],
                        [0.492, -0.492, 0.45]
                    ]
                },
                'keypoints': {}
            }
        }
    },
    'personAssociation': {
        'likelihood_threshold_association': 0.3,
        'single_person': {
            'reproj_error_threshold_association': 20,
            'tracked_keypoint': 'Neck'
        },
        'multi_person': {
            'reconstruction_error_threshold': 0.1,
            'min_affinity': 0.2
        }
    },
    'triangulation': {
        'reproj_error_threshold_triangulation': 15,
        'likelihood_threshold_triangulation': 0.3,
        'min_cameras_for_triangulation': 2,
        'interp_if_gap_smaller_than': 20,
        'max_distance_m': 1.0,
        'interpolation': 'linear',
        'remove_incomplete_frames': False,
        'sections_to_keep': 'all',
        'min_chunk_size': 10,
        'fill_large_gaps_with': 'last_value',
        'show_interp_indices': True,
        'make_c3d': True
    },
    'filtering': {
        'reject_outliers': True,
        'filter': True,
        'type': 'butterworth',
        'display_figures': False,
        'save_filt_plots': True,
        'make_c3d': True,
        'butterworth': {'cut_off_frequency': 6, 'order': 4},
        'kalman': {'trust_ratio': 500, 'smooth': True},
        'gcv_spline': {'cut_off_frequency': 'auto', 'smoothing_factor': 1.0},
        'loess': {'nb_values_used': 5},
        'gaussian': {'sigma_kernel': 1},
        'median': {'kernel_size': 3},
        'butterworth_on_speed': {'order': 4, 'cut_off_frequency': 10}
    },
    'markerAugmentation': {
        'feet_on_floor': False,
        'make_c3d': True
    },
    'kinematics': {
        'use_augmentation': True,
        'use_simple_model': False,
        'right_left_symmetry': True,
        'default_height': 1.7,
        'remove_individual_scaling_setup': True,
        'remove_individual_ik_setup': True,
        'fastest_frames_to_remove_percent': 0.1,
        'close_to_zero_speed_m': 0.2,
        'large_hip_knee_angles': 45,
        'trimmed_extrema_percent': 0.5
    }
}

# Default skeleton edges
DEFAULT_SKELETON_EDGES: List[Tuple[str, str]] = [
    ("Head", "Nose"),
    ("Nose", "Neck"),
    ("Neck", "RShoulder"),
    ("RShoulder", "RElbow"),
    ("RElbow", "RWrist"),
    ("Neck", "LShoulder"),
    ("LShoulder", "LElbow"),
    ("LElbow", "LWrist"),
    ("Neck", "Hip"),
    ("Hip", "RHip"),
    ("RHip", "RKnee"),
    ("RKnee", "RAnkle"),
    ("RAnkle", "RHeel"),
    ("RAnkle", "RBigToe"),
    ("RBigToe", "RSmallToe"),
    ("Hip", "LHip"),
    ("LHip", "LKnee"),
    ("LKnee", "LAnkle"),
    ("LAnkle", "LHeel"),
    ("LAnkle", "LBigToe"),
    ("LBigToe", "LSmallToe"),
]


@dataclass
class MotionSettings:
    """GUI state that can fully generate a Pose2Sim-like config dict without Config.toml.

    - UI는 '자주 쓰는' 항목만 노출하고,
    - 노출하지 않는 항목은 DEFAULT_POSE2SIM_CONFIG 값을 그대로 사용합니다.
    """

    # -------- Output / Paths --------
    output_dir: str = ""
    camera_calibration_path: str = ""  # calibration.toml (file or folder)

    # -------- Project --------
    multi_person: bool = False
    analysis_type: str = "single"  # 'single' | 'multi'
    participant_height: str = "auto"   # 'auto' or numeric string (meters)
    participant_mass_kg: float = float(DEFAULT_POSE2SIM_CONFIG["project"]["participant_mass"])
    frame_rate: str = str(DEFAULT_POSE2SIM_CONFIG["project"]["frame_rate"])     # 'auto' or int string
    frame_range: str = str(DEFAULT_POSE2SIM_CONFIG["project"]["frame_range"])   # 'auto'/'all'/'[start,end]'

    # -------- Pose (YOLO + RTMPOSE) --------
    vid_img_extension: str = str(DEFAULT_POSE2SIM_CONFIG["pose"]["vid_img_extension"])
    mode: str = "balanced"               # 'balanced' | 'performance' (GUI 선택)
    det_frequency: int = int(DEFAULT_POSE2SIM_CONFIG["pose"]["det_frequency"])
    device: str = str(DEFAULT_POSE2SIM_CONFIG["pose"]["device"])               # auto/CPU/CUDA/MPS/ROCM
    backend: str = str(DEFAULT_POSE2SIM_CONFIG["pose"]["backend"])             # auto/openvino/onnxruntime/opencv
    tracking_mode: str = str(DEFAULT_POSE2SIM_CONFIG["pose"]["tracking_mode"]) # none/sports2d/deepsort
    max_distance_px: int = int(DEFAULT_POSE2SIM_CONFIG["pose"]["max_distance_px"])
    deepsort_params: str = str(DEFAULT_POSE2SIM_CONFIG["pose"]["deepsort_params"])
    display_detection: bool = bool(DEFAULT_POSE2SIM_CONFIG["pose"]["display_detection"])
    overwrite_pose: bool = bool(DEFAULT_POSE2SIM_CONFIG["pose"]["overwrite_pose"])
    save_video: str = str(DEFAULT_POSE2SIM_CONFIG["pose"]["save_video"])
    output_format: str = str(DEFAULT_POSE2SIM_CONFIG["pose"]["output_format"])

    # -------- Pipeline toggles (GUI에서 단계 On/Off) --------
    do_synchronization: bool = False
    do_person_association: bool = False
    do_triangulation: bool = True
    do_filtering: bool = True
    do_marker_augmentation: bool = True
    do_kinematic: bool = True
    kinematic_source: str = "augmented"  # 'augmented' | 'raw'

    # -------- Synchronization (auto default) --------
    synchronization_gui: bool = bool(DEFAULT_POSE2SIM_CONFIG["synchronization"]["synchronization_gui"])
    display_sync_plots: bool = bool(DEFAULT_POSE2SIM_CONFIG["synchronization"]["display_sync_plots"])
    save_sync_plots: bool = bool(DEFAULT_POSE2SIM_CONFIG["synchronization"]["save_sync_plots"])
    keypoints_to_consider: str = "RWrist"  # UI에서는 CSV 형태로 입력 (예: RWrist,RElbow)
    approx_time_maxspeed: str = str(DEFAULT_POSE2SIM_CONFIG["synchronization"]["approx_time_maxspeed"])
    time_range_around_maxspeed: float = float(DEFAULT_POSE2SIM_CONFIG["synchronization"]["time_range_around_maxspeed"])
    sync_likelihood_threshold: float = float(DEFAULT_POSE2SIM_CONFIG["synchronization"]["likelihood_threshold"])
    sync_filter_cutoff: float = float(DEFAULT_POSE2SIM_CONFIG["synchronization"]["filter_cutoff"])
    sync_filter_order: int = int(DEFAULT_POSE2SIM_CONFIG["synchronization"]["filter_order"])

    # -------- Person Association --------
    reproj_error_threshold_association: float = float(DEFAULT_POSE2SIM_CONFIG["personAssociation"]["single_person"]["reproj_error_threshold_association"])
    tracked_keypoint: str = str(DEFAULT_POSE2SIM_CONFIG["personAssociation"]["single_person"]["tracked_keypoint"])

    # -------- Triangulation --------
    reproj_error_threshold_triangulation: float = float(DEFAULT_POSE2SIM_CONFIG["triangulation"]["reproj_error_threshold_triangulation"])
    likelihood_threshold_triangulation: float = float(DEFAULT_POSE2SIM_CONFIG["triangulation"]["likelihood_threshold_triangulation"])
    min_cameras_for_triangulation: int = int(DEFAULT_POSE2SIM_CONFIG["triangulation"]["min_cameras_for_triangulation"])
    interp_if_gap_smaller_than: int = int(DEFAULT_POSE2SIM_CONFIG["triangulation"]["interp_if_gap_smaller_than"])
    interpolation: str = str(DEFAULT_POSE2SIM_CONFIG["triangulation"]["interpolation"])

    # -------- Filtering (기본: butterworth) --------
    reject_outliers: bool = bool(DEFAULT_POSE2SIM_CONFIG["filtering"]["reject_outliers"])
    filter_on: bool = bool(DEFAULT_POSE2SIM_CONFIG["filtering"]["filter"])
    filter_type: str = str(DEFAULT_POSE2SIM_CONFIG["filtering"]["type"])
    butter_cutoff_hz: float = float(DEFAULT_POSE2SIM_CONFIG["filtering"]["butterworth"]["cut_off_frequency"])
    butter_order: int = int(DEFAULT_POSE2SIM_CONFIG["filtering"]["butterworth"]["order"])

    # -------- Kinematics --------
    use_simple_model: bool = bool(DEFAULT_POSE2SIM_CONFIG["kinematics"]["use_simple_model"])
    right_left_symmetry: bool = bool(DEFAULT_POSE2SIM_CONFIG["kinematics"]["right_left_symmetry"])
    default_height: float = float(DEFAULT_POSE2SIM_CONFIG["kinematics"]["default_height"])


def parse_csv_list(s) -> list[str]:
    """Parse CSV string or pass-through list/tuple into list[str]."""
    if s is None:
        return []
    # Already a list/tuple of keypoints
    if isinstance(s, (list, tuple)):
        out = []
        for x in s:
            if x is None:
                continue
            sx = str(x).strip()
            if sx:
                out.append(sx)
        return out
    # String-like
    s = str(s).strip()
    if not s:
        return []
    return [x.strip() for x in s.split(",") if x.strip()]


def build_pose2sim_config(settings: MotionSettings) -> dict:
    """Config.toml 없이도 Pose2Sim 파이프라인이 돌아가도록 config dict를 생성한다."""
    cfg = copy.deepcopy(DEFAULT_POSE2SIM_CONFIG)

    # ---- helpers ----
    def _strip_scalar(v, default=""):
        """Return a stripped scalar string; if list is given, use the first element."""
        if v is None:
            return default
        if isinstance(v, list):
            if len(v) == 0:
                return default
            v = v[0]
        if isinstance(v, str):
            s = v.strip()
            return s if s != "" else default
        s = str(v).strip()
        return s if s != "" else default

    def _normalize_participant_height(v):
        """Pose2Sim: 'auto', float, or list of floats."""
        if v is None or v == "":
            return "auto"
        if isinstance(v, list):
            out = []
            for x in v:
                if x is None or str(x).strip() == "":
                    continue
                if isinstance(x, (int, float)):
                    out.append(float(x))
                    continue
                sx = str(x).strip()
                if sx.lower() == "auto":
                    return "auto"
                try:
                    out.append(float(sx))
                except Exception:
                    out.append(sx)
            return out if len(out) > 0 else "auto"
        if isinstance(v, (int, float)):
            return float(v)
        sv = str(v).strip()
        if sv.lower() == "auto" or sv == "":
            return "auto"
        try:
            return float(sv)
        except Exception:
            return sv

    def _normalize_frame_rate(v):
        """Pose2Sim: 'auto' or int."""
        if v is None or v == "":
            return "auto"
        if isinstance(v, list):
            if len(v) == 0:
                return "auto"
            v = v[0]
        if isinstance(v, (int, float)):
            return int(v)
        sv = str(v).strip()
        if sv.lower() == "auto" or sv == "":
            return "auto"
        try:
            return int(float(sv))
        except Exception:
            return sv

    def _normalize_frame_range(v):
        """Pose2Sim: 'auto'/'all' or list [start,end]."""
        if v is None or v == "":
            return "auto"
        if isinstance(v, list):
            try:
                if len(v) == 2:
                    return [int(v[0]), int(v[1])]
            except Exception:
                pass
            return v
        if isinstance(v, tuple):
            return list(v)
        sv = str(v).strip()
        if sv == "" or sv.lower() in ("auto", "all"):
            return sv.lower() if sv != "" else "auto"
        try:
            if sv.startswith("[") and sv.endswith("]"):
                parsed = ast.literal_eval(sv)
                if isinstance(parsed, (list, tuple)) and len(parsed) == 2:
                    return [int(parsed[0]), int(parsed[1])]
                return parsed
            if "," in sv:
                a, b = sv.split(",", 1)
                return [int(float(a.strip())), int(float(b.strip()))]
        except Exception:
            pass
        return sv

    def _normalize_save_video(v):
        """Pose2Sim: 'none'/'to_video'/'to_images' or list of those."""
        if v is None:
            return "none"
        if isinstance(v, list):
            cleaned = []
            for x in v:
                sx = str(x).strip()
                if sx == "" or sx.lower() == "none":
                    continue
                cleaned.append(sx)
            if len(cleaned) == 0:
                return "none"
            if len(cleaned) == 1:
                return cleaned[0]
            seen = set()
            out = []
            for x in cleaned:
                if x not in seen:
                    out.append(x)
                    seen.add(x)
            return out
        sv = _strip_scalar(v, default="none")
        if sv.lower() == "none" or sv == "":
            return "none"
        if sv.startswith("[") and sv.endswith("]"):
            try:
                parsed = ast.literal_eval(sv)
                if isinstance(parsed, list):
                    return _normalize_save_video(parsed)
            except Exception:
                pass
        if sv in ("to_video", "to_images"):
            return sv
        if sv.lower().replace(" ", "") in ("to_video+to_images", "to_video,to_images", "to_images,to_video"):
            return ["to_video", "to_images"]
        return sv

    # ---- project ----
    cfg["project"]["multi_person"] = (str(settings.analysis_type).strip().lower() == "multi") or bool(settings.multi_person)
    cfg["project"]["participant_height"] = _normalize_participant_height(settings.participant_height)
    cfg["project"]["participant_mass"] = float(settings.participant_mass_kg)
    cfg["project"]["frame_rate"] = _normalize_frame_rate(settings.frame_rate)
    cfg["project"]["frame_range"] = _normalize_frame_range(settings.frame_range)

    # ---- pose ----
    cfg["pose"]["vid_img_extension"] = _strip_scalar(settings.vid_img_extension, default=cfg["pose"]["vid_img_extension"])
    # Pose2Sim(MMPose) 기준 문자열
    # pose_model is fixed to Body_with_feet (HALPE_26)
    cfg["pose"]["pose_model"] = "Body_with_feet"
    cfg["pose"]["engine"] = "yolo_onnx"
    _mode = _strip_scalar(settings.mode, default="balanced")
    if (_mode or '').lower().strip() == "normal":
        _mode = "balanced"
    cfg["pose"]["mode"] = _mode
    cfg["pose"]["det_frequency"] = int(settings.det_frequency)
    cfg["pose"]["device"] = _strip_scalar(settings.device, default=cfg["pose"]["device"])
    cfg["pose"]["backend"] = _strip_scalar(settings.backend, default=cfg["pose"]["backend"])
    cfg["pose"]["tracking_mode"] = _strip_scalar(settings.tracking_mode, default=cfg["pose"]["tracking_mode"])
    cfg["pose"]["max_distance_px"] = int(settings.max_distance_px)
    cfg["pose"]["deepsort_params"] = settings.deepsort_params
    cfg["pose"]["display_detection"] = bool(settings.display_detection)
    cfg["pose"]["overwrite_pose"] = bool(settings.overwrite_pose)
    # save_video: can be "to_video", "to_images", "none", or ["to_video","to_images"]
    cfg["pose"]["save_video"] = _normalize_save_video(settings.save_video)
    cfg["pose"]["output_format"] = _strip_scalar(settings.output_format, default=cfg["pose"]["output_format"])

    # ---- synchronization ----
    cfg["synchronization"]["synchronization_gui"] = bool(settings.synchronization_gui)
    cfg["synchronization"]["display_sync_plots"] = bool(settings.display_sync_plots)
    cfg["synchronization"]["save_sync_plots"] = bool(settings.save_sync_plots)
    cfg["synchronization"]["keypoints_to_consider"] = parse_csv_list(settings.keypoints_to_consider) or cfg["synchronization"]["keypoints_to_consider"]
    cfg["synchronization"]["approx_time_maxspeed"] = _strip_scalar(settings.approx_time_maxspeed, default=cfg["synchronization"]["approx_time_maxspeed"]) if settings.approx_time_maxspeed else cfg["synchronization"]["approx_time_maxspeed"]
    cfg["synchronization"]["time_range_around_maxspeed"] = float(settings.time_range_around_maxspeed)
    cfg["synchronization"]["likelihood_threshold"] = float(settings.sync_likelihood_threshold)
    cfg["synchronization"]["filter_cutoff"] = float(settings.sync_filter_cutoff)
    cfg["synchronization"]["filter_order"] = int(settings.sync_filter_order)

    # ---- personAssociation ----
    cfg["personAssociation"]["single_person"]["reproj_error_threshold_association"] = float(settings.reproj_error_threshold_association)
    cfg["personAssociation"]["single_person"]["tracked_keypoint"] = _strip_scalar(settings.tracked_keypoint, default=cfg["personAssociation"]["single_person"]["tracked_keypoint"]) or cfg["personAssociation"]["single_person"]["tracked_keypoint"]

    # ---- triangulation ----
    cfg["triangulation"]["reproj_error_threshold_triangulation"] = float(settings.reproj_error_threshold_triangulation)
    cfg["triangulation"]["likelihood_threshold_triangulation"] = float(settings.likelihood_threshold_triangulation)
    cfg["triangulation"]["min_cameras_for_triangulation"] = int(settings.min_cameras_for_triangulation)
    cfg["triangulation"]["interp_if_gap_smaller_than"] = int(settings.interp_if_gap_smaller_than)
    cfg["triangulation"]["interpolation"] = _strip_scalar(settings.interpolation, default=cfg["triangulation"]["interpolation"]) or cfg["triangulation"]["interpolation"]

    # ---- filtering ----
    cfg["filtering"]["reject_outliers"] = bool(settings.reject_outliers)
    cfg["filtering"]["filter"] = bool(settings.filter_on)
    cfg["filtering"]["type"] = _strip_scalar(settings.filter_type, default=cfg["filtering"]["type"]) or cfg["filtering"]["type"]
    cfg["filtering"]["butterworth"]["cut_off_frequency"] = float(settings.butter_cutoff_hz)
    cfg["filtering"]["butterworth"]["order"] = int(settings.butter_order)

    # ---- kinematics ----
    cfg["kinematics"]["use_simple_model"] = bool(settings.use_simple_model)
    cfg["kinematics"]["right_left_symmetry"] = bool(settings.right_left_symmetry)
    cfg["kinematics"]["default_height"] = float(settings.default_height)

    # calibration path is not in original toml: we store it for pipeline runner
    cfg.setdefault("_ui", {})
    cfg["_ui"]["output_dir"] = settings.output_dir
    cfg["_ui"]["calibration_path"] = settings.camera_calibration_path

    # pipeline toggles
    cfg["_ui"]["do_synchronization"] = bool(settings.do_synchronization)
    cfg["_ui"]["do_person_association"] = bool(settings.do_person_association)
    cfg["_ui"]["do_triangulation"] = bool(settings.do_triangulation)
    cfg["_ui"]["do_filtering"] = bool(settings.do_filtering)
    cfg["_ui"]["do_marker_augmentation"] = bool(settings.do_marker_augmentation)
    cfg["_ui"]["do_kinematic"] = bool(settings.do_kinematic)
    cfg["_ui"]["kinematic_source"] = settings.kinematic_source

    return cfg

