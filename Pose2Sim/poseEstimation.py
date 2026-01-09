#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
###########################################################################
## POSE ESTIMATION                                                       ##
###########################################################################

    Pose2Sim pose estimation stage with optional Ultralytics YOLO detector.

    - USE_YOLO = True:
        * Detection: Ultralytics YOLOv11 (.pt) -> person bboxes (xyxy)
        * Pose: RTMLib RTMPose (local ONNX) using the detected bboxes
        * Downstream (NMS/tracking/json/video) stays the same.

    - USE_YOLO = False:
        * Original Pose2Sim behavior using RTMLib PoseTracker (det+pose)

    Notes:
    - This file is intended to be a drop-in replacement for Pose2Sim/poseEstimation.py
    - Ultralytics is imported lazily; if USE_YOLO=False you don't need it installed.
"""

# =========================
# User switches (top-level)
# =========================
USE_YOLO = True          # True: Ultralytics YOLOv11 person det + RTMPose pose. False: original PoseTracker pipeline.
DET_CONF = 0.25          # Ultralytics detection confidence threshold
DET_IOU = 0.70           # Ultralytics NMS IoU threshold
DET_IMGSZ = 640          # Ultralytics inference image size

# =========================
# INIT
# =========================
import os
import glob
import json
import re
import logging
import ast
from functools import partial
from tqdm import tqdm
from anytree.importer import DictImporter
from anytree import RenderTree
import numpy as np
import cv2

from rtmlib import PoseTracker, BodyWithFeet, Wholebody, Body, Hand, Custom, draw_skeleton
from rtmlib.tools.object_detection.post_processings import nms
from Pose2Sim.common import (
    natural_sort_key, sort_people_sports2d, sort_people_deepsort,
    colors, thickness, draw_bounding_box, draw_keypts, draw_skel, bbox_xyxy_compute,
    get_screen_size, calculate_display_size
)
from Pose2Sim.skeletons import *

np.set_printoptions(legacy='1.21')  # otherwise prints np.float64(3.0) rather than 3.0

# Silence numpy and CoreML warnings
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, message="Mean of empty slice")
warnings.filterwarnings("ignore", category=RuntimeWarning, message="All-NaN slice encountered")
warnings.filterwarnings("ignore", category=RuntimeWarning, message="invalid value encountered in scalar divide")
warnings.filterwarnings("ignore", message=".*Input.*has a dynamic shape.*but the runtime shape.*has zero elements.*")

# Not safe, but to be used until OpenMMLab/RTMlib's SSL certificates are updated
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

# AUTHORSHIP INFORMATION (keep original)
__author__ = "HunMin Kim, David Pagnon"
__copyright__ = "Copyright 2021, Pose2Sim"
__credits__ = ["HunMin Kim", "David Pagnon"]
__license__ = "BSD 3-Clause License"
from importlib.metadata import version, PackageNotFoundError
try:
    __version__ = version('pose2sim')
except PackageNotFoundError:
    __version__ = 'unknown'
__maintainer__ = "David Pagnon"
__email__ = "contact@david-pagnon.com"
__status__ = "Development"


# =========================
# Ultralytics helpers
# =========================
def _get_app_root():
    """Return project root (folder containing 'models')."""
    return os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))


def _select_ultralytics_pt(mode: str) -> str:
    """Select YOLOv11 .pt path based on mode ('balanced'|'performance')."""
    mode_folder = 'performance' if str(mode).lower() == 'performance' else 'balanced'
    models_root = os.path.join(_get_app_root(), 'models')
    pt_name = 'yolo11l.pt' if mode_folder == 'performance' else 'yolo11m.pt'
    return os.path.join(models_root, mode_folder, pt_name)


def _find_extracted_onnx(mode: str, kind: str):
    """
    Find extracted end2end.onnx under models/<mode>/_extracted by substring match.
    kind: 'rtmpose' or 'yolox' (or any substring).
    """
    mode_folder = 'performance' if str(mode).lower() == 'performance' else 'balanced'
    extracted_root = os.path.join(_get_app_root(), 'models', mode_folder, '_extracted')
    if not os.path.isdir(extracted_root):
        return None
    cands = glob.glob(os.path.join(extracted_root, '*', 'end2end.onnx'))
    kind = (kind or '').lower()
    for c in cands:
        if kind and kind in c.lower():
            return c
    return cands[0] if cands else None


def _ultra_device_from(pose_device: str):
    """Map pose device string to ultralytics device argument."""
    try:
        import torch
        has_cuda = torch.cuda.is_available()
    except Exception:
        has_cuda = False

    d = (pose_device or '').lower()
    if d in ('cuda', 'gpu') and has_cuda:
        return 0  # cuda:0
    if d in ('cpu',):
        return 'cpu'
    # auto
    return 0 if has_cuda else 'cpu'


class _UltraDetWrapper:
    """Callable wrapper returning Nx4 xyxy bboxes as float32 numpy array."""
    def __init__(self, yolo_model, yolo_cfg):
        self.model = yolo_model
        self.cfg = dict(yolo_cfg)

    def __call__(self, frame_bgr: np.ndarray) -> np.ndarray:
        res = self.model.predict(frame_bgr, classes=[0], **self.cfg)[0]  # person only (COCO class 0)
        if res.boxes is None or len(res.boxes) == 0:
            return np.zeros((0, 4), dtype=np.float32)
        return res.boxes.xyxy.detach().cpu().numpy().astype(np.float32)


def setup_ultralytics_detector(mode: str, pose_device: str):
    """Initialize Ultralytics YOLO model and config."""
    pt_path = _select_ultralytics_pt(mode)
    if not os.path.exists(pt_path):
        raise FileNotFoundError(f"Ultralytics YOLO model not found: {pt_path}")

    try:
        from ultralytics import YOLO
    except Exception as e:
        raise ImportError("Ultralytics is required when USE_YOLO=True. Install with: pip install ultralytics") from e

    yolo_model = YOLO(pt_path)
    yolo_cfg = dict(
        device=_ultra_device_from(pose_device),
        imgsz=DET_IMGSZ,
        conf=DET_CONF,
        iou=DET_IOU,
        verbose=False
    )
    return yolo_model, yolo_cfg


def setup_pose_solver_for_ultra(mode: str, backend: str, device: str):
    """
    Create an RTMPose solver that accepts bboxes. (No det model is created.)
    Returns a callable: pose_solver(frame_bgr, bboxes=xyxy) -> (keypoints, scores)
    """
    pose_onnx = _find_extracted_onnx(mode, kind='rtmpose')
    if not pose_onnx:
        raise FileNotFoundError(
            f"Cannot locate extracted RTMPose ONNX under models/"
            f"{'performance' if str(mode).lower()=='performance' else 'balanced'}/_extracted"
        )

    # RTMPose import (rtmlib version dependent)
    try:
        from rtmlib.tools.pose_estimation.rtmpose import RTMPose
    except Exception:
        try:
            from rtmlib.tools.pose_estimation import RTMPose
        except Exception as e:
            raise ImportError("Cannot import RTMPose from rtmlib. Please check your rtmlib version.") from e

    pose_solver = RTMPose(
        pose_onnx,
        model_input_size=(192, 256),
        to_openpose=False,
        backend=backend,
        device=device
    )
    return pose_solver


# =========================
# Original Pose2Sim helpers
# =========================
def setup_backend_device(backend='auto', device='auto'):
    """
    Set up backend & device.

    If device and backend are not specified, they are determined automatically:
      1) CUDA + onnxruntime
      2) ROCm + onnxruntime
      3) MPS/CoreML + onnxruntime
      4) CPU + openvino (fallback)
    """
    if device != 'auto' and backend != 'auto':
        device = device.lower()
        backend = backend.lower()

    if device == 'auto' or backend == 'auto':
        if (device == 'auto' and backend != 'auto') or (device != 'auto' and backend == 'auto'):
            logging.warning("If you set device or backend to 'auto', set the other to 'auto' too. Auto-detecting both.")

        try:
            import torch
            import onnxruntime as ort
            if torch.cuda.is_available() and 'CUDAExecutionProvider' in ort.get_available_providers():
                device = 'cuda'
                backend = 'onnxruntime'
                logging.info("\nValid CUDA installation found: using ONNXRuntime backend with GPU.")
            elif torch.cuda.is_available() and 'ROCMExecutionProvider' in ort.get_available_providers():
                device = 'rocm'
                backend = 'onnxruntime'
                logging.info("\nValid ROCM installation found: using ONNXRuntime backend with GPU.")
            else:
                raise RuntimeError("No CUDA/ROCM provider")
        except Exception:
            try:
                import onnxruntime as ort
                if 'MPSExecutionProvider' in ort.get_available_providers() or 'CoreMLExecutionProvider' in ort.get_available_providers():
                    device = 'mps'
                    backend = 'onnxruntime'
                    logging.info("\nValid MPS installation found: using ONNXRuntime backend with GPU.")
                else:
                    raise RuntimeError("No MPS/CoreML provider")
            except Exception:
                device = 'cpu'
                backend = 'openvino'
                logging.info("\nNo valid CUDA installation found: using OpenVINO backend with CPU.")
    return backend, device


def setup_pose_tracker(ModelClass, det_frequency, mode, tracking, backend, device):
    """Set up the RTMLib PoseTracker (original pipeline)."""
    backend, device = setup_backend_device(backend=backend, device=device)
    pose_tracker = PoseTracker(
        ModelClass,
        det_frequency=det_frequency,
        mode=mode,
        backend=backend,
        device=device,
        tracking=tracking,
        to_openpose=False
    )
    return pose_tracker


def setup_model_class_mode(pose_model, mode, config_dict={}):
    """
    Determine skeleton tree (pose_model), RTMLib solution class (ModelClass), and mode string.
    This is largely the original Pose2Sim logic; we keep it for skeleton definitions.
    """
    if pose_model.upper() in ('HALPE_26', 'BODY_WITH_FEET'):
        model_name = 'HALPE_26'
        ModelClass = BodyWithFeet  # 26 keypoints (halpe26)
        logging.info(f"Using HALPE_26 model (body and feet) for pose estimation in {mode} mode.")
    elif pose_model.upper() in ('COCO_133', 'WHOLE_BODY', 'WHOLE_BODY_WRIST'):
        model_name = 'COCO_133'
        ModelClass = Wholebody
        logging.info(f"Using COCO_133 model (body, feet, hands, and face) for pose estimation in {mode} mode.")
    elif pose_model.upper() in ('COCO_17', 'BODY'):
        model_name = 'COCO_17'
        ModelClass = Body
        logging.info(f"Using COCO_17 model (body) for pose estimation in {mode} mode.")
    elif pose_model.upper() == 'HAND':
        model_name = 'HAND_21'
        ModelClass = Hand
        logging.info(f"Using HAND_21 model for pose estimation in {mode} mode.")
    else:
        model_name = pose_model.upper()
        logging.info(f"Using model {model_name} for pose estimation in {mode} mode.")

    try:
        pose_model = eval(model_name)
    except Exception:
        try:
            model_name = pose_model.upper()
            pose_model = DictImporter().import_(config_dict.get('pose').get(pose_model)[0])
            if pose_model.id == 'None':
                pose_model.id = None
            logging.info(f"Using model {model_name} for pose estimation.")
        except Exception:
            raise NameError(f'{pose_model} not found in skeletons.py nor in Config.toml')

    # Validate mode
    if mode not in ['lightweight', 'balanced', 'performance']:
        logging.warning("Invalid mode. Must be 'lightweight', 'balanced', or 'performance'. Using 'balanced'.")
        mode = 'balanced'

    return pose_model, ModelClass, mode


def save_to_openpose(json_file_path, keypoints, scores):
    """Save keypoints/scores to OpenPose-like JSON."""
    nb_detections = len(keypoints)
    detections = []
    for i in range(nb_detections):  # nb of detected people
        keypoints_with_confidence_i = []
        for kp, score in zip(keypoints[i], scores[i]):
            x = float(kp[0]) if not np.isnan(kp[0]) else float('nan')
            y = float(kp[1]) if not np.isnan(kp[1]) else float('nan')
            s = float(score) if not np.isnan(score) else float('nan')
            keypoints_with_confidence_i.extend([x, y, s])

        detections.append({
            "person_id": [-1],
            "pose_keypoints_2d": keypoints_with_confidence_i,
            "face_keypoints_2d": [],
            "hand_left_keypoints_2d": [],
            "hand_right_keypoints_2d": [],
            "pose_keypoints_3d": [],
            "face_keypoints_3d": [],
            "hand_left_keypoints_3d": [],
            "hand_right_keypoints_3d": []
        })

    json_output = {"version": 1.3, "people": detections}

    json_output_dir = os.path.abspath(os.path.join(json_file_path, '..'))
    if not os.path.isdir(json_output_dir):
        os.makedirs(json_output_dir, exist_ok=True)
    with open(json_file_path, 'w') as json_file:
        json.dump(json_output, json_file)


def _nms_and_filter(frame_shape, keypoints, scores):
    """Pose-level NMS and filtering (original logic)."""
    if keypoints is None or scores is None:
        return keypoints, scores
    if len(keypoints) == 0:
        return keypoints, scores

    mask_scores = np.mean(scores, axis=1) > 0.2
    likely_keypoints = np.where(mask_scores[:, np.newaxis, np.newaxis], keypoints, np.nan)
    likely_scores = np.where(mask_scores[:, np.newaxis], scores, np.nan)
    likely_bboxes = bbox_xyxy_compute(frame_shape, likely_keypoints, padding=0)
    score_likely_bboxes = np.nanmean(likely_scores, axis=1)

    valid_indices = np.where(~np.isnan(score_likely_bboxes))[0]
    if len(valid_indices) > 0:
        valid_bboxes = likely_bboxes[valid_indices]
        valid_scores = score_likely_bboxes[valid_indices]
        keep_valid = nms(valid_bboxes, valid_scores, nms_thr=0.45)
        keep = valid_indices[keep_valid]
    else:
        keep = []
    return likely_keypoints[keep], likely_scores[keep]


def _draw_overlay(frame, keypoints, scores, pose_model):
    """Draw skeleton (Sports2D fallback style)."""
    valid_X, valid_Y, valid_scores = [], [], []
    for person_keypoints, person_scores in zip(keypoints, scores):
        person_X, person_Y = person_keypoints[:, 0], person_keypoints[:, 1]
        valid_X.append(person_X)
        valid_Y.append(person_Y)
        valid_scores.append(person_scores)

    img_show = frame.copy()
    img_show = draw_bounding_box(img_show, valid_X, valid_Y, colors=colors, fontSize=2, thickness=thickness)
    img_show = draw_keypts(img_show, valid_X, valid_Y, valid_scores, cmap_str='RdYlGn')
    img_show = draw_skel(img_show, valid_X, valid_Y, pose_model)
    return img_show


def process_video(
    video_path,
    pose_tracker,
    pose_model,
    output_format,
    save_video,
    save_images,
    display_detection,
    frame_range,
    tracking_mode,
    max_distance_px,
    deepsort_tracker,
    use_yolo=False,
    ultra_det=None,
    pose_solver=None,
    pose_dir=None
):
    """Estimate pose from a video file."""
    cap = cv2.VideoCapture(video_path)
    cap.read()
    if cap.read()[0] is False:
        raise NameError(f"{video_path} is not a video. Images must be put in one subdirectory per camera.")

    # Use provided pose_dir if available, otherwise fall back to relative path calculation
    if pose_dir is None:
        pose_dir = os.path.abspath(os.path.join(video_path, '..', '..', 'pose'))
    if not os.path.isdir(pose_dir):
        os.makedirs(pose_dir, exist_ok=True)
    video_name_wo_ext = os.path.splitext(os.path.basename(video_path))[0]
    json_output_dir = os.path.join(pose_dir, f'{video_name_wo_ext}_json')
    output_video_path = os.path.join(pose_dir, f'{video_name_wo_ext}_pose.mp4')
    img_output_dir = os.path.join(pose_dir, f'{video_name_wo_ext}_img')

    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if save_video:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = round(cap.get(cv2.CAP_PROP_FPS)) or 30
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (W, H))

    if display_detection:
        screen_width, screen_height = get_screen_size()
        display_width, display_height = calculate_display_size(W, H, screen_width, screen_height, margin=50)
        cv2.namedWindow(f"Pose Estimation {os.path.basename(video_path)}", cv2.WINDOW_NORMAL)
        cv2.resizeWindow(f"Pose Estimation {os.path.basename(video_path)}", display_width, display_height)

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    f_range = [0, total_frames] if frame_range in ('all', 'auto', []) else frame_range
    cap.set(cv2.CAP_PROP_POS_FRAMES, f_range[0])
    frame_idx = f_range[0]

    # Retrieve keypoint names from model
    keypoints_ids = [node.id for _, _, node in RenderTree(pose_model) if node.id is not None]
    kpt_id_max = max(keypoints_ids) + 1

    with tqdm(iterable=range(*f_range), desc=f'Processing {os.path.basename(video_path)}') as pbar:
        while cap.isOpened():
            if frame_idx in range(*f_range):
                success, frame = cap.read()
                if not success:
                    break

                try:
                    # --- Pose inference ---
                    if use_yolo:
                        if ultra_det is None or pose_solver is None:
                            raise RuntimeError("use_yolo=True but ultra_det/pose_solver not provided.")
                        bboxes = ultra_det(frame)  # Nx4 xyxy
                        if bboxes.shape[0] == 0:
                            keypoints = np.full((1, kpt_id_max, 2), np.nan, dtype=np.float32)
                            scores = np.full((1, kpt_id_max), np.nan, dtype=np.float32)
                        else:
                            keypoints, scores = pose_solver(frame, bboxes=bboxes)
                    else:
                        keypoints, scores = pose_tracker(frame)

                    # --- Pose-level NMS/filter ---
                    keypoints, scores = _nms_and_filter(frame.shape, keypoints, scores)

                    # --- Track poses across frames (optional) ---
                    if tracking_mode == 'deepsort':
                        keypoints, scores = sort_people_deepsort(keypoints, scores, deepsort_tracker, frame, frame_idx)
                    elif tracking_mode == 'sports2d':
                        if 'prev_keypoints' not in locals():
                            prev_keypoints = keypoints
                        prev_keypoints, keypoints, scores = sort_people_sports2d(
                            prev_keypoints, keypoints, scores=scores, max_dist=max_distance_px
                        )

                except Exception as e:
                    logging.exception(f"[PoseEst] frame={frame_idx} failed: {e}")
                    keypoints = np.full((1, kpt_id_max, 2), fill_value=np.nan, dtype=np.float32)
                    scores = np.full((1, kpt_id_max), fill_value=np.nan, dtype=np.float32)

                # Save to json
                if 'openpose' in output_format:
                    json_file_path = os.path.join(json_output_dir, f'{video_name_wo_ext}_{frame_idx:06d}.json')
                    save_to_openpose(json_file_path, keypoints, scores)

                # Draw skeleton on the frame (match original Pose2Sim style)
                if display_detection or save_video or save_images:
                    img_show = _draw_overlay(frame, keypoints, scores, pose_model)

                if display_detection:
                    cv2.imshow(f"Pose Estimation {os.path.basename(video_path)}", img_show)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

                if save_video:
                    out.write(img_show)

                if save_images:
                    if not os.path.isdir(img_output_dir):
                        os.makedirs(img_output_dir, exist_ok=True)
                    cv2.imwrite(os.path.join(img_output_dir, f'{video_name_wo_ext}_{frame_idx:06d}.jpg'), img_show)

                frame_idx += 1
                pbar.update(1)

            if frame_idx >= f_range[1]:
                break

    cap.release()
    if save_video:
        out.release()
        logging.info(f"--> Output video saved to {output_video_path}.")
    if save_images:
        logging.info(f"--> Output images saved to {img_output_dir}.")
    if display_detection:
        cv2.destroyAllWindows()


def process_images(
    image_folder_path,
    vid_img_extension,
    pose_tracker,
    pose_model,
    output_format,
    fps,
    save_video,
    save_images,
    display_detection,
    frame_range,
    tracking_mode,
    max_distance_px,
    deepsort_tracker,
    use_yolo=False,
    ultra_det=None,
    pose_solver=None,
    pose_dir=None
):
    """Estimate pose from an image folder."""
    # Use provided pose_dir if available, otherwise fall back to relative path calculation
    if pose_dir is None:
        pose_dir = os.path.abspath(os.path.join(image_folder_path, '..', '..', 'pose'))
    if not os.path.isdir(pose_dir):
        os.makedirs(pose_dir, exist_ok=True)
    json_output_dir = os.path.join(pose_dir, f'{os.path.basename(image_folder_path)}_json')
    output_video_path = os.path.join(pose_dir, f'{os.path.basename(image_folder_path)}_pose.mp4')
    img_output_dir = os.path.join(pose_dir, f'{os.path.basename(image_folder_path)}_img')

    image_files = glob.glob(os.path.join(image_folder_path, '*' + vid_img_extension))
    image_files = sorted(image_files, key=natural_sort_key)

    if save_video:
        logging.warning('Using default framerate of 60 fps.')
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        W, H = cv2.imread(image_files[0]).shape[:2][::-1]
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (W, H))

    if display_detection:
        screen_width, screen_height = get_screen_size()
        display_width, display_height = calculate_display_size(W, H, screen_width, screen_height, margin=50)
        cv2.namedWindow(f"Pose Estimation {os.path.basename(image_folder_path)}", cv2.WINDOW_NORMAL)
        cv2.resizeWindow(f"Pose Estimation {os.path.basename(image_folder_path)}", display_width, display_height)

    # Retrieve keypoint names from model
    keypoints_ids = [node.id for _, _, node in RenderTree(pose_model) if node.id is not None]
    kpt_id_max = max(keypoints_ids) + 1

    f_range = [0, len(image_files)] if frame_range in ('all', 'auto', []) else frame_range
    for frame_idx, image_file in enumerate(tqdm(image_files, desc=f'\nProcessing {os.path.basename(img_output_dir)}')):
        if frame_idx in range(*f_range):
            frame = cv2.imread(image_file)
            if frame is None:
                raise NameError(f"{image_file} is not an image. Videos must be put in the video directory, not in subdirectories.")

            try:
                if use_yolo:
                    if ultra_det is None or pose_solver is None:
                        raise RuntimeError("use_yolo=True but ultra_det/pose_solver not provided.")
                    bboxes = ultra_det(frame)
                    if bboxes.shape[0] == 0:
                        keypoints = np.full((1, kpt_id_max, 2), np.nan, dtype=np.float32)
                        scores = np.full((1, kpt_id_max), np.nan, dtype=np.float32)
                    else:
                        keypoints, scores = pose_solver(frame, bboxes=bboxes)
                else:
                    keypoints, scores = pose_tracker(frame)

                keypoints, scores = _nms_and_filter(frame.shape, keypoints, scores)

                if tracking_mode == 'deepsort':
                    keypoints, scores = sort_people_deepsort(keypoints, scores, deepsort_tracker, frame, frame_idx)
                elif tracking_mode == 'sports2d':
                    if 'prev_keypoints' not in locals():
                        prev_keypoints = keypoints
                    prev_keypoints, keypoints, scores = sort_people_sports2d(prev_keypoints, keypoints, scores=scores, max_dist=max_distance_px)

            except Exception as e:
                logging.exception(f"[PoseEst] frame={frame_idx} failed: {e}")
                keypoints = np.full((1, kpt_id_max, 2), fill_value=np.nan, dtype=np.float32)
                scores = np.full((1, kpt_id_max), fill_value=np.nan, dtype=np.float32)

            if 'openpose' in output_format:
                json_file_path = os.path.join(json_output_dir, f"{os.path.splitext(os.path.basename(image_file))[0]}_{frame_idx:06d}.json")
                save_to_openpose(json_file_path, keypoints, scores)

            if display_detection or save_video or save_images:
                img_show = _draw_overlay(frame, keypoints, scores, pose_model)

            if display_detection:
                cv2.imshow(f"Pose Estimation {os.path.basename(image_folder_path)}", img_show)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            if save_video:
                out.write(img_show)

            if save_images:
                if not os.path.isdir(img_output_dir):
                    os.makedirs(img_output_dir, exist_ok=True)
                cv2.imwrite(os.path.join(img_output_dir, f'{os.path.splitext(os.path.basename(image_file))[0]}_{frame_idx:06d}.png'), img_show)

    if save_video:
        out.release()
        logging.info(f"--> Output video saved to {output_video_path}.")
    if save_images:
        logging.info(f"--> Output images saved to {img_output_dir}.")
    if display_detection:
        cv2.destroyAllWindows()


def estimate_pose_all(config_dict):
    """
    Entry point used by Pose2Sim pipeline.
    """
    project_dir = config_dict['project']['project_dir']
    session_dir = os.path.realpath(os.path.join(project_dir, '..'))
    session_dir = session_dir if 'Config.toml' in os.listdir(session_dir) else os.getcwd()

    frame_range = config_dict.get('project').get('frame_range')
    multi_person = config_dict.get('project').get('multi_person')
    # Use video_dir from config paths if available, otherwise fall back to project_dir/videos
    if 'paths' in config_dict and 'video_dir' in config_dict['paths']:
        video_dir = config_dict['paths']['video_dir']
    else:
        video_dir = os.path.join(project_dir, 'videos')
    pose_dir = os.path.join(project_dir, 'pose')

    pose_model = config_dict['pose']['pose_model']
    mode = config_dict['pose']['mode']
    vid_img_extension = config_dict['pose']['vid_img_extension']

    output_format = config_dict['pose']['output_format']
    save_video = True if 'to_video' in config_dict['pose']['save_video'] else False
    save_images = True if 'to_images' in config_dict['pose']['save_video'] else False
    display_detection = config_dict['pose']['display_detection']
    overwrite_pose = config_dict['pose']['overwrite_pose']
    det_frequency = config_dict['pose']['det_frequency']
    tracking_mode = config_dict.get('pose').get('tracking_mode')
    max_distance_px = config_dict.get('pose').get('max_distance_px', None)

    # DeepSORT setup
    if tracking_mode == 'deepsort' and multi_person:
        deepsort_params = config_dict.get('pose').get('deepsort_params')
        try:
            deepsort_params = ast.literal_eval(deepsort_params)
        except Exception:
            deepsort_params = deepsort_params.strip("'").replace('\n', '').replace(" ", "").replace(",", '", "').replace(":", '":"').replace("{", '{"').replace("}", '"}').replace('":"/', ':/').replace('":"\\', ':\\')
            deepsort_params = re.sub(r'"\[([^"]+)",\s?"([^"]+)\]"', r'[\1,\2]', deepsort_params)
            deepsort_params = json.loads(deepsort_params)
        from deep_sort_realtime.deepsort_tracker import DeepSort
        deepsort_tracker = DeepSort(**deepsort_params)
    else:
        deepsort_tracker = None

    backend = config_dict['pose']['backend']
    device = config_dict['pose']['device']

    # Determine frame rate
    video_files = glob.glob(os.path.join(video_dir, '*' + vid_img_extension))
    frame_rate = config_dict.get('project').get('frame_rate')
    if frame_rate == 'auto':
        try:
            cap = cv2.VideoCapture(video_files[0])
            cap.read()
            if cap.read()[0] is False:
                raise RuntimeError("Cannot read video")
            frame_rate = round(cap.get(cv2.CAP_PROP_FPS))
        except Exception:
            logging.warning("Cannot read video. Frame rate will be set to 30 fps.")
            frame_rate = 30

    # Detection frequency validation
    if det_frequency > 1:
        logging.info(f'Inference run only every {det_frequency} frames. Inbetween, pose estimation tracks previously detected points.')
    elif det_frequency == 1:
        logging.info('Inference run on every single frame.')
    else:
        raise ValueError(f"Invalid det_frequency: {det_frequency}. Must be an integer >= 1.")

    logging.info('\nEstimating pose...')
    pose_model_name = pose_model
    pose_model, ModelClass, mode = setup_model_class_mode(pose_model, mode, config_dict)

    backend, device = setup_backend_device(backend=backend, device=device)

    # Skip if pose already exists and overwrite is false
    try:
        pose_listdirs_names = next(os.walk(pose_dir))[1]
        os.listdir(os.path.join(pose_dir, pose_listdirs_names[0]))[0]
        if not overwrite_pose:
            logging.info('Skipping pose estimation as it has already been done. Set overwrite_pose=true to run again.')
            return
        else:
            logging.info('Overwriting previous pose estimation (overwrite_pose=true).')
            raise RuntimeError("overwrite requested")
    except Exception:
        pass

    # Tracking mode validation
    if tracking_mode not in ['deepsort', 'sports2d']:
        logging.warning(f"Tracking mode {tracking_mode} not recognized. Using sports2d.")
        tracking_mode = 'sports2d'

    logging.info(f'\nPose tracking set up for "{pose_model_name}" model.')
    logging.info(f'Mode: {mode}.')
    logging.info(f'Tracking is performed with {tracking_mode}.\n')

    # Build solvers
    pose_tracker = None
    det_solver = None
    pose_solver = None

    if USE_YOLO:
        yolo_model, yolo_cfg = setup_ultralytics_detector(mode, device)
        det_solver = _UltraDetWrapper(yolo_model, yolo_cfg)
        pose_solver = setup_pose_solver_for_ultra(mode, backend, device)
    else:
        try:
            pose_tracker = setup_pose_tracker(ModelClass, det_frequency, mode, False, backend, device)
        except Exception:
            logging.error('Error: Pose estimation failed. Check in Config.toml that pose_model and mode are valid.')
            raise ValueError('Error: Pose estimation failed. Check in Config.toml that pose_model and mode are valid.')

    video_files = sorted(glob.glob(os.path.join(video_dir, '*' + vid_img_extension)))
    if len(video_files) != 0:
        logging.info(f'Found video files with {vid_img_extension} extension.')
        for video_path in video_files:
            if not USE_YOLO:
                pose_tracker.reset()
                if tracking_mode == 'deepsort' and deepsort_tracker is not None:
                    deepsort_tracker.tracker.delete_all_tracks()

            process_video(
                video_path,
                pose_tracker,
                pose_model,
                output_format,
                save_video,
                save_images,
                display_detection,
                frame_range,
                tracking_mode,
                max_distance_px,
                deepsort_tracker,
                use_yolo=USE_YOLO,
                ultra_det=det_solver,
                pose_solver=pose_solver,
                pose_dir=pose_dir
            )
    else:
        image_folders = sorted([os.path.join(video_dir, f) for f in os.listdir(video_dir) if os.path.isdir(os.path.join(video_dir, f))])
        empty_folders = [folder for folder in image_folders if len(glob.glob(os.path.join(folder, '*' + vid_img_extension))) == 0]
        if len(empty_folders) != 0:
            raise NameError(f'No image files with {vid_img_extension} extension found in {empty_folders}.')
        elif len(image_folders) == 0:
            raise NameError(f'No image folders containing files with {vid_img_extension} extension found in {video_dir}.')
        else:
            logging.info(f'Found image folders with {vid_img_extension} extension.')
            for image_folder_path in image_folders:
                if not USE_YOLO:
                    pose_tracker.reset()
                    if tracking_mode == 'deepsort' and deepsort_tracker is not None:
                        deepsort_tracker.tracker.delete_all_tracks()

                process_images(
                    image_folder_path,
                    vid_img_extension,
                    pose_tracker,
                    pose_model,
                    output_format,
                    frame_rate,
                    save_video,
                    save_images,
                    display_detection,
                    frame_range,
                    tracking_mode,
                    max_distance_px,
                    deepsort_tracker,
                    use_yolo=USE_YOLO,
                    ultra_det=det_solver,
                    pose_solver=pose_solver,
                    pose_dir=pose_dir
                )
