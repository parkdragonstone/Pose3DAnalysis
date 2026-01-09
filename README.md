# Pose3DAnalysis 프로젝트 구조

## 개요
이 프로젝트는 Pose2Sim 을 GUI 기반으로 이용하기 위해 만들어졌음음

## 디렉토리 구조

```
Pose3DAnalysis_v4/
├── app.py                          # 메인 엔트리 포인트 (루트에서 실행)
├── src/                            # 소스 코드 (새로 추가)
│   └── pose3danalysis/             # 메인 패키지
│       ├── __init__.py
│       ├── __main__.py             # 모듈 실행 엔트리 포인트
│       ├── core/                   # 핵심 기능 모듈
│       │   ├── __init__.py
│       │   ├── calib_io.py         # 캘리브레이션 I/O
│       │   ├── frame_extract.py    # 프레임 추출
│       │   ├── point_picker.py     # 포인트 선택
│       │   ├── utils.py            # 유틸리티 함수
│       │   ├── video.py            # 비디오 처리
│       │   └── zoom_preview.py     # 줌 프리뷰
│       ├── gui/                    # GUI 모듈
│       │   ├── __init__.py
│       │   ├── main_app.py         # 메인 앱
│       │   ├── tabs/               # 탭 모듈
│       │   │   ├── __init__.py
│       │   │   ├── calibration_tab.py
│       │   │   ├── motion_analysis_tab.py
│       │   │   └── batch_processing_tab.py
│       │   ├── components/         # GUI 컴포넌트
│       │   │   ├── __init__.py
│       │   │   ├── calibration_preview.py
│       │   │   ├── multicam_preview.py
│       │   │   ├── scrollable_frame.py
│       │   │   ├── side_settings_panel.py
│       │   │   └── viewer3d_panel.py
│       │   ├── calibration/        # 캘리브레이션 관련 모듈
│       │   │   ├── __init__.py
│       │   │   ├── camera_state.py
│       │   │   ├── extrinsic_processor.py
│       │   │   ├── intrinsic_processor.py
│       │   │   ├── preview_renderer.py
│       │   │   └── session_manager.py
│       │   └── utils/              # GUI 유틸리티
│       │       ├── __init__.py
│       │       ├── calibration_utils.py
│       │       ├── helpers.py
│       │       ├── logging_utils.py
│       │       ├── motion_settings.py
│       │       ├── pose_utils.py
│       │       └── video_utils.py
│       └── runner/                 # 실행 관련 모듈
│           ├── __init__.py
│           ├── config_builder.py
│           └── pose2sim_runner.py
├── Pose2Sim/                       # Pose2Sim 라이브러리 (외부, 필요한 경우 일부 수정하였음)
├── models/                         # 모델 파일 (기존의 방식은 Fine-Tuning 한 모델을 사용하기 힘들었는데 Local 부분에 업로드해서 본인의 모델을 사용하기 용이하게 함)
├── docker/                         # Docker 설정
├── tests/                          # 테스트
│   ├── __init__.py
│   └── test_helpers.py
├── pyproject.toml                  # 프로젝트 설정
└── README.md                       # 프로젝트 문서
```


### 1. Installation

#### 가상환경 설치
```bash
conda create --name p3a python=3.12 -y
conda activate p3a
```

#### Opensim Package 다운로드
```bash
conda install -c opensim-org opensim -y
```

#### Pytorch 설치치
```bash
# ROCM 6.1 (Linux only)
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/rocm6.1
# CUDA 11.8
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu118
# CUDA 12.1
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu121
# CUDA 12.4
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu124
# CPU only
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cpu
```

### 2. 실행 방법

#### 방법 1: app.py 사용 (기존 방식)
```bash
python app.py
```

#### 방법 2: 모듈로 실행 (권장)
```bash
python -m pose3danalysis
```

#### 방법 3: 설치 후 실행
```bash
pip install -e .
pose3danalysis
```


```bash
docker build -f docker/Dockerfile -t p3a .
echo $DISPLAY
docker run --gpus all --name --shm-size 16g -it --rm \
  -e DISPLAY=$DISPLAY \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v /mnt/c/Users/USER/Desktop/demo:/demo \
  -v "$PWD":/app p3a

docker run --gpus all --name p3a --shm-size 16g -it --rm \
  -e DISPLAY=$DISPLAY \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v /mnt/c/Users/USER/Desktop/demo:/demo \
  -v "$PWD":/app p3a bash
```

## Features (Calibration)

- Load videos and preview side-by-side
- Intrinsic calibration (checkerboard) -> `lens_calibration.toml`
  - finishes with popup showing reprojection errors (px) per camera
- Extrinsic calibration
  - board method (checkerboard) or scene/object method
  - scene/object method supports manual clicking of 2D points via OpenCV windows
  - finishes with popup showing reprojection error (clicked vs projected) per camera

## Notes

- For scene method, you must provide at least 6 3D points (x,y,z) in **mm**.
- Point picking uses the **current frame** shown in the preview.
  You can pause at a good frame, then click points.

## Acknowledgments

This project is built on top of [Pose2Sim](https://github.com/perfanalytics/pose2sim), a free and open-source workflow for 3D markerless kinematics. Pose2Sim provides the core functionality for pose estimation, triangulation, filtering, and OpenSim integration.

### Pose2Sim

**Pose2Sim** is a workflow for 3D markerless kinematics (human or animal), providing an alternative to traditional marker-based MoCap methods. It is free and open-source, requiring low-cost hardware but with research-grade accuracy and production-grade robustness.

- **Repository**: [https://github.com/perfanalytics/pose2sim](https://github.com/perfanalytics/pose2sim)
- **License**: BSD-3-Clause
- **Documentation**: [https://perfanalytics.github.io/pose2sim/](https://perfanalytics.github.io/pose2sim/)

### What This Project Adds

Pose3DAnalysis extends Pose2Sim by providing:

- **GUI-based Interface**: A user-friendly desktop application built with Tkinter for easier workflow management
- **Enhanced Calibration Tools**: Improved multi-camera calibration interface with real-time preview
- **Batch Processing**: Process multiple folders sequentially with shared settings
- **3D Visualization**: Interactive 3D viewer for marker data with coordinate system controls
- **Local Model Support**: Easier integration of custom fine-tuned models

### Citation

If you use Pose2Sim in your research, please cite the original Pose2Sim project. For details on how to cite Pose2Sim, please refer to the [Pose2Sim repository](https://github.com/perfanalytics/pose2sim).

## License

This project is licensed under the BSD 3-Clause License - see the [LICENSE](LICENSE) file for details.

## Authors

- **Yongseok Park** (pys9610@gmail.com) - Initial development and GUI implementation

See the [AUTHORS](AUTHORS) file for a complete list of contributors and acknowledgments.
