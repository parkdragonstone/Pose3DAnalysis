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


### 3. 실행 방법

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

### 4. pyproject.toml 업데이트
- `[tool.setuptools]` 섹션에 `packages = {find = {where = ["src"]}}` 추가
- `[project.scripts]`에 `pose3danalysis` entry point 추가

## 모듈 구조

### core/
핵심 기능 모듈:
- `calib_io.py`: 캘리브레이션 데이터 I/O
- `frame_extract.py`: 비디오 프레임 추출
- `point_picker.py`: 포인트 선택 UI
- `utils.py`: 공통 유틸리티 함수
- `video.py`: 비디오 처리
- `zoom_preview.py`: 줌 가능한 프리뷰 위젯

### gui/
GUI 모듈:
- `main_app.py`: 메인 애플리케이션 클래스
- `tabs/`: 각 탭 구현
  - `calibration_tab.py`: 캘리브레이션 탭
  - `motion_analysis_tab.py`: 모션 분석 탭
  - `batch_processing_tab.py`: 배치 처리 탭
- `components/`: 재사용 가능한 GUI 컴포넌트
- `calibration/`: 캘리브레이션 관련 로직
- `utils/`: GUI 유틸리티 함수

### runner/
실행 관련 모듈:
- `config_builder.py`: Pose2Sim 설정 빌더
- `pose2sim_runner.py`: Pose2Sim 파이프라인 실행

## 개발 환경 설정

### 개발 모드 설치
```bash
pip install -e .
```

이렇게 하면 소스 코드 변경사항이 즉시 반영됩니다.

### 테스트 실행
```bash
python -m pytest tests/
```

## 향후 개선 방향

1. **추가 모듈 분리**
   - 더 세분화된 모듈 구조
   - 명확한 책임 분리

2. **테스트 확장**
   - 각 모듈에 대한 단위 테스트 추가
   - 통합 테스트 추가

3. **문서화**
   - 각 모듈에 docstring 추가
   - API 문서 생성 (Sphinx 등)

4. **타입 힌팅**
   - 모든 함수에 타입 힌팅 추가
   - mypy를 사용한 타입 체크
