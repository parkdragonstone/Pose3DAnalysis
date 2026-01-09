# (추정) 기반이 CUDA 12.4.1 + cuDNN9 + Ubuntu 20.04 계열로 보임
FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC

RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common ca-certificates curl wget gnupg lsb-release \
    build-essential cmake ninja-build git tzdata \
    libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 \
 && add-apt-repository ppa:deadsnakes/ppa -y \
 && apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3.10-dev python3.10-venv \
 && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 \
 && ln -sf /usr/bin/python3 /usr/bin/python \
 && python -V

RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.10 \
 && python3 -m pip install --no-cache-dir --upgrade pip wheel setuptools cython

RUN pip install --no-cache-dir \
    torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 \
    --index-url https://download.pytorch.org/whl/cu124

RUN pip install -U openmim
RUN mim install "mmengine" "mmcv<2.2.0"
RUN mim install "mmdet<3.3.0"

RUN conda run -n p3a python -c "import mmcv; import mmcv._ext; print('mmcv._ext OK', mmcv.__version__, mmcv._ext.__file__)"

RUN pip install mmpose
RUN pip install ultralytics

WORKDIR /mmpose
RUN git clone https://github.com/open-mmlab/mmpose.git /mmpose \
 && git checkout main
RUN pip install -r requirements.txt
RUN pip install --no-cache-dir -e .

# numpy 핀 고정(중간에 uninstall/reinstall 흔적 존재)
RUN pip uninstall -y numpy
RUN pip install numpy==1.26.4

WORKDIR /workspace