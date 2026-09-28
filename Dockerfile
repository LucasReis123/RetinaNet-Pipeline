# Ambiente de treinamento do RetinaNet (Detectron2).
#
# O Detectron2 não tem wheel pronta: ele é compilado aqui. Por isso a imagem base precisa do
# nvcc (nvidia/cuda:*-devel), e a versão do CUDA da base tem que bater com a do PyTorch.
#
#   CUDA_VERSION / TORCH_CUDA   12.6.3 / cu126 (padrão), 12.8.1 / cu128, 12.4.1 / cu124, 11.8.0 / cu118
#   TORCH_CUDA_ARCH_LIST        arquiteturas de GPU compiladas (8.6 = RTX 30xx, 8.9 = RTX 40xx, 12.0 = RTX 50xx)
#   BASE_IMAGE                  sobrescreve a base inteira (ex.: ubuntu:22.04 para CPU)

ARG CUDA_VERSION=12.6.3
ARG BASE_IMAGE=nvidia/cuda:${CUDA_VERSION}-devel-ubuntu22.04
FROM ${BASE_IMAGE}

ARG PYTHON_VERSION=3.12
ARG TORCH_CUDA=cu126
ARG TORCH_VERSION=2.9.1
ARG TORCHVISION_VERSION=0.24.1
ARG DETECTRON2_REF=main
ARG FORCE_CUDA=1
ARG TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6;8.9;9.0+PTX"
ARG MAX_JOBS=8

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    UV_PYTHON_INSTALL_DIR=/opt/python \
    VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:$PATH \
    HOME=/tmp \
    MPLBACKEND=Agg \
    MPLCONFIGDIR=/tmp/matplotlib \
    FVCORE_CACHE=/workspace/app/.cache/fvcore

# Compiladores e bibliotecas do OpenCV
RUN apt-get update \
    && apt-get install -y --no-install-recommends git build-essential ninja-build ca-certificates libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Python isolado (a versão do Ubuntu não importa)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
RUN uv venv --seed --python ${PYTHON_VERSION} ${VIRTUAL_ENV}

RUN pip install \
        "torch==${TORCH_VERSION}" \
        "torchvision==${TORCHVISION_VERSION}" \
        --index-url "https://download.pytorch.org/whl/${TORCH_CUDA}"

# setuptools<81: o model_zoo do Detectron2 ainda usa pkg_resources
RUN pip install "setuptools<81" wheel ninja

RUN FORCE_CUDA=${FORCE_CUDA} TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST}" MAX_JOBS=${MAX_JOBS} \
    pip install --no-build-isolation "git+https://github.com/facebookresearch/detectron2.git@${DETECTRON2_REF}"

WORKDIR /workspace/app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY *.py ./

CMD ["python", "main.py"]
