# syntax=docker/dockerfile:1.7
FROM nvcr.io/nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04 AS builder

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

ARG DEBIAN_FRONTEND=noninteractive
ARG PYTHON_VERSION=3.12
ARG TORCH_CUDA_ARCH_LIST=7.0

ARG PYTORCH_REPO=https://github.com/pedestrianlove/pytorch.git
ARG PYTORCH_REF=0e270059155c368f9fbe6fac20253a1e0ce73d4e
ARG NCCL_REPO=https://github.com/pedestrianlove/nccl.git
ARG NCCL_REF=361915904b456d397e6e1578f8f65ea1a45bdd28
ARG APEX_REF=ba32a259b7aa4a7d797369543ead466fe4a760a5
ARG NVRX_REF=432d5844eaddcbb7a1c6a3ad6b9c3d2697effc7a
ARG EMERGING_REF=8d12ddf22e5a7092ae2c21ea5eceaf70eb7f79b3
ARG GALORE_REF=2cc66f88cce189e505affbb91042a8e77f5bf4e9
ARG Q_GALORE_REF=5bd2f1c5a1aded31223d7f15c4604c4d56cfe1e6
ARG APOLLO_PKG_REF=830f7646472497c742ffd5863e18e5156a46a5cc
ARG MEGATRON_REPO=https://github.com/pedestrianlove/Megatron-LM.git
ARG MEGATRON_BRANCH=nonuniform-tp
ARG MEGATRON_REF=01e3ac2b6a68f504aa1f8e8b3ccbd404d387476b

ENV TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST} \
    CUDA_HOME=/usr/local/cuda \
    CUDA_PATH=/usr/local/cuda \
    CUDNN_PATH=/usr/local/cuda \
    MAGMA_HOME=/usr/local/cuda-12.8/magma \
    CMAKE_PREFIX_PATH=/usr/local:/usr/local/cuda \
    CUDA_INC_PATH=/usr/local/cuda/include:/usr/local/include \
    CUDA_LIB_PATH=/usr/local/cuda/lib64:/usr/local/lib \
    OMP_NUM_THREADS=4 \
    PIP_ROOT_USER_ACTION=ignore \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:${PATH} \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64:/opt/src/pytorch/build/nccl/lib:/opt/venv/lib/python3.12/site-packages/torch/lib

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        git \
        build-essential \
        pkg-config \
        ccache \
        patchelf \
        libssl-dev \
        zlib1g-dev \
        libbz2-dev \
        libreadline-dev \
        libsqlite3-dev \
        libffi-dev \
        liblzma-dev \
        python${PYTHON_VERSION} \
        python${PYTHON_VERSION}-dev \
        python${PYTHON_VERSION}-venv && \
    rm -rf /var/lib/apt/lists/*

RUN python${PYTHON_VERSION} -m venv /opt/venv && \
    curl -fsSL https://bootstrap.pypa.io/get-pip.py -o /tmp/get-pip.py && \
    /opt/venv/bin/python /tmp/get-pip.py && \
    rm /tmp/get-pip.py && \
    ln -sf /opt/venv/bin/python /usr/local/bin/python && \
    ln -sf /opt/venv/bin/python /usr/local/bin/python3 && \
    ln -sf /opt/venv/bin/pip /usr/local/bin/pip && \
    ln -sf /opt/venv/bin/pip /usr/local/bin/pip3

RUN python --version && \
    pip --version

RUN pip install --upgrade \
        pip==25.1.1 \
        'setuptools<80' \
        wheel && \
    pip install \
        cmake==4.0.3 \
        ninja \
        numpy \
        'packaging>=24.2' \
        pyyaml \
        requests \
        six \
        'typing-extensions>=4.15.0' \
        pybind11 \
        'mkl-static>=2025.1.0,<2026' \
        'mkl-include>=2025.1.0,<2026'

WORKDIR /opt/src

RUN git clone --branch main --recursive ${PYTORCH_REPO} pytorch && \
    git -C pytorch checkout ${PYTORCH_REF} && \
    git -C pytorch submodule sync && \
    git -C pytorch submodule update --init --recursive && \
    git clone --branch master ${NCCL_REPO} pytorch/third_party/nccl && \
    git -C pytorch/third_party/nccl checkout ${NCCL_REF}

WORKDIR /opt/src/pytorch

RUN pip install -v --group dev

RUN bash .ci/docker/common/install_magma.sh 12.8

RUN --mount=type=cache,id=ccache-ubuntu2404,target=/root/.cache/ccache \
    --mount=type=cache,id=pytorch-build-ubuntu2404,target=/opt/src/pytorch/build \
    BUILD_JOBS="$(nproc)" && \
    export MAX_JOBS="${BUILD_JOBS}" && \
    export CMAKE_BUILD_PARALLEL_LEVEL="${BUILD_JOBS}" && \
    export CCACHE_DIR=/root/.cache/ccache && \
    export CMAKE_C_COMPILER_LAUNCHER=ccache && \
    export CMAKE_CXX_COMPILER_LAUNCHER=ccache && \
    make -j"${BUILD_JOBS}" triton

RUN --mount=type=cache,id=ccache-ubuntu2404,target=/root/.cache/ccache \
    --mount=type=cache,id=pytorch-build-ubuntu2404,target=/opt/src/pytorch/build \
    BUILD_JOBS="$(nproc)" && \
    export MAX_JOBS="${BUILD_JOBS}" && \
    export CMAKE_BUILD_PARALLEL_LEVEL="${BUILD_JOBS}" && \
    export CCACHE_DIR=/root/.cache/ccache && \
    export CMAKE_C_COMPILER_LAUNCHER=ccache && \
    export CMAKE_CXX_COMPILER_LAUNCHER=ccache && \
    export USE_NUMPY=1 && \
    export USE_CUDA=1 && \
    export USE_CUDNN=1 && \
    export USE_NCCL=1 && \
    export USE_XPU=0 && \
    export USE_DISTRIBUTED=1 && \
    export USE_SYSTEM_NCCL=0 && \
    export CMAKE_POLICY_VERSION_MINIMUM=3.5 && \
    pip install --no-build-isolation -v .

WORKDIR /opt/src

RUN pip install torch-c-dlpack-ext && \
    python - <<'PY'
import torch
print(torch.__version__)
print(torch.version.cuda)
PY

RUN BUILD_JOBS="$(nproc)" && \
    export CPLUS_INCLUDE_PATH="/opt/src/pytorch/build/nccl/include:/usr/local/cuda/include:${CPLUS_INCLUDE_PATH}" && \
    export LIBRARY_PATH="/opt/src/pytorch/build/nccl/lib:${LIBRARY_PATH}" && \
    export CUDA_HOME=/usr/local/cuda && \
    export APEX_PARALLEL_BUILD="${BUILD_JOBS}" && \
    export APEX_CPP_EXT=1 && \
    export APEX_CUDA_EXT=1 && \
    export NVCC_APPEND_FLAGS="--threads ${BUILD_JOBS}" && \
    pip install -v --no-build-isolation git+https://github.com/pedestrianlove/apex.git@${APEX_REF}

RUN BUILD_JOBS="$(nproc)" && \
    export CPLUS_INCLUDE_PATH="/opt/src/pytorch/build/nccl/include:/usr/local/cuda/include:${CPLUS_INCLUDE_PATH}" && \
    export LIBRARY_PATH="/opt/src/pytorch/build/nccl/lib:${LIBRARY_PATH}" && \
    export MAX_JOBS="${BUILD_JOBS}" && \
    pip install -v --no-build-isolation 'transformer_engine[pytorch]<2.12.0'

RUN pip install git+https://github.com/pedestrianlove/nvidia-resiliency-ext.git@${NVRX_REF}

RUN pip install --no-build-isolation git+https://github.com/pedestrianlove/Emerging-Optimizers.git@${EMERGING_REF}

WORKDIR /opt/src

RUN git clone --branch ${MEGATRON_BRANCH} ${MEGATRON_REPO} Megatron-LM && \
    git -C Megatron-LM checkout ${MEGATRON_REF}

WORKDIR /opt/src/Megatron-LM

RUN pip install --no-build-isolation -e .

RUN pip install 'nvidia-nvcomp-cu12>=5.0.0.6,<6'

RUN pip install \
    git+https://github.com/pedestrianlove/GaLore.git@${GALORE_REF} \
    git+https://github.com/pedestrianlove/Q-GaLore.git@${Q_GALORE_REF} \
    git+https://github.com/pedestrianlove/APOLLO.git@${APOLLO_PKG_REF}

RUN cat <<'EOF' >/usr/local/bin/stack-smoke-test
#!/usr/bin/env bash
set -euo pipefail

python - <<'PY'
import ctypes
import importlib

for candidate in ("libnccl.so", "libnccl.so.2"):
    try:
        ctypes.CDLL(candidate)
        print("nccl_lib", candidate)
        break
    except OSError:
        pass
else:
    raise OSError("Unable to load libnccl.so or libnccl.so.2")

torch = importlib.import_module("torch")
apex = importlib.import_module("apex")
te = importlib.import_module("transformer_engine.pytorch")
megatron_core = importlib.import_module("megatron.core")
nvrx = importlib.import_module("nvidia_resiliency_ext")
nvcomp = importlib.import_module("nvidia.nvcomp")

print("torch", torch.__version__)
print("torch_cuda", torch.version.cuda)
print("cuda_available", torch.cuda.is_available())
print("device_count", torch.cuda.device_count())
print("nccl_version", torch.cuda.nccl.version())
print("apex", apex.__file__)
print("transformer_engine", te.__file__)
print("megatron_core", megatron_core.__file__)
print("nvidia_resiliency_ext", nvrx.__file__)
print("nvidia_nvcomp", nvcomp.__file__)

if torch.cuda.is_available():
    capability = torch.cuda.get_device_capability(0)
    print("device0_cc", capability)
    if capability[0] >= 7:
        x = torch.randn(4, device="cuda")
        print("cuda_tensor", x.tolist())
    else:
        print("cuda_tensor_skipped", f"host GPU CC {capability[0]}.{capability[1]} is below sm_70 build target")
PY

pip show torch
pip show apex
pip show transformer-engine || true
pip show nvidia-resiliency-ext
pip show megatron-core
pip show nvidia-nvcomp-cu12
pip show galore-torch
pip show q-galore-torch
pip show apollo-torch
EOF

RUN chmod +x /usr/local/bin/stack-smoke-test

WORKDIR /workspace

CMD ["bash"]

FROM nvcr.io/nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04 AS runtime

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

ARG DEBIAN_FRONTEND=noninteractive
ARG PYTHON_VERSION=3.12
ARG TORCH_CUDA_ARCH_LIST=7.0

ENV TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST} \
    CUDA_HOME=/usr/local/cuda \
    CUDA_PATH=/usr/local/cuda \
    CUDNN_PATH=/usr/local/cuda \
    CMAKE_PREFIX_PATH=/usr/local:/usr/local/cuda \
    CUDA_INC_PATH=/usr/local/cuda/include:/usr/local/include \
    CUDA_LIB_PATH=/usr/local/cuda/lib64:/usr/local/lib \
    OMP_NUM_THREADS=4 \
    PIP_ROOT_USER_ACTION=ignore \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:${PATH} \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64:/opt/venv/lib/python3.12/site-packages/torch/lib

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        ca-certificates \
        libgomp1 \
        python${PYTHON_VERSION} \
        python${PYTHON_VERSION}-dev && \
    rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /opt/src/Megatron-LM /opt/src/Megatron-LM
COPY --from=builder /usr/local/bin/stack-smoke-test /usr/local/bin/stack-smoke-test

RUN rm -rf /opt/src/Megatron-LM/.git && \
    ln -sf /opt/venv/bin/python /usr/local/bin/python && \
    ln -sf /opt/venv/bin/python /usr/local/bin/python3 && \
    ln -sf /opt/venv/bin/pip /usr/local/bin/pip && \
    ln -sf /opt/venv/bin/pip /usr/local/bin/pip3 && \
    chmod +x /usr/local/bin/stack-smoke-test

WORKDIR /workspace

CMD ["bash"]
