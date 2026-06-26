# ============================================================================
# Patched Dockerfile for KS-PlaNet (Fixes CMake and Bazel Wheel Build Errors)
# ============================================================================

FROM nvcr.io/nvidia/tensorflow:20.10-tf1-py3

LABEL maintainer="KS-PlaNet"
LABEL description="PlaNet for KS + Kolmogorov PDE world-model learning"

ARG http_proxy
ARG https_proxy
ENV HTTP_PROXY=$http_proxy
ENV HTTPS_PROXY=$https_proxy

# Step 1: Force upgrade pip/setuptools and install wheel/cmake binaries
RUN python -m pip install --no-cache-dir --upgrade pip=="21.3.1" setuptools wheel && \
    python -m pip install --no-cache-dir cmake

# CRITICAL FIX: Ensure the new pip-installed CMake takes absolute priority over /usr/bin/cmake
ENV PATH="/usr/local/bin:${PATH}"

# Step 2: Install cloudpickle & TF Probability forcefully bypassing strict version checks
RUN pip install --no-cache-dir 'cloudpickle==1.3.0' && \
    pip install --no-cache-dir --no-deps 'tensorflow_probability==0.8.0'

# Step 3: Install structural components with explicitly pinned Python 3.6 binary wheels
RUN pip install --no-cache-dir \
        'gym>=0.17.0,<=0.21.0' \
        'dm-tree==0.1.6' \
        'labmaze==1.0.3' \
        'dm_control' \
        'ruamel.yaml>=0.15.0,<0.16.0' \
        'scikit-image>=0.17.0' \
        'opencv-python-headless==4.3.0.38'

WORKDIR /workspace

ENV TF_CPP_MIN_LOG_LEVEL=2

CMD ["/bin/bash"]