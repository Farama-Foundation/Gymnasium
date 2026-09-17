# A Dockerfile that sets up a full Gymnasium install with test dependencies
ARG PYTHON_VERSION
FROM python:$PYTHON_VERSION
ARG NUMPY_VERSION=">=1.21,<2.0"

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

RUN apt-get -y update \
    && apt-get install --no-install-recommends -y \
    libglu1-mesa-dev libgl1-mesa-dev libosmesa6-dev \
    xvfb unzip patchelf ffmpeg cmake swig \
    && apt-get autoremove -y \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN pip install uv

COPY . /usr/local/gymnasium/
WORKDIR /usr/local/gymnasium/

# Specify the numpy version to cover both 1.x and 2.x
# Test with PyTorch CPU build, since CUDA is not available in CI anyway
# Regression: [all] must not pull torch or nvidia-* packages
RUN uv pip install --system .[all,testing] "numpy$NUMPY_VERSION" --no-cache-dir \
    && python -c "from importlib.metadata import distributions; \
        names = {(dist.metadata.get('Name') or '').lower().replace('_', '-') for dist in distributions()}; \
        leaked = sorted(name for name in names if name == 'torch' or name.startswith('nvidia-')); \
        assert not leaked, f'torch/nvidia packages leaked into [all]: {leaked}'" \
    && uv pip install --system .[torch] --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu \
    && python -c "import torch; \
        assert torch.version.cuda is None, f'CUDA-enabled torch installed: {torch.__version__}'; \
        print(f'torch {torch.__version__} CPU OK')"

ENTRYPOINT ["/usr/local/gymnasium/bin/docker_entrypoint"]
