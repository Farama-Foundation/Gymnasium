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
    && python -c "import importlib, sys; \
        assert importlib.util.find_spec('torch') is None, 'torch leaked into [all]'; \
        mods = [m for m in sys.modules if m.startswith('nvidia')]; \
        assert not mods, f'nvidia packages leaked into [all]: {mods}'" \
    && uv pip install --system .[torch] --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu \
    && python -c "import torch; print(f'torch {torch.__version__} OK')"

ENTRYPOINT ["/usr/local/gymnasium/bin/docker_entrypoint"]
