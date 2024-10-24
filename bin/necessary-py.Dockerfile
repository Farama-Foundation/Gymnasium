# A Dockerfile that sets up a full Gymnasium install with test dependencies
ARG PYTHON_VERSION
FROM python:$PYTHON_VERSION

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

RUN apt-get -y update \
    && apt-get install --no-install-recommends -y \
    unzip \
    libglu1-mesa-dev \
    libgl1-mesa-dev \
    libosmesa6-dev \
    xvfb \
    patchelf \
    ffmpeg cmake \
    && apt-get autoremove -y \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

COPY . /usr/local/gymnasium/
WORKDIR /usr/local/gymnasium/

RUN pip install uv
RUN uv pip install --system --upgrade "numpy>=1.21,<2.0"
RUN uv pip install --system .[testing] --no-cache-dir

ENTRYPOINT ["/usr/local/gymnasium/bin/docker_entrypoint"]
