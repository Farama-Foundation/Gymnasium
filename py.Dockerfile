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
    && rm -rf /var/lib/apt/lists/* \
    # Download mujoco
    && mkdir /root/.mujoco \
    && cd /root/.mujoco \
    && wget -qO- 'https://github.com/deepmind/mujoco/releases/download/2.1.0/mujoco210-linux-x86_64.tar.gz' | tar -xzvf -

ENV LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/root/.mujoco/mujoco210/bin"

COPY . /usr/local/gymnasium/
WORKDIR /usr/local/gymnasium/

RUN if [ "python:${PYTHON_VERSION}" = "python:3.6.15" ] ; then pip install .[box2d,classic_control,toy_text,other] pytest=="7.0.1" gym>="0.26" --no-cache-dir; else pip install .[testing] --no-cache-dir; fi

ENTRYPOINT ["/usr/local/gymnasium/bin/docker_entrypoint"]
