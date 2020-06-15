FROM nvidia/cuda:10.0-cudnn7-runtime-ubuntu18.04

# Pick up some TF dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
        cuda-command-line-tools-10-0 \
        curl \
        libcap-dev \
        libxext6 \
        libxrender1 \
        software-properties-common \
        unzip \
        && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Some TF tools expect a "python" binary
RUN ln -s $(which python3) /usr/local/bin/python

COPY requirements.yaml install_miniconda.sh /tmp/

RUN /tmp/install_miniconda.sh \
    && rm -f /tmp/requirements.yaml /tmp/setup-ray.sh \
    && mkdir /workbench

COPY entrypoint.sh /usr/local/bin/entrypoint.sh

WORKDIR /workbench
VOLUME ["/workbench"]
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
