FROM nvcr.io/nvidia/cuda:12.2.0-devel-ubuntu22.04

LABEL sponsor="Hivetrain"

WORKDIR /hivetrain

ENV DEBIAN_FRONTEND="noninteractive"

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
    git \
    python3-dev \
    python3-pip \
    python3-packaging \
    python3-venv \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements.txt

RUN pip install -r requirements.txt && \
    pip cache purge

COPY ./ .

RUN pip install -r requirements.docker.txt && \
    pip cache purge

ENTRYPOINT ["bash", "./entrypoint-miner.sh"]