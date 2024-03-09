FROM nvcr.io/nvidia/cuda:12.2.0-devel-ubuntu22.04

LABEL sponsor="Hivetrain"

ENV DEBIAN_FRONTEND="noninteractive"

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
    git \
    python3-dev \
    python3-pip \
    python3-packaging \
    python3-venv \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt requirements.txt

RUN pip install -r requirements.txt && \
    pip cache purge

COPY ./ /app

RUN pip install git+https://github.com/LuciferianInk/lightning-Hivemind.git

RUN pip install -e . && \
    pip cache purge

RUN python3 post_install.py

ENTRYPOINT "bash ./entrypoint.sh"