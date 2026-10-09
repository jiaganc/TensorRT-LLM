# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        git \
        git-lfs \
        python3.12 \
        python3.12-dev \
        python3.12-venv \
        libopenmpi-dev \
        libzmq3-dev \
        cmake \
        ninja-build \
        ccache \
        clang-format \
        build-essential \
        pkg-config \
        sudo \
        docker.io \
        fuse-overlayfs \
        iptables \
        libnvidia-compute-550 \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.12 /usr/local/bin/python3 \
    && git lfs install --system

USER ubuntu
