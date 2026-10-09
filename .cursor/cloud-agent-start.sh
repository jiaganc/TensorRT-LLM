#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

if ! command -v dockerd >/dev/null 2>&1; then
  exit 0
fi

if sudo docker info >/dev/null 2>&1; then
  exit 0
fi

sudo mkdir -p /etc/docker
if [[ ! -f /etc/docker/daemon.json ]]; then
  echo '{"storage-driver":"fuse-overlayfs"}' | sudo tee /etc/docker/daemon.json >/dev/null
fi
sudo update-alternatives --set iptables /usr/sbin/iptables-legacy 2>/dev/null || true

if ! pgrep -x dockerd >/dev/null 2>&1; then
  sudo dockerd >/tmp/dockerd.log 2>&1 &
fi

for _ in $(seq 1 60); do
  if sudo docker info >/dev/null 2>&1; then
    break
  fi
  sleep 1
done

sudo usermod -aG docker "${USER:-ubuntu}" 2>/dev/null || true
