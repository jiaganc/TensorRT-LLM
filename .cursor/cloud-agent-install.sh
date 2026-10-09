#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

if command -v git-lfs >/dev/null 2>&1; then
  git lfs install --local 2>/dev/null || git lfs install
  git lfs pull
fi

if [[ ! -d .venv ]]; then
  python3 -m venv .venv
fi
# shellcheck source=/dev/null
source .venv/bin/activate

python -m pip install -U pip wheel setuptools
pip install -r requirements-dev.txt
git config --local --unset-all core.hooksPath 2>/dev/null || true
pre-commit install --install-hooks

# Optional: compile and editable-install when a GPU driver is present.
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
  ARCH="${TRTLLM_CUDA_ARCH:-}"
  if [[ -z "${ARCH}" ]]; then
    case "$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d '.')" in
      100) ARCH="100-real" ;;
      90) ARCH="90-real" ;;
      89) ARCH="89-real" ;;
      80) ARCH="80-real" ;;
      *) ARCH="90-real" ;;
    esac
  fi
  python3 scripts/build_wheel.py \
    --use_ccache \
    -a "${ARCH}" \
    --skip_building_wheel \
    --linking_install_binary
  pip install -e .
fi
