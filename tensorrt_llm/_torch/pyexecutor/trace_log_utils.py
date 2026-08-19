# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Gated trace/log utilities for pyexecutor.

Leaf module — no other pyexecutor file is imported here, so any consumer
(``_util``, ``model_engine``, ``model_loader``, ``resource_manager``)
can import freely without creating circular dependencies.
"""

import os
import threading
import time
from contextlib import contextmanager
from typing import Iterator

import torch

from tensorrt_llm.logger import logger

_GIB = 1 << 30
_MIB = 1 << 20
_PROC_STATUS_MEMORY_FIELDS = {
    "VmHWM": "host_rss_peak",
    "VmRSS": "host_rss",
    "RssAnon": "host_rss_anon",
    "RssFile": "host_rss_file",
    "RssShmem": "host_rss_shmem",
}


def _read_proc_status_memory() -> dict[str, int]:
    """Read Linux process RSS counters and return their values in bytes."""
    values = {}
    try:
        with open("/proc/self/status", encoding="utf-8") as status:
            for line in status:
                key, separator, value = line.partition(":")
                if not separator or key not in _PROC_STATUS_MEMORY_FIELDS:
                    continue
                values[_PROC_STATUS_MEMORY_FIELDS[key]] = int(value.split()[0]) * 1024
    except OSError:
        return {}
    return values


@contextmanager
def log_mem_delta(tag: str, *, min_delta_mib: int = 64, **metadata: object) -> Iterator[None]:
    """Log significant host RSS changes across one diagnostic function call."""
    enabled = (
        os.environ.get("TLLM_LOG_MEM_PROFILE", "") == "1"
        and os.environ.get("TLLM_LOG_MEM_PROFILE_DETAIL", "") == "1"
    )
    if not enabled:
        yield
        return

    before = _read_proc_status_memory()
    start = time.perf_counter()
    try:
        yield
    finally:
        after = _read_proc_status_memory()
        keys = ("host_rss", "host_rss_anon", "host_rss_file", "host_rss_shmem")
        deltas = {key: after.get(key, 0) - before.get(key, 0) for key in keys}
        if max((abs(delta) for delta in deltas.values()), default=0) >= min_delta_mib * _MIB:
            extras = "".join(f" {key}={value}" for key, value in metadata.items())
            logger.info(
                f"[mem-profile-detail/{tag}] "
                f"rss_delta={deltas['host_rss'] / _GIB:+.2f}GiB "
                f"anon_delta={deltas['host_rss_anon'] / _GIB:+.2f}GiB "
                f"file_delta={deltas['host_rss_file'] / _GIB:+.2f}GiB "
                f"shmem_delta={deltas['host_rss_shmem'] / _GIB:+.2f}GiB "
                f"rss={after.get('host_rss', 0) / _GIB:.2f}GiB "
                f"anon={after.get('host_rss_anon', 0) / _GIB:.2f}GiB "
                f"elapsed={time.perf_counter() - start:.3f}s "
                f"thread={threading.get_ident()}{extras}"
            )


def log_mem_snapshot(tag: str) -> None:
    """Log host RSS counters and Torch/CUDA memory usage.

    Gated by ``TLLM_LOG_MEM_PROFILE=1``; default OFF (zero overhead).

    Prints these fields:

    - ``host_rss``            = current process resident set
    - ``host_rss_peak``       = peak process resident set
    - ``host_rss_anon``       = resident anonymous mappings
    - ``host_rss_file``       = resident file-backed mappings
    - ``host_rss_shmem``      = resident shared-memory mappings
    - ``torch_alloc``         = :func:`torch.cuda.memory_allocated`
    - ``torch_reserved``      = :func:`torch.cuda.memory_reserved`
    - ``torch_alloc_peak``    = :func:`torch.cuda.max_memory_allocated`
    - ``torch_reserved_peak`` = :func:`torch.cuda.max_memory_reserved`
    - ``free``                = ``cuMemGetInfo().free``
    - ``total``               = ``cuMemGetInfo().total``

    Derived quantities the reader may need:

    - ``used      = total - free`` — whole-process GPU consumption
    - ``slack     = reserved - alloc`` — Torch caching allocator free blocks
    - ``non_torch = used - reserved`` — bytes outside Torch (KV pool C++
      cudaMalloc, NCCL buffers, cuBLAS workspace, CUDA driver context,
      CUDA graph mempool, etc.)
    """
    if os.environ.get("TLLM_LOG_MEM_PROFILE", "") != "1":
        return
    fields = _read_proc_status_memory()
    parts = [f"{name}={value / _GIB:.2f}GiB" for name, value in fields.items()]
    if not torch.cuda.is_initialized():
        logger.info(f"[mem-profile/{tag}] " + " ".join(parts))
        return
    free, total = torch.cuda.mem_get_info()
    alloc = torch.cuda.memory_allocated()
    reserved = torch.cuda.memory_reserved()
    alloc_peak = torch.cuda.max_memory_allocated()
    reserved_peak = torch.cuda.max_memory_reserved()
    parts.extend(
        [
            f"torch_alloc={alloc / _GIB:.2f}GiB",
            f"torch_reserved={reserved / _GIB:.2f}GiB",
            f"torch_alloc_peak={alloc_peak / _GIB:.2f}GiB",
            f"torch_reserved_peak={reserved_peak / _GIB:.2f}GiB",
            f"free={free / _GIB:.2f}GiB",
            f"total={total / _GIB:.2f}GiB",
        ]
    )
    logger.info(f"[mem-profile/{tag}] " + " ".join(parts))


def log_tensor_size(tag: str, tensor: torch.Tensor, **extra) -> None:
    """Log a single tensor's footprint (shape / dtype / bytes) at a tag.

    Gated by ``TLLM_LOG_MEM_PROFILE=1``; default OFF (zero overhead).

    Bytes = ``numel * element_size``. Any keyword arguments are appended
    as ``key=value`` for caller-specific context (e.g. routing config).
    """
    if os.environ.get("TLLM_LOG_MEM_PROFILE", "") != "1":
        return
    size_bytes = tensor.numel() * tensor.element_size()
    extras = "".join(f" {k}={v}" for k, v in extra.items())
    logger.info(
        f"[mem-profile/{tag}] "
        f"shape={tuple(tensor.shape)} dtype={tensor.dtype} "
        f"size={size_bytes / 1024 / 1024:.2f}MiB{extras}"
    )
