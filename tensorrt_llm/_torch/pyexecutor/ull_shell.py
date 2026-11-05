# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Switches of the low-latency decode-step shell (everything around the model's layers).

Each switch is read once per process.

* ``TRTLLM_K3_ULL_STEP_PREP=1`` (default off): one-launch step preparation. The batched
  input commit (``input_commit.py``) runs for CUDA-graph generation steps, also on engines
  whose MLA layers use the FP4 KV cache (``TrtllmAttentionMetadata.fp4_mla_state``, the
  Kimi-K3 helix path), which ``TRTLLM_BATCHED_INPUT_COMMIT=1`` alone leaves on the
  copy-by-copy path.
* ``TRTLLM_K3_ULL_PIN_EXECUTOR=auto`` (default off): pin the executor loop thread to one
  physical core of the process's affinity set (one core per GPU of the node, never CPU 0's
  core) and keep the process's other threads off that core and its SMT siblings
  (:class:`ExecutorThreadPin`).
* ``TRTLLM_K3_ULL_GIL_SWITCH_US=<us>`` (default unset): the interpreter's thread switch
  interval (``sys.setswitchinterval``, default 5000 us). A thread that wants the GIL asks its
  holder to drop it only after this interval, so a CPU-bound background thread can hold the
  executor for up to the interval.

Two switches of the same shell select what this executor already does for every one-model
speculative engine, so they have nothing left to enable (:func:`report_default_on_switches`
logs that when they are set):

* ``TRTLLM_K3_ULL_GRAPH_INPUT_ALIAS``: the decode CUDA graphs read their input ids and
  positions straight from the engine's buffers (``DecoderRunner._cuda_graph_runner_config``
  passes ``static_input_ids`` / ``static_position_ids`` for speculative engines without
  MRoPE), so a replay copies nothing in front of the graph.
* ``TRTLLM_K3_ULL_SAMPLE_EXEC_STREAM``: the one-model speculative sampler (``SpecSampler``)
  runs on the execution stream in ``PyExecutor._executor_loop_overlap``.
"""

import os
import sys
import threading
from typing import Optional, Set

from tensorrt_llm.logger import logger

STEP_PREP_ENV = "TRTLLM_K3_ULL_STEP_PREP"
GRAPH_INPUT_ALIAS_ENV = "TRTLLM_K3_ULL_GRAPH_INPUT_ALIAS"
SAMPLE_EXEC_STREAM_ENV = "TRTLLM_K3_ULL_SAMPLE_EXEC_STREAM"
PIN_EXECUTOR_ENV = "TRTLLM_K3_ULL_PIN_EXECUTOR"
GIL_SWITCH_US_ENV = "TRTLLM_K3_ULL_GIL_SWITCH_US"

_STEP_PREP = os.environ.get(STEP_PREP_ENV, "0") == "1"


def step_prep_enabled() -> bool:
    """Whether the batched input commit runs, and also covers the FP4 MLA KV path."""
    return _STEP_PREP


def report_default_on_switches() -> None:
    """Log the shell switches that select this executor's default behavior, when set."""
    defaults = {
        GRAPH_INPUT_ALIAS_ENV: "decode CUDA graphs of one-model speculative engines read "
        "the engine's input id and position buffers",
        SAMPLE_EXEC_STREAM_ENV: "the one-model speculative sampler runs on the execution stream",
    }
    for name, behavior in defaults.items():
        value = os.environ.get(name)
        if value is not None:
            logger.info(f"{name}={value}: always on here ({behavior}); the switch has no effect")


def apply_gil_switch_interval() -> Optional[float]:
    """Set ``sys.setswitchinterval`` from ``TRTLLM_K3_ULL_GIL_SWITCH_US``.

    Returns:
        The interval set, in seconds, or None when the switch is unset or ``0``.

    Raises:
        ValueError: The value is negative.
    """
    value = os.environ.get(GIL_SWITCH_US_ENV, "").strip()
    if not value or value == "0":
        return None
    seconds = float(value) * 1e-6
    if seconds <= 0:
        raise ValueError(f"{GIL_SWITCH_US_ENV} must be > 0, got {value!r}")
    sys.setswitchinterval(seconds)
    logger.info(
        f"{GIL_SWITCH_US_ENV}: thread switch interval {sys.getswitchinterval() * 1e6:.0f} us"
    )
    return sys.getswitchinterval()


def _physical_gpu_index() -> int:
    """The node-local index of this process's GPU (0 without CUDA)."""
    try:
        import torch

        if not torch.cuda.is_available():
            return 0
        device = torch.cuda.current_device()
    except (ImportError, RuntimeError):
        return 0
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
    if device < len(visible) and visible[device].strip().isdigit():
        return int(visible[device])
    return device


def pin_executor_cpu(
    available: Optional[Set[int]] = None, gpu_index: Optional[int] = None
) -> Optional[int]:
    """The CPU the executor thread is pinned to, or None when the switch is off.

    One physical core per GPU: the ranks of a node often share one NUMA-wide set, so the core
    is the ``gpu_index``-th of the set's physical cores (lowest CPU of each SMT group),
    skipping the core of CPU 0 (housekeeping interrupts).

    Args:
        available: The process's affinity set (``os.sched_getaffinity(0)`` by default).
        gpu_index: The node-local index of this process's GPU (default: from the current
            CUDA device and ``CUDA_VISIBLE_DEVICES``).

    Raises:
        ValueError: ``TRTLLM_K3_ULL_PIN_EXECUTOR`` is set to something other than ``auto``,
            ``0`` or the empty string.
    """
    value = os.environ.get(PIN_EXECUTOR_ENV, "").strip()
    if not value or value == "0":
        return None
    if value != "auto":
        raise ValueError(f"{PIN_EXECUTOR_ENV} must be auto, got {value!r}")
    cpus = sorted(available if available is not None else os.sched_getaffinity(0))
    if not cpus:
        return None
    cores, seen = [], set()
    for cpu in cpus:
        if cpu in seen:
            continue
        siblings = _smt_siblings(cpu)
        seen |= siblings
        if 0 not in siblings:
            cores.append(cpu)
    if not cores:
        return cpus[0]
    if gpu_index is None:
        gpu_index = _physical_gpu_index()
    return cores[gpu_index % len(cores)]


def _smt_siblings(cpu: int) -> Set[int]:
    """``cpu`` and its SMT siblings (just ``cpu`` when sysfs does not say)."""
    path = f"/sys/devices/system/cpu/cpu{cpu}/topology/thread_siblings_list"
    try:
        with open(path) as f:
            text = f.read().strip()
    except OSError:
        return {cpu}
    siblings: Set[int] = set()
    for part in text.split(","):
        if "-" in part:
            a, b = part.split("-")
            siblings.update(range(int(a), int(b) + 1))
        elif part:
            siblings.add(int(part))
    return siblings | {cpu}


class ExecutorThreadPin:
    """Keeps the executor loop thread alone on one CPU (``TRTLLM_K3_ULL_PIN_EXECUTOR``).

    The calling thread gets ``{cpu}``; every other thread of the process gets the process's
    CPU set without ``cpu`` and its SMT siblings. Threads started later inherit the affinity
    of the thread that starts them, so :meth:`maybe_refresh` moves new threads off the CPU
    every ``refresh_iters`` iterations (a scan of ``/proc/self/task``).

    Args:
        cpu: The CPU the calling thread is pinned to.
        available: The process's affinity set.
        refresh_iters: Iterations between two scans for new threads.
    """

    def __init__(self, cpu: int, available: Set[int], refresh_iters: int = 512):
        self.cpu = cpu
        self.tid = threading.get_native_id()
        reserved = _smt_siblings(cpu)
        self.others = set(available) - reserved
        if not self.others:
            # A one-core set: nothing to move the other threads to.
            self.others = set(available) - {cpu} or set(available)
        self.refresh_iters = max(1, refresh_iters)
        self._known: Set[int] = set()

    @classmethod
    def create(cls) -> Optional["ExecutorThreadPin"]:
        """Pin the calling thread when the switch is on; None otherwise or on failure."""
        if os.environ.get(PIN_EXECUTOR_ENV, "").strip() in ("", "0"):
            return None
        try:
            available = os.sched_getaffinity(0)
            cpu = pin_executor_cpu(available)
            if cpu is None:
                return None
            pin = cls(cpu, available)
            os.sched_setaffinity(0, {cpu})
            pin.refresh()
        except (OSError, ValueError) as error:
            logger.warning(f"{PIN_EXECUTOR_ENV}: executor thread not pinned ({error})")
            return None
        logger.info(
            f"{PIN_EXECUTOR_ENV}: executor thread {pin.tid} pinned to CPU {cpu}; "
            f"{len(pin.others)} CPUs left for the other threads"
        )
        return pin

    def refresh(self) -> int:
        """Move the threads not seen before off the executor's CPU; returns how many moved."""
        moved = 0
        try:
            tids = [int(t) for t in os.listdir("/proc/self/task")]
        except OSError:
            return 0
        for tid in tids:
            if tid == self.tid or tid in self._known:
                continue
            try:
                os.sched_setaffinity(tid, self.others)
                moved += 1
            except OSError:
                # The thread exited, or its affinity is not ours to change.
                pass
            self._known.add(tid)
        self._known.intersection_update(tids)
        return moved

    def maybe_refresh(self, iteration: int) -> None:
        """Run :meth:`refresh` every ``refresh_iters`` iterations."""
        if iteration % self.refresh_iters == 0:
            self.refresh()
