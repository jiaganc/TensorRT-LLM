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
"""One-launch upload of a decode step's input buffers (``TRTLLM_BATCHED_INPUT_COMMIT=1``).

Input preparation for a CUDA-graph decode step fills small device buffers from host data
(positions, KV and prompt lengths, helix offsets, sampling seeds, ...) and, under the
overlap scheduler, gathers the previous step's tokens and accepted lengths on the device.
Issued one by one, these copies and gathers form a chain of small operations between two
graph replays, each waiting for the previous replay because it writes buffers that replay
reads. ``InputCommit`` collects the host values into a pinned arena while the step is
prepared and writes every destination with one kernel right before the replay (the kernel
reads the arena directly over the host link). The overlap-scheduler gathers run in the same
kernel.

Contract for staged copies: between ``stage`` and ``commit`` no device operation may read or
write a staged destination. Call sites stage only buffers that the input preparation does
not read back on the device; the device-side gathers that do read them (the previous step's
tokens) are part of the commit kernel.

Switches (read once per process, default off):

* ``TRTLLM_BATCHED_INPUT_COMMIT=1``: the decoder runner opens a commit around the input
  preparation of an eligible step (see ``DecoderRunner._input_commit_for_step``).
  ``TRTLLM_K3_ULL_STEP_PREP=1`` (``ull_shell.py``) turns it on as well.
* ``TRTLLM_BATCHED_INPUT_COMMIT_VERIFY=<n>``: after each of the first ``n`` commits (``-1``:
  all), synchronize and compare every destination with the staged bytes and with a torch
  reference of the overlap gathers; raises on a mismatch and logs once when the first ``n``
  commits matched.
"""

import os
import threading
from contextlib import contextmanager
from typing import Iterator, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import triton
import triton.language as tl

from tensorrt_llm._utils import prefer_pinned
from tensorrt_llm.logger import logger

BATCHED_INPUT_COMMIT_ENV = "TRTLLM_BATCHED_INPUT_COMMIT"
BATCHED_INPUT_COMMIT_VERIFY_ENV = "TRTLLM_BATCHED_INPUT_COMMIT_VERIFY"

_ENABLED = os.environ.get(BATCHED_INPUT_COMMIT_ENV, "0") == "1"
_VERIFY_COMMITS = int(os.environ.get(BATCHED_INPUT_COMMIT_VERIFY_ENV, "0") or "0")

# torch dtype -> (numpy dtype, element size) of the staged values.
_TORCH_TO_NUMPY = {
    torch.int32: (np.int32, 4),
    torch.int64: (np.int64, 8),
    torch.bool: (np.bool_, 1),
    torch.float32: (np.float32, 4),
    torch.uint8: (np.uint8, 1),
    torch.int8: (np.int8, 1),
}

# Bytes copied by one program of the commit kernel.
_COPY_CHUNK = 512
# Elements per program of the overlap fixup.
_FIX_BLOCK = 256

# The commit collecting the current step's uploads, per thread: only the thread that
# prepares the step may stage into it.
_SCOPE = threading.local()


def _cdiv(a: int, b: int) -> int:
    # Plain integer ceil-div: triton.cdiv is a JIT function, slow to call from Python.
    return -(-a // b)


def batched_input_commit_enabled() -> bool:
    """Whether ``TRTLLM_BATCHED_INPUT_COMMIT=1`` is set."""
    return _ENABLED


def active_input_commit() -> Optional["InputCommit"]:
    """The commit collecting the current step's uploads, or None (copy directly)."""
    return getattr(_SCOPE, "commit", None)


@contextmanager
def input_commit_scope(commit: Optional["InputCommit"]) -> Iterator[Optional["InputCommit"]]:
    """Make ``commit`` the active one on this thread for the duration of the block."""
    previous = active_input_commit()
    _SCOPE.commit = commit
    try:
        yield commit
    finally:
        _SCOPE.commit = previous


@contextmanager
def input_commit_step(commit: Optional["InputCommit"]) -> Iterator[Optional["InputCommit"]]:
    """Open ``commit`` for the step prepared in the block and launch it when the block ends.

    With ``commit`` None the block runs as is (every upload copies directly). On an
    exception the staged values are dropped and nothing is launched.
    """
    if commit is None:
        yield None
        return
    commit.begin()
    try:
        with input_commit_scope(commit):
            yield commit
    except BaseException:
        commit.abort()
        raise
    commit.commit()


def overlap_fixup_supported(
    new_tokens: Optional[torch.Tensor],
    new_tokens_lens: Optional[torch.Tensor],
    next_draft_tokens: Optional[torch.Tensor],
) -> bool:
    """Whether :meth:`InputCommit.set_overlap_fixup` covers these sampler stores.

    ``new_tokens`` must be an int32 ``[T_store, num_slots, 1]`` CUDA tensor,
    ``new_tokens_lens`` an int32 ``[num_slots]`` and ``next_draft_tokens`` an int32
    ``[num_slots, D_store]`` with unit stride in its last dimension.
    """
    tensors = (new_tokens, new_tokens_lens, next_draft_tokens)
    return (
        all(isinstance(t, torch.Tensor) and t.is_cuda and t.dtype == torch.int32 for t in tensors)
        and new_tokens.dim() == 3
        and (new_tokens.shape[2] == 1 or new_tokens.stride(2) == 1)
        and new_tokens_lens.dim() == 1
        and next_draft_tokens.dim() == 2
        and next_draft_tokens.stride(1) == 1
    )


@triton.jit(
    do_not_specialize=[
        "num_desc",
        "num_prev",
        "tokens_per_req",
        "draft_width",
        "new_tokens_stride_t",
        "new_tokens_stride_s",
        "next_draft_stride_s",
        "input_ids_start",
        "draft_tokens_start",
        "pos_off_len",
        "pos_off_start",
        "kv_off_len",
        "kv_off_start",
    ]
)
def _input_commit_kernel(
    desc_ptr,
    num_desc,
    slots_ptr,
    num_prev,
    tokens_per_req,
    draft_width,
    new_tokens_ptr,
    new_tokens_stride_t,
    new_tokens_stride_s,
    new_tokens_lens_ptr,
    next_draft_ptr,
    next_draft_stride_s,
    input_ids_ptr,
    input_ids_start,
    draft_tokens_ptr,
    draft_tokens_start,
    pos_off_ptr,
    pos_off_len,
    pos_off_start,
    kv_off_ptr,
    kv_off_len,
    kv_off_start,
    CHUNK: tl.constexpr,
    FIX_BLOCK: tl.constexpr,
    HAS_FIX: tl.constexpr,
):
    pid = tl.program_id(0)
    if pid < num_desc:
        # Staged host->device copy: (dst address, src address, bytes <= CHUNK).
        desc = desc_ptr + pid * 3
        dst = tl.load(desc).to(tl.pointer_type(tl.uint8))
        src = tl.load(desc + 1).to(tl.pointer_type(tl.uint8))
        nbytes = tl.load(desc + 2)
        byte_offs = tl.arange(0, CHUNK)
        byte_mask = byte_offs < nbytes
        tl.store(dst + byte_offs, tl.load(src + byte_offs, mask=byte_mask), mask=byte_mask)
    elif HAS_FIX:
        # Overlap-scheduler inputs of the requests that ran in the previous step
        # (slot = their sequence slot in the sampler store):
        #   input_ids[input_ids_start + p*T + t] = new_tokens[t, slot_p]
        #   draft_tokens[draft_tokens_start + p*D + j] = next_draft[slot_p, j]
        #   pos_off = 0 except pos_off[pos_off_start + p*T + t] = new_tokens_lens[slot_p]
        #   kv_off = 0 except kv_off[kv_off_start + p] = new_tokens_lens[slot_p] - T
        idx = (pid - num_desc) * FIX_BLOCK + tl.arange(0, FIX_BLOCK)
        num_tok = num_prev * tokens_per_req

        m_tok = idx < num_tok
        p_tok = idx // tokens_per_req
        t_tok = idx - p_tok * tokens_per_req
        slot_tok = tl.load(slots_ptr + p_tok, mask=m_tok, other=0).to(tl.int64)
        tok = tl.load(
            new_tokens_ptr + t_tok * new_tokens_stride_t + slot_tok * new_tokens_stride_s,
            mask=m_tok,
            other=0,
        )
        tl.store(input_ids_ptr + input_ids_start + idx, tok, mask=m_tok)

        m_pos = idx < pos_off_len
        j_pos = idx - pos_off_start
        in_pos = m_pos & (j_pos >= 0) & (j_pos < num_tok)
        p_pos = tl.where(in_pos, j_pos, 0) // tokens_per_req
        slot_pos = tl.load(slots_ptr + p_pos, mask=in_pos, other=0).to(tl.int64)
        lens_pos = tl.load(new_tokens_lens_ptr + slot_pos, mask=in_pos, other=0)
        tl.store(pos_off_ptr + idx, tl.where(in_pos, lens_pos, 0), mask=m_pos)

        num_draft = num_prev * draft_width
        m_dr = idx < num_draft
        p_dr = idx // tl.maximum(draft_width, 1)
        j_dr = idx - p_dr * draft_width
        slot_dr = tl.load(slots_ptr + p_dr, mask=m_dr, other=0).to(tl.int64)
        dr = tl.load(next_draft_ptr + slot_dr * next_draft_stride_s + j_dr, mask=m_dr, other=0)
        tl.store(draft_tokens_ptr + draft_tokens_start + idx, dr, mask=m_dr)

        m_kv = idx < kv_off_len
        j_kv = idx - kv_off_start
        in_kv = m_kv & (j_kv >= 0) & (j_kv < num_prev)
        slot_kv = tl.load(slots_ptr + tl.where(in_kv, j_kv, 0), mask=in_kv, other=0).to(tl.int64)
        lens_kv = tl.load(new_tokens_lens_ptr + slot_kv, mask=in_kv, other=0)
        tl.store(kv_off_ptr + idx, tl.where(in_kv, lens_kv - tokens_per_req, 0), mask=m_kv)


class _PinnedArena:
    """A pinned host byte buffer with typed numpy views for staging values."""

    def __init__(self, capacity: int):
        # The commit kernel reads the arena over the host link, so it must be pinned;
        # InputCommit is only created where prefer_pinned() holds.
        self.tensor = torch.empty(capacity, dtype=torch.uint8, pin_memory=prefer_pinned())
        self.bytes = self.tensor.numpy()
        self.base = self.tensor.data_ptr()
        self.cursor = 0
        # Whole-arena typed views; staged values start at 16-byte aligned offsets.
        self.views = {
            np_dtype: self.bytes[: capacity - capacity % size].view(np_dtype)
            for np_dtype, size in _TORCH_TO_NUMPY.values()
        }

    @property
    def capacity(self) -> int:
        return self.bytes.shape[0]

    def alloc(self, nbytes: int) -> int:
        offset = self.cursor
        self.cursor = (offset + nbytes + 15) & ~15
        return offset


class InputCommit:
    """Collects one step's host->device input copies and writes them with one kernel.

    ``begin()`` opens a step, ``stage()`` records a copy (the host values are copied into a
    pinned arena immediately, so the caller may reuse its source right away),
    ``set_overlap_fixup()`` records the previous-step gathers, and ``commit()`` launches the
    kernel on the current stream. Two arenas alternate between steps; an event recorded
    after each commit kernel guards an arena's reuse.

    Args:
        capacity: Initial size in bytes of each of the two pinned arenas; an arena grows
            when a step stages more.
    """

    def __init__(self, capacity: int = 1 << 20):
        self._arenas = [_PinnedArena(capacity), _PinnedArena(capacity)]
        self._events: List[Optional[torch.cuda.Event]] = [None, None]
        self._index = 0
        self._arena = self._arenas[0]
        self._entries: List[Tuple[int, int, int]] = []
        self._fixup: Optional[dict] = None
        self._open = False
        self._verify_dsts: List[Tuple[torch.Tensor, int, int]] = []
        self._num_commits = 0
        # Off while warmup_input_commit_kernels runs: its commits are neither counted nor
        # verified.
        self._counting = True

    @property
    def is_open(self) -> bool:
        return self._open

    @property
    def num_commits(self) -> int:
        """How many step commits have launched their kernel (warmup excluded)."""
        return self._num_commits

    def _verifying(self) -> bool:
        return self._counting and (_VERIFY_COMMITS < 0 or self._num_commits < _VERIFY_COMMITS)

    def begin(self) -> None:
        """Open a step; waits until the kernel that last read the step's arena has run."""
        assert not self._open, "InputCommit.begin() called twice without commit()"
        self._index ^= 1
        event = self._events[self._index]
        if event is not None:
            event.synchronize()
        self._arena = self._arenas[self._index]
        self._arena.cursor = 0
        self._entries.clear()
        self._fixup = None
        self._verify_dsts.clear()
        self._open = True

    def abort(self) -> None:
        """Close the step without launching anything."""
        self._entries.clear()
        self._fixup = None
        self._verify_dsts.clear()
        self._open = False

    def _reserve(self, nbytes: int) -> int:
        if self._arena.cursor + nbytes > self._arena.capacity:
            self._grow(self._arena.cursor + nbytes)
        return self._arena.alloc(nbytes)

    def _grow(self, needed: int) -> None:
        # Rare. Replaces the arena of the open step (nothing reads it yet) by a larger one
        # and rebases the host addresses recorded so far.
        old = self._arena
        new = _PinnedArena(max(2 * old.capacity, 2 * needed))
        new.bytes[: old.cursor] = old.bytes[: old.cursor]
        new.cursor = old.cursor
        delta = new.base - old.base
        self._entries = [(dst, src + delta, n) for dst, src, n in self._entries]
        if self._fixup is not None:
            self._fixup["slots_addr"] += delta
        self._arenas[self._index] = new
        self._arena = new

    def stage(self, dst: torch.Tensor, values: Union[Sequence, np.ndarray, torch.Tensor]) -> None:
        """Record ``dst.copy_(values)``.

        Args:
            dst: Contiguous CUDA tensor of one of the dtypes in ``_TORCH_TO_NUMPY``.
            values: ``dst.numel()`` host values: a sequence, a numpy array or a CPU tensor
                (converted to ``dst.dtype``). Copied into the arena before returning.
        """
        assert self._open
        assert dst.is_cuda and dst.is_contiguous(), "staged destinations must be contiguous"
        n = dst.numel()
        if n == 0:
            return
        np_dtype, itemsize = _TORCH_TO_NUMPY[dst.dtype]
        nbytes = n * itemsize
        offset = self._reserve(nbytes)
        arena = self._arena
        first = offset // itemsize
        view = arena.views[np_dtype][first : first + n]
        if isinstance(values, torch.Tensor):
            src = values.reshape(-1)
            if src.dtype != dst.dtype:
                src = src.to(dst.dtype)
            view[:] = src.numpy()
        else:
            view[:] = values
        self._entries.append((dst.data_ptr(), arena.base + offset, nbytes))
        if self._verifying():
            self._verify_dsts.append((dst, offset, nbytes))

    def stage_host_values(self, values: Sequence[int], dtype=np.int32) -> Tuple[int, int]:
        """Copy ``values`` into the arena without a destination; returns (address, offset)."""
        assert self._open
        arr = np.asarray(values, dtype=dtype)
        offset = self._reserve(max(arr.nbytes, 1))
        self._arena.bytes[offset : offset + arr.nbytes].view(dtype)[:] = arr
        return self._arena.base + offset, offset

    def set_overlap_fixup(
        self,
        *,
        previous_slots: Sequence[int],
        tokens_per_req: int,
        draft_width: int,
        new_tokens: torch.Tensor,
        new_tokens_lens: torch.Tensor,
        next_draft_tokens: torch.Tensor,
        input_ids: torch.Tensor,
        input_ids_start: int,
        draft_tokens: torch.Tensor,
        draft_tokens_start: int,
        pos_offsets: torch.Tensor,
        pos_offsets_start: int,
        kv_offsets: torch.Tensor,
        kv_offsets_start: int,
    ) -> None:
        """Record the overlap-scheduler gathers of the decoder input preparation.

        For previous-step request ``p`` with sampler slot ``s = previous_slots[p]``, writes
        ``input_ids[input_ids_start + p*T + t] = new_tokens[t, s, 0]``,
        ``draft_tokens[draft_tokens_start + p*D + j] = next_draft_tokens[s, j]``, and the
        whole ``pos_offsets`` / ``kv_offsets`` buffers: zero except
        ``pos_offsets[pos_offsets_start + p*T + t] = new_tokens_lens[s]`` and
        ``kv_offsets[kv_offsets_start + p] = new_tokens_lens[s] - T``
        (``T = tokens_per_req``, ``D = draft_width``).

        Args:
            previous_slots: Host sampler slots of the requests that ran in the previous step.
            tokens_per_req: Tokens per generation request ``T``.
            draft_width: Draft tokens per request ``D`` (``T - 1``).
            new_tokens: int32 ``[T_store, num_slots, 1]`` sampler store.
            new_tokens_lens: int32 ``[num_slots]`` accepted lengths.
            next_draft_tokens: int32 ``[num_slots, D_store]`` next drafts.
            input_ids: Contiguous int32 input-id buffer.
            input_ids_start: First row the gathered tokens go to.
            draft_tokens: Contiguous int32 draft-token buffer.
            draft_tokens_start: First row the gathered drafts go to.
            pos_offsets: Contiguous int32 per-token position corrections.
            pos_offsets_start: First token of the previous-step requests.
            kv_offsets: Contiguous int32 per-request KV-length corrections.
            kv_offsets_start: First previous-step request.
        """
        assert self._open and self._fixup is None
        assert overlap_fixup_supported(new_tokens, new_tokens_lens, next_draft_tokens)
        for t in (input_ids, draft_tokens, pos_offsets, kv_offsets):
            assert t.is_cuda and t.dtype == torch.int32 and t.is_contiguous()
        num_prev = len(previous_slots)
        assert num_prev * tokens_per_req <= input_ids.numel() - input_ids_start
        assert num_prev * draft_width <= draft_tokens.numel() - draft_tokens_start
        slots_addr, _ = self.stage_host_values(previous_slots if num_prev else [0])
        self._fixup = dict(
            slots_addr=slots_addr,
            num_prev=num_prev,
            tokens_per_req=tokens_per_req,
            draft_width=draft_width,
            new_tokens=new_tokens,
            new_tokens_lens=new_tokens_lens,
            next_draft_tokens=next_draft_tokens,
            input_ids=input_ids,
            input_ids_start=input_ids_start,
            draft_tokens=draft_tokens,
            draft_tokens_start=draft_tokens_start,
            pos_offsets=pos_offsets,
            pos_offsets_start=pos_offsets_start,
            kv_offsets=kv_offsets,
            kv_offsets_start=kv_offsets_start,
        )

    def commit(self) -> None:
        """Launch the commit kernel on the current stream and close the step."""
        assert self._open
        if not self._entries and self._fixup is None:
            self._open = False
            return
        num_desc = sum(_cdiv(nbytes, _COPY_CHUNK) for _, _, nbytes in self._entries)
        # Reserving may grow the arena and rebase the recorded host addresses, so the
        # entries and the fixup are read afterwards.
        desc_offset = self._reserve(max(24 * num_desc, 8))
        self._open = False
        entries = self._entries
        fixup = self._fixup
        arena = self._arena
        descs: List[int] = []
        for dst, src, nbytes in entries:
            for start in range(0, nbytes, _COPY_CHUNK):
                descs.extend((dst + start, src + start, min(_COPY_CHUNK, nbytes - start)))
        if descs:
            arena.bytes[desc_offset : desc_offset + 8 * len(descs)].view(np.int64)[:] = descs
        desc_tensor = arena.tensor[desc_offset : desc_offset + max(8 * len(descs), 8)].view(
            torch.int64
        )
        if fixup is not None:
            slots = _host_view(arena, fixup["slots_addr"], max(fixup["num_prev"], 1))
            num_fix = max(
                fixup["num_prev"] * fixup["tokens_per_req"],
                fixup["pos_offsets"].numel(),
                fixup["num_prev"] * fixup["draft_width"],
                fixup["kv_offsets"].numel(),
            )
            num_fix_programs = _cdiv(max(num_fix, 1), _FIX_BLOCK)
            new_tokens = fixup["new_tokens"]
            args = (
                slots,
                fixup["num_prev"],
                fixup["tokens_per_req"],
                fixup["draft_width"],
                new_tokens,
                new_tokens.stride(0),
                new_tokens.stride(1),
                fixup["new_tokens_lens"],
                fixup["next_draft_tokens"],
                fixup["next_draft_tokens"].stride(0),
                fixup["input_ids"],
                fixup["input_ids_start"],
                fixup["draft_tokens"],
                fixup["draft_tokens_start"],
                fixup["pos_offsets"],
                fixup["pos_offsets"].numel(),
                fixup["pos_offsets_start"],
                fixup["kv_offsets"],
                fixup["kv_offsets"].numel(),
                fixup["kv_offsets_start"],
            )
        else:
            num_fix_programs = 0
            dummy = desc_tensor
            args = (
                dummy,
                0,
                1,
                0,
                dummy,
                0,
                0,
                dummy,
                dummy,
                0,
                dummy,
                0,
                dummy,
                0,
                dummy,
                0,
                0,
                dummy,
                0,
                0,
            )
        _input_commit_kernel[(num_desc + num_fix_programs,)](
            desc_tensor,
            num_desc,
            *args,
            CHUNK=_COPY_CHUNK,
            FIX_BLOCK=_FIX_BLOCK,
            HAS_FIX=fixup is not None,
        )
        event = self._events[self._index]
        if event is None:
            event = torch.cuda.Event()
            self._events[self._index] = event
        event.record()
        if self._verifying():
            self._verify(fixup)
            if self._num_commits + 1 == _VERIFY_COMMITS:
                logger.info(f"InputCommit verify: the first {_VERIFY_COMMITS} commits matched")
        if self._counting:
            self._num_commits += 1
        self._entries = []
        self._fixup = None

    def _verify(self, fixup: Optional[dict]) -> None:
        torch.cuda.current_stream().synchronize()
        arena = self._arena
        for dst, offset, nbytes in self._verify_dsts:
            expected = torch.from_numpy(arena.bytes[offset : offset + nbytes].copy())
            got = dst.reshape(-1).view(torch.uint8).cpu()
            if not torch.equal(got, expected):
                raise RuntimeError(
                    f"InputCommit verify: staged copy mismatch for a {dst.dtype} "
                    f"destination of {dst.numel()} elements"
                )
        if fixup is None:
            return
        num_prev = fixup["num_prev"]
        tokens = fixup["tokens_per_req"]
        width = fixup["draft_width"]
        slots_view = _host_view(arena, fixup["slots_addr"], max(num_prev, 1))[:num_prev]
        slots = slots_view.to(device="cuda", dtype=torch.long)
        lens = fixup["new_tokens_lens"]
        ref_ids = fixup["new_tokens"].transpose(0, 1)[slots, :tokens].flatten()
        ref_draft = fixup["next_draft_tokens"][slots, :width].flatten()
        ref_pos = torch.zeros_like(fixup["pos_offsets"])
        start = fixup["pos_offsets_start"]
        ref_pos[start : start + num_prev * tokens] = lens[slots.repeat_interleave(tokens)]
        ref_kv = torch.zeros_like(fixup["kv_offsets"])
        start = fixup["kv_offsets_start"]
        ref_kv[start : start + num_prev] = lens[slots] - tokens
        ids_start = fixup["input_ids_start"]
        draft_start = fixup["draft_tokens_start"]
        checks = (
            ("input_ids", fixup["input_ids"][ids_start : ids_start + num_prev * tokens], ref_ids),
            (
                "draft_tokens",
                fixup["draft_tokens"][draft_start : draft_start + num_prev * width],
                ref_draft,
            ),
            ("pos_offsets", fixup["pos_offsets"], ref_pos),
            ("kv_offsets", fixup["kv_offsets"], ref_kv),
        )
        for name, got, ref in checks:
            if not torch.equal(got, ref.to(got.dtype)):
                raise RuntimeError(f"InputCommit verify: overlap fixup mismatch in {name}")


def warmup_input_commit_kernels(commit: InputCommit) -> None:
    """Compile the commit kernel (with and without the overlap fixup) ahead of serving.

    Warmup and capture forwards do not use the commit, so its first use would otherwise
    pay the JIT compile. Does not count towards ``TRTLLM_BATCHED_INPUT_COMMIT_VERIFY``.
    """
    device = torch.device("cuda", torch.cuda.current_device())
    buf = torch.zeros(64, dtype=torch.int32, device=device)
    new_tokens = torch.zeros((2, 2, 1), dtype=torch.int32, device=device)
    lens = torch.ones(2, dtype=torch.int32, device=device)
    draft = torch.zeros((2, 1), dtype=torch.int32, device=device)
    pos_off = torch.zeros(4, dtype=torch.int32, device=device)
    kv_off = torch.zeros(2, dtype=torch.int32, device=device)
    commit._counting = False
    try:
        for with_fixup in (False, True):
            with input_commit_step(commit):
                commit.stage(buf[:4], [1, 2, 3, 4])
                if with_fixup:
                    commit.set_overlap_fixup(
                        previous_slots=[0, 1],
                        tokens_per_req=2,
                        draft_width=1,
                        new_tokens=new_tokens,
                        new_tokens_lens=lens,
                        next_draft_tokens=draft,
                        input_ids=buf[16:32],
                        input_ids_start=0,
                        draft_tokens=buf[32:48],
                        draft_tokens_start=0,
                        pos_offsets=pos_off,
                        pos_offsets_start=0,
                        kv_offsets=kv_off,
                        kv_offsets_start=0,
                    )
    finally:
        commit._counting = True
    torch.cuda.synchronize()


def _host_view(arena: _PinnedArena, address: int, count: int) -> torch.Tensor:
    offset = address - arena.base
    return arena.tensor[offset : offset + 4 * count].view(torch.int32)
