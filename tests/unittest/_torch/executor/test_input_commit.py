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
"""Bit-exactness of the batched input commit (``input_commit.py``).

Every check compares against the operations the commit replaces: ``Tensor.copy_`` for
staged host->device copies, and the overlap-scheduler gather sequence of
``DecoderInputPreparer._copy_token_ids_to_device`` for the fixup.
"""

import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from tensorrt_llm._torch.pyexecutor import input_commit as input_commit_module
from tensorrt_llm._torch.pyexecutor.input_commit import (
    InputCommit,
    active_input_commit,
    input_commit_scope,
    input_commit_step,
    overlap_fixup_supported,
    warmup_input_commit_kernels,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")


def _random_values(dtype: torch.dtype, n: int, rng: random.Random):
    if dtype == torch.bool:
        return [rng.random() < 0.5 for _ in range(n)]
    if dtype == torch.float32:
        return [rng.uniform(-3.0, 3.0) for _ in range(n)]
    if dtype == torch.int64:
        return [rng.randint(-(1 << 40), 1 << 40) for _ in range(n)]
    return [rng.randint(-(1 << 30), 1 << 30) for _ in range(n)]


@pytest.mark.parametrize("capacity", [1 << 20, 64])
def test_staged_copies_match_direct_copies(capacity: int) -> None:
    """Every dtype and size (several chunks per copy), list or tensor sources, and an
    arena that has to grow (capacity 64) mid-step."""
    rng = random.Random(capacity)
    commit = InputCommit(capacity=capacity)
    dtypes = [torch.int32, torch.int64, torch.bool, torch.float32, torch.int32]
    sizes = [1, 3, 14, 112, 129, 700, 2049]
    for step in range(6):
        dsts, refs = [], []
        commit.begin()
        for i, size in enumerate(sizes):
            dtype = dtypes[(i + step) % len(dtypes)]
            buf = torch.full((size + 9,), 7, dtype=dtype, device="cuda")
            ref = buf.clone()
            values = _random_values(dtype, size, rng)
            off = (i + step) % 5
            if i % 2:
                src = torch.tensor(values, dtype=dtype)
                commit.stage(buf[off : off + size], src)
            else:
                commit.stage(buf[off : off + size], values)
            ref[off : off + size].copy_(torch.tensor(values, dtype=dtype))
            dsts.append(buf)
            refs.append(ref)
        commit.commit()
        torch.cuda.synchronize()
        for buf, ref in zip(dsts, refs):
            assert torch.equal(buf, ref)


def test_staged_copy_converts_tensor_dtype() -> None:
    commit = InputCommit()
    dst = torch.zeros(5, dtype=torch.int32, device="cuda")
    commit.begin()
    commit.stage(dst, torch.arange(5, dtype=torch.int64) * 3)
    commit.commit()
    torch.cuda.synchronize()
    assert dst.cpu().tolist() == [0, 3, 6, 9, 12]


def _reference_fixup(
    new_tokens,
    lens,
    next_draft,
    slots,
    tokens,
    input_ids,
    ids_start,
    draft_tokens,
    draft_start,
    pos_off,
    pos_start,
    kv_off,
    kv_start,
):
    """The device sequence of _copy_token_ids_to_device for requests with a previous step."""
    pos_off.zero_()
    kv_off.zero_()
    width = tokens - 1
    if not slots:
        return
    previous_slots = torch.tensor(slots, dtype=torch.int, device="cuda")
    num_prev = len(slots)
    flat = new_tokens.transpose(0, 1)[previous_slots, :tokens].flatten()
    input_ids[ids_start : ids_start + num_prev * tokens].copy_(flat)
    if width > 0:
        draft_tokens[draft_start : draft_start + num_prev * width].copy_(
            next_draft[previous_slots, :width].flatten()
        )
    kv_len_offsets = lens - tokens
    pos_indices = torch.tensor(
        [s for s in slots for _ in range(tokens)], dtype=torch.int, device="cuda"
    )
    pos_off[pos_start : pos_start + num_prev * tokens].copy_(lens[pos_indices])
    kv_off[kv_start : kv_start + num_prev].copy_(kv_len_offsets[previous_slots])


@pytest.mark.parametrize("tokens", [1, 2, 7, 8])
@pytest.mark.parametrize("num_prev,num_new", [(0, 2), (1, 0), (2, 0), (2, 1), (37, 3), (256, 0)])
def test_overlap_fixup_matches_gather_sequence(tokens: int, num_prev: int, num_new: int) -> None:
    rng = random.Random(tokens * 1000 + num_prev * 10 + num_new)
    num_slots = num_prev + num_new + 5
    store_tokens = tokens + 1
    store_width = max(tokens, 2)
    new_tokens = torch.randint(0, 160000, (store_tokens, num_slots, 1), dtype=torch.int32).cuda()
    lens = torch.randint(1, tokens + 1, (num_slots,), dtype=torch.int32).cuda()
    next_draft = torch.randint(0, 160000, (num_slots, store_width), dtype=torch.int32).cuda()
    slots = rng.sample(range(num_slots), num_prev)
    max_tokens = (num_prev + num_new) * tokens + 11
    ids_start = num_new * tokens
    draft_start = num_new * (tokens - 1)
    num_extend = num_prev + num_new
    pos_start = (num_extend - num_prev) * tokens
    kv_start = num_extend - num_prev

    def buffers():
        g = torch.Generator().manual_seed(5)
        return [
            torch.randint(-9, 9, (n,), generator=g, dtype=torch.int32).cuda()
            for n in (max_tokens, max_tokens, max_tokens, num_extend + 3)
        ]

    ids_r, draft_r, pos_r, kv_r = buffers()
    _reference_fixup(
        new_tokens,
        lens,
        next_draft,
        slots,
        tokens,
        ids_r,
        ids_start,
        draft_r,
        draft_start,
        pos_r,
        pos_start,
        kv_r,
        kv_start,
    )
    ids_c, draft_c, pos_c, kv_c = buffers()
    commit = InputCommit()
    commit.begin()
    commit.set_overlap_fixup(
        previous_slots=slots,
        tokens_per_req=tokens,
        draft_width=tokens - 1,
        new_tokens=new_tokens,
        new_tokens_lens=lens,
        next_draft_tokens=next_draft,
        input_ids=ids_c,
        input_ids_start=ids_start,
        draft_tokens=draft_c,
        draft_tokens_start=draft_start,
        pos_offsets=pos_c,
        pos_offsets_start=pos_start,
        kv_offsets=kv_c,
        kv_offsets_start=kv_start,
    )
    commit.commit()
    torch.cuda.synchronize()
    for got, ref in ((ids_c, ids_r), (draft_c, draft_r), (pos_c, pos_r), (kv_c, kv_r)):
        assert torch.equal(got, ref)


def test_overlap_fixup_supported_layouts() -> None:
    new_tokens = torch.zeros((8, 4, 1), dtype=torch.int32, device="cuda")
    lens = torch.zeros(4, dtype=torch.int32, device="cuda")
    draft = torch.zeros((4, 7), dtype=torch.int32, device="cuda")
    assert overlap_fixup_supported(new_tokens, lens, draft)
    assert not overlap_fixup_supported(new_tokens.long(), lens, draft)
    assert not overlap_fixup_supported(new_tokens, lens.long(), draft)
    assert not overlap_fixup_supported(new_tokens, lens, draft.cpu())
    assert not overlap_fixup_supported(new_tokens[:, :, 0], lens, draft)
    assert not overlap_fixup_supported(new_tokens, lens, draft.t().contiguous().t())


def test_arena_reuse_waits_and_scope_restores() -> None:
    commit = InputCommit(capacity=256)
    out = torch.zeros(4096, dtype=torch.int64, device="cuda")
    for step in range(8):
        commit.begin()
        with input_commit_scope(commit):
            assert active_input_commit() is commit
            # Grows past the 256-byte arena on the first steps.
            commit.stage(out, np.arange(step, step + out.numel(), dtype=np.int64))
        assert active_input_commit() is None
        commit.commit()
    torch.cuda.synchronize()
    assert torch.equal(out.cpu(), torch.arange(7, 7 + out.numel(), dtype=torch.int64))
    commit.begin()
    commit.abort()
    assert not commit.is_open


def test_input_commit_step_commits_or_drops() -> None:
    dst = torch.zeros(3, dtype=torch.int32, device="cuda")
    with input_commit_step(None) as none:
        assert none is None and active_input_commit() is None
    commit = InputCommit()
    with input_commit_step(commit) as opened:
        assert opened is commit and active_input_commit() is commit and commit.is_open
        commit.stage(dst, [1, 2, 3])
    assert active_input_commit() is None and not commit.is_open
    torch.cuda.synchronize()
    assert dst.cpu().tolist() == [1, 2, 3]
    with pytest.raises(KeyError):
        with input_commit_step(commit):
            commit.stage(dst, [7, 8, 9])
            raise KeyError("prepare failed")
    torch.cuda.synchronize()
    assert active_input_commit() is None and not commit.is_open
    assert dst.cpu().tolist() == [1, 2, 3]
    assert commit.num_commits == 1


def test_scope_is_per_thread() -> None:
    import threading

    commit = InputCommit(capacity=64)
    seen = []
    with input_commit_scope(commit):
        thread = threading.Thread(target=lambda: seen.append(active_input_commit()))
        thread.start()
        thread.join()
    assert seen == [None]


def test_verify_logs_once_and_raises_on_mismatch(monkeypatch: pytest.MonkeyPatch) -> None:
    logged = []
    monkeypatch.setattr(input_commit_module, "_VERIFY_COMMITS", 2)
    monkeypatch.setattr(input_commit_module, "logger", SimpleNamespace(info=logged.append))
    commit = InputCommit()
    warmup_input_commit_kernels(commit)
    assert commit.num_commits == 0
    dst = torch.zeros(4, dtype=torch.int32, device="cuda")
    for step in range(3):
        commit.begin()
        commit.stage(dst, [step] * 4)
        commit.commit()
    assert logged == ["InputCommit verify: the first 2 commits matched"]
    # Two staged copies to the same bytes: whichever lands, the other's check fails.
    monkeypatch.setattr(input_commit_module, "_VERIFY_COMMITS", -1)
    commit.begin()
    commit.stage(dst, [1, 1, 1, 1])
    commit.stage(dst, [2, 2, 2, 2])
    with pytest.raises(RuntimeError, match="staged copy mismatch"):
        commit.commit()


# ---------------------------------------------------------------------------
# DecoderInputPreparer call sites: the same buffers with and without a commit
# ---------------------------------------------------------------------------


def _preparer(max_num_tokens: int, max_batch_size: int, seed: int):
    """A stand-in for the DecoderInputPreparer state the copy stages read, with buffers
    filled with the same garbage for every call of the same seed."""
    g = torch.Generator().manual_seed(seed)

    def garbage(n):
        return torch.randint(-9, 9, (n,), generator=g, dtype=torch.int32).cuda()

    buffers = SimpleNamespace(
        input_ids_cuda=garbage(max_num_tokens),
        draft_tokens_cuda=garbage(max_num_tokens),
        position_ids_cuda=garbage(max_num_tokens),
        previous_batch_indices_cuda=garbage(max_num_tokens),
        previous_pos_indices_cuda=garbage(max_num_tokens),
        previous_pos_id_offsets_cuda=garbage(max_num_tokens),
        previous_kv_lens_offsets_cuda=garbage(max_batch_size),
    )
    return SimpleNamespace(
        _buffers=buffers,
        _config=SimpleNamespace(
            max_num_tokens=max_num_tokens,
            disable_overlap_scheduler=False,
            max_beam_width=1,
            use_mrope=False,
        ),
        _staged_previous_batch_indices=None,
        _staged_previous_pos_indices=None,
        _step_input_gather=None,
        get_runtime_tokens_per_gen_step=lambda draft_len: draft_len + 1,
        model=SimpleNamespace(),
    )


@pytest.mark.parametrize("draft_len", [0, 3, 7])
@pytest.mark.parametrize("num_new,num_prev", [(0, 1), (0, 4), (2, 0), (1, 3)])
def test_preparer_token_and_position_copies_match_with_commit(
    draft_len: int, num_new: int, num_prev: int
) -> None:
    """``_copy_token_ids_to_device`` / ``_copy_position_ids_to_device`` write the same
    model inputs whether the step copies directly or stages into a commit (the generation
    rows of a CUDA-graph step, with and without requests lacking a previous step)."""
    from tensorrt_llm._torch.pyexecutor.engine.runners.decoder.inputs import DecoderInputPreparer

    tokens = draft_len + 1
    rng = random.Random(draft_len * 100 + num_new * 10 + num_prev)
    num_slots = num_new + num_prev + 3
    new_tokens = torch.randint(0, 160000, (tokens + 1, num_slots, 1), dtype=torch.int32).cuda()
    lens = torch.randint(1, tokens + 1, (num_slots,), dtype=torch.int32).cuda()
    next_draft = torch.randint(0, 160000, (num_slots, max(draft_len, 1)), dtype=torch.int32).cuda()
    previous_slots = rng.sample(range(num_slots), num_prev)
    input_ids = [rng.randrange(160000) for _ in range(num_new * tokens)]
    draft_tokens = [rng.randrange(160000) for _ in range(num_new * draft_len)]
    total_tokens = (num_new + num_prev) * tokens
    position_ids = [rng.randrange(1 << 20) for _ in range(total_tokens)]
    max_num_tokens, max_batch_size = 64, 16
    extend = [object() for _ in range(num_new + num_prev)]

    def run(with_commit):
        prep = _preparer(max_num_tokens, max_batch_size, seed=11)
        host = SimpleNamespace(
            lists=SimpleNamespace(
                input_ids=list(input_ids),
                draft_tokens=list(draft_tokens),
                position_ids=list(position_ids),
                draft_lens=[draft_len] * (num_new + num_prev),
                previous_batch_indices=list(previous_slots),
                previous_pos_indices=[s for s in previous_slots for _ in range(tokens)],
                extend_requests=extend,
                extend_dummy_requests=[],
                generation_requests=[],
                unwritten_input_id_rows=[],
            ),
            total_num_tokens=total_tokens,
        )
        batch = SimpleNamespace(is_attention_dp_phase_idle=False, num_context_requests=0)
        commit = InputCommit() if with_commit else None
        with input_commit_step(commit):
            DecoderInputPreparer._copy_token_ids_to_device(
                prep,
                host,
                batch,
                new_tokens,
                lens,
                next_draft,
                draft_len,
                attn_metadata=SimpleNamespace(is_cuda_graph=True),
                enable_spec_decode=True,
            )
            DecoderInputPreparer._copy_position_ids_to_device(prep, host)
        torch.cuda.synchronize()
        assert host.host_position_ids.tolist() == position_ids
        return prep._buffers

    direct, committed = run(False), run(True)
    assert torch.equal(
        direct.input_ids_cuda[:total_tokens], committed.input_ids_cuda[:total_tokens]
    )
    num_draft = (num_new + num_prev) * draft_len
    assert torch.equal(
        direct.draft_tokens_cuda[:num_draft], committed.draft_tokens_cuda[:num_draft]
    )
    assert torch.equal(
        direct.position_ids_cuda[:total_tokens], committed.position_ids_cuda[:total_tokens]
    )
    # The preprocessing reads the offsets of the step's rows; the commit also zeroes the
    # rest, which the direct path skips when every row has a previous step.
    rows = num_new + num_prev
    assert torch.equal(
        direct.previous_pos_id_offsets_cuda[: rows * tokens],
        committed.previous_pos_id_offsets_cuda[: rows * tokens],
    )
    assert torch.equal(
        direct.previous_kv_lens_offsets_cuda[:rows], committed.previous_kv_lens_offsets_cuda[:rows]
    )
    assert not committed.previous_pos_id_offsets_cuda[rows * tokens :].any()
    assert not committed.previous_kv_lens_offsets_cuda[rows:].any()
