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
"""CPU tests of the unified standalone draft cache in KVCacheManagerV2.

A standalone drafter whose context KV the target manager owns: its history
contract, the draft storage cost, the KvCacheCreator gating and kwargs, and
the engine hooks that publish the draft history after sampling.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from tensorrt_llm._torch.pyexecutor._util import KvCacheCreator
from tensorrt_llm._torch.pyexecutor.engine.runners.decoder.runner import DecoderRunner
from tensorrt_llm._torch.pyexecutor.kv_cache import kv_cache_manager_v2 as kvm2
from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import KVCacheManagerV2
from tensorrt_llm._torch.pyexecutor.kv_cache.standalone_draft_cache import (
    DraftHistoryUpdate,
    StandaloneDraftHistory,
)
from tensorrt_llm._torch.pyexecutor.py_executor import PyExecutor
from tensorrt_llm._torch.speculative.interface import SpecWorkerBase
from tensorrt_llm.bindings import DataType
from tensorrt_llm.mapping import Mapping

pytestmark = pytest.mark.cpu_only

# K3 DSpark MLA drafter: five layers of one 576-wide latent per token.
MLA_LAYERS = 5
MLA_LATENT = 576


# --- History contract -------------------------------------------------------


@pytest.mark.parametrize("valid_length, position", [(0, 0), (0, 7), (5, 5), (5, 9)])
def test_draft_history_accepts_valid_ranges(valid_length, position):
    history = StandaloneDraftHistory(valid_length, position)
    assert (history.valid_length, history.position) == (valid_length, position)


@pytest.mark.parametrize("valid_length, position", [(-1, 0), (6, 5), (1.0, 2), (True, 2)])
def test_draft_history_rejects_invalid_ranges(valid_length, position):
    with pytest.raises(ValueError):
        StandaloneDraftHistory(valid_length, position)


class _FakeManager:
    def __init__(self, request_ids):
        self.kv_cache_map = {rid: SimpleNamespace(is_active=True) for rid in request_ids}
        self.published = []

    def set_draft_history(self, request_id, valid_length, position):
        self.published.append((request_id, valid_length, position))


def test_draft_history_update_requires_one_pair_per_request():
    manager = _FakeManager([1, 2])
    with pytest.raises(ValueError, match="one length/position pair"):
        DraftHistoryUpdate.capture(manager, (1, 2), torch.zeros(2, 3, dtype=torch.long))


def test_draft_history_update_publishes_only_live_unreplaced_requests():
    manager = _FakeManager([1, 2, 3, 4])
    values = torch.tensor([[3, 10], [4, 11], [5, 12], [6, 13]])
    update = DraftHistoryUpdate.capture(manager, (1, 2, 3, 4), values)
    # The readback is a copy: later device writes do not change it.
    values.zero_()
    del manager.kv_cache_map[2]  # released
    manager.kv_cache_map[3] = SimpleNamespace(is_active=True)  # replaced
    manager.kv_cache_map[4].is_active = False  # suspended
    update.publish()
    assert manager.published == [(1, 3, 10)]


# --- Draft storage cost -----------------------------------------------------


def test_draft_layer_sizes():
    assert kvm2._get_draft_layer_sizes() == ()
    assert (
        kvm2._get_draft_layer_sizes(
            draft_num_layers=MLA_LAYERS,
            draft_num_kv_heads=1,
            draft_head_dim=MLA_LATENT,
            draft_dtype=DataType.FP8,
            draft_kv_factor=1,
            draft_scratch_tokens=8,  # other draft kwargs are ignored
        )
        == (MLA_LATENT,) * MLA_LAYERS
    )
    assert (
        kvm2._get_draft_layer_sizes(
            draft_num_layers=2,
            draft_num_kv_heads=4,
            draft_head_dim=128,
            draft_dtype=DataType.BF16,
            draft_kv_factor=2,
        )
        == (4 * 128 * 2 * 2,) * 2
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"draft_num_layers": -1},
        {"draft_num_layers": 1, "draft_num_kv_heads": 0, "draft_head_dim": 8},
        {
            "draft_num_layers": 1,
            "draft_num_kv_heads": 1,
            "draft_head_dim": 8,
            "draft_dtype": DataType.FLOAT,
        },
        {
            "draft_num_layers": 1,
            "draft_num_kv_heads": 1,
            "draft_head_dim": 8,
            "draft_kv_factor": 3,
        },
    ],
)
def test_draft_layer_sizes_reject_invalid_storage(kwargs):
    with pytest.raises(ValueError):
        kvm2._get_draft_layer_sizes(**kwargs)


@pytest.mark.parametrize("helix_cp_size", [1, 8])
def test_draft_cost_charges_history_scratch_and_helix_subpages(helix_cp_size):
    sizes = (MLA_LATENT,) * MLA_LAYERS
    context, generation, per_request = kvm2._estimate_draft_cache_size_components(
        sizes,
        32,
        generation_capacity_headroom=1,
        scratch_tokens=8,
        helix_cp_size=helix_cp_size,
    )
    # Full history: every rank-local ledger token costs helix_cp_size draft
    # tokens (the rank keeps one page per CP rank's block); scratch is per request.
    assert context == generation == sum(sizes) * helix_cp_size
    assert per_request == 8 * sum(sizes)


def test_static_cache_cost_adds_unified_draft_layers(monkeypatch):
    target_sizes = [1000] * 4
    monkeypatch.setattr(
        kvm2,
        "_get_static_cache_size_layer_components",
        lambda *args, **kwargs: (list(target_sizes), [None] * len(target_sizes)),
    )
    mapping = MagicMock()
    mapping.has_cp_helix.return_value = False

    def cost(**draft_kwargs):
        return KVCacheManagerV2.get_cache_size_per_token(
            MagicMock(),
            mapping,
            tokens_per_block=32,
            max_batch_size=4,
            max_num_tokens=64,
            **draft_kwargs,
        )

    base_slope, base_fixed = cost()
    slope, fixed = cost(
        draft_num_layers=MLA_LAYERS,
        draft_num_kv_heads=1,
        draft_head_dim=MLA_LATENT,
        draft_dtype=DataType.FP8,
        draft_kv_factor=1,
        draft_scratch_tokens=8,
    )
    draft_bytes = MLA_LAYERS * MLA_LATENT
    assert slope - base_slope == draft_bytes
    # The scratch is reserved in every layer, target and draft, of each request.
    assert fixed - base_fixed == 4 * 8 * (sum(target_sizes) + draft_bytes)


def test_draft_ratio_for_token_parity():
    # One target group of 90 B per block: a 10 B draft block holds as many
    # tokens with a tenth of the quota.
    assert kvm2._draft_ratio_for_token_parity(10, [1.0], [90]) == pytest.approx(0.1)
    # Groups that do not grow with tokens do not bound the target.
    assert kvm2._draft_ratio_for_token_parity(10, [0.5, 0.5], [90, None]) == pytest.approx(5 / 95)
    assert kvm2._draft_ratio_for_token_parity(10, [1.0], [None]) is None
    assert kvm2._draft_ratio_for_token_parity(0, [1.0], [90]) is None


# --- Manager draft history --------------------------------------------------


def _manager(*, window=None, capacity=100, draft_layers=(61,)):
    manager = object.__new__(KVCacheManagerV2)
    manager.kv_cache_map = {7: SimpleNamespace(is_active=True, capacity=capacity)}
    manager.draft_history = {}
    manager.draft_window_size = window
    manager._standalone_draft_layer_ids = tuple(draft_layers)
    manager.get_draft_transfer_identity = lambda: {"num_layers": MLA_LAYERS, "dtype": "fp8"}
    manager.get_draft_block_table = lambda request_ids, histories: None
    return manager


def test_set_draft_history_checks_the_allocation():
    manager = _manager(window=16)
    manager.set_draft_history(7, 10, 40)
    assert manager.get_draft_history(7) == StandaloneDraftHistory(10, 40)
    with pytest.raises(ValueError, match="exceeds allocated capacity"):
        manager.set_draft_history(7, 10, 101)
    with pytest.raises(ValueError, match="retention window"):
        manager.set_draft_history(7, 17, 40)
    with pytest.raises(ValueError, match="no active cache"):
        manager.set_draft_history(8, 0, 0)
    manager.kv_cache_map[7].is_active = False
    with pytest.raises(ValueError, match="no active cache"):
        manager.set_draft_history(7, 0, 0)


def test_draft_history_export_restore_round_trip():
    sender = _manager()
    assert _manager(draft_layers=()).export_draft_history(7) is None
    with pytest.raises(ValueError, match="no valid history to transfer"):
        sender.export_draft_history(7)
    sender.set_draft_history(7, 12, 30)
    metadata = sender.export_draft_history(7)

    receiver = _manager()
    receiver.restore_draft_history(7, metadata)
    assert receiver.get_draft_history(7) == StandaloneDraftHistory(12, 30)


def test_draft_history_restore_rejects_another_layout():
    sender = _manager()
    sender.set_draft_history(7, 12, 30)
    metadata = sender.export_draft_history(7)
    receiver = _manager()
    receiver.get_draft_transfer_identity = lambda: {"num_layers": MLA_LAYERS, "dtype": "bf16"}
    with pytest.raises(ValueError, match="layout does not match"):
        receiver.restore_draft_history(7, metadata)
    with pytest.raises(ValueError, match="layout does not match"):
        _manager(draft_layers=()).restore_draft_history(7, metadata)
    assert receiver.get_draft_history(7) is None


@pytest.mark.parametrize("draft_cp_size, expansion, draft_cp_rank", [(8, 1, 3), (1, 8, 0)])
def test_draft_pages_under_helix(draft_cp_size, expansion, draft_cp_rank):
    manager = object.__new__(KVCacheManagerV2)
    manager.mapping = SimpleNamespace(tp_rank=1)
    manager._helix_cp_size = 8
    manager._helix_cp_rank = 3
    manager._draft_cp_size = draft_cp_size
    assert manager._draft_page_expansion == expansion
    assert manager.draft_cp_rank == draft_cp_rank


# --- KvCacheCreator ---------------------------------------------------------


def _creator(
    *,
    v2=True,
    dspark=True,
    embedded=False,
    opt_in=True,
    disagg=False,
    context_kv_only=False,
    context_kv_dtype="auto",
    draft=None,
):
    creator = object.__new__(KvCacheCreator)
    creator._speculative_config = SimpleNamespace(
        spec_dec_mode=SimpleNamespace(is_dspark=lambda: dspark, is_dflash=lambda: not dspark),
        draft_is_embedded_in_target=embedded,
        _use_shared_kv_cache=False,
        context_kv_only=context_kv_only,
        context_kv_dtype=context_kv_dtype,
        max_draft_len=7,
        attention_backend="AUTO",
    )
    if draft is None:
        draft = SimpleNamespace(supports_unified_draft_cache=opt_in)
    creator._model_engine = SimpleNamespace(model=SimpleNamespace(draft_model=draft))
    creator._is_kv_cache_manager_v2 = v2
    creator._is_disagg = disagg
    return creator


def test_unified_draft_cache_is_opt_in():
    assert _creator()._uses_unified_standalone_draft_cache()
    # The target manager then owns the drafter's context KV.
    assert not _creator()._should_create_separate_draft_kv_cache()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"v2": False},
        {"dspark": False},
        {"embedded": True},
        {"opt_in": False},
        # Only a literal True opts in, not any truthy attribute.
        {"draft": MagicMock()},
        {"draft": SimpleNamespace()},
    ],
    ids=["v1", "dflash", "embedded", "opted_out", "mock_drafter", "plain_drafter"],
)
def test_unified_draft_cache_gating(kwargs):
    assert not _creator(**kwargs)._uses_unified_standalone_draft_cache()


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"context_kv_only": True}, "context_kv_only"),
        ({"context_kv_only": True, "disagg": True, "opt_in": False}, "context_kv_only"),
        ({"context_kv_dtype": "fp8", "opt_in": False}, "context_kv_dtype"),
        ({"context_kv_dtype": "fp8", "v2": False}, "context_kv_dtype"),
        ({"disagg": True, "v2": False}, "use_kv_cache_manager_v2"),
    ],
    ids=["aggregated", "no_unified_cache", "dtype_opted_out", "dtype_v1", "disagg_v1"],
)
def test_context_kv_options_need_a_unified_draft_cache(kwargs, match):
    with pytest.raises(ValueError, match=match):
        _creator(**kwargs)._validate_standalone_draft_cache()


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"disagg": True},
        {"disagg": True, "context_kv_only": True, "context_kv_dtype": "fp8"},
        {"context_kv_dtype": "fp8"},
        # A drafter without a unified cache keeps main's behavior.
        {"opt_in": False, "disagg": True},
    ],
)
def test_context_kv_options_accepted(kwargs):
    _creator(**kwargs)._validate_standalone_draft_cache()


def _mla_drafter(*, helix_mapping=None, context_kv_fp8=True):
    return SimpleNamespace(
        supports_unified_draft_cache=True,
        _helix_mapping=helix_mapping,
        _kv_factor=1,
        _num_attn_layers=MLA_LAYERS,
        _head_dim=MLA_LATENT,
        dflash_attention_backend="TRTLLM",
        config=SimpleNamespace(max_position_embeddings=262144),
        fc=SimpleNamespace(weight=torch.empty(0, dtype=torch.bfloat16)),
        context_kv_fp8=context_kv_fp8,
    )


@pytest.mark.parametrize(
    "helix_mapping, draft_cp_size",
    [(None, 1), (SimpleNamespace(has_cp_helix=lambda: True, cp_size=8), 8)],
    ids=["no_helix", "helix8"],
)
@pytest.mark.parametrize(
    "context_kv_fp8, dtype", [(True, DataType.FP8), (False, DataType.BF16)], ids=["fp8", "bf16"]
)
def test_unified_draft_cache_kwargs_of_an_mla_drafter(
    helix_mapping, draft_cp_size, context_kv_fp8, dtype
):
    creator = _creator(
        draft=_mla_drafter(helix_mapping=helix_mapping, context_kv_fp8=context_kv_fp8)
    )
    creator._draft_config = SimpleNamespace(pretrained_config=SimpleNamespace(), mapping=Mapping())
    kwargs = creator._get_unified_draft_cache_kwargs()
    assert kwargs == dict(
        draft_num_layers=MLA_LAYERS,
        draft_kv_factor=1,
        draft_cp_size=draft_cp_size,
        draft_num_kv_heads=1,
        draft_head_dim=MLA_LATENT,
        draft_dtype=dtype,
        draft_total_num_kv_heads=1,
        draft_scratch_tokens=8,  # max_draft_len + 1
        draft_attention_backend="TRTLLM",
        draft_max_position_embeddings=262144,
    )
    # The manager and its static estimator take the same kwargs.
    latent_bytes = MLA_LATENT if context_kv_fp8 else 2 * MLA_LATENT
    assert kvm2._get_draft_layer_sizes(**kwargs) == (latent_bytes,) * MLA_LAYERS


def test_context_server_spec_reserve_needs_context_kv_only():
    assert _creator()._get_context_server_spec_reserve(0.9) == 0


# --- Engine hooks -----------------------------------------------------------


def test_spec_worker_hooks_default_to_nothing():
    # SpecWorkerBase is abstract; call the defaults unbound.
    assert SpecWorkerBase.prepare_managed_draft_cache(None, None, None, None, None) is None
    assert SpecWorkerBase.snapshot_managed_draft_history(None) is None


@pytest.mark.parametrize("wrapper_attr", [None, "language_model", "llm"])
def test_runner_finds_the_speculative_worker(wrapper_attr):
    spec_model = SimpleNamespace(spec_worker=object(), draft_model=object())
    runner = object.__new__(DecoderRunner)
    runner._model = (
        spec_model if wrapper_attr is None else SimpleNamespace(**{wrapper_attr: spec_model})
    )
    assert runner._speculative_worker() == (spec_model.spec_worker, spec_model.draft_model)


def test_runner_without_speculative_worker():
    runner = object.__new__(DecoderRunner)
    runner._model = SimpleNamespace(spec_worker=None)
    assert runner._speculative_worker() == (None, None)


def test_executor_publishes_draft_history_after_updating_requests():
    events = []
    executor = object.__new__(PyExecutor)
    executor.sampler = SimpleNamespace(
        update_requests=lambda state, resource_manager: events.append("update_requests")
    )
    update = SimpleNamespace(publish=lambda: events.append("publish"))
    state = SimpleNamespace(requests=[], draft_history_update=update)
    executor._update_requests(state)
    assert events == ["update_requests", "publish"]
    assert state.draft_history_update is None
