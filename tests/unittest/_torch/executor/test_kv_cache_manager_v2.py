# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from tensorrt_llm._torch.pyexecutor import kv_cache_manager_v2 as kv_cache_v2_module
from tensorrt_llm._torch.pyexecutor.kv_cache_manager_v2 import BlockReusePolicy, KVCacheManagerV2
from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequestState
from tensorrt_llm.bindings.internal.batch_manager import CacheType as CacheTypeCpp
from tensorrt_llm.llmapi.llm_args import KvCacheConfig
from tensorrt_llm.runtime.kv_cache_manager_v2 import GpuCacheTierConfig, KVCacheManagerConfig


class _FakeKVCache:
    def __init__(self, num_committed_tokens: int) -> None:
        self.num_committed_tokens = num_committed_tokens
        self.committed_tokens: list[int] | None = None
        self.stopped_committing = False

    def commit(self, tokens: list[int]) -> None:
        self.committed_tokens = tokens
        self.num_committed_tokens += len(tokens)

    def stop_committing(self) -> None:
        self.stopped_committing = True


def _build_cache_config_for_test(
    kv_cache_config: KvCacheConfig, *, is_draft: bool = False
) -> KVCacheManagerConfig:
    cache_manager = object.__new__(KVCacheManagerV2)
    cache_manager.kv_cache_type = CacheTypeCpp.SELFKONLY
    cache_manager.head_dim_per_layer = [128]
    cache_manager.enable_swa_scratch_reuse = False
    cache_manager.num_extra_kv_tokens = 0
    cache_manager.enable_stats = False
    cache_manager.block_reuse_policy = BlockReusePolicy(kv_cache_config.block_reuse_policy)
    cache_manager.is_draft = is_draft
    cache_manager.num_local_layers = 1
    cache_manager.pp_layers = [0]
    cache_manager.max_attention_window_vec = [None]
    cache_manager.get_layer_bytes_per_token = lambda **_: 128

    return cache_manager._build_cache_config(
        kv_cache_config,
        tokens_per_block=128,
        vocab_size=129280,
        cache_tiers=[GpuCacheTierConfig(quota=1 << 30)],
    )


@pytest.mark.parametrize(
    ("enable_block_reuse", "block_reuse_policy", "is_draft", "commit_min_snapshot"),
    [
        (True, "all_reusable", False, False),
        (True, "per_request", False, True),
        (False, "per_request", False, False),
        (True, "per_request", True, True),
    ],
)
def test_commit_min_snapshot_follows_block_reuse_policy(
    enable_block_reuse: bool,
    block_reuse_policy: str,
    is_draft: bool,
    commit_min_snapshot: bool,
) -> None:
    config = _build_cache_config_for_test(
        KvCacheConfig(
            enable_block_reuse=enable_block_reuse,
            block_reuse_policy=block_reuse_policy,
            enable_partial_reuse=True,
        ),
        is_draft=is_draft,
    )

    assert config.commit_min_snapshot is commit_min_snapshot
    assert config.enable_partial_reuse


@pytest.mark.parametrize("enable_partial_reuse", [False, True])
def test_propagates_partial_reuse_config(enable_partial_reuse: bool) -> None:
    config = _build_cache_config_for_test(KvCacheConfig(enable_partial_reuse=enable_partial_reuse))

    assert config.enable_partial_reuse is enable_partial_reuse


def test_try_commit_blocks_commits_partial_block_at_context_end() -> None:
    request = SimpleNamespace(
        py_request_id=1,
        is_dummy_request=False,
        context_current_position=10,
        context_remaining_length=0,
        get_tokens=lambda beam_id: list(range(10)),
    )
    kv_cache = _FakeKVCache(num_committed_tokens=4)
    manager = object.__new__(KVCacheManagerV2)
    manager.enable_block_reuse = True
    manager.is_draft = False
    manager.kv_cache_map = {request.py_request_id: kv_cache}
    manager._augment_tokens_for_block_reuse = lambda tokens, request, start, end: tokens[start:end]

    manager.try_commit_blocks(request)

    assert kv_cache.committed_tokens == [4, 5, 6, 7, 8, 9]
    assert kv_cache.num_committed_tokens == 10
    assert kv_cache.stopped_committing


def test_iteration_stats_reports_physical_pool_groups_without_window_metadata() -> None:
    manager = object.__new__(KVCacheManagerV2)
    manager.enable_stats = True
    snapshot_delta = SimpleNamespace(
        iter_snapshot_lookups=2,
        iter_snapshot_hits=1,
        iter_snapshot_misses=1,
        iter_reused_tokens=32,
        iter_unreused_tokens=16,
        iter_aligned_snapshot_hits=1,
        iter_unaligned_snapshot_hits=0,
    )
    manager.impl = SimpleNamespace(
        cache_tier_list=[object()],
        get_and_reset_iteration_stats=lambda: {},
        get_and_reset_ssm_snapshot_iteration_stats=lambda: {3: snapshot_delta},
    )
    manager._stats_life_cycle_metadata = lambda: {3: (1, None, "ssm")}
    manager._storage_pool_groups_by_window = lambda: {}
    manager._get_and_reset_iteration_peak_block_stats = lambda _level: [None, None]
    manager._get_storage_statistics = lambda _level: [object(), object()]
    manager._build_pool_group_iteration_stats = lambda pool_group_id, *_args: pool_group_id

    stats = manager.get_iteration_stats()

    assert stats.by_pool_group == {0: 0, 1: 1}
    ssm_stats = stats.by_life_cycle[3]
    assert ssm_stats.kind == "ssm"
    assert ssm_stats.pool_group_id == 1
    assert ssm_stats.snapshot_stats.iter_snapshot_hit_rate == 0.5
    assert ssm_stats.snapshot_stats.iter_reused_tokens == 32


@pytest.mark.parametrize(
    ("is_draft", "expected_capacity"),
    [(True, 201), (False, 230)],
    ids=["draft-reclaims-reserve", "target-has-no-reserve"],
)
def test_dynamic_tree_reserved_capacity(
    monkeypatch: pytest.MonkeyPatch,
    is_draft: bool,
    expected_capacity: int,
) -> None:
    monkeypatch.setattr(
        kv_cache_v2_module,
        "_update_kv_cache_draft_token_location",
        MagicMock(),
    )
    manager = object.__new__(KVCacheManagerV2)
    manager.is_draft = is_draft
    manager.kv_compression_manages_history = False
    manager._kv_reserve_draft_tokens = 60
    request = SimpleNamespace(
        py_request_id=1,
        py_rewind_len=26,
        py_num_accepted_draft_tokens=5,
        max_beam_num_tokens=201,
        state=LlmRequestState.GENERATION_IN_PROGRESS,
    )
    cache = MagicMock(capacity=256, is_active=True)
    cache.resize.return_value = True
    manager.kv_cache_map = {request.py_request_id: cache}

    manager.update_resources(SimpleNamespace(generation_requests=[request]))

    cache.resize.assert_called_once_with(expected_capacity, 200)
