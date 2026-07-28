# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kimi KDA cache-manager specialization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Optional

import torch

from tensorrt_llm._utils import prefer_pinned
from tensorrt_llm.logger import logger

from ...pyexecutor.mamba_cache_manager import MambaHybridCacheManagerV2, PythonMambaCacheManager
from ._kda_kernels import is_kda_mtp_verify_available

if TYPE_CHECKING:
    from ...attention_backend.interface import AttentionMetadata


@dataclass(frozen=True, kw_only=True)
class KimiKDASpeculativeState(PythonMambaCacheManager.SpeculativeState):
    """Per-layer KDA state consumed by the fused verification kernel."""

    kda_conv_q: torch.Tensor
    kda_conv_k: torch.Tensor
    kda_conv_v: torch.Tensor
    kda_qkg_cache: torch.Tensor
    kda_v_cache: torch.Tensor
    kda_beta_cache: torch.Tensor


class KimiKDAHybridCacheManagerV2(MambaHybridCacheManagerV2):
    """V2 hybrid cache manager with Kimi KDA speculative replay state."""

    def __init__(
        self,
        *args,
        spec_config=None,
        conv_state_layout: str = "q_k_v",
        **kwargs,
    ) -> None:
        if conv_state_layout != "q_k_v":
            raise ValueError("Kimi KDA requires the q_k_v convolution state layout")

        self._kda_replay_num_spec: Optional[int] = None
        if spec_config is not None and is_kda_mtp_verify_available():
            if kwargs.get("use_replay_state_update", False):
                raise ValueError("KDA replay and generic Mamba replay are mutually exclusive")
            self._kda_replay_num_spec = spec_config.tokens_per_gen_step - 1
            if self._kda_replay_num_spec <= 0:
                raise ValueError("KDA replay requires at least one speculative token")

        super().__init__(
            *args,
            spec_config=spec_config,
            conv_state_layout=conv_state_layout,
            **kwargs,
        )

    @property
    def _use_kda_replay_update(self) -> bool:
        return self._kda_replay_num_spec is not None

    def _setup_mtp_intermediate_states(self, spec_config, max_batch_size: int) -> None:
        if not self._use_kda_replay_update:
            super()._setup_mtp_intermediate_states(spec_config, max_batch_size)
            return

        self.spec_config = spec_config
        self.intermediate_ssm_states = None
        self.intermediate_conv_states = None
        self.intermediate_state_indices = None

    def _setup_replay_buffers(self, spec_config) -> None:
        super()._setup_replay_buffers(spec_config)
        self.kda_conv_q = None
        self.kda_conv_k = None
        self.kda_conv_v = None
        self.kda_qkg_cache = None
        self.kda_v_cache = None
        self.kda_beta_cache = None

        if not self._use_kda_replay_update or self.local_num_mamba_layers == 0:
            return

        if len(set(self.conv_section_dims)) != 1:
            raise ValueError("KDA replay requires equal q, k, and v convolution sections")

        cache_size = self.all_ssm_states[0].shape[0]
        device = self.all_ssm_states[0].device
        section_dim = self.conv_section_dims[0]
        conv_width = self.conv_state_shape[1]
        extended_width = conv_width - 1 + self._kda_replay_num_spec
        num_heads = self.ssm_state_shape[0]

        def allocate_conv_cache() -> torch.Tensor:
            return torch.zeros(
                self.local_num_mamba_layers,
                cache_size,
                extended_width,
                section_dim,
                dtype=torch.float32,
                device=device,
            ).transpose(-1, -2)

        self.prev_num_accepted_tokens = torch.zeros(cache_size, dtype=torch.int32, device=device)
        self.kda_conv_q = allocate_conv_cache()
        self.kda_conv_k = allocate_conv_cache()
        self.kda_conv_v = allocate_conv_cache()
        self.kda_qkg_cache = torch.zeros(
            self.local_num_mamba_layers,
            cache_size,
            self._kda_replay_num_spec,
            3,
            section_dim,
            dtype=torch.float32,
            device=device,
        )
        self.kda_v_cache = torch.zeros(
            self.local_num_mamba_layers,
            cache_size,
            self._kda_replay_num_spec,
            section_dim,
            dtype=torch.float32,
            device=device,
        )
        self.kda_beta_cache = torch.zeros(
            self.local_num_mamba_layers,
            cache_size,
            self._kda_replay_num_spec,
            num_heads,
            dtype=torch.float32,
            device=device,
        )

        mask_capacity = self._host_state_indices.shape[0]
        self._dummy_request_mask = torch.zeros(mask_capacity, dtype=torch.bool, device=device)
        self._dummy_request_mask_host = torch.zeros(
            mask_capacity, dtype=torch.bool, pin_memory=prefer_pinned()
        )
        logger.info(
            "Kimi KDA V2 replay cache is allocated with "
            f"{cache_size} slots and {self._kda_replay_num_spec} draft tokens"
        )

    def mamba_layer_cache(self, layer_idx: int):
        if not self._use_kda_replay_update:
            return super().mamba_layer_cache(layer_idx)

        layer_offset = self.mamba_layer_offsets[layer_idx]
        return KimiKDASpeculativeState(
            conv=self.all_conv_states[layer_offset],
            temporal=self.all_ssm_states[layer_offset],
            prev_num_accepted_tokens=self.prev_num_accepted_tokens,
            mamba_ssm_rand_seed=self.mamba_ssm_rand_seed,
            kda_conv_q=self.kda_conv_q[layer_offset],
            kda_conv_k=self.kda_conv_k[layer_offset],
            kda_conv_v=self.kda_conv_v[layer_offset],
            kda_qkg_cache=self.kda_qkg_cache[layer_offset],
            kda_v_cache=self.kda_v_cache[layer_offset],
            kda_beta_cache=self.kda_beta_cache[layer_offset],
        )

    def _reset_context_mamba_slots(self, num_contexts: int) -> None:
        super()._reset_context_mamba_slots(num_contexts)
        if self._use_kda_replay_update and num_contexts > 0:
            context_slots = self.cuda_state_indices[:num_contexts].long()
            self.prev_num_accepted_tokens[context_slots] = 0

    def prepare_resources(self, scheduled_batch) -> None:
        super().prepare_resources(scheduled_batch)
        if self._use_kda_replay_update:
            self.seed_kda_replay_caches_for_disagg_gen(
                [request.py_request_id for request in scheduled_batch.context_requests]
            )

    def add_dummy_requests(self, request_ids: List[int], *args, **kwargs):
        requests = super().add_dummy_requests(request_ids, *args, **kwargs)
        if self._use_kda_replay_update:
            slots = [
                self._request_id_to_state_index[request_id]
                for request_id in request_ids
                if request_id in self._request_id_to_state_index
            ]
            if slots:
                self.prev_num_accepted_tokens[slots] = 0
        return requests

    @torch.inference_mode()
    def seed_kda_replay_caches_for_disagg_gen(self, request_ids: List[int]) -> None:
        """Seed replay convolution windows from transferred V2 state."""
        if not self._use_kda_replay_update:
            return

        slots = [
            self._request_id_to_state_index[request_id]
            for request_id in request_ids
            if request_id in self._request_id_to_state_index
        ]
        if not slots:
            return

        slot_indices = torch.tensor(
            sorted(set(slots)),
            dtype=torch.long,
            device=self.all_conv_states[0].device,
        )
        section_dim = self.conv_section_dims[0]
        committed_width = self.conv_state_shape[1] - 1
        replay_conv_caches = (
            self.kda_conv_q,
            self.kda_conv_k,
            self.kda_conv_v,
        )
        for layer_offset, conv_state in enumerate(self.all_conv_states):
            selected = conv_state.index_select(0, slot_indices)
            sections = (
                selected[:, :section_dim],
                selected[:, section_dim : 2 * section_dim],
                selected[:, 2 * section_dim :],
            )
            for cache, section in zip(replay_conv_caches, sections):
                seeded = torch.zeros(
                    (slot_indices.numel(),) + cache.shape[2:],
                    dtype=cache.dtype,
                    device=cache.device,
                )
                seeded[:, :, :committed_width] = section[:, :, 1:].to(cache.dtype)
                cache[layer_offset].index_copy_(0, slot_indices, seeded)

        for replay_cache in (
            self.kda_qkg_cache,
            self.kda_v_cache,
            self.kda_beta_cache,
        ):
            replay_cache.index_fill_(1, slot_indices, 0)
        self.prev_num_accepted_tokens[slot_indices] = 0

    def update_mamba_states(
        self,
        attn_metadata: "AttentionMetadata",
        num_accepted_tokens: torch.Tensor,
        state_indices: Optional[torch.Tensor] = None,
        accepted_leaf_positions: Optional[torch.Tensor] = None,
    ) -> None:
        if not self._use_kda_replay_update:
            super().update_mamba_states(
                attn_metadata,
                num_accepted_tokens,
                state_indices,
                accepted_leaf_positions,
            )
            return
        if self.local_num_mamba_layers == 0:
            return

        num_contexts = attn_metadata.num_contexts
        num_generations = attn_metadata.num_seqs - num_contexts
        if state_indices is None:
            state_indices = self.get_state_indices()
        generation_slots = state_indices[num_contexts : num_contexts + num_generations].long()
        is_dummy = self._dummy_request_mask[num_contexts : num_contexts + num_generations]
        current = self.prev_num_accepted_tokens[generation_slots]
        accepted_drafts = (
            (num_accepted_tokens[num_contexts : num_contexts + num_generations] - 1)
            .to(torch.int32)
            .clamp(min=0)
        )
        self.prev_num_accepted_tokens[generation_slots] = torch.where(
            is_dummy, current, accepted_drafts
        )

    def shutdown(self) -> None:
        self.kda_conv_q = None
        self.kda_conv_k = None
        self.kda_conv_v = None
        self.kda_qkg_cache = None
        self.kda_v_cache = None
        self.kda_beta_cache = None
        super().shutdown()
