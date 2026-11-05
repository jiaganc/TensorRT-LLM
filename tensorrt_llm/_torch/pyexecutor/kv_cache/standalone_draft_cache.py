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
"""History contract of a standalone drafter whose context KV the target manager owns."""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import torch

from tensorrt_llm._utils import prefer_pinned

if TYPE_CHECKING:
    from .kv_cache_manager_v2 import KVCacheManagerV2


@dataclass(frozen=True)
class StandaloneDraftHistory:
    """Committed draft context length and the next absolute position.

    ``valid_length`` tokens ending just before ``position`` hold drafter
    context KV. Both are independent of the target's history and of any
    scratch capacity reserved for the next draft forward.
    """

    valid_length: int
    position: int

    def __post_init__(self) -> None:
        if type(self.valid_length) is not int or type(self.position) is not int:
            raise ValueError("Standalone draft history requires integer length and position")
        if self.valid_length < 0 or self.position < self.valid_length:
            raise ValueError("Invalid standalone draft history length or position")


@dataclass(frozen=True)
class DraftHistoryUpdate:
    """One iteration's draft history, published after its sampling event completes."""

    manager: "KVCacheManagerV2"
    request_ids: tuple[int, ...]
    cache_instances: tuple[object, ...]
    values_host: torch.Tensor
    # Recorded after the device-to-host copy of a CUDA ``values``.
    copied: Optional[torch.cuda.Event] = None

    @classmethod
    def capture(
        cls,
        manager: "KVCacheManagerV2",
        request_ids: tuple[int, ...],
        values: torch.Tensor,
    ) -> "DraftHistoryUpdate":
        """Queue a length/position readback on the current stream.

        Args:
            manager: The unified cache manager owning the draft layers.
            request_ids: Requests of the rows of ``values``.
            values: ``[len(request_ids), 2]`` integer tensor of
                ``(valid_length, position)`` per request.
        """
        if tuple(values.shape) != (len(request_ids), 2):
            raise ValueError("Draft history update requires one length/position pair per request")
        values_host = torch.empty(
            values.shape,
            dtype=values.dtype,
            device="cpu",
            pin_memory=values.is_cuda and prefer_pinned(),
        )
        values_host.copy_(values, non_blocking=True)
        copied = None
        if values.is_cuda:
            copied = torch.cuda.Event()
            copied.record()
        return cls(
            manager,
            tuple(request_ids),
            tuple(manager.kv_cache_map[request_id] for request_id in request_ids),
            values_host,
            copied,
        )

    def publish(self) -> None:
        """Publish the readback without reviving released or replaced requests.

        Call once the iteration's sampling has completed: the readback is then
        done and waiting on it does not block.
        """
        if self.copied is not None:
            self.copied.synchronize()
        for request_id, cache, (length, position) in zip(
            self.request_ids, self.cache_instances, self.values_host.tolist()
        ):
            current = self.manager.kv_cache_map.get(request_id)
            if current is cache and current.is_active:
                self.manager.set_draft_history(request_id, length, position)
