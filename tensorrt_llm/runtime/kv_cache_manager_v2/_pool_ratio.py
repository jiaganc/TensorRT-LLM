# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
import warnings

from ._config import KVCacheManagerConfig, LayerGroupMatch, LayerGroupType
from ._life_cycle_registry import AttnLifeCycle, LifeCycle, LifeCycleRegistry


def _describe(selector: LayerGroupMatch) -> str:
    fields = []
    if selector.type is not None:
        fields.append(f"type: {selector.type.name.lower()}")
    if selector.window_size is not None:
        fields.append(f"window_size: {selector.window_size}")
    if selector.sink_blocks is not None:
        fields.append(f"sink_blocks: {selector.sink_blocks}")
    return "{" + ", ".join(fields) + "}"


def _descriptor(lifecycle: LifeCycle) -> LayerGroupMatch:
    if isinstance(lifecycle, AttnLifeCycle):
        return LayerGroupMatch(
            LayerGroupType.SWA
            if lifecycle.window_size is not None
            else LayerGroupType.FULL_ATTENTION,
            lifecycle.window_size,
            lifecycle.num_sink_blocks,
        )
    return LayerGroupMatch(LayerGroupType.SSM)


def _matches(selector: LayerGroupMatch, group: LayerGroupMatch) -> bool:
    return (
        (selector.type is None or selector.type == group.type)
        and (
            selector.window_size is None
            or (group.type == LayerGroupType.SWA and selector.window_size == group.window_size)
        )
        and (
            selector.sink_blocks is None
            or (group.type != LayerGroupType.SSM and selector.sink_blocks == group.sink_blocks)
        )
    )


def resolve_pool_ratios(
    config: KVCacheManagerConfig, registry: LifeCycleRegistry
) -> list[float] | None:
    """Resolve against this manager's assigned IDs before allocating storage."""
    config.validate_pool_ratios()
    if config.initial_pool_ratio is not None:
        warnings.warn(
            "KVCacheManagerV2: initial_pool_ratio is deprecated; use initial_pool_ratio_descriptors "
            "to configure GPU cache ratios without depending on layer-group ID order. "
            "For the LLM API, replace kv_cache_config.pool_ratio with "
            "kv_cache_config.pool_ratio_descriptors.",
            FutureWarning,
            stacklevel=3,
        )
        return config.initial_pool_ratio
    entries = config.initial_pool_ratio_descriptors
    if entries is None:
        return None
    catalog = [_descriptor(lifecycle) for lifecycle in registry]
    available = "\n  ".join(_describe(group) for group in catalog)
    ratios = [0.0] * len(catalog)
    owners: dict[int, int] = {}
    for index, entry in enumerate(entries):
        matches = [
            group_id for group_id, group in enumerate(catalog) if _matches(entry.match, group)
        ]
        label = f"initial_pool_ratio_descriptors[{index}].match {_describe(entry.match)}"
        if not matches:
            raise ValueError(
                f"{label} matches no layer groups. Available layer groups:\n  {available}"
            )
        if len(matches) > 1:
            groups = [catalog[group_id] for group_id in matches]
            distinguishing = []
            if len({group.type for group in groups}) > 1:
                distinguishing.append("type")
            if len({group.window_size for group in groups}) > 1:
                distinguishing.append("window_size")
            if len({group.sink_blocks for group in groups}) > 1:
                distinguishing.append("sink_blocks")
            details = "\n  ".join(_describe(group) for group in groups)
            raise ValueError(
                f"{label} matches {len(matches)} layer groups:\n  {details}\n"
                f"Specify {', '.join(distinguishing)} to distinguish them."
            )
        group_id = matches[0]
        if group_id in owners:
            raise ValueError(
                f"initial_pool_ratio_descriptors entries {owners[group_id]} and {index} "
                f"target the same layer group {_describe(catalog[group_id])}"
            )
        owners[group_id] = index
        ratios[group_id] = entry.ratio
    missing = [_describe(group) for group_id, group in enumerate(catalog) if group_id not in owners]
    if missing:
        raise ValueError(
            "initial_pool_ratio_descriptors missing layer groups: " + ", ".join(missing)
        )
    logging.getLogger(__name__).info(
        "KVCacheManagerV2 initial GPU byte shares:\n%s",
        "\n".join(
            f"  {_describe(group)} -> layer_group_id={group_id}, ratio={ratios[group_id]:.9g}"
            for group_id, group in enumerate(catalog)
        ),
    )
    return ratios
