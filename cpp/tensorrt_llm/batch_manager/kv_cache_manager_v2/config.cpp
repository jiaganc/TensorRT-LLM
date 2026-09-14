/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "kv_cache_manager_v2/config.h"
#include "kv_cache_manager_v2/exceptions.h"

#include <cmath>
#include <filesystem>
#include <set>
#include <stdexcept>

namespace tensorrt_llm::batch_manager::kv_cache_manager_v2
{

void DiskCacheTierConfig::assertValid() const
{
    if (quota == 0)
    {
        throw std::invalid_argument("DiskCacheTierConfig: quota must be > 0");
    }
    if (!std::filesystem::is_directory(path))
    {
        throw std::invalid_argument("DiskCacheTierConfig: path '" + path + "' is not a directory");
    }
}

void LayerGroupMatch::validate() const
{
    if (type && *type != LayerType::ATTENTION && *type != LayerType::SSM)
    {
        throw std::invalid_argument("type must be attention or ssm");
    }
    if (windowSize && (!windowSizeSpecified || *windowSize <= 0))
    {
        throw std::invalid_argument("window_size must be positive and window_size_specified must be true");
    }
    if (sinkBlocks && *sinkBlocks < 0)
    {
        throw std::invalid_argument("sink_blocks must be a nonnegative integer");
    }
    if (type == LayerType::SSM && (windowSizeSpecified || sinkBlocks))
    {
        throw std::invalid_argument("SSM selectors cannot specify window_size or sink_blocks");
    }
}

void PoolRatioDescriptor::validate() const
{
    match.validate();
    if (!std::isfinite(ratio) || ratio <= 0)
    {
        throw std::invalid_argument("descriptor ratio must be finite and positive");
    }
}

void KVCacheManagerConfig::validatePoolRatios() const
{
    if (initialPoolRatio && initialPoolRatioDescriptors)
    {
        throw std::invalid_argument("initial_pool_ratio and initial_pool_ratio_descriptors are mutually exclusive");
    }
    if (initialPoolRatioDescriptors)
    {
        if (initialPoolRatioDescriptors->empty())
        {
            throw std::invalid_argument("initial_pool_ratio_descriptors must be a nonempty descriptor list");
        }
        double sum = 0;
        for (auto const& entry : *initialPoolRatioDescriptors)
        {
            entry.validate();
            sum += entry.ratio;
        }
        if (!std::isfinite(sum) || std::abs(sum - 1.0) > 1e-6)
        {
            throw std::invalid_argument("initial_pool_ratio_descriptors ratios must sum to 1.0");
        }
    }
}

void KVCacheManagerConfig::validate() const
{
    validatePoolRatios();
    if (swaScratchReuse.has_value())
    {
        swaScratchReuse->validate();
    }

    // These mirror Python's KVCacheManagerConfig.__post_init__ asserts, so they
    // throw AssertionError (translated in the binding layer) rather than ValueError.
    if (cacheTiers.empty() || cacheTierOf(cacheTiers[0]) != CacheTier::GPU_MEM)
    {
        throw AssertionError("KVCacheManagerConfig: first cache tier must be GPU memory");
    }

    // Check for duplicate layer ids.
    std::set<LayerId> seenLayerIds;
    for (auto const& layer : layers)
    {
        std::visit(
            [&](auto const& cfg)
            {
                if (!seenLayerIds.insert(cfg.layerId).second)
                {
                    throw AssertionError("KVCacheManagerConfig: duplicate layer id");
                }
                for (auto const& buf : cfg.buffers)
                {
                    if (buf.tokensPerBlockOverride.has_value()
                        && (*buf.tokensPerBlockOverride <= 0 || tokensPerBlock % *buf.tokensPerBlockOverride != 0))
                    {
                        throw AssertionError(
                            "KVCacheManagerConfig: tokensPerBlockOverride must be a divisor of "
                            "tokensPerBlock");
                    }
                }
            },
            layer);
    }

    // SSM-specific validation.
    bool hasSSM = false;
    for (auto const& layer : layers)
    {
        if (std::holds_alternative<SsmLayerConfig>(layer))
        {
            hasSSM = true;
            break;
        }
    }
    if (hasSSM)
    {
        if (!commitMinSnapshot)
            throw AssertionError("KVCacheManagerConfig: commit_min_snapshot must be True when SSM layers are present");
    }
}

} // namespace tensorrt_llm::batch_manager::kv_cache_manager_v2
