//
// SPDX-FileCopyrightText: Copyright 2021-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the License); you may
// not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an AS IS BASIS, WITHOUT
// WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//

// #define LOG_TRACE_ENABLE TD_1
#include "cascade_builder.hpp"

#include "common/logging.hpp"

#include "common/numeric_util.hpp"
#include "common/shape.hpp"
#include "op_type.hpp"
#include "scheduler.hpp"
#include "scheduler_operation.hpp"

#include <memory>
#include <unordered_map>
#include <vector>

namespace regor
{

class BufferMap
{
    using Key = std::pair<UniqueId, UniqueId>;
    struct KeyHash
    {
        size_t operator()(const Key &k) const { return (k.first << 8) ^ k.second; }
    };

private:
    std::unordered_map<Key, CascadeBuffer, KeyHash> _cache;

public:
    CascadeBuffer GetBuffer(SchedulerOperation *producer, SchedulerOperation *consumer, const Schedule *refSchedule)
    {
        auto key = Key(producer ? *producer : 0, consumer ? *consumer : 0);
        auto pos = _cache.find(key);
        if ( pos != _cache.end() )
        {
            return pos->second;
        }

        Shape bufferShape;
        int bufferSize = 0;
        // No cached buffer between these two SchedulerOperations
        if ( consumer == nullptr )
        {
            auto ofm = producer->OFM();
            // There are either no consumers or multiple consumers - FeatureMap needs to be stored in full
            bufferShape = ofm->shape;
            bufferSize = ofm->tensor->AllocationSizeBytes();
        }
        else if ( producer == nullptr )
        {
            auto ifm = consumer->IFM(consumer->PrimaryIfmIndex());
            // First Op in subgraph or cascade - FeatureMap needs to be stored in full
            bufferShape = ifm->shape;
            bufferSize = ifm->tensor->AllocationSizeBytes();
        }
        else
        {
            auto ofm = producer->OFM();
            auto ifm = consumer->IFM(consumer->PrimaryIfmIndex());

            if ( ofm->requireFullTensor || ifm->requireFullTensor )
            {
                // FeatureMap needs to be stored in full
                bufferShape = Shape::Max(ofm->shape, ifm->shape);
                bufferSize = std::max(ofm->tensor->AllocationSizeBytes(), ifm->tensor->AllocationSizeBytes());
            }
            else
            {
                // Use a rolling buffer
                auto producerCost = refSchedule->Cost(producer);
                auto consumerCost = refSchedule->Cost(consumer);

                bufferShape = RollingBufferShape(producerCost->stripe, consumerCost->stripeInput[0]);
                bufferSize = DataTypeStorageSizeBytes(ofm->Type(), bufferShape.Elements());
            }
        }
        _cache.emplace(key, CascadeBuffer(bufferShape, bufferSize));

        return CascadeBuffer(bufferShape, bufferSize);
    }

    Shape RollingBufferShape(const Shape &producerStripeShape, const Shape &consumerStripeShape)
    {
        // Calculates the storage shape of the rolling buffer between two SchedulerOperations in a Cascade
        int buffer_height = RoundAway(producerStripeShape.Height() + consumerStripeShape.Height(), consumerStripeShape.Height());
        // Rolling buffers have to conform to NHCWB16 alignment
        return consumerStripeShape.With(-3, buffer_height).With(-1, RoundAway(producerStripeShape.Depth(), 16));
    }
};


CascadeBuilder::CascadeBuilder(vector_span<std::unique_ptr<SchedulerOperation>> ops,
    const std::unordered_map<UniqueId, int> &nonLocalMemUsage, const std::unordered_map<UniqueId, int> &opLocalMemUsage,
    const std::unordered_map<UniqueId, LiveRangeSummary> &tensorLiveRanges, bool spilling) :
        _ops(ops),
        _nonLocalMemUsage(nonLocalMemUsage), _opLocalMemUsage(opLocalMemUsage), _tensorLiveRanges(tensorLiveRanges)

{
    _spilling = spilling;
}


void CascadeBuilder::BuildCascades(Schedule *refSchedule, Schedule *fallbackSchedule, Address guidingStagingLimit)
{
    BufferMap buffers;
    SchedulerCostMap costs;
    std::unordered_map<int, CascadeInfo> cascadeMap;


    LOG_TRACE1("Build Cascades for '{}' with limit of {} bytes\n", refSchedule->Name(), guidingStagingLimit);
    // Peak memory usage so far - updated continuously, except where spilling makes this a hard limit
    int peakStagingUsage = int(std::min(INT64_C(1) << 30, guidingStagingLimit));
    auto pos = _ops.begin();
    while ( pos != _ops.end() )
    {
        SchedulerOperation *op = pos->get();
        if ( !op->IsNpuOp() )
        {
            pos++;
            continue;
        }

        // Already processed this Op if it has a cost
        if ( costs.find(*op) != costs.end() )
        {
            pos++;
            continue;
        }

        auto fallbackCost = fallbackSchedule->Cost(op);

        SchedulerConnection *ifm = op->IFM(op->PrimaryIfmIndex());

        // If Op is not a candidate for cascading - assign fallback cost
        if ( !IsCascadable(op, ifm, refSchedule->Cost(op)) )
        {
            costs[*op] = std::make_unique<SchedulerOpInfo>(*fallbackCost);
            if ( !_spilling )
            {
                peakStagingUsage = std::max(EstimateUncascadedBufferUsage(op, fallbackCost), peakStagingUsage);
            }
            pos++;
            continue;
        }

        // Propose a cascade starting with this Op
        // Keep track of which Ops are in the proposed cascade as well as the best cascade so far
        int cascadeStart = op->Index();
        std::vector<SchedulerOperation *> opsInCascade = {op};
        std::vector<SchedulerOperation *> opsInBestCascade = {op};

        // Get the size of the weight buffer
        auto *refCost = refSchedule->Cost(op);
        int weightBufferSize = refCost->bufferedWeightTensor.AllocatedSize();

        // The first IFM is stored in full unless spilling disables it.
        const int ifmStoredSize = _spilling ? 0 : ifm->tensor->AllocationSizeBytes();

        // Sum of all intermediate cascade buffers (including weight buffers)
        int cascadeBuffersSize = weightBufferSize;

        // Best cascade size - Initially it's the fallback cost of the first Op in the cascade
        int bestCascadeSize = EstimateUncascadedBufferUsage(op, fallbackCost);
        int bestCascadeLocalSize = 0;

        int rangeStart = 1;
        int rangeEnd = 0;
        int rangeSize = 0;
        UniqueId startIfmRangeId = INVALID_UID;
        auto liveRangeIt = _tensorLiveRanges.find(ifm->tensor->equivalenceId);
        if ( liveRangeIt != _tensorLiveRanges.end() )
        {
            rangeStart = liveRangeIt->second.startTime;
            rangeEnd = liveRangeIt->second.endTime;
            rangeSize = liveRangeIt->second.size;
            startIfmRangeId = liveRangeIt->second.rangeId;
        }

        // Start op keeps its IFM local, so preserve its non-local usage separately.
        const int startNonLocal = std::max(0, NonLocalUsage(*op));
        int activeAdjustedMax = 0;
        int inactiveMax = 0;
        int segmentNonLocalMax = 0;
        auto updateSegmentNonLocal = [&](SchedulerOpInfo *cost, int nonLocalBytes, bool isStartOp, bool startIfmRangeAlreadyLocal)
        {
            int clampedNonLocal = std::max(0, nonLocalBytes);
            bool inRange = rangeSize > 0 && cost && cost->timeIndex >= rangeStart && cost->timeIndex <= rangeEnd;
            if ( inRange )
            {
                if ( !isStartOp )
                {
                    // Subtract the start-IFM live range only if it is actually part of this op's non-local usage.
                    // When ReuseIFM has fused the start-IFM live range with this op's local buffers, NonLocalUsage
                    // already excludes it.
                    int adjustedNonLocal = startIfmRangeAlreadyLocal ? clampedNonLocal : std::max(0, clampedNonLocal - rangeSize);
                    activeAdjustedMax = std::max(activeAdjustedMax, adjustedNonLocal);
                }
            }
            else
            {
                inactiveMax = std::max(inactiveMax, clampedNonLocal);
            }
            segmentNonLocalMax = std::max(inactiveMax, std::max(startNonLocal, activeAdjustedMax));
        };
        updateSegmentNonLocal(refCost, startNonLocal, true, false);

        // Op is the producer of the OFM consumed by the next Op to consider
        auto producer = op;
        while ( true )
        {
            auto &dependants = producer->OFM()->tensor->consumers;

            if ( dependants.size() != 1u )
            {
                // producer is either the last Op in the schedule or the start of a branch
                break;
            }

            SchedulerOperation *currentOp = dependants[0];
            refCost = refSchedule->Cost(currentOp);

            auto currentIfm = currentOp->IFM(currentOp->PrimaryIfmIndex());
            auto producerOfm = producer->OFM();

            if ( costs.find(*currentOp) != costs.end() || (refCost == nullptr) || !IsCascadable(currentOp, currentIfm, refCost) ||
                 producer->OFM()->shape != currentIfm->shape || currentIfm->requireFullTensor || producerOfm->requireFullTensor ||
                 currentIfm->tensor->needsLinearFormat || producerOfm->tensor->needsLinearFormat )
            {
                // Current op has already been processed or cannot be cascaded
                break;
            }
            if ( currentOp->Index() != producer->Index() + 1 )
            {
                // Cascading is possible, but requires reordering of operations in the schedule,
                // this is currently not supported
                break;
            }

            // Get the size of the FeatureMap buffers between current and neighbouring Ops
            int opFullIfmSize = currentIfm->tensor->AllocationSizeBytes();
            int opFullOfmSize = currentOp->OFM()->tensor->AllocationSizeBytes();

            auto bufferInfo = buffers.GetBuffer(producer, currentOp, refSchedule);
            int ifmBufferSize = bufferInfo.sizeBytes;

            // Get the size of the weight buffer
            int opWeightBuffer = refCost->bufferedWeightTensor.AllocatedSize();

            // Add current Op to cascade
            opsInCascade.push_back(currentOp);

            // Increase the accumulated intermediate buffers in the cascade
            cascadeBuffersSize += ifmBufferSize + opWeightBuffer;

            bool startIfmRangeAlreadyLocal = false;
            if ( rangeSize > 0 && startIfmRangeId != INVALID_UID )
            {
                for ( const auto *tensor : {currentIfm->tensor.get(), currentOp->OFM()->tensor.get()} )
                {
                    auto tensorLrIt = _tensorLiveRanges.find(tensor->equivalenceId);
                    if ( tensorLrIt != _tensorLiveRanges.end() && tensorLrIt->second.rangeId == startIfmRangeId )
                    {
                        startIfmRangeAlreadyLocal = true;
                        break;
                    }
                }
            }
            updateSegmentNonLocal(refCost, NonLocalUsage(*currentOp), false, startIfmRangeAlreadyLocal);
            LOG_TRACE1("\tAppend '{0}:{1}' to cascade\n", currentOp->Index(), OpTypeToString(currentOp->Type()));
            LOG_TRACE1("\t\tFull Primary IFM [{0}] bytes = {1}, Full OFM bytes [{2}] = {3}\n",
                currentIfm->shape.ToString(), opFullIfmSize, currentOp->OFM()->shape.ToString(), opFullOfmSize);
            LOG_TRACE1("\t\tCascade buffer bytes = {0} - [{1}]\n", cascadeBuffersSize, bufferInfo.shape.ToString());

            if ( _spilling )
            {
                // Set uncascadedStagingUsage to usage if the op where to be run fully in staging
                int uncascadedStagingUsage = opFullIfmSize + opFullOfmSize + NonLocalUsage(*currentOp);
                bool uncascadedFits = uncascadedStagingUsage < peakStagingUsage;
                bool buffersExceedPeak = cascadeBuffersSize > peakStagingUsage;
                if ( uncascadedFits || buffersExceedPeak )
                {
                    // Cascade until an Op fits in its entirety or the accumulated buffers no longer fit
                    break;
                }
                else
                {
                    opsInBestCascade = opsInCascade;
                    bestCascadeSize = cascadeBuffersSize;
                    bestCascadeLocalSize = cascadeBuffersSize;
                }
            }
            else
            {
                int uncascadedStagingUsage = EstimateUncascadedBufferUsage(currentOp, fallbackSchedule->Cost(currentOp));
                // Calculate the total size of the current cascade
                int cascadeLocalSize = ifmStoredSize + cascadeBuffersSize + opFullOfmSize;
                int cascadeSize = cascadeLocalSize + segmentNonLocalMax;

                // Determine if current cascade is the best so far
                // Allow larger cascades if they reduce total staging usage versus uncascaded ops.
                if ( cascadeSize < bestCascadeSize || cascadeSize < uncascadedStagingUsage )
                {
                    bestCascadeSize = cascadeSize;
                    bestCascadeLocalSize = cascadeLocalSize;
                    opsInBestCascade = opsInCascade;
                }
                // Determine if cascading search should stop
                int cascadePrefix = ifmStoredSize + segmentNonLocalMax + cascadeBuffersSize;
                if ( ((uncascadedStagingUsage < peakStagingUsage) && (bestCascadeSize < peakStagingUsage)) || (cascadePrefix > bestCascadeSize) )
                {
                    // Both the existing cascade and current Op fits, or the cascade cannot shrink further.
                    break;
                }
            }

            producer = currentOp;
        }

        if ( opsInBestCascade.size() > 1 )
        {
            // A cascade was created - assign cascade and ref_cost to all of the Ops
            int cascadeEnd = cascadeStart + int(opsInBestCascade.size()) - 1;  // Inclusive end

            std::unordered_map<UniqueId, CascadeBuffer> buffersInCascade;
            SchedulerOperation *prevOp = nullptr;
            for ( auto cascadedOp : opsInBestCascade )
            {
                assert(cascadedOp->Index() <= cascadeEnd);
                auto cascadedCost = std::make_unique<SchedulerOpInfo>(*refSchedule->Cost(cascadedOp));
                cascadedCost->cascade = cascadeEnd;
                cascadedCost->firstInCascade = (*cascadedOp == *opsInBestCascade[0]);
                costs.emplace(*cascadedOp, std::move(cascadedCost));

                if ( prevOp )
                {
                    auto const &buffer = buffers.GetBuffer(prevOp, cascadedOp, refSchedule);
                    buffersInCascade[*cascadedOp] = buffer;
                }

                prevOp = cascadedOp;
            }

            // Create a CascadeInfo for the cascade
            cascadeMap.emplace(cascadeEnd,
                CascadeInfo(cascadeStart, cascadeEnd, bestCascadeSize, bestCascadeLocalSize, std::move(buffersInCascade)));
            if ( !_spilling )
            {
                // Update peak memory usage
                peakStagingUsage = std::max(bestCascadeSize, peakStagingUsage);
            }
        }
        else
        {
            // Assign fallback cost to the initial Op
            costs.emplace(*op, std::make_unique<SchedulerOpInfo>(*fallbackCost));
            if ( !_spilling )
            {
                peakStagingUsage = std::max(EstimateUncascadedBufferUsage(op, fallbackCost), peakStagingUsage);
            }
        }
    }
    // Update costing and cascade information for the ref_schedule
    refSchedule->UpdateCosts(costs);
    refSchedule->cascades = std::move(cascadeMap);
}


bool CascadeBuilder::IsCascadable(const SchedulerOperation *op, SchedulerConnection *ifmConn, SchedulerOpInfo *cost) const
{
    OpType type = op->Type();
    auto ifm = ifmConn->tensor;

    if ( ifm->IsConstant() )
    {
        return false;
    }

    if ( op->IsReordering() )
    {
        LOG_TRACE1("Not cascading Transpose/Reverse");
        return false;
    }

    // TODO MLBEDSW-11387: Support cascading for ReduceSum
    // TODO MLBEDSW-7003: Resampling mode is not supported for cascaded convolutions
    return (cost->stripe.Height() < op->OFM()->shape.Height()) &&
           ((IsConvolution(type) && (ifmConn->resamplingMode == ArchResampling::None)) || IsElementwise(type) ||
               (IsPooling(type) && type != OpType::ReduceSum));
}


int CascadeBuilder::EstimateUncascadedBufferUsage(SchedulerOperation *op, SchedulerOpInfo *) const
{
    // Use the exact local-op memory captured alongside non-local usage.
    return OpLocalUsage(*op) + NonLocalUsage(*op);
}


int CascadeBuilder::NonLocalUsage(UniqueId uid) const
{
    auto opPos = _nonLocalMemUsage.find(uid);
    if ( opPos != _nonLocalMemUsage.end() )
    {
        return opPos->second;
    }

    return 0;
}

int CascadeBuilder::OpLocalUsage(UniqueId uid) const
{
    auto opPos = _opLocalMemUsage.find(uid);
    if ( opPos != _opLocalMemUsage.end() )
    {
        return opPos->second;
    }
    return 0;
}

}  // namespace regor
