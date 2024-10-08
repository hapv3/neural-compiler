//
// SPDX-FileCopyrightText: Copyright 2021-2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
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
                bufferSize = DataTypeStorageSizeBytes(ofm->tensor->dataType, bufferShape.Elements());
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


CascadeBuilder::CascadeBuilder(vector_span<std::unique_ptr<SchedulerOperation>> ops, const std::unordered_map<UniqueId, int> &nonLocalMemUsage, bool spilling) :
        _ops(ops), _nonLocalMemUsage(nonLocalMemUsage)

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
                peakStagingUsage = std::max(EstimateBufferUsage(op, fallbackCost), peakStagingUsage);
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
        int weightBufferSize = 0;
        auto refCost = refSchedule->Cost(op);
        if ( refCost->bufferedWeightTensor.tensor )
        {
            weightBufferSize = refCost->bufferedWeightTensor.tensor->AllocationSizeBytes();
        }

        // The first IFM needs to be stored in full
        int cascadeIFMSize = _spilling ? 0 : ifm->tensor->AllocationSizeBytes();

        // Add non-local memory usage
        cascadeIFMSize += NonLocalUsage(*op);

        // Sum of all intermediate cascade buffers (including weight buffers)
        int cascadeBuffersSize = weightBufferSize;

        // Best cascade size - Initially it's the fallback cost of the first Op in the cascade
        int bestCascadeSize = EstimateBufferUsage(op, fallbackCost);

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
            int opWeightBuffer = 0;
            if ( refCost->bufferedWeightTensor.tensor )
            {
                opWeightBuffer = refCost->bufferedWeightTensor.tensor->AllocationSizeBytes();
            }

            // Calculate the uncascaded memory requirement for current Op
            int uncascadedStagingUsage = opFullIfmSize + opFullOfmSize + NonLocalUsage(*currentOp);

            // Add current Op to cascade
            opsInCascade.push_back(currentOp);

            // Increase the accumulated intermediate buffers in the cascade
            cascadeBuffersSize += ifmBufferSize + opWeightBuffer;

            LOG_TRACE1("\tAppend '{0}:{1}' to cascade\n", currentOp->Index(), OpTypeToString(currentOp->Type()));
            LOG_TRACE1("\t\tFull Primary IFM [{0}] bytes = {1}, Full OFM bytes [{2}] = {3}\n",
                currentIfm->shape.ToString(), opFullIfmSize, currentOp->OFM()->shape.ToString(), opFullOfmSize);
            LOG_TRACE1("\t\tCascade buffer bytes = {0} - [{1}]\n", cascadeBuffersSize, bufferInfo.shape.ToString());

            if ( _spilling )
            {
                if ( (uncascadedStagingUsage < peakStagingUsage) || (cascadeBuffersSize > peakStagingUsage) )
                {
                    // Cascade until an Op fits in its entirety or the accumulated buffers no longer fit
                    break;
                }
                else
                {
                    opsInBestCascade = opsInCascade;
                    bestCascadeSize = cascadeBuffersSize;
                }
            }
            else
            {
                // Calculate the total size of the current cascade
                int cascadeSize = cascadeIFMSize + cascadeBuffersSize + opFullOfmSize;

                // Determine if current cascade is the best so far
                if ( cascadeSize < bestCascadeSize )
                {
                    bestCascadeSize = cascadeSize;
                    opsInBestCascade = opsInCascade;
                }
                // Determine if cascading search should stop
                if ( ((uncascadedStagingUsage < peakStagingUsage) && (bestCascadeSize < peakStagingUsage)) ||
                     (cascadeIFMSize + cascadeBuffersSize) > bestCascadeSize )
                {
                    // Both the existing cascade and current Op fits
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
                costs.emplace(*cascadedOp, std::move(cascadedCost));

                if ( prevOp )
                {
                    auto const &buffer = buffers.GetBuffer(prevOp, cascadedOp, refSchedule);
                    buffersInCascade[*cascadedOp] = buffer;
                }

                prevOp = cascadedOp;
            }

            // Create a CascadeInfo for the cascade
            cascadeMap.emplace(cascadeEnd, CascadeInfo(cascadeStart, cascadeEnd, bestCascadeSize, std::move(buffersInCascade)));
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
                peakStagingUsage = std::max(EstimateBufferUsage(op, fallbackCost), peakStagingUsage);
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

    // ReduceSum: sum over the entire IFM - full shape needed
    // TransposeConv: Uses resampling mode which is not supported in cascades
    return (cost->stripe.Height() < op->OFM()->shape.Height()) &&
           ((IsConvolution(type) && (ifmConn->resamplingMode == ArchResampling::None)) || IsElementwise(type) ||
               (IsPooling(type) && type != OpType::ReduceSum));
}


int CascadeBuilder::EstimateBufferUsage(SchedulerOperation *op, SchedulerOpInfo *) const
{
    // Estimate the RAM required for the Op if all FeatureMaps are in RAM
    int size = NonLocalUsage(*op);

    for ( auto usage : {TensorUsage::IFM, TensorUsage::IFM1, TensorUsage::OFM} )
    {
        SchedulerConnection *fm = IsOFM(usage) ? op->Output(usage) : op->TryInput(usage);
        if ( !fm )
        {
            continue;
        }

        if ( fm->requireFullTensor )
        {
            size += fm->tensor->AllocationSizeBytes();
        }
        else
        {
            size += fm->PartialAllocationSizeBytes();
            size = RoundAway(size, 16);
        }
    }

    return size;
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

}  // namespace regor
