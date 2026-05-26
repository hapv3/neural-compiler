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

#include "ethos_u85_performance.hpp"

#include "common/common.hpp"
#include "common/logging.hpp"

#include "architecture/architecture.hpp"
#include "compiler/shape_util.hpp"
#include "ethos_u85.hpp"

namespace regor
{

static const Point2i s_SubkernelLimits[] = {
    {0, 0},  // No kernel
    {8, 8},  // Convolution
    {8, 8},  // Depthwise
    {1, 1},  // VectorProduct
    {8, 8},  // Pooling
    {8, 8},  // ReduceSum
    {8, 8},  // ReduceMinMax
    {8, 8},  // ArgMax
    {1, 1},  // Elementwise
    {1, 1},  // Resize
    {1, 1},  // Dma
};

static constexpr bool OpUsesMacs(EthosU85NpuOp npuOp)
{
    return (npuOp != EthosU85NpuOp::Elementwise && npuOp != EthosU85NpuOp::Resize && npuOp != EthosU85NpuOp::Dma &&
            npuOp != EthosU85NpuOp::Branch && npuOp != EthosU85NpuOp::None);
}

EthosU85Performance::EthosU85Performance(ArchEthosU85 *arch, const EthosU85PerfInfo *perfInfo) : _arch(arch)
{
    _perfInfo = perfInfo;
}

CycleCost EthosU85Performance::MeasureCycleCost(const PerformanceQuery &query)
{
    CycleCost cycles;
    EthosU85Cycles cycleComponents;
    auto npuOp = _arch->GetHWOp(query.type);
    if ( npuOp == EthosU85NpuOp::Dma || npuOp == EthosU85NpuOp::Branch )
    {
        // DMA and Branch operations have no NPU cycle cost
        cycles.opCycles = 0;
        cycles.macs = 0;
    }
    else if ( npuOp == EthosU85NpuOp::Resize )
    {
        // TODO: Implement for Resize
        cycles.opCycles = 0;
        cycles.macs = 0;
    }
    else if ( OpUsesMacs(npuOp) )
    {
        // MAC operation cycle calculation
        cycleComponents = EstimateMacOpCycles(query);
        cycles.opCycles = cycleComponents.cycles;
        cycles.macs = cycleComponents.macs;
    }
    else if ( npuOp == EthosU85NpuOp::Elementwise )
    {
        // Elementwise operation cycle calculation
        cycleComponents = EstimateElementwiseCycles(query);
        cycles.opCycles = cycleComponents.cycles;
        cycles.macs = 0;
    }
    else
    {
        assert(false && "Unknown operator cycle costing");
    }

    if ( _db && _nextId != -1 )
    {
        // Record to debug database
        assert(_mainTable != -1);
        EthosU85OpConfig *opConfig = static_cast<EthosU85OpConfig *>(query.config);

        std::vector<std::string> row = {
            OpUsesMacs(npuOp) ? std::to_string(cycleComponents.macCycles) : "",
            std::to_string(cycleComponents.aoCycles),
            std::to_string(cycleComponents.cmdCycles),
            opConfig ? EnumToString(opConfig->Traversal()) : "",
        };
        auto shapeToStrings = [&row](const std::vector<int> &shape)
        {
            std::transform(shape.begin(), shape.end(), std::back_inserter(row),
                [](int n) -> std::string { return n ? std::to_string(n) : ""; });
        };

        shapeToStrings(ReshapeToNHWC(opConfig ? opConfig->IfmBlock() : Shape()).ToList<int>());
        shapeToStrings(ReshapeToNHWC(opConfig ? opConfig->OfmBlock() : Shape()).ToList<int>());
        shapeToStrings(ReshapeToNHWC(opConfig ? opConfig->OfmUBlock() : Shape()).ToList<int>());

        _db->AddRow(_mainTable, _nextId, std::move(row));
        _nextId = -1;
    }

    return cycles;
}

int64_t EthosU85Performance::MemToMemCycles(const ArchitectureMemory *dest, const ArchitectureMemory *source, int sizeBytes)
{
    int64_t fromCycles = int64_t(float(sizeBytes) / ChannelBW(source, MemChannel::Mem2Mem));
    fromCycles += source->ReadLatency();
    // TODO: Below shouldn't use the OFM channel. See MLBEDSW-9384.
    int64_t toCycles = int64_t(float(sizeBytes) / ChannelBW(dest, MemChannel::Write));
    toCycles += dest->WriteLatency();
    return std::max(fromCycles, toCycles);
}

namespace
{

int64_t EstimateMemoryTransfer(bool isRead, ArchitectureMemory *memory, TensorFormat format, int elementBits,
    const Shape &block, const Shape &shape, int64_t elementsToTransfer)
{
    int burstLen = 8;

    if ( format == TensorFormat::NHCWB16 )
    {
        int zStride = (shape.Width() * elementBits * 16) / 8;
        if ( zStride == block.Depth() )
        {
            burstLen = elementBits * block.ElementsWC();
        }
        else if ( isRead )
        {
            burstLen = 16 * elementBits * block.Width();
        }
        else
        {
            burstLen = 16 * elementBits * block.Width();
        }
    }
    else if ( format == TensorFormat::NHWC )
    {
        int xStride = (shape.Depth() * elementBits) / 8;
        if ( isRead )
        {
            if ( xStride == block.Depth() )
            {
                burstLen = elementBits * block.ElementsWC();
            }
            else
            {
                burstLen = elementBits * block.Depth();
            }
        }
        else
        {
            if ( (block.Depth() <= 16) && xStride == block.Depth() )
            {
                burstLen = elementBits * block.ElementsWC();
            }
            else
            {
                burstLen = std::min(std::min(64 * 8, 16 * elementBits), block.Depth() * elementBits);
            }
        }
    }

    int64_t bytesToTransfer = (elementsToTransfer * elementBits) / 8;
    int burstLenBytes = std::min(memory->MaxBurstLength(), burstLen / 8);
    assert(burstLenBytes > 0 && "Burst length cannot be zero");
    return (bytesToTransfer * memory->MaxBurstLength()) / burstLenBytes;
}

int64_t MinimumIfmCycles(const PerformanceQuery &query)
{
    EthosU85OpConfig *opConfig = static_cast<EthosU85OpConfig *>(query.config);

    int ifmBits = DataTypeSizeBits(query.ifmType[0]);  // All inputs expect same bit width
    const int ifmCount = query.ifmShape[1] ? int(std::size(query.ifmShape)) : 1;
    int64_t cyclesIfm = 0;
    for ( int i = 0; i < ifmCount; i++ )
    {
        // Input block HW transfer (only for elements present)
        int64_t ifmElements = Shape::Min(query.ifmShape[i], opConfig->IfmBlock()).Elements64();
        int64_t cyclesIfmBlk = query.ifmMemory[i]->ReadLatency();
        int64_t tx = EstimateMemoryTransfer(true, query.ifmMemory[i], query.ifmFormat[i], ifmBits, opConfig->IfmBlock(),
            query.ifmShape[i], ifmElements);
        cyclesIfmBlk += int64_t(float(tx) / query.ifmMemory[i]->Bandwidth());

        cyclesIfm = std::max(cyclesIfm, cyclesIfmBlk);
    }
    return cyclesIfm;
}

int64_t MinimumOfmCycles(const PerformanceQuery &query)
{
    EthosU85OpConfig *opConfig = static_cast<EthosU85OpConfig *>(query.config);

    // Output block HW transfer (only for elements present)
    int ofmBits = DataTypeSizeBits(query.ofmType);
    int64_t ofmElements = Shape::Min(query.ofmShape, opConfig->OfmBlock()).Elements64();
    int64_t cyclesOfm = query.ofmMemory->WriteLatency();
    int64_t tx = EstimateMemoryTransfer(false, query.ofmMemory, query.ofmFormat, ofmBits, opConfig->OfmBlock(), query.ofmShape, ofmElements);
    cyclesOfm += int64_t(float(tx) / query.ofmMemory->Bandwidth());

    return cyclesOfm;
}

}  // namespace

int64_t EthosU85Performance::EstimateMacCyclesPerBlock(const PerformanceQuery &query)
{
    EthosU85OpConfig *opConfig = static_cast<EthosU85OpConfig *>(query.config);
    auto npuOp = _arch->GetHWOp(query.type);
    assert(npuOp != EthosU85NpuOp::None);

    // Clip blocks to FM shapes in case the block boundary exceeds the full FM shape in any dimension.
    // This prevents estimation of microblocks which are never actually processed.
    Shape ifmBlock = Shape::Min(query.ifmShape[0], opConfig->IfmBlock());
    Shape ofmBlock = Shape::Min(query.ofmShape, opConfig->OfmBlock());
    Shape ofmUBlock = opConfig->OfmUBlock();

    int ifmBits = DataTypeSizeBits(query.ifmType[0]);
    Shape numUBlocks = Shape::DivRoundUp(ofmBlock, ofmUBlock);
    bool use48BitAcc = opConfig->Acc() == EthosU85Accumulator::Acc48;

    int64_t cyclesDpuBlk = 0;
    // Ideal weight decode estimate
    int cyclesWb = 32 * ofmUBlock.Depth() / (8 * _arch->_cores);

    int subKernelWidth = s_SubkernelLimits[int(npuOp)].x;
    int subKernelHeight = s_SubkernelLimits[int(npuOp)].y;
    const Point2i kernelSize = query.kernel->Size();
    bool isConvolutionMxN = (npuOp == EthosU85NpuOp::Convolution);

    for ( int x = 0; x < kernelSize.x; x += subKernelWidth )
    {
        for ( int y = 0; y < kernelSize.y; y += subKernelHeight )
        {
            int subKernelElements = std::min(kernelSize.y - y, subKernelHeight);
            subKernelElements *= std::min(kernelSize.x - x, subKernelWidth);

            // Calculate processing cycles
            int numKernelSteps = 0;
            int cycles = 0;
            if ( npuOp == EthosU85NpuOp::Pooling || npuOp == EthosU85NpuOp::ReduceMinMax || npuOp == EthosU85NpuOp::ArgMax )
            {
                numKernelSteps = 1;
                cycles = std::max(4, subKernelElements) * numUBlocks.Elements() * (ifmBits / 2);
            }
            else if ( npuOp == EthosU85NpuOp::Depthwise )
            {
                numKernelSteps = DivRoundUp(subKernelElements, 4);
                cycles = 4 * numUBlocks.ElementsWH() * (ifmBits / 8);
                cycles = std::max(cyclesWb, cycles) * numKernelSteps * numUBlocks.Depth();
            }
            else if ( (isConvolutionMxN && opConfig->Traversal() != EthosU85Traversal::PartKernel) ||
                      npuOp == EthosU85NpuOp::VectorProduct || npuOp == EthosU85NpuOp::ReduceSum )
            {
                numKernelSteps = subKernelElements;
                cycles = std::max(cyclesWb, ifmBlock.Depth() / 8 * numUBlocks.ElementsWH()) * numKernelSteps *
                         numUBlocks.Depth() * (ifmBits / 8);
                cycles /= query.weightFormat & WeightFormat::Sparse2_4 ? 2 : 1;
            }
            else
            {
                assert(opConfig->Traversal() == EthosU85Traversal::PartKernel);
                int divider = (ifmBits == 16) ? 2 : 4;
                numKernelSteps = DivRoundUp(subKernelElements, divider);
                cycles = std::max(cyclesWb, 4 * numUBlocks.ElementsWH()) * numKernelSteps * numUBlocks.Depth() *
                         DivRoundUp(ifmBlock.Depth(), 8);
                cycles /= query.weightFormat & WeightFormat::Sparse2_4 ? 2 : 1;
            }

            // Calculate delay
            int delayCycles = 0;
            int delay = (use48BitAcc && (_arch->_macs <= 128)) ? 3 : 2;

            if ( numUBlocks.ElementsWH() == 1 )
            {
                if ( numUBlocks.Depth() == 1 )
                {
                    delayCycles = delay * numKernelSteps;
                }
                else if ( numKernelSteps > 1 )
                {
                    delayCycles = delay * (numKernelSteps - 1) * numUBlocks.Depth();
                }
            }

            if ( isConvolutionMxN && opConfig->Traversal() == EthosU85Traversal::PartKernel )
            {
                delayCycles *= DivRoundUp(ifmBlock.Depth(), 8);
            }

            cyclesDpuBlk += cycles;
            cyclesDpuBlk += delayCycles;
        }
    }

    if ( npuOp == EthosU85NpuOp::Convolution || npuOp == EthosU85NpuOp::VectorProduct || npuOp == EthosU85NpuOp::ReduceSum )
    {
        cyclesDpuBlk *= DivRoundUp(query.ifmShape[0].Depth(), ifmBlock.Depth());
    }

    return cyclesDpuBlk;
}

EthosU85Cycles EthosU85Performance::EstimateMacOpCycles(const PerformanceQuery &query)
{
    EthosU85OpConfig *opConfig = static_cast<EthosU85OpConfig *>(query.config);
    auto npuOp = _arch->GetHWOp(query.type);
    assert(npuOp != EthosU85NpuOp::None);
    assert(OpUsesMacs(npuOp));

    // Calculate number of fractional OFM blocks (clipped to OFM shape) and aligned to microblocks
    Shape ofmBlock = Shape::Min(query.ofmShape, opConfig->OfmBlock());
    Shape ofmUBlock = opConfig->OfmUBlock();
    int uBlocksInOfmBlock = Shape::DivRoundUp(ofmBlock, ofmUBlock).Elements();
    assert(uBlocksInOfmBlock > 0);
    double numOfmBlks = double(Shape::DivRoundUp(query.ofmShape, ofmUBlock).Elements()) / uBlocksInOfmBlock;

    // Estimate AO cycles
    const double aoCyclesPerElem = EstimateAOCyclesPerElement(query);
    const double aoComputeCyclesPerBlock = std::ceil(aoCyclesPerElem * ofmBlock.Elements());

    // Estimate scale and bias read cycles if present
    double biasCyclesPerBlock = 0;
    if ( (npuOp == EthosU85NpuOp::Convolution || npuOp == EthosU85NpuOp::Depthwise || npuOp == EthosU85NpuOp::VectorProduct) &&
         query.constShape.Size() > 0 && query.constShape.Depth() > 0 )
    {
        auto *fromMem = query.weightStagingMemory ? query.weightStagingMemory : query.constMemory;
        biasCyclesPerBlock = double(10) * ofmBlock.Depth() * fromMem->ReadLatency() / 256;
    }
    const double aoCyclesPerBlock = std::max(aoComputeCyclesPerBlock, biasCyclesPerBlock);

    // Estimate MAC cycles
    const double macCyclesPerBlock = EstimateMacCyclesPerBlock(query);

    // Estimate the command issuing limit cycles
    int64_t ifmBlockCycles = MinimumIfmCycles(query);
    int64_t ofmBlockCycles = MinimumOfmCycles(query);
    int64_t minMemCycles = (numOfmBlks > 1) ? std::abs(ofmBlockCycles - ifmBlockCycles) : 0;
    const double cmdIssueLimitCycles = (minMemCycles + macCyclesPerBlock + aoCyclesPerBlock) / 4;  // Per DPU

    // Estimate full Op cycles
    const double balancedMacCyclesPerBlock = std::max(macCyclesPerBlock, cmdIssueLimitCycles);
    const double balancedAoCyclesPerBlock = std::max(aoCyclesPerBlock, cmdIssueLimitCycles);

    int64_t totalCycles = 0;
    if ( balancedMacCyclesPerBlock > balancedAoCyclesPerBlock )
    {
        totalCycles = int64_t(balancedMacCyclesPerBlock * numOfmBlks);
        if ( query.scheduling & OpScheduling::Last ) totalCycles += balancedAoCyclesPerBlock;
    }
    else
    {
        totalCycles = balancedAoCyclesPerBlock * numOfmBlks + balancedMacCyclesPerBlock;
    }

    if ( query.scheduling & OpScheduling::Last ) totalCycles += ofmBlockCycles;
    if ( query.scheduling & OpScheduling::First ) totalCycles += ifmBlockCycles;

    // Estimate total number of MACs
    int64_t totalMacs = int64_t(query.kernel->ElementsWH()) * query.ofmShape.Elements64();
    if ( !(npuOp == EthosU85NpuOp::Depthwise || npuOp == EthosU85NpuOp::Pooling || npuOp == EthosU85NpuOp::ReduceMinMax || npuOp == EthosU85NpuOp::ArgMax) )
    {
        totalMacs *= query.ifmShape[0].Depth();
    }
    totalMacs /= query.weightFormat & WeightFormat::Sparse2_4 ? 2 : 1;

    const int64_t totalMacCycles = macCyclesPerBlock * numOfmBlks + aoCyclesPerBlock;
    const int64_t totalAoCycles = aoComputeCyclesPerBlock * numOfmBlks + macCyclesPerBlock;

    EthosU85Cycles cycleComponents;
    cycleComponents.cycles = totalCycles;
    cycleComponents.macCycles = totalMacCycles;
    cycleComponents.aoCycles = totalAoCycles;
    cycleComponents.cmdCycles = int64_t(cmdIssueLimitCycles);
    cycleComponents.macs = totalMacs;

    return cycleComponents;
}

EthosU85Cycles EthosU85Performance::EstimateElementwiseCycles(const PerformanceQuery &query)
{
    EthosU85OpConfig *opConfig = static_cast<EthosU85OpConfig *>(query.config);
    assert(_arch->GetHWOp(query.type) == EthosU85NpuOp::Elementwise);

    auto ofmShape =
        (query.ofmFormat == TensorFormat::NHCWB16) ? Shape::RoundAway(query.ofmShape, Shape(1, 1, 1, 16)) : query.ofmShape;
    const int64_t elements = ofmShape.Elements64();

    // Estimate AO cycles
    const double aoCyclesPerElem = EstimateAOCyclesPerElement(query);
    const double aoCycles = std::ceil(aoCyclesPerElem * elements);

    // Estimate the command issuing limit cycles
    const int ofmBlockElements = opConfig->OfmBlock().Elements();
    assert(ofmBlockElements > 0);
    // Assumes overlapped I/O
    const double blockCycles = std::max(MinimumOfmCycles(query), MinimumIfmCycles(query));
    const double cmdCyclesPerElem = (blockCycles / ofmBlockElements + aoCyclesPerElem) / 4.0;  // per DPU
    const double cmdIssueLimitCycles = std::ceil(cmdCyclesPerElem * elements);

    const int64_t totalCycles = std::max(cmdIssueLimitCycles, aoCycles);

    EthosU85Cycles cycleComponents;
    cycleComponents.cycles = totalCycles;
    cycleComponents.aoCycles = int64_t(aoCycles);
    cycleComponents.cmdCycles = int64_t(cmdIssueLimitCycles);
    cycleComponents.macCycles = 0;
    cycleComponents.macs = 0;
    return cycleComponents;
}


double EthosU85Performance::GetActivationCyclesPerElement(ReverseType reverse, TransposeType transpose)
{
    size_t activationPerfIndex = 0;
    if ( transpose == TransposeType::NWHC || transpose == TransposeType::NWCH || transpose == TransposeType::NCWH )
    {
        // H <-> W transposes has half throughput
        activationPerfIndex = 1;
    }
    else if ( reverse == ReverseType::W )
    {
        // Reversing in W-dimension has half throughput
        activationPerfIndex = 1;
    }

    assert(activationPerfIndex < std::size(_perfInfo->activationCycles));
    return _perfInfo->activationCycles[activationPerfIndex];
}

double EthosU85Performance::GetOutputCyclesPerElement(OpType opType, DataType ifmType, DataType ofmType, bool writesToCB)
{
    if ( opType == OpType::Div )
    {
        return 33.0f;
    }

    int ifmBits = DataTypeSizeBits(ifmType);
    int ofmBits = DataTypeSizeBits(ofmType);
    size_t outputPerfIndex = 0;

    if ( writesToCB && (ofmBits == 32 || ifmBits == 32) )
    {
        outputPerfIndex = 4;
    }
    else if ( ofmBits == 64 )
    {
        outputPerfIndex = 3;
    }
    else if ( IsBinaryElementwise(opType) && ifmBits == 32 )
    {
        outputPerfIndex = ofmBits == 32 ? 5 : 4;
    }
    else if ( ofmBits == 32 )
    {
        outputPerfIndex = 2;
    }
    else if ( ofmBits == 16 )
    {
        outputPerfIndex = 1;
    }
    else
    {
        assert(ofmBits == 8);
        outputPerfIndex = 0;
    }

    assert(outputPerfIndex < std::size(_perfInfo->outputCycles));
    return _perfInfo->outputCycles[outputPerfIndex];
}

double EthosU85Performance::EstimateAOCyclesPerElement(const PerformanceQuery &query)
{
    double cyclesPerElement = 0.0;
    std::vector<double> cyclesPerOp;
    EthosU85OpGroup *opGroup = static_cast<EthosU85OpGroup *>(query.opGroup);
    assert(opGroup);
    for ( const auto &opInfo : *opGroup )
    {
        OpType opType = opInfo.type;
        auto ofm = opInfo.ofm;
        if ( IsActivation(opType) || opType == OpType::Reverse || opType == OpType::Transpose )
        {
            // Activations are done through the same pass of the AO, push the cycles to the list
            cyclesPerOp.push_back(GetActivationCyclesPerElement(ofm.reverse, ofm.transpose));
        }
        else
        {
            // New chained operation, add the max of the cycles per element for all the previous ops
            // It can contain 0 or 1 output ops and 0 or more activations
            cyclesPerElement += cyclesPerOp.empty() ? 0.0 : *std::max_element(cyclesPerOp.begin(), cyclesPerOp.end());
            cyclesPerOp.clear();
            // If the OFM does not require allocation it means this op writes to a chaining buffer
            bool writesToCB = !opGroup->NeedsAllocation(ofm.key);
            cyclesPerOp.push_back(GetOutputCyclesPerElement(opType, opInfo.ifm[0].type, ofm.type, writesToCB));
        }
    }

    // Add the cycles for the last op in the chain (and possibly activations)
    cyclesPerElement += cyclesPerOp.empty() ? 0.0 : *std::max_element(cyclesPerOp.begin(), cyclesPerOp.end());
    return cyclesPerElement;
}

ElementAccess EthosU85Performance::MeasureElementAccess(const PerformanceQuery &query)
{
    ElementAccess access;
    EthosU85OpConfig *opConfig = static_cast<EthosU85OpConfig *>(query.config);
    auto npuOp = _arch->GetHWOp(query.type);
    assert(npuOp != EthosU85NpuOp::None);

    Shape ifmRounding = _arch->GetStorageRounding(query.ifmFormat[0]);

    // Convolution & pooling
    if ( OpUsesMacs(npuOp) )
    {
        Shape ifmBlock = Shape::Min(query.ifmShape[0], opConfig->IfmBlock());
        Shape ofmBlock = Shape::Min(query.ofmShape, opConfig->OfmBlock());

        // Number of ofm blocks in the overall output shape
        Shape ofmBlocks = Shape::DivRoundUp(query.ofmShape, ofmBlock);

        int ofmBlockDepth = ofmBlock.Depth();
        if ( npuOp == EthosU85NpuOp::Depthwise || npuOp == EthosU85NpuOp::Pooling ||
             npuOp == EthosU85NpuOp::ReduceMinMax || npuOp == EthosU85NpuOp::ArgMax )
        {
            ofmBlocks = ofmBlocks.WithDepth(1);
            ofmBlockDepth = query.ifmShape[0].Depth();
        }

        // Number of sub kernels
        int subKernelWidth = s_SubkernelLimits[int(npuOp)].x;
        int subKernelHeight = s_SubkernelLimits[int(npuOp)].y;
        int subkernels = DivRoundUp(query.kernel->Size().x, subKernelWidth) * DivRoundUp(query.kernel->Size().y, subKernelHeight);

        int ifmFetch =
            (Shape::RoundAway(ifmBlock, ifmRounding).ElementsWH() * Shape::RoundAway(query.ifmShape[0], ifmRounding).Depth());

        int ofmBlockCount = ofmBlocks.Elements();

        access.ifmRead[0] = ifmFetch * subkernels * ofmBlockCount;

        // Calculate weight and bias/scale reads
        if ( npuOp == EthosU85NpuOp::Convolution || npuOp == EthosU85NpuOp::Depthwise || npuOp == EthosU85NpuOp::VectorProduct )
        {
            int kernelRead = query.kernel->Size().AreaXY();
            if ( npuOp != EthosU85NpuOp::Depthwise )
            {
                kernelRead *= query.ifmShape[0].Depth();
            }

            int weightFetch = kernelRead * ofmBlockDepth * ofmBlockCount;
            access.constRead[0] = weightFetch;
            access.constRead[1] = query.ofmShape.Depth();  // Scales & biases
            access.weightsRefetch = ofmBlocks.ElementsWH();
        }
    }
    else if ( npuOp == EthosU85NpuOp::Elementwise )
    {
        bool encodedScalar = false;
        for ( size_t i = 0; i < std::size(query.ifmShape); i++ )
        {
            if ( query.ifmShape[i] && (query.ifmShape[i].Elements64() > 1 || encodedScalar) )
            {
                access.ifmRead[i] = Shape::RoundAway(query.ifmShape[i], ifmRounding).Elements64();
            }
            else if ( query.ifmShape[i] )
            {
                // Only one scalar can be encoded
                encodedScalar = encodedScalar || (query.ifmShape[i].Elements64() == 1);
            }
        }
    }
    else if ( npuOp == EthosU85NpuOp::Resize )
    {
        // TODO: Implement for Resize
        access.ifmRead[0] = Shape::RoundAway(query.ifmShape[0], ifmRounding).Elements64();
    }
    else if ( npuOp == EthosU85NpuOp::Dma )
    {
        if ( query.type == OpType::Gather )
        {
            // One element from IFM0 (positions) is read per element in IFM1 (index)
            access.ifmRead[0] = Shape::RoundAway(query.ifmShape[1], ifmRounding).Elements64();

            // Complete IFM1 (index) is read
            access.ifmRead[1] = Shape::RoundAway(query.ifmShape[1], ifmRounding).Elements64();
        }
        else
        {
            LOG_WARN("Missing element access estimation for DMA op {}\n", OpTypeToString(query.type).c_str());
        }
    }
    else if ( npuOp == EthosU85NpuOp::Branch )
    {
    }
    else
    {
        assert(false);
    }

    // Measure acces for external FMs of chained operations
    EthosU85OpGroup *opGroup = static_cast<EthosU85OpGroup *>(query.opGroup);
    assert(opGroup);
    for ( const auto &opInfo : *opGroup )
    {
        auto it = std::find_if(query.featureMapRecords.begin(), query.featureMapRecords.end(),
            [&opInfo](const FeatureMapRecord &record) { return record.opId == opInfo.opId; });
        if ( it != query.featureMapRecords.end() )
        {
            const FeatureMapRecord &fmRecord = *it;
            if ( !fmRecord.access )  // Let the caller control whether it wants results
                continue;

            if ( IsIFM(fmRecord.usage) )
            {
                int ifmIdx = GetUsageIndex(fmRecord.usage);
                assert(size_t(ifmIdx) < std::size(fmRecord.access->ifmRead));
                // If the feature map is a scalar it will be enocded and won't require any external read
                if ( fmRecord.shape.Elements64() > 1 )
                {
                    Shape extIfmRounding = _arch->GetStorageRounding(fmRecord.format);
                    fmRecord.access->ifmRead[ifmIdx] = Shape::RoundAway(fmRecord.shape, extIfmRounding).Elements64();
                }
            }
            else
            {
                assert(IsOFM(fmRecord.usage) && "Unexpected usage for external FM");
                Shape extOfmRounding = _arch->GetStorageRounding(fmRecord.format);
                fmRecord.access->ofmWrite = Shape::RoundAway(fmRecord.shape, extOfmRounding).Elements64();
            }
        }
    }

    // Complete OFM is written as long as it needs to be allocated
    if ( opGroup->NeedsAllocation(opGroup->begin()->ofm.key) )
    {
        Shape ofmRounding = _arch->GetStorageRounding(query.ofmFormat);
        access.ofmWrite = Shape::RoundAway(query.ofmShape, ofmRounding).Elements64();
    }

    return access;
}


ElementAccess EthosU85Performance::ElementTransferToBytes(const PerformanceQuery &query, const ElementAccess &access)
{
    EthosU85OpConfig *opConfig = static_cast<EthosU85OpConfig *>(query.config);
    auto ifmBlock = opConfig ? opConfig->IfmBlock() : Shape(1, 1, 1, 1);
    auto ofmBlock = opConfig ? opConfig->OfmBlock() : Shape(1, 1, 1, 1);

    ElementAccess result = access;

    // IFM bytes transferred
    const int ifmCount = query.ifmShape[1] ? int(std::size(query.ifmShape)) : 1;
    for ( int i = 0; i < ifmCount; i++ )
    {
        result.ifmRead[i] = EstimateMemoryTransfer(true, query.ifmMemory[i], query.ifmFormat[i],
            DataTypeSizeBits(query.ifmType[i]), ifmBlock, query.ifmShape[i], access.ifmRead[i]);
    }

    // OFM bytes transferred
    result.ofmWrite = EstimateMemoryTransfer(false, query.ofmMemory, query.ofmFormat, DataTypeSizeBits(query.ofmType),
        ofmBlock, query.ofmShape, access.ofmWrite);

    // Use encoded information from query to estimate weight reads if present
    result.constRead[0] = result.constRead[1] = 0;
    if ( query.encodedWeightSize )
    {
        result.constRead[0] = access.weightsRefetch * query.encodedWeightSize;
        result.constRead[1] = access.weightsRefetch * query.encodedScaleSize;
        result.weightsRefetch = 1;
    }

    // External FMs of chained operations
    EthosU85OpGroup *opGroup = static_cast<EthosU85OpGroup *>(query.opGroup);
    assert(opGroup);
    for ( const auto &opInfo : *opGroup )
    {
        auto it = std::find_if(query.featureMapRecords.begin(), query.featureMapRecords.end(),
            [&opInfo](const FeatureMapRecord &record) { return record.opId == opInfo.opId; });
        if ( it != query.featureMapRecords.end() )
        {
            const FeatureMapRecord &fmRecord = *it;
            if ( !fmRecord.access ) continue;
            ElementAccess &extAccess = *fmRecord.access;
            if ( IsIFM(fmRecord.usage) )
            {
                int ifmIdx = GetUsageIndex(fmRecord.usage);
                assert(size_t(ifmIdx) < std::size(extAccess.ifmRead) && size_t(ifmIdx) < std::size(opInfo.ifm));
                extAccess.ifmRead[ifmIdx] = EstimateMemoryTransfer(true, fmRecord.memory, fmRecord.format,
                    DataTypeSizeBits(opInfo.ifm[ifmIdx].type), ifmBlock, fmRecord.shape, extAccess.ifmRead[ifmIdx]);
            }
            else
            {
                assert(IsOFM(fmRecord.usage) && "Unexpected usage for external FM");
                extAccess.ofmWrite = EstimateMemoryTransfer(false, fmRecord.memory, fmRecord.format,
                    DataTypeSizeBits(opInfo.ofm.type), ofmBlock, fmRecord.shape, extAccess.ofmWrite);
            }
        }
    }

    return result;
}

int64_t EthosU85Performance::WeightDecodeCycles(
    const PerformanceQuery &, const WeightStats &weights, Flags<WeightFormat> format, ArchitectureMemory *weightsMemory)
{
    int weightsPerCycle;
    if ( format % WeightFormat::Fast )
    {
        weightsPerCycle = (weights.distinctWeights < 16) ? 64 : 32;
    }
    else
    {
        assert(weights.size > 0);
        float zeroRate = std::min(float(weights.zeroCount) / weights.size, 0.9f);
        zeroRate = std::max(zeroRate, 0.5f);
        int weightsPerCore = 8 + (zeroRate - 0.5) * (32 - 8) / 0.4;
        weightsPerCycle = weightsPerCore * _arch->_cores;
    }
    int64_t decodeCycles = weights.size / weightsPerCycle;
    if ( _db && _nextId != -1 )
    {
        assert(_wdTable != -1);
        _db->AddRow(_wdTable, _nextId, {std::to_string(decodeCycles)});
        _nextId = -1;
    }

    MemChannel channel = (format % WeightFormat::Fast) ? MemChannel::FastWeight : MemChannel::Weight;
    int64_t dmaCycles = int64_t(float(weights.encodedSize) / ChannelBW(weightsMemory, channel));
    dmaCycles += weightsMemory->ReadLatency();
    return std::max(decodeCycles, dmaCycles);
}

float EthosU85Performance::ChannelBW(const ArchitectureMemory *mem, const MemChannel channel)
{
    assert(mem->PortsUsed() && mem->MaxBurstLength() && mem->Bandwidth() > 0 && mem->MaxReads() > 0);

    int burstLenWords = std::max(mem->MaxBurstLength() / 16, 1);

    float read_rb_lim;
    int maxOutstanding;
    int latency;
    if ( channel == MemChannel::None )
    {
        latency = mem->ReadLatency();
        maxOutstanding = mem->MaxReads();
        read_rb_lim = std::numeric_limits<float>::max();
    }
    else if ( channel == MemChannel::Write )
    {
        maxOutstanding = mem->MaxWrites();
        latency = mem->WriteLatency();
        read_rb_lim = std::numeric_limits<float>::max();
    }
    else
    {
        maxOutstanding = mem->MaxReads();
        latency = mem->ReadLatency();
        auto channelIdx = std::max(static_cast<int>(channel) - 1, 0);
        int channelRB = _arch->_channelRBs->at(channelIdx);
        read_rb_lim = static_cast<float>(channelRB) / burstLenWords;
    }

    float transactionUtil = std::min(read_rb_lim, static_cast<float>(maxOutstanding * mem->PortsUsed() * 0.8));
    float channelBW = std::min(mem->Bandwidth(), static_cast<float>(mem->MaxBurstLength() * transactionUtil / latency * 0.8));

    return channelBW;
}

void EthosU85Performance::InitDatabase(Database *optDB)
{
    _db = optDB;
    _mainTable = _db->AddTable("perf_debug_main");
    _wdTable = _db->AddTable("perf_debug_wd");

    std::vector<std::string> columns = {
        "mac_cycles",
        "ao_cycles",
        "cmd_cycles",
        "traversal",
    };

    std::vector<std::string> shapes = {"ifm_block", "ofm_block", "ofm_ublock"};

    for ( auto &shape : shapes )
    {
        columns.push_back(shape + "_n");
        columns.push_back(shape + "_h");
        columns.push_back(shape + "_w");
        columns.push_back(shape + "_c");
    }
    _db->AddColumns(_mainTable, std::move(columns));
    _db->AddColumns(_wdTable, {"wd_cycles"});
}

void EthosU85Performance::RecordToDB(int opId)
{
    if ( _db )
    {
        _nextId = opId;
    }
}

MemChannel EthosU85Performance::LookupChannel(OpType type, TensorUsage usage, bool fastWeights)
{
    if ( usage == TensorUsage::Weights )
    {
        if ( fastWeights )
        {
            return MemChannel::FastWeight;
        }
        else
        {
            return MemChannel::Weight;
        }
    }
    else if ( usage == TensorUsage::Scales )
    {
        return MemChannel::Scale;
    }
    else if ( IsIFM(usage) )
    {
        if ( (usage == TensorUsage::IFM1 && type == OpType::MatMul) || type == OpType::Resize || IsElementwise(type) )
        {
            return MemChannel::IFMStream;
        }
        else
        {
            return MemChannel::IFM;
        }
    }
    else if ( IsOFM(usage) )
    {
        return MemChannel::Write;
    }
    else if ( usage == TensorUsage::Scratch )
    {
        return MemChannel::IFMStream;
    }
    else
    {
        return MemChannel::None;
    }
}

int64_t EthosU85Performance::MinReadCycles(ArchitectureMemory *mem, int64_t size, TensorUsage usage, OpType type, bool fastWeights)
{
    auto channel = LookupChannel(type, usage, fastWeights);
    auto transferCycles = size / double(ChannelBW(mem, channel));
    // Add on latency since this function returns the cycle count for the transfer itself which is not necessarily the
    // same as the cycle count that the operation attributes to this transfer.
    return transferCycles + mem->ReadLatency();
}

int64_t EthosU85Performance::MinWriteCycles(ArchitectureMemory *mem, int64_t size)
{
    auto channel = MemChannel::Write;
    auto transferCycles = size / double(ChannelBW(mem, channel));
    // Add on latency since this function returns the cycle count for the transfer itself which is not necessarily the
    // same as the cycle count that the operation attributes to this transfer.
    return transferCycles + mem->WriteLatency();
}

std::unordered_map<const ArchitectureMemory *, AccessCycles>
EthosU85Performance::MeasureAccessCycles(const PerformanceQuery &query, const ElementAccess &byteAccess)
{
    enum class TransferGroup
    {
        FeatureMaps,
        Weights,
        Scales,
    };
    std::unordered_map<const ArchitectureMemory *, AccessCycles> memoryAccessCycles;
    std::unordered_map<const ArchitectureMemory *, std::unordered_map<MemChannel, std::unordered_map<TransferGroup, int64_t>>> channelTransferBytes;
    // IFM
    auto channel = LookupChannel(query.type, TensorUsage::IFM, false);
    channelTransferBytes[query.ifmMemory[0]][channel][TransferGroup::FeatureMaps] += byteAccess.ifmRead[0];
    // IFM2
    if ( !query.ifmShape[1].IsEmpty() )
    {
        channel = LookupChannel(query.type, TensorUsage::IFM1, false);
        channelTransferBytes[query.ifmMemory[1]][channel][TransferGroup::FeatureMaps] += byteAccess.ifmRead[1];
    }
    // OFM
    channelTransferBytes[query.ofmMemory][MemChannel::Write][TransferGroup::FeatureMaps] += byteAccess.ofmWrite;
    // External FMs of chained operations
    EthosU85OpGroup *opGroup = static_cast<EthosU85OpGroup *>(query.opGroup);
    assert(opGroup);
    for ( const auto &opInfo : *opGroup )
    {
        auto it = std::find_if(query.featureMapRecords.begin(), query.featureMapRecords.end(),
            [&opInfo](const FeatureMapRecord &record) { return record.opId == opInfo.opId; });
        if ( it != query.featureMapRecords.end() )
        {
            const FeatureMapRecord &fmRecord = *it;
            assert(fmRecord.access);
            if ( IsIFM(fmRecord.usage) )
            {
                int ifmIdx = GetUsageIndex(fmRecord.usage);
                assert(size_t(ifmIdx) < std::size(fmRecord.access->ifmRead));
                channel = LookupChannel(opInfo.type, MakeTensorUsage(TensorUsage::IFM, ifmIdx), false);
                channelTransferBytes[fmRecord.memory][channel][TransferGroup::FeatureMaps] += fmRecord.access->ifmRead[ifmIdx];
            }
            else
            {
                assert(IsOFM(fmRecord.usage) && "Unexpected usage for external FM");
                channelTransferBytes[fmRecord.memory][MemChannel::Write][TransferGroup::FeatureMaps] += fmRecord.access->ofmWrite;
            }
        }
    }

    if ( query.constMemory )
    {
        // Weights
        channel = LookupChannel(query.type, TensorUsage::Weights, query.weightFormat & WeightFormat::Fast);
        if ( query.weightStagingMemory )
        {
            // Concurrent DMA Weights
            auto nonPreBufferedWeightsSize = std::max(int64_t(query.encodedWeightSize) - int64_t(query.firstWeightDMASize), int64_t(0));
            channelTransferBytes[query.constMemory][MemChannel::Mem2Mem][TransferGroup::Weights] += nonPreBufferedWeightsSize;
            channelTransferBytes[query.weightStagingMemory][MemChannel::Write][TransferGroup::Weights] += nonPreBufferedWeightsSize;
            channelTransferBytes[query.weightStagingMemory][channel][TransferGroup::Weights] += byteAccess.constRead[0];
        }
        else
        {
            channelTransferBytes[query.constMemory][MemChannel::Weight][TransferGroup::Weights] += byteAccess.constRead[0];
        }
        // Scales
        channel = LookupChannel(query.type, TensorUsage::Scales, false);
        channelTransferBytes[query.constMemory][channel][TransferGroup::Scales] += byteAccess.constRead[1];
    }
    // DMA
    if ( query.tmpMemory )
    {
        channel = LookupChannel(query.type, TensorUsage::Scratch, false);
        channelTransferBytes[query.tmpMemory][channel][TransferGroup::FeatureMaps] += byteAccess.tmpRead;
        channelTransferBytes[query.tmpMemory][MemChannel::Write][TransferGroup::FeatureMaps] += byteAccess.tmpWrite;
    }

    // Total access cycles for any grouping:
    // Group access cycles = max(group read + group write/mem bw, max group channel cycles)
    // Where group channel cycles is the channel transfer cycles attributable to that group.
    for ( auto &[mem, channels] : channelTransferBytes )
    {
        AccessCycles accessCycles;

        int64_t maxChannelCycles = 0;
        std::unordered_map<TransferGroup, int64_t> maxGroupChannelCycles;
        int64_t totalBytes = 0;
        std::unordered_map<TransferGroup, int64_t> totalGroupBytes;

        for ( auto &[memChannel, groups] : channels )
        {
            int64_t channelCycles = 0;
            for ( auto &[group, bytes] : groups )
            {
                int64_t cycles = bytes / ChannelBW(mem, memChannel);
                if ( cycles > maxGroupChannelCycles[group] )
                {
                    maxGroupChannelCycles[group] = cycles;
                }
                totalGroupBytes[group] += bytes;
                totalBytes += bytes;
                channelCycles += cycles;
            }
            maxChannelCycles = std::max(maxChannelCycles, channelCycles);
        }

        accessCycles.fmAccessCycles =
            totalGroupBytes.count(TransferGroup::FeatureMaps) ?
                std::max(int64_t(totalGroupBytes[TransferGroup::FeatureMaps] / mem->Bandwidth()), maxGroupChannelCycles[TransferGroup::FeatureMaps]) :
                0;
        accessCycles.weightsAccessCycles =
            totalGroupBytes.count(TransferGroup::Weights) ?
                std::max(int64_t(totalGroupBytes[TransferGroup::Weights] / mem->Bandwidth()), maxGroupChannelCycles[TransferGroup::Weights]) :
                0;
        accessCycles.scalesAccessCycles =
            totalGroupBytes.count(TransferGroup::Scales) ?
                std::max(int64_t(totalGroupBytes[TransferGroup::Scales] / mem->Bandwidth()), maxGroupChannelCycles[TransferGroup::Scales]) :
                0;
        accessCycles.totalAccessCycles = std::max(int64_t(totalBytes / mem->Bandwidth()), maxChannelCycles);
        memoryAccessCycles[mem] = accessCycles;
    }
    return memoryAccessCycles;
}

}  // namespace regor
