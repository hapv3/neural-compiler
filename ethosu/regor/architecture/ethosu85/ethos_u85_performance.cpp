//
// SPDX-FileCopyrightText: Copyright 2021-2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
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
    return (npuOp != EthosU85NpuOp::Elementwise && npuOp != EthosU85NpuOp::Resize && npuOp != EthosU85NpuOp::Dma && npuOp != EthosU85NpuOp::None);
}

EthosU85Performance::EthosU85Performance(ArchEthosU85 *arch, const EthosU85PerfInfo *perfInfo) : _arch(arch)
{
    _perfInfo = perfInfo;
}

CycleCost EthosU85Performance::MeasureCycleCostForSparsity(const PerformanceQuery &query, const std::vector<FusionQuery> &)
{
    // Temporary until we have a better performance estimate for U85
    CycleCost cycles;
    EthosU85OpConfig *opConfig = static_cast<EthosU85OpConfig *>(query.config);
    cycles.opCycles = opConfig->OptimalDepthGranule();  // Maybe can find a better metric? MLBEDSW-9227
    return cycles;
}

CycleCost EthosU85Performance::MeasureCycleCost(const PerformanceQuery &query, const std::vector<FusionQuery> &fused)
{
    CycleCost cycles;
    EthosU85Cycles cycleComponents;

    auto npuOp = _arch->GetHWOp(query.type);
    const bool sparse = query.weightFormat & WeightFormat::Sparse2_4;
    const bool recordToDb = _db && _nextId != -1;
    // Convolution/Vector product cycle calculation
    if ( OpUsesMacs(npuOp) )
    {
        if ( npuOp == EthosU85NpuOp::Depthwise || npuOp == EthosU85NpuOp::Pooling ||
             npuOp == EthosU85NpuOp::ReduceMinMax || npuOp == EthosU85NpuOp::ArgMax )
        {
            cycles.macs = int64_t(query.kernel->ElementsWH()) * query.ofmShape.Elements();
        }
        else
        {
            cycles.macs = int64_t(query.kernel->ElementsWH()) * query.ofmShape.Elements() * query.ifmShape[0].Depth();
        }
        cycles.macs /= sparse ? 2 : 1;

        cycleComponents = EstimateConvCycles(query, fused);
        cycles.opCycles = cycleComponents.cycles;
    }
    // Elementwise cycle calculation
    else if ( npuOp == EthosU85NpuOp::Elementwise )
    {
        auto [totCCPerElem, aoCCPerElem, cmdCCPerElem] = EstimateOutputCyclesPerElement(query, fused);
        auto ofmShape =
            (query.ofmFormat == TensorFormat::NHCWB16) ? Shape::RoundAway(query.ofmShape, Shape(1, 1, 1, 16)) : query.ofmShape;
        cycles.opCycles = int64_t(totCCPerElem * float(ofmShape.Elements()));
        if ( recordToDb )
        {
            cycleComponents.aoCycles = int64_t(aoCCPerElem * float(ofmShape.Elements()));
            cycleComponents.cmdCycles = int64_t(cmdCCPerElem * float(ofmShape.Elements()));
        }
    }
    // Resize cycle calculation
    else if ( npuOp == EthosU85NpuOp::Resize )
    {
        // TODO: Implement for Resize
        cycles.opCycles = 0;
    }
    // DMA cycle calculation
    else if ( npuOp == EthosU85NpuOp::Dma )
    {
        // TODO: below is incorrect (MLBEDSW-8400)

        auto ofmShape =
            (query.ofmFormat == TensorFormat::NHCWB16) ? Shape::RoundAway(query.ofmShape, Shape(1, 1, 1, 16)) : query.ofmShape;
        cycles.opCycles = 0;
    }
    else
    {
        assert(false && "Unknown operator cycle costing");
    }

    if ( recordToDb )
    {
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
    int64_t toCycles = int64_t(float(sizeBytes) / ChannelBW(dest, MemChannel::OFM));
    toCycles += source->WriteLatency();
    return std::max(fromCycles, toCycles);
}

EthosU85Cycles EthosU85Performance::EstimateConvCycles(const PerformanceQuery &query, const std::vector<FusionQuery> &fused)
{
    EthosU85OpConfig *opConfig = static_cast<EthosU85OpConfig *>(query.config);
    auto npuOp = _arch->GetHWOp(query.type);
    assert(npuOp != EthosU85NpuOp::None);

    Shape ifmBlock = Shape::Min(query.ifmShape[0], opConfig->IfmBlock());
    Shape ofmBlock = Shape::Min(query.ofmShape, opConfig->OfmBlock());
    Shape ofmUBlock = opConfig->OfmUBlock();

    // HW Optimisation check
    if ( (ofmUBlock.Height() == 2) && (npuOp == EthosU85NpuOp::Convolution || npuOp == EthosU85NpuOp::VectorProduct) &&
         (query.ofmShape.Height() == 1) && (query.ofmShape.Width() % 2 == 0) &&  // Optimisation only applies for even
                                                                                 // width tensors
         (query.kernel->Size().y == 1) )
    {
        ofmUBlock = Shape(1, 1, 4, ofmUBlock.Depth());
        ofmBlock = ofmBlock.WithHeight(1);
    }

    int ifmBits = DataTypeSizeBits(query.ifmType[0]);
    Shape numUBlocks = Shape::DivRoundUp(ofmBlock, ofmUBlock);
    bool use48BitAcc = opConfig->Acc() == EthosU85Accumulator::Acc48;

    int64_t cyclesDpuBlk = 0;
    int cyclesWb = 32 * ofmUBlock.Depth() / 8;

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

    // Estimate output cycles
    float numOfmBlks = 1;
    for ( int i = 0; i < std::min(query.ofmShape.Size(), ofmBlock.Size()); i++ )
    {
        numOfmBlks *= std::max(static_cast<float>(query.ofmShape[i]) / ofmBlock[i], 1.0f);
    }
    auto [totCCPerElem, aoCCPerElem, cmdCCPerElem] = EstimateOutputCyclesPerElement(query, fused);
    auto aoCycles = int64_t(aoCCPerElem * float(ofmBlock.Elements()));
    auto cmdCycles = int64_t(cmdCCPerElem * float(ofmBlock.Elements()));
    auto cyclesOutputBlk = int64_t(totCCPerElem * float(ofmBlock.Elements()));

    // Scale and bias tensor
    if ( query.constShape.Size() > 0 && query.constShape.Depth() > 0 )
    {
        int cyclesBiasBlk = (10 * ofmBlock.Depth() * query.constMemory->ReadLatency() / 256);
        cyclesOutputBlk = std::max(cyclesOutputBlk, int64_t(cyclesBiasBlk));
    }

    int64_t cmdCycles2 = EstimateMinimumMemoryCycles(query);
    cmdCycles2 = (cmdCycles2 + cyclesOutputBlk + cyclesDpuBlk) / 4;  // Per DPU

    int64_t cyclesAO = aoCycles * numOfmBlks + cyclesDpuBlk;
    int64_t cyclesDpu = cyclesDpuBlk * numOfmBlks + cyclesOutputBlk;

    cmdCycles = std::max(cmdCycles, cmdCycles2);
    cyclesDpuBlk = std::max(cyclesDpuBlk, cmdCycles2);
    cyclesOutputBlk = std::max(cyclesOutputBlk, cmdCycles2);

    int64_t totalCycles = 0;
    if ( cyclesDpuBlk > cyclesOutputBlk )
    {
        totalCycles = int64_t(cyclesDpuBlk * numOfmBlks) + cyclesOutputBlk;
    }
    else
    {
        totalCycles = int64_t(cyclesOutputBlk * numOfmBlks) + cyclesDpuBlk;
        cmdCycles = cmdCycles * numOfmBlks + cyclesDpuBlk;
    }

    return {totalCycles, cyclesDpu, cyclesAO, cmdCycles};
}

static int64_t EstimateMemoryTransfer(int cores, bool isRead, ArchitectureMemory *memory, TensorFormat format,
    int elementBits, const Shape &block, const Shape &shape, int toTransfer)
{
    int burstLen = 8;

    if ( format == TensorFormat::NHCWB16 )
    {
        int zStride = (shape.Width() * elementBits * 16) / 8;
        if ( zStride == block.Depth() )
        {
            burstLen = elementBits * block.Depth() * block.Width();
        }
        else if ( isRead )
        {
            burstLen = 16 * elementBits * block.Width();
        }
        else
        {
            burstLen = 16 * elementBits * block.Width() * cores;
        }
    }
    else if ( format == TensorFormat::NHWC )
    {
        int xStride = (shape.Depth() * elementBits) / 8;
        if ( isRead )
        {
            if ( xStride == block.Depth() )
            {
                burstLen = elementBits * block.Depth() * block.Width();
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
                burstLen = elementBits * block.Depth() * block.Width();
            }
            else
            {
                burstLen = std::min(std::min(64 * 8, 16 * elementBits * cores), block.Depth() * elementBits);
            }
        }
    }

    burstLen = std::min(memory->MaxBurstLength(), burstLen / 8);
    assert(burstLen > 0 && "Burst length cannot be zero");
    int64_t memTransfer = (int64_t(toTransfer) * memory->MaxBurstLength()) / burstLen;
    return memTransfer;
}


int64_t EthosU85Performance::EstimateMinimumMemoryCycles(const PerformanceQuery &query)
{
    EthosU85OpConfig *opConfig = static_cast<EthosU85OpConfig *>(query.config);

    int ifmBits = DataTypeSizeBits(query.ifmType[0]);  // All inputs expect same bit width
    const int ifmCount = query.ifmShape[1].Elements() > 0 ? int(std::size(query.ifmShape)) : 1;
    int64_t cyclesIfm = 0;
    for ( int i = 0; i < ifmCount; i++ )
    {
        // Input block HW transfer (only for elements present)
        int ifmBytes = Shape::Min(query.ifmShape[i], opConfig->IfmBlock()).Elements() * ifmBits / 8;
        int64_t cyclesIfmBlk = query.ifmMemory[i]->ReadLatency();
        int64_t tx = EstimateMemoryTransfer(_arch->_cores, true, query.ifmMemory[i], query.ifmFormat[i], ifmBits,
            opConfig->IfmBlock(), query.ifmShape[i], ifmBytes);
        cyclesIfmBlk += int64_t(float(tx) / query.ifmMemory[i]->Bandwidth());

        cyclesIfm = std::max(cyclesIfm, cyclesIfmBlk);
    }

    // Output block HW transfer (only for elements present)
    int ofmBits = DataTypeSizeBits(query.ofmType);
    int ofmBytes = Shape::Min(query.ofmShape, opConfig->OfmBlock()).Elements() * ofmBits / 8;
    int64_t cyclesOfm = query.ofmMemory->WriteLatency();
    int64_t tx = EstimateMemoryTransfer(_arch->_cores, false, query.ofmMemory, query.ofmFormat, ofmBits,
        opConfig->OfmBlock(), query.ofmShape, ofmBytes);
    cyclesOfm += int64_t(float(tx) / query.ofmMemory->Bandwidth());

    return cyclesIfm + cyclesOfm;
}


EthosU85ElementCycles EthosU85Performance::EstimateOutputCyclesPerElement(const PerformanceQuery &query, const std::vector<FusionQuery> &fused)
{
    EthosU85OpConfig *opConfig = static_cast<EthosU85OpConfig *>(query.config);
    auto npuOp = _arch->GetHWOp(query.type);
    assert(npuOp != EthosU85NpuOp::None);
    int ifmBits = DataTypeSizeBits(query.ifmType[0]);
    int ofmBits = DataTypeSizeBits(query.ofmType);
    int outputPerfIndex = 0;

    if ( ifmBits == 32 )
    {
        outputPerfIndex = 0;
    }
    else if ( query.type == OpType::Mul && ofmBits == 32 )
    {
        outputPerfIndex = 1;
    }

    int activationPerfIndex = 0;
    assert(fused.size() <= 1 && "multiple op performance not available");
    for ( const FusionQuery &fusedOp : fused )
    {
        if ( fusedOp.type == OpType::Sigmoid || fusedOp.type == OpType::Tanh || fusedOp.type == OpType::LookupTable )
        {
            activationPerfIndex = 0;
        }
        else if ( fusedOp.type == OpType::Relu || fusedOp.type == OpType::Relu6 || fusedOp.type == OpType::ReluN1To1 )
        {
            activationPerfIndex = 1;
        }
        else
        {
            activationPerfIndex = 2;
        }
    }

    float cyclesPerElement = std::max(_perfInfo->outputCycles[outputPerfIndex], _perfInfo->activationCycles[activationPerfIndex]);
    float cycleCmd = 0;
    float aoCyclesPerElement = cyclesPerElement;
    if ( npuOp == EthosU85NpuOp::Elementwise )
    {
        int numElemsBlk = opConfig->OfmBlock().Elements();
        assert(numElemsBlk > 0);
        cycleCmd = (float(EstimateMinimumMemoryCycles(query)) / float(numElemsBlk) + cyclesPerElement) / 4.0f;  // per
                                                                                                                // DPU
        cyclesPerElement = std::max(cyclesPerElement, cycleCmd);
    }

    return {cyclesPerElement, aoCyclesPerElement, cycleCmd};
}

ElementAccess EthosU85Performance::MeasureElementAccess(const PerformanceQuery &query)
{
    ElementAccess access;
    EthosU85OpConfig *opConfig = static_cast<EthosU85OpConfig *>(query.config);
    auto npuOp = _arch->GetHWOp(query.type);
    assert(npuOp != EthosU85NpuOp::None);

    Shape ifmRounding = _arch->GetStorageRounding(query.ifmFormat[0]);
    Shape ofmRounding = _arch->GetStorageRounding(query.ofmFormat);

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

        int kernelRead = query.kernel->Size().AreaXY();
        if ( npuOp != EthosU85NpuOp::Depthwise && npuOp != EthosU85NpuOp::Pooling &&
             npuOp != EthosU85NpuOp::ReduceMinMax && npuOp != EthosU85NpuOp::ArgMax )
        {
            kernelRead *= query.ifmShape[0].Depth();
        }

        int ofmBlockCount = ofmBlocks.Elements();

        access.ifmRead[0] = ifmFetch * subkernels * ofmBlockCount;

        if ( (npuOp != EthosU85NpuOp::Pooling) && (npuOp != EthosU85NpuOp::ReduceSum) )
        {
            int weightFetch = kernelRead * ofmBlockDepth * ofmBlockCount;
            access.constRead[0] = weightFetch;
            access.constRead[1] = query.ofmShape.Depth();  // Scales & biases
            access.weightsRefetch = ofmBlocks.ElementsWH();
        }
    }
    else if ( npuOp == EthosU85NpuOp::Elementwise )
    {
        // IFM1 is scalar
        if ( query.ifmShape[0].Elements() == 1 )
        {
            if ( DataTypeSizeBits(query.ifmType[0]) > 8 )  // IFM1 is a non 8-bit scalar
            {
                access.ifmRead[0] = Shape::RoundAway(query.ifmShape[0], ifmRounding).Elements();
            }
            else if ( query.ifmShape[1].Elements() > 0 )
            {
                access.ifmRead[1] = Shape::RoundAway(query.ofmShape, ifmRounding).Elements();
            }
        }
        else  // IFM1 is not scalar
        {
            access.ifmRead[0] = Shape::RoundAway(query.ofmShape, ifmRounding).Elements();
            if ( query.ifmShape[1].Elements() > 0 )
            {
                // IFM2 is not scalar
                if ( query.ifmShape[1].Elements() > 1 )
                {
                    access.ifmRead[1] = access.ifmRead[0];
                }
                else if ( DataTypeSizeBits(query.ifmType[1]) > 8 )  // IFM2 is a non 8-bit scalar
                {
                    access.ifmRead[1] = Shape::RoundAway(query.ifmShape[1], ifmRounding).Elements();
                }
            }
        }
    }
    else if ( npuOp == EthosU85NpuOp::Resize )
    {
        // TODO: Implement for Resize
        access.ifmRead[0] = Shape::RoundAway(query.ifmShape[0], ifmRounding).Elements();
        access.ofmWrite = Shape::RoundAway(query.ofmShape[0], ofmRounding).Elements();
    }
    else if ( npuOp == EthosU85NpuOp::Dma )
    {
        if ( query.type == OpType::Gather )
        {
            // One element from IFM0 (positions) is read per element in IFM1 (index)
            access.ifmRead[0] = Shape::RoundAway(query.ifmShape[1], ifmRounding).Elements();

            // Complete IFM1 (index) is read
            access.ifmRead[1] = Shape::RoundAway(query.ifmShape[1], ifmRounding).Elements();

            // Complete OFM is written
            access.ofmWrite = Shape::RoundAway(query.ofmShape[0], ofmRounding).Elements();
        }
        else if ( query.type == OpType::Tile )
        {
            // IFM0 is read multiple times to cover all elements in ofmShape
            access.ifmRead[0] = Shape::RoundAway(query.ofmShape[0], ofmRounding).Elements();
            // Complete OFM is written
            access.ofmWrite = Shape::RoundAway(query.ofmShape[0], ofmRounding).Elements();
        }
        else
        {
            LOG_WARN("Missing element access estimation for DMA op {}\n", OpTypeToString(query.type).c_str());
        }
    }
    else
    {
        assert(false);
    }

    access.ofmWrite = Shape::RoundAway(query.ofmShape, ofmRounding).Elements();

    return access;
}


ElementAccess EthosU85Performance::ElementTransferToBytes(const PerformanceQuery &query, const ElementAccess &access)
{
    EthosU85OpConfig *opConfig = static_cast<EthosU85OpConfig *>(query.config);
    auto ifmBlock = opConfig ? opConfig->IfmBlock() : Shape(1, 1, 1, 1);
    auto ofmBlock = opConfig ? opConfig->OfmBlock() : Shape(1, 1, 1, 1);

    ElementAccess result = access;

    // IFM bytes transferred
    const int ifmCount = query.ifmShape[1].Elements() > 0 ? int(std::size(query.ifmShape)) : 1;
    for ( int i = 0; i < ifmCount; i++ )
    {
        result.ifmRead[i] = EstimateMemoryTransfer(_arch->_cores, true, query.ifmMemory[i], query.ifmFormat[i],
            DataTypeSizeBits(query.ifmType[i]), ifmBlock, query.ifmShape[i], access.ifmRead[i]);
    }

    // OFM bytes transferred
    result.ofmWrite = EstimateMemoryTransfer(_arch->_cores, false, query.ofmMemory, query.ofmFormat,
        DataTypeSizeBits(query.ofmType), ofmBlock, query.ofmShape, access.ofmWrite);

    // These requires compression ratio information
    result.constRead[0] = 0;
    result.constRead[1] = 0;

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
    int burstLenWords = std::max(mem->MaxBurstLength() / 16, 1);

    float read_rb_lim;
    int maxOutstanding;
    int latency;

    if ( channel == MemChannel::OFM )
    {
        maxOutstanding = mem->MaxWrites();
        latency = mem->WriteLatency();
        read_rb_lim = std::numeric_limits<float>::max();
    }
    else
    {
        maxOutstanding = mem->MaxReads();
        latency = mem->ReadLatency();
        int channelRB = _arch->_channelRBs->at(static_cast<int>(channel));
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

}  // namespace regor
