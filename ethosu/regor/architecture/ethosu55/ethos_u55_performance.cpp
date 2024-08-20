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

#include "ethos_u55_performance.hpp"

#include "common/common.hpp"

#include "architecture/architecture.hpp"
#include "ethos_u55.hpp"

namespace regor
{

static const Point2i s_SubkernelLimits[] = {
    {0, 0},  // No kernel
    {8, 8},  // Convolution
    {8, 8},  // Depthwise
    {1, 1},  // VectorProduct
    {8, 8},  // Pooling
    {8, 8},  // ReduceSum
    {1, 1},  // Elementwise
};

static constexpr bool OpUsesMacs(EthosU55NpuOp npuOp)
{
    return (npuOp != EthosU55NpuOp::Elementwise && npuOp != EthosU55NpuOp::None);
}

EthosU55Performance::EthosU55Performance(ArchEthosU55 *arch, const EthosU55PerfInfo *perfInfo) : _arch(arch)
{
    _perfInfo = perfInfo;
}

CycleCost EthosU55Performance::MeasureCycleCostForSparsity(const PerformanceQuery &query, const std::vector<FusionQuery> &fused)
{
    return MeasureCycleCost(query, fused);
}

CycleCost EthosU55Performance::MeasureCycleCost(const PerformanceQuery &query, const std::vector<FusionQuery> &fused)
{
    CycleCost cycles;
    auto npuOp = _arch->GetHWOp(query.type);

    // Convolution/Vector product cycle calculation
    if ( OpUsesMacs(npuOp) )
    {
        if ( (npuOp == EthosU55NpuOp::Depthwise) || (npuOp == EthosU55NpuOp::Pooling) )
        {
            cycles.macs = int64_t(query.kernel->ElementsWH()) * query.ofmShape.Elements() * 1;
        }
        else
        {
            cycles.macs = int64_t(query.kernel->ElementsWH()) * query.ofmShape.Elements() * query.ifmShape[0].Depth();
        }

        cycles.opCycles = EstimateConvCycles(query, fused);
    }
    // Elementwise cycle calculation
    else if ( npuOp == EthosU55NpuOp::Elementwise )
    {
        auto ofmShape =
            (query.ofmFormat == TensorFormat::NHCWB16) ? Shape::RoundAway(query.ofmShape, Shape(1, 1, 1, 16)) : query.ofmShape;
        cycles.opCycles = int64_t(EstimateOutputCyclesPerElement(query, fused) * float(ofmShape.Elements()));
    }
    else
    {
        assert(false && "Unknown operator cycle costing");
    }

    return cycles;
}

int64_t EthosU55Performance::MemToMemCycles(const ArchitectureMemory *dest, const ArchitectureMemory *source, int sizeBytes)
{
    int64_t fromCycles = int64_t(float(sizeBytes) / source->Bandwidth());
    fromCycles += source->ReadLatency();
    int64_t toCycles = int64_t(float(sizeBytes) / dest->Bandwidth());
    toCycles += source->WriteLatency();
    return std::max(fromCycles, toCycles);
}

int64_t EthosU55Performance::EstimateConvCycles(const PerformanceQuery &query, const std::vector<FusionQuery> &fused)
{
    EthosU55OpConfig *opConfig = static_cast<EthosU55OpConfig *>(query.config);
    auto npuOp = _arch->GetHWOp(query.type);
    assert(npuOp != EthosU55NpuOp::None);

    Shape ifmBlock = Shape::Min(query.ifmShape[0], opConfig->IfmBlock());
    Shape ofmBlock = Shape::Min(query.ofmShape, opConfig->OfmBlock());
    Shape ofmUBlock = _arch->OfmUBlock();

    // HW Optimisation check
    if ( (ofmUBlock.Height() == 2) && (npuOp == EthosU55NpuOp::Convolution || npuOp == EthosU55NpuOp::VectorProduct) &&
         (query.ofmShape.Height() == 1) && (query.ofmShape.Width() % 2 == 0) &&  // Optimisation only applies for even
                                                                                 // width tensors
         (query.kernel->Size().y == 1) )
    {
        ofmUBlock = Shape(1, 1, 4, ofmUBlock.Depth());
        ofmBlock = ofmBlock.WithHeight(1);
    }

    int ifmBits = DataTypeSizeBits(query.ifmType[0]);
    Shape numUBlocks = Shape::DivRoundUp(ofmBlock, ofmUBlock);
    bool use40BitAcc = opConfig->Acc() == EthosU55SHRamElements::SHRAM_Acc40;

    int64_t cyclesDpuBlk = 0;
    int cyclesWb = 32 * ofmUBlock.Depth() / 8;

    int subKernelWidth = s_SubkernelLimits[int(npuOp)].x;
    int subKernelHeight = s_SubkernelLimits[int(npuOp)].y;
    const Point2i kernelSize = query.kernel->Size();
    bool isConvolutionMxN = (npuOp == EthosU55NpuOp::Convolution);

    for ( int x = 0; x < kernelSize.x; x += subKernelWidth )
    {
        for ( int y = 0; y < kernelSize.y; y += subKernelHeight )
        {
            int subKernelElements = std::min(kernelSize.y - y, subKernelHeight);
            subKernelElements *= std::min(kernelSize.x - x, subKernelWidth);

            // Calculate processing cycles
            int numKernelSteps = 0;
            int cycles = 0;
            if ( npuOp == EthosU55NpuOp::Pooling )
            {
                numKernelSteps = 1;
                cycles = std::max(4, subKernelElements) * numUBlocks.Elements();
                if ( !_arch->IsU55_32() )
                {
                    cycles = cycles * (ifmBits / 2);
                }
            }
            else if ( npuOp == EthosU55NpuOp::Depthwise )
            {
                numKernelSteps = DivRoundUp(subKernelElements, 4);
                cycles = 4 * numUBlocks.ElementsWH() * (ifmBits / 8);
                cycles = std::max(cyclesWb, cycles) * numKernelSteps * numUBlocks.Depth();
            }
            else if ( (isConvolutionMxN && opConfig->Traversal() != EthosUTraversal::PartKernel) ||
                      npuOp == EthosU55NpuOp::VectorProduct || npuOp == EthosU55NpuOp::ReduceSum )
            {
                numKernelSteps = subKernelElements;
                cycles = std::max(cyclesWb, 4 * numUBlocks.ElementsWH()) * numKernelSteps * numUBlocks.Depth();
            }
            else
            {
                assert(opConfig->Traversal() == EthosUTraversal::PartKernel);
                int divider = (ifmBits == 16) ? 2 : 4;
                numKernelSteps = DivRoundUp(subKernelElements, divider);
                cycles = std::max(cyclesWb, 4 * numUBlocks.ElementsWH()) * numKernelSteps * numUBlocks.Depth() *
                         DivRoundUp(ifmBlock.Depth(), 8);
            }

            // Calculate delay
            int delayCycles = 0;
            if ( _arch->IsU55_32() )
            {
                int delay = use40BitAcc ? 7 : 3;
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

                if ( (numUBlocks.Width() == 1 || numUBlocks.Height() == 1) && (numUBlocks.Depth() > 1) && use40BitAcc )
                {
                    delayCycles += delay * numUBlocks.Depth();
                }
            }
            else
            {
                int delay = (use40BitAcc && (_arch->_macs <= 128)) ? 3 : 2;

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
            }

            if ( isConvolutionMxN && opConfig->Traversal() == EthosUTraversal::PartKernel )
            {
                delayCycles *= DivRoundUp(ifmBlock.Depth(), 8);
            }

            cyclesDpuBlk += cycles;
            cyclesDpuBlk += delayCycles;
        }
    }

    if ( npuOp == EthosU55NpuOp::Convolution || npuOp == EthosU55NpuOp::VectorProduct || npuOp == EthosU55NpuOp::ReduceSum )
    {
        cyclesDpuBlk *= DivRoundUp(query.ifmShape[0].Depth(), ifmBlock.Depth());
    }

    cyclesDpuBlk /= _arch->_cores;

    // Estimate output cycles
    int numOfmBlks = Shape::DivRoundUp(query.ofmShape, ofmBlock).Elements();
    int64_t cyclesOutputBlk = int64_t(EstimateOutputCyclesPerElement(query, fused) * float(ofmBlock.Elements()));

    // Scale and bias tensor
    if ( query.constShape.Size() > 0 && query.constShape.Depth() > 0 )
    {
        int cyclesBiasBlk = (10 * ofmBlock.Depth() * query.constMemory->ReadLatency() / 256);
        cyclesOutputBlk = std::max(cyclesOutputBlk, int64_t(cyclesBiasBlk));
    }

    int64_t cycles_cmd = EstimateMinimumMemoryCycles(query);
    cycles_cmd = (cycles_cmd + cyclesOutputBlk + cyclesDpuBlk) / 4;  // Per DPU

    cyclesDpuBlk = std::max(cyclesDpuBlk, cycles_cmd);
    cyclesOutputBlk = std::max(cyclesOutputBlk, cycles_cmd);

    int64_t totalCycles = 0;
    if ( cyclesDpuBlk > cyclesOutputBlk )
    {
        totalCycles = cyclesDpuBlk * numOfmBlks + cyclesOutputBlk;
    }
    else
    {
        totalCycles = cyclesOutputBlk * numOfmBlks + cyclesDpuBlk;
    }

    return totalCycles;
}

static int EstimateMemoryTransfer(int cores, bool isRead, ArchitectureMemory *memory, TensorFormat format,
    int elementBits, Shape block, Shape shape, int toTransfer)
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
    return (toTransfer * memory->MaxBurstLength()) / burstLen;
}


int64_t EthosU55Performance::EstimateMinimumMemoryCycles(const PerformanceQuery &query)
{
    EthosU55OpConfig *opConfig = static_cast<EthosU55OpConfig *>(query.config);

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


float EthosU55Performance::EstimateOutputCyclesPerElement(const PerformanceQuery &query, const std::vector<FusionQuery> &fused)
{
    EthosU55OpConfig *opConfig = static_cast<EthosU55OpConfig *>(query.config);
    auto npuOp = _arch->GetHWOp(query.type);
    assert(npuOp != EthosU55NpuOp::None);
    int ifmBits = DataTypeSizeBits(query.ifmType[0]);
    int ofmBits = DataTypeSizeBits(query.ofmType);
    int outputPerfIndex = 0;

    if ( (npuOp == EthosU55NpuOp::Elementwise) && (ifmBits == 32) )
    {
        // Unary op else Binary op
        outputPerfIndex = query.ifmShape[1].Elements() > 0 ? 1 : 0;
    }
    else if ( query.type == OpType::Mul && ofmBits == 32 )
    {
        outputPerfIndex = 2;
    }
    else if ( (query.type == OpType::Mul) || ((npuOp != EthosU55NpuOp::Elementwise) && opConfig->Acc() == EthosU55SHRamElements::SHRAM_Acc40) )
    {
        outputPerfIndex = 3;
    }
    else if ( query.type == OpType::Add || query.type == OpType::Sub )
    {
        if ( false )
        {
            // Simple Add/Sub
            outputPerfIndex = 4;
        }
        else
        {
            // Advanced Add/Sub TODO: Add as perf selection as operator variant
            outputPerfIndex = 5;
        }
    }
    else if ( query.type == OpType::MaxPool )
    {
        outputPerfIndex = 6;
    }
    else
    {
        outputPerfIndex = 7;
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

    if ( npuOp == EthosU55NpuOp::Elementwise )
    {
        int numElemsBlk = opConfig->OfmBlock().Elements();
        assert(numElemsBlk > 0);
        float cycleCmd = (float(EstimateMinimumMemoryCycles(query)) / float(numElemsBlk) + cyclesPerElement) / 4.0f;  // per DPU
        cyclesPerElement = std::max(cyclesPerElement, cycleCmd);
    }

    return cyclesPerElement;
}

ElementAccess EthosU55Performance::MeasureElementAccess(const PerformanceQuery &query)
{
    ElementAccess access;
    EthosU55OpConfig *opConfig = static_cast<EthosU55OpConfig *>(query.config);
    auto npuOp = _arch->GetHWOp(query.type);
    assert(npuOp != EthosU55NpuOp::None);

    Shape ifmBlock = Shape::Min(query.ifmShape[0], opConfig->IfmBlock());
    Shape ofmBlock = Shape::Min(query.ofmShape, opConfig->OfmBlock());

    Shape ifmRounding = _arch->GetStorageRounding(query.ifmFormat[0]);
    Shape ofmRounding = _arch->GetStorageRounding(query.ofmFormat);

    // Number of ofm blocks in the overall output shape
    Shape ofmBlocks = Shape::DivRoundUp(query.ofmShape, ofmBlock);

    int ofmBlockDepth = ofmBlock.Depth();
    if ( npuOp == EthosU55NpuOp::Depthwise || npuOp == EthosU55NpuOp::Pooling )
    {
        ofmBlocks = ofmBlocks.WithDepth(1);
        ofmBlockDepth = query.ifmShape[0].Depth();
    }

    // Convolution & pooling
    if ( OpUsesMacs(npuOp) )
    {
        // Number of sub kernels
        int subKernelWidth = s_SubkernelLimits[int(npuOp)].x;
        int subKernelHeight = s_SubkernelLimits[int(npuOp)].y;
        int subkernels = DivRoundUp(query.kernel->Size().x, subKernelWidth) * DivRoundUp(query.kernel->Size().y, subKernelHeight);

        int ifmFetch =
            (Shape::RoundAway(ifmBlock, ifmRounding).ElementsWH() * Shape::RoundAway(query.ifmShape[0], ifmRounding).Depth());

        int kernelRead = query.kernel->Size().AreaXY();
        if ( (npuOp != EthosU55NpuOp::Depthwise) && (npuOp != EthosU55NpuOp::Pooling) )
        {
            kernelRead *= query.ifmShape[0].Depth();
        }

        int ofmBlockCount = ofmBlocks.Elements();

        access.ifmRead[0] = ifmFetch * subkernels * ofmBlockCount;

        if ( (npuOp != EthosU55NpuOp::Pooling) && (npuOp != EthosU55NpuOp::ReduceSum) )
        {
            int weightFetch = kernelRead * ofmBlockDepth * ofmBlockCount;
            access.constRead[0] = weightFetch;
            access.constRead[1] = query.ofmShape.Depth();  // Scales & biases
            access.weightsRefetch = ofmBlocks.ElementsWH();
        }
    }
    else if ( npuOp == EthosU55NpuOp::Elementwise )
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
    else
    {
        assert(false);
    }

    access.ofmWrite = Shape::RoundAway(query.ofmShape, ofmRounding).Elements();

    return access;
}


ElementAccess EthosU55Performance::ElementTransferToBytes(const PerformanceQuery &query, const ElementAccess &access)
{
    EthosU55OpConfig *opConfig = static_cast<EthosU55OpConfig *>(query.config);

    ElementAccess result = access;

    // IFM bytes transferred
    int ifmBits = DataTypeSizeBits(query.ifmType[0]);  // All inputs expect same bit width
    const int ifmCount = query.ifmShape[1].Elements() > 0 ? int(std::size(query.ifmShape)) : 1;
    for ( int i = 0; i < ifmCount; i++ )
    {
        result.ifmRead[i] = EstimateMemoryTransfer(_arch->_cores, true, query.ifmMemory[i], query.ifmFormat[i], ifmBits,
            opConfig->IfmBlock(), query.ifmShape[i], access.ifmRead[i]);
    }

    // OFM bytes transferred
    result.ofmWrite = EstimateMemoryTransfer(_arch->_cores, false, query.ofmMemory, query.ofmFormat,
        DataTypeSizeBits(query.ofmType), opConfig->OfmBlock(), query.ofmShape, access.ofmWrite);

    // These requires compression ratio information
    result.constRead[0] = 0;
    result.constRead[1] = 0;

    return result;
}

int64_t EthosU55Performance::WeightDecodeCycles(
    const PerformanceQuery &, const WeightStats &weights, Flags<WeightFormat>, ArchitectureMemory *weightsMemory)
{
    int64_t dmaCycles = int64_t(float(weights.encodedSize) / weightsMemory->Bandwidth());
    dmaCycles += weightsMemory->ReadLatency();
    return dmaCycles;
}

float EthosU55Performance::ChannelBW(const ArchitectureMemory *mem, const MemChannel channel)
{
    UNUSED(channel);
    return mem->Bandwidth();
}
}  // namespace regor
