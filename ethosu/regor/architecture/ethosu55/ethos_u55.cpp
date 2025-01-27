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

#include "ethos_u55.hpp"

#include "common/common.hpp"
#include "common/logging.hpp"

#include "common/bit_flags.hpp"
#include "common/numeric_util.hpp"
#include "ethos_u55_constraints.hpp"
#include "ethos_u55_performance.hpp"
#include "ethos_u55_register_cs_generator.hpp"
#include "ethos_u55_weight_encoder.hpp"

#include <algorithm>
#include <iterator>
#include <limits>
#include <unordered_map>
#include <unordered_set>

BEGIN_ENUM_TABLE(regor::EthosU55SHRamElements)
    ADD_ENUM_NAME(SHRAM_IFM8)
    ADD_ENUM_NAME(SHRAM_IFM16)
    ADD_ENUM_NAME(SHRAM_IFM8_Elementwise)
    ADD_ENUM_NAME(SHRAM_IFM16_Elementwise)
    ADD_ENUM_NAME(SHRAM_IFM32)
    ADD_ENUM_NAME(SHRAM_Acc16)
    ADD_ENUM_NAME(SHRAM_Acc32)
    ADD_ENUM_NAME(SHRAM_Acc40)
END_ENUM_TABLE()

BEGIN_ENUM_TABLE(regor::EthosUTraversal)
    ADD_ENUM_NAME(DepthFirst)
    ADD_ENUM_NAME(PartKernel)
    ADD_ENUM_NAME(Depthwise)
END_ENUM_TABLE()

namespace regor
{

static const EthosU55PerfInfo s_EthosU55PerfInfo[] = {
    // Accelerator.Ethos_U55_32
    {{2.0, 3.0, 3.0, 3.0, 4.0, 6.0, 1.0, 2.0}, {1.0, 1.0, 0.0}},
    // Accelerator.Ethos_U55_64
    {{1.0, 1.5, 1.5, 1.5, 2.0, 3.0, 0.5, 1.0}, {1.0, 1.0, 0.0}},
    // Accelerator.Ethos_U55_128
    {{0.75, 1.25, 0.75, 0.75, 1.0, 1.5, 0.25, 0.5}, {1.0, 0.5, 0.0}},
    // Accelerator.Ethos_U55_256
    {{0.625, 1.125, 0.5, 0.375, 0.5, 0.75, 0.125, 0.25}, {1.0, 0.25, 0.0}},
};

static const ArchEthosU55::AcceleratorConfig s_EthosU55Configs[] = {
    // Accelerator.Ethos_U55_32
    {32, 1, Shape(1, 1, 4), Shape(1, 1, 8), 16, {2, 2, 2, 2, 4, 4, 4, 4}, 1, &s_EthosU55PerfInfo[0]},
    // Accelerator.Ethos_U55_64
    {64, 1, Shape(1, 1, 8), Shape(1, 1, 8), 16, {2, 2, 2, 2, 4, 4, 4, 8}, 2, &s_EthosU55PerfInfo[1]},
    // Accelerator.Ethos_U55_128
    {128, 1, Shape(1, 2, 8), Shape(1, 2, 8), 24, {4, 4, 4, 4, 8, 4, 8, 12}, 4, &s_EthosU55PerfInfo[2]},
    // Accelerator.Ethos_U55_256
    {256, 1, Shape(2, 2, 8), Shape(2, 2, 8), 48, {8, 8, 8, 8, 16, 8, 16, 20}, 8, &s_EthosU55PerfInfo[3]},
};

enum class ElementwiseUsage
{
    No = 0,
    Full = 1,
    Scalar = 2,
};

static const int s_SHRAMElementBits[] = {
    8,   // IFM8
    16,  // IFM16
    8,   // IFM8_Elementwise
    16,  // IFM16_Elementwise
    32,  // IFM32
    16,  // Acc16
    32,  // Acc32
    40,  // Acc40
};

static_assert(std::size(s_SHRAMElementBits) == int(SHRAM_Last) + 1, "Bad element mapping");

// max size for tensor axes
const static Shape MAX_SHAPE(nullptr, 8, 65536);

ArchEthosU55::ArchEthosU55() : _subkernelMax(8, 8, 65536), _ofmBlockMax(32, 64, 128)
{
    _weightEncoder = std::make_unique<EthosU55WeightEncoder>(this);
    _constraints = std::make_unique<EthosU55Constraints>(this);
}

uint32_t ArchEthosU55::Version()
{
    return EthosU55RCSGenerator::IdRegister();
}

bool ArchEthosU55::ParseConfig(IniReader *reader)
{
    // Parse architecture configuration
    std::string key;
    int macs = 0;
    while ( reader->Begin(key) )
    {
        if ( key == "macs" )
        {
            macs = reader->Get<int>();
        }
        reader->End();
    }

    // Find the requested MAC configuration for this accelerator
    auto cfg = std::find_if(s_EthosU55Configs, std::cend(s_EthosU55Configs),
        [&](const AcceleratorConfig &config) { return config.macs == macs; });
    if ( cfg == std::cend(s_EthosU55Configs) )
    {
        assert(macs == 32 || macs == 64 || macs == 128 || macs == 256);
        LOG_TRACE0("Unable to find U55 accelerator for macs={}", macs);
        return false;
    }

    ApplyConfig(cfg);

    return true;
}

void ArchEthosU55::ApplyConfig(const AcceleratorConfig *cfg)
{
    // Basic configuration
    _cores = cfg->cores;
    _macs = cfg->macs;
    _ifmUBlock = cfg->ifmUblock;
    _ofmUBlock = cfg->ofmUBlock;

    // All SHRAM granules
    _shramGranules = cfg->shramGranules;

    // Bank granules organised by bit width
    _ifmBankGranules[0] = cfg->shramGranules[SHRAM_IFM8];
    _ifmBankGranules[1] = cfg->shramGranules[SHRAM_IFM16];
    _ifmBankGranules[2] = 0;
    _ifmBankGranules[3] = cfg->shramGranules[SHRAM_IFM32];

    // Elementwise bank granules organised by bit width
    _ifmEWBankGranules[0] = cfg->shramGranules[SHRAM_IFM8_Elementwise];
    _ifmEWBankGranules[1] = cfg->shramGranules[SHRAM_IFM16_Elementwise];
    _ifmEWBankGranules[2] = 0;
    _ifmEWBankGranules[3] = cfg->shramGranules[SHRAM_IFM32];

    // SHRAM layout information
    _shram.reservedOutputBanks = 2;
    _shram.bankSizeBytes = 1024, _shram.totalBanks = cfg->shramBanks;
    _shram.reservedEndBanks = (_shram.totalBanks > 16) ? 2 : 0;

    _shramMemory = std::make_unique<ArchitectureMemory>("shram", _shram.bankSizeBytes * _shram.totalBanks);
    // TODO: Change SHRAM to use an internal memory representation rather than an external one (ArchitectureMemory)
    //  where most of the parameters are not relevant. It is currently this way to enable easy
    //  integration of the LUT memory
    _shramMemory->SetParameters(1, 0, 0, 1, 1, 1000, 1000);
    _lutMemory = _shramMemory.get();
    _performance = std::unique_ptr<ArchitecturePerformance>(new EthosU55Performance(this, cfg->perfInfo));
    _rcsGenerator = std::make_unique<EthosU55RCSGenerator>(this);
}


std::unique_ptr<ArchitectureOpConfig> ArchEthosU55::GetOpConfig(OpType opType, const ArchitectureConfigQuery &query)
{
    auto config = FindBlockConfig(opType, query);
    return config;
}


std::unique_ptr<ArchitectureOpGroup> ArchEthosU55::CreateOpGroup(const ArchitectureOpGroupQuery &op)
{
    LOG_TRACE1("Trying to create ArchEthosU55 OpGroup for {}\n", OpTypeToString(op.type));

    auto group = std::make_unique<EthosU55OpGroup>();
    if ( !group->Add(op) )
    {
        return nullptr;
    }

    return group;
}

std::vector<uint32_t> ArchEthosU55::ConfigRegisters()
{
    return std::vector<uint32_t>(1, ConfigRegister(0));
}

int ArchEthosU55::UpscaleAndRounding(ArchResampling resampling, int &rounding)
{
    rounding = (resampling == ArchResampling::Nearest) ? 1 : 0;
    return (resampling == ArchResampling::None) ? 1 : 2;
}

AxisMask ArchEthosU55::CanSubdivide(OpType opType, TransposeType transpose, ReverseType reverse)
{
    if ( (opType == OpType::FullyConnected || IsConvolution(opType) || IsElementwise(opType) || IsPooling(opType)) &&
         IsNone(transpose) && (reverse != ReverseType::H) )
    {
        return AxisMask::AxisY;
    }
    return AxisMask::None;
}

bool ArchEthosU55::SupportsScalar(OpType opType, DataType dataType, TensorUsage usage)
{
    bool supportedType(dataType == DataType::Int8 || dataType == DataType::UInt8 || dataType == DataType::Int16);
    return EthosU55RCSGenerator::IsSupportedElementwise(opType) && supportedType && IsIFM(usage);
}

Flags<WeightFormat> ArchEthosU55::SupportedWeightFormat(OpType)
{
    return WeightFormat::Default;
}

bool ArchEthosU55::UseAvgPoolNop(OpType type)
{
    return IsActivation(type) || type == OpType::Quantize || type == OpType::MemoryCopy;
}

static bool ChooseKernelMethod(const Shape &ifmShape, int ifmBits, const Kernel *kernel)
{
    if ( ifmShape.Depth() <= 8 )
    {
        return true;
    }

    // Compare part-kernel to depth-kernel and choose the one with best utilisation
    int kernelElements = kernel->ElementsWH();
    double depthUtilisation = ifmShape.Depth() / double(RoundAway(ifmShape.Depth(), ifmBits == 8 ? 32 : 16));
    double partUtilisation =
        (ifmShape.Depth() / double(RoundAway(ifmShape.Depth(), 8)) *
            (kernelElements / double(RoundAway(kernelElements, ifmBits == 8 ? 4 : 2))));

    return partUtilisation >= depthUtilisation;
}


static Shape GetArchIFMBlockSize(const Shape &ofmBlock, const Kernel *kernel, const Shape &ublock,
    const Shape &subkernelLimit, int upscale, int rounding)
{
    Point2i dilatedSize = kernel->DilatedWH();

    // IFM block height
    int h = RequiredInputSize(ofmBlock.Height(), kernel->Stride().y, std::min(dilatedSize.y, subkernelLimit.Height()), upscale, rounding);
    h = RoundAway(h, ublock.Height());

    // IFM block width
    int w = RequiredInputSize(ofmBlock.Width(), kernel->Stride().x, std::min(dilatedSize.x, subkernelLimit.Width()), upscale, rounding);
    w = RoundAway(w, ublock.Width());

    return Shape(1, h, w, ofmBlock.Depth());
}


static Shape FitBlockForOFM(const Shape &ofmShape, const Kernel *kernel, const Shape &block, int ublockHeight)
{
    // 256/512 __Conv1D__ optimisation (ratio of IFM:Accumulators changes) This is a specific
    // interpretation of a more general constraint that can't be applied because the
    // FindBlockConfig function must return block configs that can be applied to any OFM shape.
    if ( (ofmShape.Height() == 1) && (kernel->Size().y == 1) && (ublockHeight == 2) )
    {
        return Shape(1, std::min(block.Height(), ofmShape.Height()), block.Width(), block.Depth());
    }
    return block;
}


std::unique_ptr<ArchitectureOpConfig> ArchEthosU55::FindBlockConfig(OpType opType, const ArchitectureConfigQuery &query)
{
    assert(query.ifmBits > 0 && query.ifmBits <= 32);
    assert(query.ofmShape.Size() > 2 && "Insufficient dimensions to search for block config");
    assert(query.kernel != nullptr);

    const int OFMSplitDepth = 16;  // Specific to this architecture

    // Elementwise larger-volume correction
    const Shape &ifmShape = (query.ifmShape[1].Elements() > query.ifmShape[0].Elements()) ? query.ifmShape[1] : query.ifmShape[0];

    EthosU55NpuOp npuOp = GetHWOp(opType);
    assert(npuOp != EthosU55NpuOp::None);

    // Figure out if SHRAM should be portioned for elementwise
    ElementwiseUsage ewUsage = ElementwiseUsage::No;
    if ( npuOp == EthosU55NpuOp::Elementwise )
    {
        bool usesScalar = query.ifmShape[0].Elements() == 1;
        if ( query.ifmShape[1].IsValid() )
        {
            usesScalar = usesScalar || query.ifmShape[1].Elements() == 1;
        }

        ewUsage = (usesScalar && (query.ifmBits <= 16)) ? ElementwiseUsage::Scalar : ElementwiseUsage::Full;
    }

    // Operator typing help
    bool isPooling = npuOp == EthosU55NpuOp::Pooling || npuOp == EthosU55NpuOp::ReduceSum;
    bool isReduceSum = npuOp == EthosU55NpuOp::ReduceSum;
    bool isDepthwise = npuOp == EthosU55NpuOp::Depthwise;
    bool isEqualDepthOp = (ewUsage != ElementwiseUsage::No) || (isPooling && !isReduceSum) || isDepthwise;
    bool isPartKernel = npuOp == EthosU55NpuOp::Convolution && ChooseKernelMethod(ifmShape, query.ifmBits, query.kernel);

    // Operator configuration to be returned
    auto config = std::make_unique<EthosU55OpConfig>();
    config->_bankSize = _shram.bankSizeBytes;
    // IFM is not broadcasted for pooling and depthwise ops and for elementwise
    // when there's no elementwise-broadcasting in depth
    int ifmDepthBufScaling =
        isPooling || isDepthwise || IsUnaryElementwise(opType) ||
                (IsBinaryElementwise(opType) && (query.ifmShape[0].Depth() == query.ifmShape[1].Depth())) ?
            _cores :
            1;
    config->_ifmDepthBufScaling = ifmDepthBufScaling;
    config->_traversal = isDepthwise ? EthosUTraversal::Depthwise : (isPartKernel ? EthosUTraversal::PartKernel : EthosUTraversal::DepthFirst);

    // Accumulator & granule settings
    EthosU55SHRamElements accType = SHRAM_Acc32;
    if ( query.ifmBits == 16 && (!isPooling || isReduceSum) && query.scaled )
    {
        accType = SHRAM_Acc40;
    }
    config->_accumulatorType = accType;

    // Memory rounding granules
    int accGranule = _shramGranules[accType];
    int accBits = s_SHRAMElementBits[accType];
    int ifmGranule = 0;
    if ( ewUsage != ElementwiseUsage::No )
    {
        ifmGranule = _ifmEWBankGranules[query.ifmBits / 8 - 1];
    }
    else
    {
        ifmGranule = _ifmBankGranules[query.ifmBits / 8 - 1];
    }

    int rounding;
    int upscale = UpscaleAndRounding(query.ifmResampling, rounding);
    int lutBanks = std::max(DivRoundUp(query.lutBytes, 1024), _shram.reservedEndBanks);

    // Subkernel repeats of the IFM
    Point2i dilatedWH = query.kernel->DilatedWH();
    int ifmRepeats = DivRoundUp(dilatedWH.x, _subkernelMax.Width()) * DivRoundUp(dilatedWH.y, _subkernelMax.Height());

    int ifmBlockDepth = 0;
    if ( query.ifmBits == 16 )
    {
        ifmBlockDepth = RoundAway(std::min(ifmShape.Depth(), 16), 4);
    }
    else
    {
        ifmBlockDepth = RoundAway(std::min(ifmShape.Depth(), isPartKernel ? 16 : 32), _ifmUBlock.Depth());
    }

    // Weights fetch (for operators that have them)
    bool hasWeights = npuOp == EthosU55NpuOp::Convolution || isDepthwise;
    int weightFetchWH = hasWeights ? query.kernel->Size().AreaXY() : 0;

    int ofmUBlockDepth = _ofmUBlock.Depth() * _cores;
    Shape searchSpace = Shape::RoundAway(Shape::Min(query.ofmShape, _ofmBlockMax), _ofmUBlock.WithDepth(ofmUBlockDepth));

    // Block WHC search, loops across the search space looking for best efficiency
    float bestCost = std::numeric_limits<float>::infinity();
    float bestCoverage = std::numeric_limits<float>::infinity();
    int ofmElements = query.ofmShape.Elements();

    int depth = std::max(ofmUBlockDepth, std::min(searchSpace.Depth(), OFMSplitDepth));
    if ( depth < query.ofmShape.Depth() )
    {
        depth = RoundAway(depth, OFMSplitDepth);
    }

    std::unordered_set<Point2i, Point2Hash<int>> wontFit;
    while ( depth <= searchSpace.Depth() )
    {
        for ( int height = _ofmUBlock.Height(); height <= searchSpace.Height(); height += _ofmUBlock.Height() )
        {
            for ( int width = _ofmUBlock.Width(); width <= searchSpace.Width(); width += _ofmUBlock.Width() )
            {
                // Avoid checking blocks that already didn't fit with a smaller depth.
                // i.e. if 8x4x16 didn't fit then 8x4x32 won't either.
                if ( wontFit.count(Point2i(height, width)) > 0 )
                {
                    continue;
                }

                // Calculate the IFM block dimensions required to feed this OFM block
                Shape ofmBlock = Shape(height, width, depth);
                Shape ifmBlock = GetArchIFMBlockSize(ofmBlock, query.kernel, _ofmUBlock, _subkernelMax, upscale, rounding);
                if ( !isEqualDepthOp )
                {
                    ifmBlock[-1] = ifmBlockDepth;
                }

                ofmBlock = FitBlockForOFM(query.ofmShape, query.kernel, ofmBlock, _ofmUBlock.Height());

                // Test if the IFM/OFM blocks fit into SHRAM
                EthosU55OpConfig::SHRAMLayout layout;
                if ( TryBlockConfig(layout, int(ewUsage), ofmBlock, ifmBlock, query.ifmBits, ifmGranule, accBits, accGranule, lutBanks, ifmDepthBufScaling) )
                {
                    Shape fullBlocks = Shape::DivRoundUp(query.ofmShape, ofmBlock);
                    Point3<float> blocks = query.ofmShape.HWC<float>() / ofmBlock.HWC<float>();

                    // Weights fetching
                    float weightFetch = float(weightFetchWH * ifmShape.Depth() * fullBlocks.ElementsWH());
                    if ( !isDepthwise )
                    {
                        weightFetch *= blocks.z * float(ofmBlock.Depth());
                    }

                    // IFM fetching
                    float ifmFetch = float(ifmBlock.ElementsWH() * ifmShape.Depth() * ifmRepeats) * blocks.x * blocks.y;
                    if ( !isEqualDepthOp )
                    {
                        ifmFetch *= float(fullBlocks.Depth());
                    }

                    // Scale relative to every output OFM element
                    float relativeCost =
                        npuOp == EthosU55NpuOp::Elementwise ? float(ofmElements) / float(height * width * depth) : (ifmFetch + weightFetch) / float(ofmElements);

                    // If the entire IFM can be encompassed by both buffers, bias to prefer this configuration
                    if ( ifmShape.Elements() < ifmBlock.Elements() * 2 )
                    {
                        relativeCost = relativeCost / 2.0f;
                    }

                    // Choose based on relative minimum cost or larger IFM area (if equal cost)
                    if ( relativeCost <= bestCost )
                    {
                        bool chooseThis = false;
                        // Check IFM coverage only when it's equal best_cost and small OFM
                        if ( relativeCost == bestCost )
                        {
                            Shape coverageShape = Shape::Min(ifmShape, ifmBlock);
                            float coverage = float(ifmShape.ElementsWH()) / float(coverageShape.ElementsWH());
                            // Small 4x4 IFM constraint found through analysis of networks
                            if ( coverage <= bestCoverage && (height <= 4 && width <= 4) )
                            {
                                bestCoverage = coverage;
                                chooseThis = true;
                            }
                        }
                        else
                        {
                            bestCoverage = std::numeric_limits<float>::infinity();
                            chooseThis = true;
                        }

                        if ( chooseThis )
                        {
                            bestCost = relativeCost;
                            config->_layout = layout;
                            config->_ifmBlock = ifmBlock;
                            config->_ofmBlock = Shape(1, height, width, depth);
                        }
                    }
                }
                else
                {
                    wontFit.emplace(height, width);
                }
            }
        }

        // Try Next block depth, rounded
        depth = depth + ofmUBlockDepth;
        if ( depth < query.ofmShape.Depth() )
        {
            depth = RoundAway(depth, OFMSplitDepth);
        }
    }

    // Return the best configuration
    if ( bestCost != std::numeric_limits<float>::infinity() )
    {
        return std::unique_ptr<ArchitectureOpConfig>(config.release());
    }

    // Didn't find a configuration
    return std::unique_ptr<ArchitectureOpConfig>();
}


bool ArchEthosU55::TryBlockConfig(EthosU55OpConfig::SHRAMLayout &layout, int ewUsage, const Shape &ofmBlock,
    const Shape &ifmBlock, int ifmBits, int ifmGranule, int accBits, int accGranule, int lutBanks, int ifmDepthBufScaling)
{
    assert((accBits > 0) && (accGranule > 0));
    assert((ifmBits >= 8) && ((ifmBits % 8) == 0) && (ifmGranule > 0));

    // Scale depth with cores
    int ifm_depth = DivRoundUp(ifmBlock.Depth(), ifmDepthBufScaling);
    int ofm_depth = DivRoundUp(ofmBlock.Depth(), _cores);

    // Always need IFM space
    int ifm_bytes = ifmBlock.ElementsWH() * RoundAway(ifm_depth * (ifmBits / 8), 8);
    int ifm_banks = DivRoundUp(ifm_bytes, _shram.bankSizeBytes) * 2;
    ifm_banks = RoundAway(ifm_banks, ifmGranule);

    // Calculate SHRAM boundaries of the IFM and Accumulators
    int lut_start = _shram.totalBanks - lutBanks;
    int ifm_end = _shram.reservedOutputBanks + ifm_banks;
    int ifm2_start = ifm_end;
    int acc_start = lut_start;

    // If not elementwise then we need accumulator space
    if ( ewUsage == int(ElementwiseUsage::No) )
    {
        int acc_bytes = (ofmBlock.ElementsWH() * RoundAway(ofm_depth, 8) * accBits) / 8;
        int acc_banks = DivRoundUp(acc_bytes, _shram.bankSizeBytes) * 2;
        acc_banks = RoundAway(acc_banks, accGranule);
        acc_start = acc_start - acc_banks;
    }
    else
    {
        int ifm2_banks = (ewUsage == int(ElementwiseUsage::Full)) ? ifm_banks : 0;
        if ( ifm2_start + ifm2_banks > acc_start )
        {
            return false;
        }
        ifm_end = acc_start;
    }

    // IFM must still fit before accumulators
    if ( ifm_end > acc_start )
    {
        return false;
    }

    // Should all fit, so return this layout
    layout.ibStart = _shram.reservedOutputBanks;
    layout.ibStart2 = ifm2_start;
    layout.ibEnd = ifm_end;
    layout.abStart = acc_start;
    layout.lutStart = lut_start;
    return true;
}


Shape ArchEthosU55::GetStorageRounding(TensorFormat format)
{
    if ( format == TensorFormat::NHCWB16 )
    {
        return Shape(1, 1, 1, 16);
    }

    return Shape(1, 1, 1, 1);
}


uint32_t ArchEthosU55::ConfigRegister(int product)
{
    uint32_t macs = _macs * _cores;
    int macsCeilLog2 = 0;
    while ( macs >>= 1 )
    {
        macsCeilLog2++;
    }
    int shramSize = _cores * (int(_shramMemory->SizeBytes()) >> 10);
    assert(macsCeilLog2 < 16);
    assert(shramSize < 256);
    return macsCeilLog2 | (shramSize << 8) | (product << 28);
}

std::unique_ptr<ArchitectureOpConfig> EthosU55OpConfig::Clone()
{
    auto config = std::make_unique<EthosU55OpConfig>();
    config->_bankSize = _bankSize;
    config->_ifmDepthBufScaling = _ifmDepthBufScaling;
    config->_traversal = _traversal;
    config->_accumulatorType = _accumulatorType;
    config->_ofmBlock = _ofmBlock;
    config->_ifmBlock = _ifmBlock;
    config->_layout = _layout;
    return std::unique_ptr<ArchitectureOpConfig>(config.release());
}

int EthosU55OpConfig::MaxIFMBuffering()
{
    return (_layout.ibEnd - _layout.ibStart) * _bankSize * _ifmDepthBufScaling;
}

Point2i EthosU55OpConfig::OptimalStripeGranule()
{
    return _ofmBlock.WH<int>();
}

int EthosU55OpConfig::OptimalDepthGranule()
{
    return _ofmBlock.Depth();
}

std::string EthosU55OpConfig::ToString(bool full)
{
    std::string tmp = fmt::format("OFM Block=[{}], IFM Block=[{}], Traversal={}, AccType={}", _ofmBlock.ToString(),
        _ifmBlock.ToString(), EnumToString(_traversal), EnumToString(_accumulatorType));
    if ( full )
    {
        tmp += fmt::format("\nSHRAM: ib={} ibE={}, ib2={}, ab={}, lut={}", _layout.ibStart, _layout.ibEnd,
            _layout.ibStart2, _layout.abStart, _layout.lutStart);
    }
    return tmp;
}

EthosU55NpuOp ArchEthosU55::GetHWOp(OpType type)
{
    static const std::unordered_map<OpType, EthosU55NpuOp> toNpuOp = {
        {OpType::DepthwiseConv2D, EthosU55NpuOp::Depthwise},
        {OpType::Conv2D, EthosU55NpuOp::Convolution},
        {OpType::FullyConnected, EthosU55NpuOp::VectorProduct},
        {OpType::MaxPool, EthosU55NpuOp::Pooling},
        {OpType::AvgPool, EthosU55NpuOp::Pooling},
        {OpType::QuantizedAvgPool, EthosU55NpuOp::Pooling},
        {OpType::QuantizedMaxPool, EthosU55NpuOp::Pooling},
        {OpType::ResizeBilinear, EthosU55NpuOp::Pooling},
        {OpType::ReduceSum, EthosU55NpuOp::ReduceSum},
        {OpType::Rescale, EthosU55NpuOp::Pooling},
        {OpType::Tile, EthosU55NpuOp::Dma},
        {OpType::Transpose, EthosU55NpuOp::Compound},
        {OpType::MatMul, EthosU55NpuOp::Compound},
    };
    auto pos = toNpuOp.find(type);
    if ( pos != toNpuOp.end() )
    {
        return pos->second;
    }
    else if ( EthosU55RCSGenerator::IsSupportedElementwise(type) )
    {
        return EthosU55NpuOp::Elementwise;
    }
    else if ( UseAvgPoolNop(type) )
    {
        return EthosU55NpuOp::Pooling;
    }
    return EthosU55NpuOp::None;
}

int EthosU55OpGroup::Add(const ArchitectureOpGroupQuery &op, const std::vector<int> &dependsOn)
{
    LOG_TRACE1("Trying to add op {}\n", OpTypeToString(op.type));

    if ( _opsCount >= 2 )
    {
        // Can only fuse 2 ops
        return 0;
    }

    if ( !CanRunOnNPU(op) )
    {
        // Can only fuse NPU ops
        return 0;
    }

    if ( _opsCount > 0 && !IsActivation(op.type) )
    {
        // Can only fuse with activation
        return 0;
    }

    for ( int dep : dependsOn )
    {
        if ( dep > 0 )
        {
            // Don't validate user-specified (positive keys) dependencies
            continue;
        }
        else if ( dep < 0 )
        {
            // Convert to group generated keys (negative keys) to array index
            dep = (-dep) - 1;
            if ( dep >= _opsCount )
            {
                // Missing dependency
                return 0;
            }
        }

        const EthosU55OpGroup::OpInfo &prevOp = _ops[dep];

        if ( prevOp.ofm.key != op.ifm[0].key && (op.inputs == 1 || prevOp.ofm.key != op.ifm[1].key) )
        {
            // Can only fuse when ops are connected
            return 0;
        }
        else
        {
            _fusedTensors.insert(prevOp.ofm.key);
        }
    }

    // Generated key
    int key = (-_opsCount) - 1;

    // Save copy of op
    _ops[_opsCount] = op;
    _opsInternal[_opsCount].dependsOn = dependsOn;
    _opsCount++;

    return key;
}
bool EthosU55OpGroup::NeedsAllocation(UniqueId tensorUID)
{
    return _fusedTensors.count(tensorUID) == 0;
}

// Table of allowed ifm/ofm data type combinations for each HWOp
static const std::unordered_map<EthosU55NpuOp, std::unordered_map<DataType, std::vector<DataType>>> s_opDataTypeSupport = {
    {EthosU55NpuOp::Convolution,  // HWOp
        {
            // IFM data type  | OFM data type(s)
            {DataType::UInt8, {DataType::UInt8, DataType::Int8, DataType::Int16, DataType::Int32}},
            {DataType::Int8, {DataType::UInt8, DataType::Int8, DataType::Int16, DataType::Int32}},
            {DataType::Int16, {DataType::UInt8, DataType::Int8, DataType::Int16, DataType::Int32}},
        }},
    {EthosU55NpuOp::Depthwise,
        {
            {DataType::UInt8, {DataType::UInt8, DataType::Int8, DataType::Int16, DataType::Int32}},
            {DataType::Int8, {DataType::UInt8, DataType::Int8, DataType::Int16, DataType::Int32}},
            {DataType::Int16, {DataType::UInt8, DataType::Int8, DataType::Int16, DataType::Int32}},
        }},
    {EthosU55NpuOp::VectorProduct,
        {
            {DataType::UInt8, {DataType::UInt8, DataType::Int8, DataType::Int16, DataType::Int32}},
            {DataType::Int8, {DataType::UInt8, DataType::Int8, DataType::Int16, DataType::Int32}},
            {DataType::Int16, {DataType::UInt8, DataType::Int8, DataType::Int16, DataType::Int32}},
        }},
    {EthosU55NpuOp::Pooling,
        {
            {DataType::UInt8, {DataType::UInt8}},
            {DataType::Int8, {DataType::Int8}},
            {DataType::Int16, {DataType::Int16}},
        }},
    {EthosU55NpuOp::ReduceSum,
        {
            {DataType::UInt8, {DataType::UInt8, DataType::Int8, DataType::Int16, DataType::Int32}},
            {DataType::Int8, {DataType::UInt8, DataType::Int8, DataType::Int16, DataType::Int32}},
            {DataType::Int16, {DataType::UInt8, DataType::Int8, DataType::Int16, DataType::Int32}},
            {DataType::Int32, {DataType::UInt8, DataType::Int8, DataType::Int16, DataType::Int32}},
        }},
};

bool EthosU55OpGroup::CanRunOnNPU(const ArchitectureOpGroupQuery &op)
{
    EthosU55NpuOp npuOp = ArchEthosU55::GetHWOp(op.type);

    if ( IsFloat(op.ifm[0].type | op.ifm[1].type | op.ofm.type) )
    {
        return false;
    }

    if ( npuOp == EthosU55NpuOp::None || npuOp > EthosU55NpuOp::Compound )
    {
        return false;
    }

    auto k = op.kernel;
    if ( k->Stride().x > 3 || k->Stride().y > 3 )
    {
        return false;
    }

    if ( k->Dilation().x > 2 || k->Dilation().y > 2 )
    {
        return false;
    }

    if ( k->DepthMultiplier() > 1 )
    {
        return false;
    }

    // Validate that input/outputs shapes don't overflow
    if ( npuOp != EthosU55NpuOp::Dma )
    {
        const auto &ifmShape = op.ifm[0].shape;
        const auto &ofmShape = op.ofm.shape;
        if ( ifmShape.GreaterMask(MAX_SHAPE) != 0 )
        {
            return false;
        }
        if ( ofmShape.GreaterMask(MAX_SHAPE) != 0 )
        {
            return false;
        }
        if ( op.inputs > 1 )
        {
            const auto &ifm2Shape = op.ifm[1].shape;
            if ( ifm2Shape.GreaterMask(MAX_SHAPE) != 0 )
            {
                return false;
            }
        }
    }

    // Check allowed ifm/ofm type mapping
    if ( npuOp != EthosU55NpuOp::Elementwise )
    {
        if ( op.type == OpType::LUT || op.type == OpType::MemoryCopy || op.type == OpType::Rescale ||
             op.type == OpType::Tile || op.type == OpType::Transpose || npuOp == EthosU55NpuOp::Compound )
        {  // TODO: LUT operations end up here due to UseAvgPoolNop although the rules are not the same as
           // for a Pooling operation, so skip checks for now.
            return true;
        }

        auto map = s_opDataTypeSupport.find(npuOp);
        if ( map == s_opDataTypeSupport.end() )
        {
            assert(false && "Data type mapping for HWOp missing");
            return false;
        }
        auto &typeMap = map->second;
        auto ifmEntry = typeMap.find(op.ifm[0].type);
        if ( ifmEntry == typeMap.end() )
        {  // Unsupported ifm data type
            return false;
        }
        auto &ofmTypes = ifmEntry->second;
        if ( 0 == std::count(ofmTypes.begin(), ofmTypes.end(), op.ofm.type) )
        {  // Unsupported ofm data type
            return false;
        }
    }
    else
    {
        std::vector<DataType> validIfmTypes;
        std::vector<DataType> validOfmTypes;
        switch ( op.type )
        {
            case OpType::Add:
            case OpType::Sub:
            case OpType::Mul:
            {
                validIfmTypes = {DataType::UInt8, DataType::Int8, DataType::Int16, DataType::Int32};
                validOfmTypes = validIfmTypes;
            }
            break;
            case OpType::Minimum:
            case OpType::Maximum:
            case OpType::LeakyRelu:
            case OpType::Abs:
            {
                validIfmTypes = {DataType::UInt8, DataType::Int8, DataType::Int16, DataType::Int32};
                validOfmTypes = {op.ifm[0].type};
            }
            break;
            case OpType::CLZ:
            case OpType::SHL:
            case OpType::Asr:
            {
                validIfmTypes = {DataType::Int32};
                validOfmTypes = {DataType::Int32};
                if ( op.type == OpType::Asr )
                {
                    validOfmTypes.insert(validOfmTypes.begin(), {DataType::UInt8, DataType::Int8, DataType::Int16});
                }
            }
            break;
            default:
                assert(false && "Unkown elementwise type");
                break;
        }

        if ( 0 == std::count(validIfmTypes.begin(), validIfmTypes.end(), op.ifm[0].type) )
        {  // Unsupported ifm data type
            return false;
        }
        if ( IsBinaryElementwise(op.type) && op.ifm[1].type != op.ifm[0].type )
        {  // ifm2 data type must match ifm data type
            return false;
        }
        if ( 0 == std::count(validOfmTypes.begin(), validOfmTypes.end(), op.ofm.type) )
        {  // Unsupported ofm data type
            return false;
        }
    }

    return true;
}

}  // namespace regor
