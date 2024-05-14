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

#include "ethos_u85.hpp"

#include "common/common.hpp"
#include "common/logging.hpp"

#include "common/bit_flags.hpp"
#include "common/numeric_util.hpp"
#include "ethos_u85_performance.hpp"
#include "ethos_u85_register_cs_generator.hpp"
#include "ethos_u85_weight_encoder.hpp"

#include <algorithm>
#include <limits>
#include <unordered_map>
#include <unordered_set>

BEGIN_ENUM_TABLE(regor::EthosU85Accumulator)
    ADD_ENUM_NAME(Acc32)
    ADD_ENUM_NAME(Acc48)
END_ENUM_TABLE()

BEGIN_ENUM_TABLE(regor::EthosU85Traversal)
    ADD_ENUM_NAME(DepthFirst)
    ADD_ENUM_NAME(PartKernel)
    ADD_ENUM_NAME(Depthwise)
END_ENUM_TABLE()

namespace regor
{

unsigned MaskForNpuOp(const EthosU85NpuOp npuOp, bool hasIfm2);
bool IsMinMaxReduction(OpType opType, const Kernel *kernel);

static const EthosU85PerfInfo s_EthosU85PerfInfo[] = {
    // Accelerator.Ethos_U85_128
    {{2.0, 3.0, 3.0, 3.0, 4.0, 6.0, 1.0, 2.0}, {1.0, 1.0, 0.0}},
    // Accelerator.Ethos_U85_256
    {{2.0, 3.0, 3.0, 3.0, 4.0, 6.0, 1.0, 2.0}, {1.0, 1.0, 0.0}},
    // Accelerator.Ethos_U85_512
    {{1.0, 1.5, 1.5, 1.5, 2.0, 3.0, 0.5, 1.0}, {1.0, 1.0, 0.0}},
    // Accelerator.Ethos_U85_1024
    {{0.75, 1.25, 0.75, 0.75, 1.0, 1.5, 0.25, 0.5}, {1.0, 0.5, 0.0}},
    // Accelerator.Ethos_U85_2048
    {{0.625, 1.125, 0.5, 0.375, 0.5, 0.75, 0.125, 0.25}, {1.0, 0.25, 0.0}},
};

static const ArchEthosU85::AcceleratorConfig s_EthosU85Configs[] = {
    // Accelerator.Ethos_U85_128
    {128, 1, {Shape(1, 2, 8), Shape(1, 1, 16)}, 2, Shape(1, 2, 8), 8 * 1024, 8 * 1024, 2, 1, 0, &s_EthosU85PerfInfo[0]},
    // Accelerator.Ethos_U85_256
    {256, 1, {Shape(1, 2, 16), Shape(1, 4, 8), Shape(2, 2, 8)}, 3, Shape(2, 2, 8), 16 * 1024, 16 * 1024, 4, 1, 0, &s_EthosU85PerfInfo[0]},
    // Accelerator.Ethos_U85_512
    {512, 2, {Shape(2, 2, 16), Shape(1, 4, 16)}, 2, Shape(2, 2, 16), 16 * 1024, 32 * 1024, 8, 1, 0, &s_EthosU85PerfInfo[1]},
    // Accelerator.Ethos_U85_1014
    {1024, 4, {Shape(2, 2, 32), Shape(1, 4, 32), Shape(2, 4, 16)}, 3, Shape(4, 2, 16), 16 * 1024, 64 * 1024, 16, 1, 1, &s_EthosU85PerfInfo[2]},
    // Accelerator.Ethos_U85_2048
    {2048, 4, {Shape(2, 2, 64), Shape(1, 4, 64), Shape(4, 4, 16)}, 3, Shape(4, 4, 16), 32 * 1024, 128 * 1024, 32, 2, 1, &s_EthosU85PerfInfo[3]},
};

enum class ElementwiseUsage
{
    No = 0,
    Full = 1,
    Scalar = 2,
};

static int AccumulatorBits(EthosU85Accumulator accType)
{
    int bits = 32;
    switch ( accType )
    {
        case EthosU85Accumulator::Acc32:
            bits = 32;
            break;
        case EthosU85Accumulator::Acc48:
            bits = 64;
            break;
        default:
            LOG_WARN("Invalid accumulator type for Ethos U85: {}\n", accType);
            assert(false);
            break;
    }
    return bits;
}


ArchEthosU85::ArchEthosU85() : _subkernelMax(8, 8, 65536), _ofmBlockMax(128, 128, 1024)
{
    _weightEncoder = std::make_unique<EthosU85WeightEncoder>(this);
    _rcsGenerator = std::make_unique<EthosU85RCSGenerator>(this);
}

uint32_t ArchEthosU85::Version()
{
    return EthosU85RCSGenerator::IdRegister();
}

bool ArchEthosU85::ParseConfig(IniReader *reader)
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
    auto cfg = std::find_if(s_EthosU85Configs, std::cend(s_EthosU85Configs),
        [&](const AcceleratorConfig &config) { return config.macs == macs; });
    if ( cfg == std::cend(s_EthosU85Configs) )
    {
        assert(macs == 128 || macs == 256 || macs == 512 || macs == 1024 || macs == 2048);
        LOG_TRACE0("Unable to find Ethos U85 accelerator for macs={}", macs);
        return false;
    }

    ApplyConfig(cfg);

    return true;
}

void ArchEthosU85::ApplyConfig(const AcceleratorConfig *cfg)
{
    // Basic configuration
    _cores = cfg->cores;
    _macs = cfg->macs;
    _ifmUBlock = cfg->ifmUBlock;
    _nOfmUBlocks = cfg->nOfmUBlocks;
    std::copy(std::begin(cfg->ofmUBlocks), std::end(cfg->ofmUBlocks), std::begin(_ofmUBlocks));

    // Internal memory
    _ifmRamSizeBytes = cfg->ifmRamSizeBytes;
    _accRamSizeBytes = cfg->accRamSizeBytes;
    _numAxiSramLog2 = cfg->numAxiSramLog2;
    _numAxiExtLog2 = cfg->numAxiExtLog2;

    _lutRam = std::make_unique<ArchitectureMemory>("lutram", 2048);
    // TODO MLBEDSW-7980 fix LUT performance parameters
    _lutRam->SetParameters(1, 0, 0, 1, 0);
    _lutMemory = _lutRam.get();
    _performance = std::unique_ptr<ArchitecturePerformance>(new EthosU85Performance(this, cfg->perfInfo));

    // Populate ofmUBlock -> NpuOp lookup table
    SetupOfmUBlockToOpTable();
    // Populate ofmUBlock -> ifmAlloc unit table
    SetupOfmUBlockToIfmAuTable();
}


std::unique_ptr<ArchitectureOpConfig> ArchEthosU85::GetOpConfig(OpType opType, const ArchitectureConfigQuery &query)
{
    auto config = FindBlockConfig(opType, query);
    return config;
}


std::unique_ptr<ArchitectureOpGroup> ArchEthosU85::CreateOpGroup(const ArchitectureOpGroupQuery &op)
{
    LOG_TRACE1("Trying to create ArchEthosU85 OpGroup for {}\n", OpTypeToString(op.type));

    auto group = std::make_unique<EthosU85OpGroup>();
    if ( !group->Add(op) )
    {
        return nullptr;
    }

    return group;
}

std::vector<uint32_t> ArchEthosU85::ConfigRegisters()
{
    return std::vector<uint32_t>(1, ConfigRegister(2));
}

int ArchEthosU85::UpscaleAndRounding(ArchResampling resampling, int &rounding)
{
    rounding = (resampling == ArchResampling::Nearest) ? 1 : 0;
    return (resampling == ArchResampling::Zeros) ? 2 : 1;
}

AxisMask ArchEthosU85::CanSubdivide(OpType opType)
{
    if ( IsConvolution(opType) || IsElementwise(opType) || IsPooling(opType) )
    {
        return AxisMask::AxisY;
    }
    return AxisMask::None;
}

bool ArchEthosU85::SupportsLeakyRelu(bool /*quantized*/, DataType /*type*/)
{
    return true;
}

bool ArchEthosU85::SupportsMatMul(OpType opType)
{
    EthosU85NpuOp npuOp = GetHWOp(opType);
    if ( npuOp == EthosU85NpuOp::None )
    {
        return false;
    }

    return true;
}

bool ArchEthosU85::SupportsTranspose(OpType opType, TransposeType transposeType)
{
    if ( IsNone(transposeType) ) return true;

    EthosU85NpuOp npuOp = GetHWOp(opType);
    if ( npuOp == EthosU85NpuOp::None || npuOp == EthosU85NpuOp::Resize || npuOp == EthosU85NpuOp::Dma )
    {
        return false;
    }
    else if ( npuOp == EthosU85NpuOp::Elementwise )
    {
        return transposeType == TransposeType::NHWC || transposeType == TransposeType::NHCW || transposeType == TransposeType::NCHW;
    }

    return transposeType == TransposeType::NHWC || transposeType == TransposeType::NWHC || transposeType == TransposeType::NHCW ||
           transposeType == TransposeType::NWCH || transposeType == TransposeType::NCHW || transposeType == TransposeType::NCWH;
}

bool ArchEthosU85::SupportsReverse(OpType opType, ReverseType reverseType)
{
    if ( reverseType == ReverseType::None ) return true;

    EthosU85NpuOp npuOp = GetHWOp(opType);
    if ( npuOp == EthosU85NpuOp::None || npuOp == EthosU85NpuOp::Elementwise || npuOp == EthosU85NpuOp::Dma )
    {
        return false;
    }

    return reverseType == ReverseType::H || reverseType == ReverseType::W || reverseType == ReverseType::C;
}

bool ArchEthosU85::SupportsGather(OpType opType)
{
    EthosU85NpuOp npuOp = GetHWOp(opType);
    if ( npuOp == EthosU85NpuOp::None )
    {
        return false;
    }

    return true;
}

bool ArchEthosU85::SupportsScatter(OpType opType)
{
    EthosU85NpuOp npuOp = GetHWOp(opType);
    if ( npuOp == EthosU85NpuOp::None )
    {
        return false;
    }

    return true;
}

bool ArchEthosU85::SupportsSigmoidTanhLutInt16(OpType opType)
{
    return (opType == OpType::Sigmoid || opType == OpType::Tanh);
}

bool ArchEthosU85::SupportsArgMax(OpType opType)
{
    EthosU85NpuOp npuOp = GetHWOp(opType);
    if ( npuOp == EthosU85NpuOp::None )
    {
        return false;
    }

    return true;
}

bool ArchEthosU85::SupportsResize(const ResizeSupportQuery &query)
{
    /* Supported operator checks for resize operations
     *
     *  * Scaling numerators must be less than or equal to 2048
     *  * Offsets must be in the range [-numerator, numerator) for each axis
     *  * The following constraints apply to upscale-factors
     *    mode REPLICATE:
     *      Any width and height upscale-factors are supported
     *    mode NEAREST:
     *      Any width and height upscale-factors are supported
     *    mode BILINEAR:
     *      if IFM W*H == 1*1:
     *        Any width and height upscale-factors are supported
     *      else:
     *        The upscale-factors need to be powers-of-two.
     */
    if ( query.ifmShape.Width() == 1 && query.ifmShape.Height() == 1 )
    {
        return true;
    }

    int n_w = query.scaleX.n;
    int d_w = query.scaleX.d;
    int n_h = query.scaleY.n;
    int d_h = query.scaleY.d;
    bool supported = true;

    if ( n_h > 2048 )
    {
        LOG_WARN("Resize height scale numerator ({}) exceeds maximum size (2048).\n", n_h);
        supported = false;
    }
    if ( n_w > 2048 )
    {
        LOG_WARN("Resize width scale numerator ({}) exceeds maximum size (2048).\n", n_w);
        supported = false;
    }
    if ( query.offsetY >= n_h || query.offsetY < -n_h )
    {
        LOG_WARN("Resize height offset: {} is outside the valid range [-height_numerator, height_numerator) = [{}, {})\n",
            query.offsetY, -n_h, n_h);
        supported = false;
    }
    if ( query.offsetX >= n_w || query.offsetX < -n_w )
    {
        LOG_WARN("Resize width offset: {} is outside the valid range [-with_numerator, width_numerator) = [{}, {})\n",
            query.offsetX, -n_w, n_w);
        supported = false;
    }

    if ( query.mode == ArchResizeMode::Bilinear )
    {
        // Get scale fractions and verify that scale-factor is a power of two.

        if ( n_w % d_w != 0 )
        {
            LOG_WARN("ResizeBilinear width scale-factor is not an integer: {}/{}\n", n_w, d_w);
            supported = false;
        }
        if ( n_h % d_h != 0 )
        {
            LOG_WARN("ResizeBilinear height scale-factor is not an integer: {}/{}\n", n_h, d_h);
            supported = false;
        }
        int scale_w = n_w / d_w;
        int scale_h = n_h / d_h;
        if ( !IsPowerOfTwo(scale_w) )
        {
            LOG_WARN("ResizeBilinear width scale-factor is not a power of two: {}\n", double(n_w) / d_w);
            supported = false;
        }
        if ( !IsPowerOfTwo(scale_h) )
        {
            LOG_WARN("ResizeBilinear height scale-factor is not a power of two: {}\n", double(n_h) / d_h);
            supported = false;
        }
        return supported;
    }
    return supported;
}

bool ArchEthosU85::SupportsAccumulatorMode(ArchAccumulatorSource source, bool outputEnabled)
{
    UNUSED(outputEnabled);
    return source == ArchAccumulatorSource::Reset || source == ArchAccumulatorSource::Acc || source == ArchAccumulatorSource::Ifm2;
}

bool ArchEthosU85::SupportsScalar(OpType opType, DataType dataType, TensorUsage usage)
{
    bool supportedType(dataType == DataType::Int8 || dataType == DataType::UInt8 || dataType == DataType::Int16 || dataType == DataType::Int32);
    return EthosU85RCSGenerator::IsSupportedElementwise(opType) && supportedType && IsIFM(usage);
}

Flags<WeightFormat> ArchEthosU85::SupportedWeightFormat(OpType op)
{
    auto hwOp = GetHWOp(op);
    if ( hwOp == EthosU85NpuOp::Convolution || hwOp == EthosU85NpuOp::VectorProduct )
    {
        return Flags<WeightFormat>(WeightFormat::Default, WeightFormat::Fast, WeightFormat::Sparse2_4);
    }
    return Flags<WeightFormat>(WeightFormat::Default);
}

bool ArchEthosU85::UseAvgPoolNop(OpType type)
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

unsigned MaskForNpuOp(const EthosU85NpuOp npuOp, bool hasIfm2 = false)
{
    if ( npuOp == EthosU85NpuOp::VectorProduct && hasIfm2 )
    {
        // first bit is reserved for matmul
        return 1;
    }
    return 1 << (int(npuOp));
}

int ArchEthosU85::IndexForOfmUBlock(const Shape &ofmUBlock)
{
    auto it = std::find(_ofmUBlocks.begin(), _ofmUBlocks.end(), ofmUBlock);
    if ( it == _ofmUBlocks.end() )
    {
        LOG_WARN("OFM microblock {} is not supported for this configuration\n", ofmUBlock.ToString());
        assert(false);
    }
    return int(std::distance(_ofmUBlocks.begin(), it));
}

void ArchEthosU85::SetupOfmUBlockToIfmAuTable()
{
    if ( _macs == 128 )
    {
        int b_1x2x8 = IndexForOfmUBlock(Shape(1, 2, 8));
        int b_1x1x16 = IndexForOfmUBlock(Shape(1, 1, 16));
        _uBlockToIfmAuTable[b_1x2x8] = {Shape(1, 2, 1), Shape(1, 1, 2), Shape(1, 1, 2)};
        _uBlockToIfmAuTable[b_1x1x16] = _uBlockToIfmAuTable[b_1x2x8];
    }
    else if ( _macs == 256 )
    {
        int b_2x2x8 = IndexForOfmUBlock(Shape(2, 2, 8));
        int b_1x4x8 = IndexForOfmUBlock(Shape(1, 4, 8));
        int b_1x2x16 = IndexForOfmUBlock(Shape(1, 2, 16));
        _uBlockToIfmAuTable[b_2x2x8] = {Shape(2, 2, 1), Shape(1, 2, 2), Shape(1, 1, 4)};
        _uBlockToIfmAuTable[b_1x2x16] = _uBlockToIfmAuTable[b_2x2x8];
        _uBlockToIfmAuTable[b_1x4x8] = {Shape(1, 4, 1), Shape(1, 2, 2), Shape(1, 1, 4)};
    }
    else if ( _macs == 512 )
    {
        int b_2x2x16 = IndexForOfmUBlock(Shape(2, 2, 16));
        int b_1x4x16 = IndexForOfmUBlock(Shape(1, 4, 16));
        _uBlockToIfmAuTable[b_2x2x16] = {Shape(2, 2, 1), Shape(1, 2, 2), Shape(1, 1, 4)};
        _uBlockToIfmAuTable[b_1x4x16] = {Shape(1, 4, 1), Shape(1, 2, 2), Shape(1, 1, 4)};
    }
    else if ( _macs == 1024 )
    {
        int b_2x2x32 = IndexForOfmUBlock(Shape(2, 2, 32));
        int b_1x4x32 = IndexForOfmUBlock(Shape(1, 4, 32));
        int b_2x4x16 = IndexForOfmUBlock(Shape(2, 4, 16));
        _uBlockToIfmAuTable[b_2x2x32] = {Shape(2, 4, 1), Shape(2, 2, 2), Shape(1, 2, 4)};
        _uBlockToIfmAuTable[b_2x4x16] = _uBlockToIfmAuTable[b_2x2x32];
        _uBlockToIfmAuTable[b_1x4x32] = {Shape(2, 4, 1), Shape(1, 4, 2), Shape(1, 2, 4)};
    }
    else
    {
        int b_2x2x64 = IndexForOfmUBlock(Shape(2, 2, 64));
        int b_1x4x64 = IndexForOfmUBlock(Shape(1, 4, 64));
        int b_4x4x16 = IndexForOfmUBlock(Shape(4, 4, 16));
        _uBlockToIfmAuTable[b_2x2x64] = {Shape(4, 4, 1), Shape(2, 4, 2), Shape(2, 2, 4)};
        _uBlockToIfmAuTable[b_4x4x16] = _uBlockToIfmAuTable[b_2x2x64];
        _uBlockToIfmAuTable[b_1x4x64] = {Shape(4, 4, 1), Shape(2, 4, 2), Shape(1, 4, 4)};
    }
}

void ArchEthosU85::SetupOfmUBlockToOpTable()
{
    unsigned conv = MaskForNpuOp(EthosU85NpuOp::Convolution);
    unsigned depthwise = MaskForNpuOp(EthosU85NpuOp::Depthwise);
    unsigned vectorprod = MaskForNpuOp(EthosU85NpuOp::VectorProduct);
    unsigned pool = MaskForNpuOp(EthosU85NpuOp::Pooling);
    unsigned reducesum = MaskForNpuOp(EthosU85NpuOp::ReduceSum);
    unsigned elementwise = MaskForNpuOp(EthosU85NpuOp::Elementwise);
    unsigned resize = MaskForNpuOp(EthosU85NpuOp::Resize);
    unsigned matmul = MaskForNpuOp(EthosU85NpuOp::VectorProduct, true);
    unsigned dma = MaskForNpuOp(EthosU85NpuOp::Dma, true);

    // clang-format off
    if ( _macs == 128 )
    {
        unsigned b_1x2x8 = IndexForOfmUBlock(Shape(1, 2, 8));
        unsigned b_1x1x16 = IndexForOfmUBlock(Shape(1, 1, 16));
        _uBlockToOpTable[b_1x2x8] = {
            // 8 bit ifm
            conv | matmul | vectorprod | reducesum | elementwise | resize,
            // 16 bit ifm
            conv | matmul | vectorprod | depthwise | pool | reducesum | elementwise | resize,
            // 32 bit ifm
            reducesum | elementwise | resize,
        };
        _uBlockToOpTable[b_1x1x16] = {
            depthwise | pool | elementwise | resize,
            vectorprod | elementwise | resize,
            elementwise | resize
        };
    }
    else if ( _macs == 256 )
    {
        unsigned b_2x2x8 = IndexForOfmUBlock(Shape(2, 2, 8));
        unsigned b_1x4x8 = IndexForOfmUBlock(Shape(1, 4, 8));
        unsigned b_1x2x16 = IndexForOfmUBlock(Shape(1, 2, 16));
        _uBlockToOpTable[b_2x2x8] = {
            conv | matmul | vectorprod | reducesum | elementwise | resize,
            conv | matmul | vectorprod | depthwise | pool | reducesum | elementwise | resize,
            reducesum | elementwise | resize
        };
        _uBlockToOpTable[b_1x4x8] = {
            conv | matmul | vectorprod | reducesum | elementwise | resize,
            conv | matmul | vectorprod | depthwise | pool | reducesum | elementwise | resize,
            reducesum | elementwise | resize
        };
        _uBlockToOpTable[b_1x2x16] = {
            depthwise | pool | elementwise | resize,
            vectorprod | elementwise | resize,
            elementwise | resize
        };
    }
    else if ( _macs == 512 )
    {
        unsigned b_2x2x16 = IndexForOfmUBlock(Shape(2, 2, 16));
        unsigned b_1x4x16 = IndexForOfmUBlock(Shape(1, 4, 16));
        _uBlockToOpTable[b_2x2x16] = {
            conv | depthwise | vectorprod | pool | reducesum | elementwise | resize | matmul,
            conv | depthwise | vectorprod | pool | reducesum | elementwise | resize | matmul,
            reducesum | elementwise | resize,
        };
        _uBlockToOpTable[b_1x4x16] = {
            conv | depthwise | vectorprod | pool | reducesum | elementwise | resize | matmul,
            conv | depthwise | vectorprod | pool | reducesum | elementwise | resize | matmul,
            reducesum | elementwise | resize,
        };
    }
    else if ( _macs == 1024 )
    {
        unsigned b_2x2x32 = IndexForOfmUBlock(Shape(2, 2, 32));
        unsigned b_1x4x32 = IndexForOfmUBlock(Shape(1, 4, 32));
        unsigned b_2x4x16 = IndexForOfmUBlock(Shape(2, 4, 16));
        _uBlockToOpTable[b_2x2x32] = {
            conv | matmul | vectorprod | elementwise,
            conv | matmul | vectorprod | elementwise,
            elementwise,
        };
        _uBlockToOpTable[b_1x4x32] = {
            conv | matmul | vectorprod | elementwise,
            conv | matmul | vectorprod | elementwise,
            elementwise,
        };
        _uBlockToOpTable[b_2x4x16] = {
            conv | vectorprod | depthwise | pool | reducesum | elementwise | resize,
            conv | vectorprod | depthwise | pool | reducesum | elementwise | resize,
            reducesum | elementwise | resize,
        };
    }
    else
    {  // 2048
        unsigned b_2x2x64 = IndexForOfmUBlock(Shape(2, 2, 64));
        unsigned b_1x4x64 = IndexForOfmUBlock(Shape(1, 4, 64));
        unsigned b_4x4x16 = IndexForOfmUBlock(Shape(4, 4, 16));
        _uBlockToOpTable[b_2x2x64] = {
            conv | matmul | vectorprod | elementwise,
            conv | matmul | vectorprod | elementwise,
            elementwise,
        };
        _uBlockToOpTable[b_1x4x64] = {
            conv | matmul | vectorprod | elementwise,
            conv | matmul | vectorprod | elementwise,
            elementwise,
        };
        _uBlockToOpTable[b_4x4x16] = {
            conv | vectorprod | depthwise | pool | reducesum | elementwise | resize,
            conv | vectorprod | depthwise | pool | reducesum | elementwise | resize,
            reducesum | elementwise | resize,
        };
    }
    // clang-format on
}

bool ArchEthosU85::IsUBlockValid(const OpType opType, int ifmBits, const Shape &ofmUBlock, bool hasIfm2)
{
    EthosU85NpuOp npuOp = GetHWOp(opType);
    if ( npuOp == EthosU85NpuOp::None )
    {
        return false;
    }

    unsigned blockIdx = IndexForOfmUBlock(ofmUBlock);
    if ( blockIdx >= _uBlockToOpTable.size() )
    {
        LOG_WARN("OFM microblock {} is not a valid block for Ethos U85-{}\n", ofmUBlock.ToString(), _macs);
        return false;
    }

    auto &bitsToOperations = _uBlockToOpTable[blockIdx];

    unsigned bitIdx = (ifmBits / 16);
    if ( bitIdx >= bitsToOperations.size() )
    {
        LOG_DEBUG("(OFM microblock validation - ifmbits: {} is not a valid ifm precision\n", ifmBits);
        return false;
    }

    // one-hot encoded mask for NpuOp operations
    unsigned opmask = MaskForNpuOp(npuOp, hasIfm2);
    return bitsToOperations[bitIdx] & opmask;
}

bool IsMinMaxReduction(OpType opType, const Kernel *kernel)
{
    // MIN/MAX Reduction over width or height is defined as a MAX/MIN-pool with 1-D kernel.
    return (opType == OpType::MaxPool || opType == OpType::Min) && (kernel->Size().x == 1 || kernel->Size().y == 1);
}

Shape ArchEthosU85::FindUBlock(OpType opType, const ArchitectureConfigQuery &query)
{
    int lookupBits = query.ifmBits;
    if ( IsMinMaxReduction(opType, query.kernel) && lookupBits == 32 )
    {
        // 16-bit microblock lookup-table is used for
        // 32-bit Min/Max reductions.
        lookupBits = 16;
    }

    const EthosU85NpuOp npuOp = GetHWOp(opType);
    assert(npuOp != EthosU85NpuOp::None);

    int bestWaste = std::numeric_limits<int>::max();
    Shape bestUblk;

    for ( int i = 0; i < _nOfmUBlocks; i++ )
    {
        const Shape &ublk = _ofmUBlocks[i];
        if ( !IsUBlockValid(opType, lookupBits, ublk, query.ifmShape[1] != Shape()) )
        {
            continue;
        }

        Shape tmp = Shape::RoundAway(query.ofmShape, ublk);
        int waste = tmp.Elements() - query.ofmShape.Elements();
        if ( waste < bestWaste )
        {
            bestUblk = ublk;
            bestWaste = waste;
        }
    }

    return bestUblk;
}

std::unique_ptr<ArchitectureOpConfig> ArchEthosU85::FindBlockConfig(OpType opType, const ArchitectureConfigQuery &query)
{
    assert(query.ifmBits > 0 && query.ifmBits <= 32);
    assert(query.ofmShape.Size() > 2 && "Insufficient dimensions to search for block config");
    assert(query.kernel != nullptr);

    if ( !SupportsAccumulatorMode(query.accSource, query.accOutputEnabled) ) return nullptr;

    const int OFMSplitDepth = 16;  // Specific to this architecture

    // Elementwise larger-volume correction
    const Shape &ifmShape = (query.ifmShape[1].Elements() > query.ifmShape[0].Elements()) ? query.ifmShape[1] : query.ifmShape[0];

    EthosU85NpuOp npuOp = GetHWOp(opType);
    assert(npuOp != EthosU85NpuOp::None);

    // Operator typing help
    bool isPooling = npuOp == EthosU85NpuOp::Pooling || npuOp == EthosU85NpuOp::ReduceSum;
    bool isReduceSum = npuOp == EthosU85NpuOp::ReduceSum;
    bool isDepthwise = npuOp == EthosU85NpuOp::Depthwise;
    bool isElementwise = npuOp == EthosU85NpuOp::Elementwise;
    bool isConvolution = npuOp == EthosU85NpuOp::Convolution || npuOp == EthosU85NpuOp::Depthwise;
    bool isResize = npuOp == EthosU85NpuOp::Resize;
    bool isDma = npuOp == EthosU85NpuOp::Dma;
    bool isPartKernel = isConvolution && ChooseKernelMethod(ifmShape, query.ifmBits, query.kernel);
    bool isEqualDepthOp = isElementwise || (isPooling && !isReduceSum) || isDepthwise || isResize;

    if ( isDma )
    {
        // DMA ops doesn't use block config
        return nullptr;
    }

    // Operator configuration to be returned
    auto config = std::make_unique<EthosU85OpConfig>();

    EthosU85Traversal traversal = isDepthwise ? EthosU85Traversal::Depthwise : (isPartKernel ? EthosU85Traversal::PartKernel : EthosU85Traversal::DepthFirst);

    // Accumulator settings
    EthosU85Accumulator accType = EthosU85Accumulator::Acc32;
    if ( query.ifmBits == 16 && (!isPooling || isReduceSum) && query.scaled )
    {
        accType = EthosU85Accumulator::Acc48;
    }
    else if ( query.ifmBits == 64 && isPooling )
    {
        // Special case for Rescale int48
        accType = EthosU85Accumulator::Acc48;
    }

    int accBits = AccumulatorBits(accType);
    int rounding;
    int upscale = UpscaleAndRounding(query.ifmResampling, rounding);
    int numBlocksInRam = 2;

    const Shape ofmUBlock = FindUBlock(opType, query);
    if ( ofmUBlock == Shape() )
    {
        // no valid ofm microblock found
        LOG_WARN("Could not find a valid OFM microblock for {} with {}-bit input.\n", OpTypeToString(opType), query.ifmBits);
        return nullptr;
    }

    // Subkernel repeats of the IFM
    Point2i dilatedWH = query.kernel->DilatedWH();
    int ifmRepeats = DivRoundUp(dilatedWH.x, _subkernelMax.Width()) * DivRoundUp(dilatedWH.y, _subkernelMax.Height());

    int ifmBlockDepth = 0;
    const bool sparse = query.weightFormat & WeightFormat::Sparse2_4;
    if ( isPartKernel )
    {
        ifmBlockDepth = 16;
    }
    else if ( query.ifmBits == 32 || ((_macs == 128 || _macs == 256) && ofmUBlock.Depth() == 16 && !sparse) )
    {
        ifmBlockDepth = 32;
    }
    else if ( sparse && traversal == EthosU85Traversal::DepthFirst && query.ifmBits == 8 )
    {
        ifmBlockDepth = 128;
    }
    else
    {
        ifmBlockDepth = 64;
    }

    // Weights fetch (for operators that have them)
    int weightFetchWH = isConvolution ? query.kernel->Size().AreaXY() : 0;

    int ofmUBlockDepth = ofmUBlock.Depth();

    // When using brick format and certain transposes, there are additional constraints to the block size, so we must
    // extend the search space to be able to find a valid block size.
    Shape ofmBlockMin = Shape(0, 0, 0);
    if ( query.ofmFormat == TensorFormat::NHCWB16 )
    {
        switch ( query.transpose )
        {
            case TransposeType::NCHW:
            case TransposeType::NHCW:
                ofmBlockMin = ofmBlockMin.WithWidth(16);
                break;
            case TransposeType::NCWH:
            case TransposeType::NWCH:
                ofmBlockMin = ofmBlockMin.WithHeight(16);
                break;
            default:
                break;
        }
    }
    Shape searchSpaceStep = Shape::Max(ofmUBlock, ofmBlockMin);
    Shape ofmBlockMaxTp = _ofmBlockMax.Untranspose(Reduce4To3(query.transpose));
    Shape searchSpaceEnd = Shape::RoundAway(Shape::Max(Shape::Min(query.ofmShape, ofmBlockMaxTp), searchSpaceStep), ofmUBlock);

    if ( isResize )
    {
        // resize operations are constrained to OFM block height 1 and depth 1-16
        // TODO MLBEDSW-8573: Improve block config search for Resize/Elementwise operations
        int resizeMaxWidth = CalcResizeMaxOfmBlockWidth(query.ifmBits, query.rescaling.scaleX.n, query.rescaling.scaleX.d);
        // reduce minimal step if max width becomes smaller than the minimal step
        if ( resizeMaxWidth < searchSpaceStep.Width() )
        {
            searchSpaceStep = searchSpaceStep.WithWidth(resizeMaxWidth);
        }
        searchSpaceStep = searchSpaceStep.WithHeight(1);
        searchSpaceEnd = searchSpaceEnd.WithHeight(1).WithDepth(16).WithWidth(resizeMaxWidth);
    }

    // At this point, OFM is already configured to NHWC but we need to limit OFM block depth as well.
    if ( query.reverse == ReverseType::C )
    {
        searchSpaceEnd = Shape::Min(searchSpaceEnd, searchSpaceEnd.WithDepth(16));
    }

    // Block WHC search, loops across the search space looking for best efficiency
    float bestCost = std::numeric_limits<float>::infinity();
    float bestCoverage = std::numeric_limits<float>::infinity();
    int ofmElements = query.ofmShape.Elements();

    int depth = std::max(ofmUBlockDepth, std::min(searchSpaceEnd.Depth(), OFMSplitDepth));
    int restartDepth = depth;
    if ( depth < query.ofmShape.Depth() )
    {
        depth = RoundAway(depth, OFMSplitDepth);
    }

    Shape ifmAllocUnit = CalcIfmAUSize(ifmBlockDepth, query.ifmBits, ofmUBlock);

    std::unordered_set<Point2i, Point2Hash<int>> wontFit;
    while ( depth <= searchSpaceEnd.Depth() )
    {
        if ( isEqualDepthOp )
        {
            // For equal depth ops, IFMBlockDepth == OFMBlockDepth
            // Recalculate the IFM AU for the new depth
            ifmBlockDepth = depth;
            ifmAllocUnit = CalcIfmAUSize(depth, query.ifmBits, ofmUBlock);
        }

        for ( int height = searchSpaceStep.Height(); height <= searchSpaceEnd.Height(); height += searchSpaceStep.Height() )
        {
            for ( int width = searchSpaceStep.Width(); width <= searchSpaceEnd.Width(); width += searchSpaceStep.Width() )
            {
                // Avoid checking W/H transposed blocks that already didn't fit. i.e. if 8x4x16 didn't
                // fit, then 4x8x16 won't either.
                if ( wontFit.count(Point2i(height, width)) > 0 )
                {
                    continue;
                }

                // Calculate the IFM block dimensions required to feed this OFM block
                Shape ofmBlock = Shape(height, width, depth);

                Shape ifmBlock = GetArchIFMBlockSize(ofmBlock, query.kernel, ifmAllocUnit, _subkernelMax, upscale, rounding);
                ifmBlock = ifmBlock.WithDepth(ifmBlockDepth);

                // Test if the IFM/OFM blocks fit into RAM
                if ( TryBlockConfig(npuOp, ofmBlock, ifmBlock, ifmShape, query.ifmBits, accBits, _ifmRamSizeBytes,
                         _accRamSizeBytes, ifmAllocUnit.Depth(), numBlocksInRam, isEqualDepthOp) )
                {
                    Shape fullBlocks = Shape::DivRoundUp(query.ofmShape, ofmBlock);
                    Point3<float> blocks = query.ofmShape.HWC<float>() / ofmBlock.HWC<float>();

                    // Weights fetching
                    float weightFetch = float(weightFetchWH) * ifmShape.Depth() * fullBlocks.ElementsWH();
                    if ( !isDepthwise )
                    {
                        weightFetch *= blocks.z * ofmBlock.Depth();
                    }

                    // IFM fetching
                    float ifmFetch = float(ifmBlock.ElementsWH()) * ifmShape.Depth() * ifmRepeats * blocks.x * blocks.y;
                    if ( !isEqualDepthOp )
                    {
                        ifmFetch *= fullBlocks.Depth();
                    }

                    // Scale relative to every output OFM element
                    float relativeCost =
                        (isElementwise || isResize) ? float(ofmElements) / (float(height) * width * depth) : (ifmFetch + weightFetch) / float(ofmElements);

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
                            config->_ifmBlock = std::move(ifmBlock);
                            config->_ofmBlock = Shape(1, height, width, depth);
                        }
                    }
                }
                else
                {
                    wontFit.emplace(width, height);
                }
            }
        }

        // Try Next block depth, rounded
        depth = depth + ofmUBlockDepth;
        if ( depth < query.ofmShape.Depth() )
        {
            depth = RoundAway(depth, OFMSplitDepth);
        }
        if ( depth > searchSpaceEnd.Depth() && bestCost == std::numeric_limits<float>::infinity() && numBlocksInRam == 2 )
        {
            numBlocksInRam = 1;
            depth = restartDepth;
        }
    }

    config->_ofmUBlock = std::move(ofmUBlock);
    config->_accumulatorType = accType;
    config->_accumulatorSource = query.accSource;
    config->_accumulatorOutputEnabled = query.accOutputEnabled;
    config->_ifmRamSizeBytes = _ifmRamSizeBytes;
    config->_traversal = traversal;

    // Return the best configuration
    if ( bestCost != std::numeric_limits<float>::infinity() )
    {
        return std::unique_ptr<ArchitectureOpConfig>(config.release());
    }

    // Didn't find a configuration
    return nullptr;
}

Shape ArchEthosU85::CalcIfmAUSize(int ifmBlkDepth, int ifmBits, Shape ofmUBlk)
{
    int ifmu = 0;
    int ifmDepthBits = ifmBlkDepth * ifmBits;
    if ( ifmDepthBits > 256 )
    {
        // ifmu3
        ifmu += 2;
    }
    else if ( ifmDepthBits > 128 )
    {
        // ifmu2
        ifmu++;
    }
    assert(ifmu < 3);
    unsigned blockIdx = IndexForOfmUBlock(ofmUBlk);
    return _uBlockToIfmAuTable[blockIdx][ifmu];
}

int ArchEthosU85::CalcResizeMaxOfmBlockWidth(int ifmBits, int scaleN, int scaleD)
{
    // Calculate the maximum OfmBlockWidth that still allows
    // the IFM block to fit in the chaining buffer
    assert(scaleN > 0);
    assert(scaleD > 0);
    int numIfmCbSlots = _macs / 16;
    if ( ifmBits == 16 )
    {
        numIfmCbSlots /= 2;
    }
    int maxOfmBlkW = int(std::ceil(((numIfmCbSlots - 2) * scaleN + 1) / double(scaleD)));
    maxOfmBlkW = std::max(1, std::min(maxOfmBlkW, _ofmBlockMax.Width()));
    return maxOfmBlkW;
}

bool ArchEthosU85::TryBlockConfig(EthosU85NpuOp npuOp, const Shape &ofmBlock, const Shape &ifmBlock, const Shape &ifmShape,
    int ifmBits, int accBits, int ifmSpace, int accSpace, int ifmAuDepth, int numBlocksInRam, bool isEqualDepthOp)
{
    assert(accBits > 0);
    assert((ifmBits >= 8) && ((ifmBits % 8) == 0));

    // Elementwise and Resize don't use IB/AB.
    if ( npuOp == EthosU85NpuOp::Elementwise || npuOp == EthosU85NpuOp::Resize )
    {
        return true;
    }

    // IFM Space
    int ifmAlignDepth = ifmAuDepth * 128 / ifmBits;
    int ifmBlockDepth = isEqualDepthOp ? ofmBlock.Depth() : std::min(ifmBlock.Depth(), ifmShape.Depth());
    ifmBlockDepth = RoundAway(ifmBlockDepth, ifmAlignDepth);
    int ifmBytes = ifmBlock.ElementsWH() * ifmBlockDepth * (ifmBits / 8) * numBlocksInRam;

    // Accumulator space
    int ofmBlockDepth = RoundAway(ofmBlock.Depth(), 16);
    int accBytes = (ofmBlock.ElementsWH() * ofmBlockDepth * accBits) / 8 * numBlocksInRam;

    if ( ifmBytes > ifmSpace || accBytes > accSpace )
    {
        return false;
    }

    return true;
}


Shape ArchEthosU85::GetStorageRounding(TensorFormat format)
{
    if ( format == TensorFormat::NHCWB16 )
    {
        return Shape(1, 1, 1, 16);
    }

    return Shape(1, 1, 1, 1);
}

uint32_t ArchEthosU85::ConfigRegister(int product)
{
    uint32_t macsLog2 = IntLog2(_macs);
    uint32_t numWdLog2 = IntLog2(_cores);

    return EthosU85RCSGenerator::ConfigRegister(macsLog2, 1, _numAxiSramLog2, _numAxiExtLog2, numWdLog2, product);
}


std::unique_ptr<ArchitectureOpConfig> EthosU85OpConfig::Clone()
{
    auto config = std::make_unique<EthosU85OpConfig>();
    config->_ifmRamSizeBytes = _ifmRamSizeBytes;
    config->_traversal = _traversal;
    config->_accumulatorType = _accumulatorType;
    config->_accumulatorSource = _accumulatorSource;
    config->_accumulatorOutputEnabled = _accumulatorOutputEnabled;
    config->_ofmBlock = _ofmBlock;
    config->_ofmUBlock = _ofmUBlock;
    config->_ifmBlock = _ifmBlock;
    return std::unique_ptr<ArchitectureOpConfig>(config.release());
}

int EthosU85OpConfig::MaxIFMBuffering()
{
    return _ifmRamSizeBytes;
}

Point2i EthosU85OpConfig::OptimalStripeGranule()
{
    return _ofmBlock.WH<int>();
}

int EthosU85OpConfig::OptimalDepthGranule()
{
    return _ofmBlock.Depth();
}

std::string EthosU85OpConfig::ToString(bool full)
{
    std::string tmp = fmt::format("OFM Block=[{}], IFM Block=[{}], OFM UBlock=[{}] Traversal={}, AccType={}", _ofmBlock.ToString(),
        _ifmBlock.ToString(), _ofmUBlock.ToString(), EnumToString(_traversal), EnumToString(_accumulatorType));
    UNUSED(full);
    return tmp;
}

EthosU85NpuOp ArchEthosU85::GetHWOp(OpType type)
{
    static const std::unordered_map<OpType, EthosU85NpuOp> toNpuOp = {
        {OpType::DepthwiseConv2DBias, EthosU85NpuOp::Depthwise},
        {OpType::Conv2D, EthosU85NpuOp::Convolution},
        {OpType::Conv2DBackpropInput, EthosU85NpuOp::Convolution},
        {OpType::Conv2DBackpropInputSwitchedBias, EthosU85NpuOp::Convolution},
        {OpType::Conv2DBias, EthosU85NpuOp::Convolution},
        {OpType::ReduceSum, EthosU85NpuOp::ReduceSum},
        {OpType::FullyConnected, EthosU85NpuOp::VectorProduct},
        {OpType::MatMul, EthosU85NpuOp::VectorProduct},
        {OpType::MaxPool, EthosU85NpuOp::Pooling},
        {OpType::AvgPool, EthosU85NpuOp::Pooling},
        {OpType::QuantizedAvgPool, EthosU85NpuOp::Pooling},
        {OpType::QuantizedMaxPool, EthosU85NpuOp::Pooling},
        {OpType::Sum, EthosU85NpuOp::Pooling},
        {OpType::Min, EthosU85NpuOp::Pooling},
        {OpType::ArgMax, EthosU85NpuOp::Pooling},
        // TODO MLBEDSW-7986 add none pooling
        {OpType::Resize, EthosU85NpuOp::Resize},
        {OpType::Gather, EthosU85NpuOp::Dma},
        {OpType::Scatter, EthosU85NpuOp::Dma},
    };

    auto pos = toNpuOp.find(type);
    if ( pos != toNpuOp.end() )
    {
        return pos->second;
    }
    else if ( EthosU85RCSGenerator::IsSupportedElementwise(type) )
    {
        return EthosU85NpuOp::Elementwise;
    }
    else if ( UseAvgPoolNop(type) )
    {
        return EthosU85NpuOp::Pooling;
    }
    return EthosU85NpuOp::None;
}

// TODO: this is activation fusing only
int EthosU85OpGroup::Add(const ArchitectureOpGroupQuery &op, const std::vector<int> &dependsOn)
{
    LOG_TRACE1("Trying to add op {}\n", OpTypeToString(op.type));

    if ( _opsCount >= 2 )
    {
        // Can only fuse 2 ops
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

        const EthosU85OpGroup::OpInfo &prevOp = _ops[dep];
        if ( prevOp.ofm.key != op.ifm.key && prevOp.ofm.key != op.ifm2.key )
        {
            // Can only fuse when ops are connected
            return 0;
        }
    }
    if ( !CanRunOnNPU(op) )
    {
        // Can only fuse NPU ops
        return 0;
    }

    if ( _opsCount > 0 )
    {
        if ( !IsActivation(op.type) )
        {
            // Can only fuse with activation
            return 0;
        }
        else if ( op.ifm.type == DataType::Int16 && (op.type == OpType::Sigmoid || op.type == OpType::Tanh) )
        {
            // Can not fuse int16 Sigmoid and Tanh LUT since they require special scaling done by AvgPoolNop
            return 0;
        }
    }

    // Generated key
    int key = (-_opsCount) - 1;

    // Save copy of op
    _ops[_opsCount].type = op.type;
    _ops[_opsCount].ifm.key = op.ifm.key;
    _ops[_opsCount].ifm.type = op.ifm.type;
    _ops[_opsCount].ifm2.key = op.ifm2.key;
    _ops[_opsCount].ifm2.type = op.ifm2.type;
    _ops[_opsCount].ofm.key = op.ofm.key;
    _ops[_opsCount].ofm.type = op.ofm.type;
    _opsInternal[_opsCount].dependsOn = dependsOn;
    _opsCount++;

    return key;
}

// TODO: This table is from the EthosU55/U65 Embedded NPU Interface Specification, it's not completely valid for
// Ethos U85 since the allowed data types depend on ifm/ofm as well as selected acc and scaling.
static const std::unordered_map<EthosU85NpuOp, std::unordered_map<DataType, std::vector<DataType>>> s_opDataTypeSupport = {
    {EthosU85NpuOp::Convolution,
        {
            {DataType::UInt8, {DataType::UInt8, DataType::Int8, DataType::Int16, DataType::Int32, DataType::Int64}},
            {DataType::Int8, {DataType::UInt8, DataType::Int8, DataType::Int16, DataType::Int32, DataType::Int64}},
            {DataType::Int16, {DataType::UInt8, DataType::Int8, DataType::Int16, DataType::Int32, DataType::Int64}},
        }},
    {EthosU85NpuOp::Depthwise,
        {
            {DataType::UInt8, {DataType::UInt8, DataType::Int8, DataType::Int16, DataType::Int32, DataType::Int64}},
            {DataType::Int8, {DataType::UInt8, DataType::Int8, DataType::Int16, DataType::Int32, DataType::Int64}},
            {DataType::Int16, {DataType::UInt8, DataType::Int8, DataType::Int16, DataType::Int32, DataType::Int64}},
        }},
    {EthosU85NpuOp::VectorProduct,
        {
            {DataType::UInt8, {DataType::UInt8, DataType::Int8, DataType::Int16, DataType::Int32, DataType::Int64}},
            {DataType::Int8, {DataType::UInt8, DataType::Int8, DataType::Int16, DataType::Int32, DataType::Int64}},
            {DataType::Int16, {DataType::UInt8, DataType::Int8, DataType::Int16, DataType::Int32, DataType::Int64}},
        }},
    {EthosU85NpuOp::Pooling,
        {
            {DataType::UInt8, {DataType::UInt8, DataType::Int32, DataType::Int64}},
            {DataType::Int8, {DataType::Int8, DataType::Int32, DataType::Int64}},
            {DataType::Int16, {DataType::Int16}},
        }},
    {EthosU85NpuOp::ReduceSum,
        {
            {DataType::UInt8, {DataType::UInt8, DataType::Int8, DataType::Int16, DataType::Int32}},
            {DataType::Int8, {DataType::UInt8, DataType::Int8, DataType::Int16, DataType::Int32}},
            {DataType::Int16, {DataType::UInt8, DataType::Int8, DataType::Int16, DataType::Int32}},
            {DataType::Int32, {DataType::UInt8, DataType::Int8, DataType::Int16, DataType::Int32}},
        }},
    {EthosU85NpuOp::Dma,
        {
            {DataType::UInt8, {DataType::UInt8}},
            {DataType::Int8, {DataType::Int8}},
            {DataType::Int16, {DataType::Int16}},
            {DataType::Int32, {DataType::Int32}},
        }},
    {EthosU85NpuOp::Resize,
        {
            {DataType::UInt8, {DataType::UInt8, DataType::Int8, DataType::Int16, DataType::Int32, DataType::Int64}},
            {DataType::Int8, {DataType::UInt8, DataType::Int8, DataType::Int16, DataType::Int32, DataType::Int64}},
            {DataType::Int16, {DataType::UInt8, DataType::Int8, DataType::Int16, DataType::Int32, DataType::Int64}},
        }},
};

bool EthosU85OpGroup::CanRunOnNPU(const ArchitectureOpGroupQuery &op)
{
    EthosU85NpuOp npuOp = ArchEthosU85::GetHWOp(op.type);

    if ( IsFloat(op.ifm.type | op.ifm2.type | op.ofm.type) )
    {
        return false;
    }

    if ( npuOp == EthosU85NpuOp::None )
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

    switch ( npuOp )
    {
        case EthosU85NpuOp::Convolution:
        case EthosU85NpuOp::Depthwise:
        case EthosU85NpuOp::VectorProduct:
        case EthosU85NpuOp::Pooling:
        case EthosU85NpuOp::ReduceSum:
        case EthosU85NpuOp::Elementwise:
        case EthosU85NpuOp::Resize:
        case EthosU85NpuOp::Dma:
            break;
        default:
            assert(false && "Unrecognized HWOp");
            return false;
    }

    // Check allowed ifm/ofm data type mapping
    if ( npuOp != EthosU85NpuOp::Elementwise )
    {
        if ( op.type == OpType::LUT || op.type == OpType::MemoryCopy )
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
        auto ifmEntry = typeMap.find(op.ifm.type);
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
        // TODO: Elementwise
    }

    return true;
}

}  // namespace regor
