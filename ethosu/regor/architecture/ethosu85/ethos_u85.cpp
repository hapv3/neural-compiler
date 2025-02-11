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

#include "ethos_u85.hpp"

#include "common/common.hpp"
#include "common/logging.hpp"

#include "common/bit_flags.hpp"
#include "common/numeric_util.hpp"
#include "ethos_u85_constraints.hpp"
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

static const EthosU85PerfInfo s_EthosU85PerfInfo[] = {
    // Accelerator.Ethos_U85_128
    {{0.5, 0.5}, {0.5, 0.5, 0.0}},
    // Accelerator.Ethos_U85_256
    {{0.25, 0.25}, {0.25, 0.25, 0.0}},
    // Accelerator.Ethos_U85_512
    {{0.125, 0.125}, {0.125, 0.125, 0.0}},
    // Accelerator.Ethos_U85_1024
    {{0.0625, 0.125}, {0.0625, 0.0625, 0.0}},
    // Accelerator.Ethos_U85_2048
    {{0.03125, 0.0625}, {0.0625, 0.03125, 0.0}},
};

static const ArchEthosU85::AcceleratorConfig s_EthosU85Configs[] = {
    // Accelerator.Ethos_U85_128
    {128, 1, {Shape(1, 2, 8), Shape(1, 1, 16)}, Shape(1, 2, 8), 2, 8192, 8192, 2048, 768, 1, 0, {64, 64, 128, 128, 104}, &s_EthosU85PerfInfo[0]},
    // Accelerator.Ethos_U85_256
    {256, 1, {Shape(1, 2, 16), Shape(1, 4, 8), Shape(2, 2, 8)}, Shape(2, 2, 8), 3, 16384, 16384, 2048, 1536, 1, 0,
        {104, 104, 128, 128, 128}, &s_EthosU85PerfInfo[1]},
    // Accelerator.Ethos_U85_512
    {512, 2, {Shape(2, 2, 16), Shape(1, 4, 16)}, Shape(2, 2, 16), 2, 16384, 32768, 4096, 3072, 1, 0, {128, 128, 256, 256, 128}, &s_EthosU85PerfInfo[2]},
    // Accelerator.Ethos_U85_1024
    {1024, 4, {Shape(2, 2, 32), Shape(1, 4, 32), Shape(2, 4, 16)}, Shape(4, 2, 16), 3, 16384, 65536, 4096, 6144, 1, 1,
        {256, 256, 416, 208, 256}, &s_EthosU85PerfInfo[3]},
    // Accelerator.Ethos_U85_2048
    {2048, 4, {Shape(2, 2, 64), Shape(1, 4, 64), Shape(4, 4, 16)}, Shape(4, 4, 16), 3, 32768, 131072, 8192, 12288, 2, 1,
        {256, 256, 512, 256, 256}, &s_EthosU85PerfInfo[4]},
};

constexpr int CB_SLOTS = 6;
constexpr int BRICK_ELEMENTS = 16;
// max size for tensors
const static Shape MAX_SHAPE(nullptr, 8, 65536);
constexpr int ACC_DEPTH_GRANULE = 16;  // Accumulator depth granularity

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
    _constraints = std::make_unique<EthosU85Constraints>(this);
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
    _cbRamSizeBytes = cfg->cbRamSizeBytes;
    _obRamSizeBytes = cfg->obRamSizeBytes;
    _numAxiSramLog2 = cfg->numAxiSramLog2;
    _numAxiExtLog2 = cfg->numAxiExtLog2;

    _lutRam = std::make_unique<ArchitectureMemory>("lutram", 2048);
    // TODO MLBEDSW-7980 fix LUT performance parameters
    _lutRam->SetParameters(1, 0, 0, 1, 1, 1000, 1000);
    _lutMemory = _lutRam.get();
    _performance = std::unique_ptr<ArchitecturePerformance>(new EthosU85Performance(this, cfg->perfInfo));
    _channelRBs = &cfg->channelRBs;
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

    auto group = std::make_unique<EthosU85OpGroup>(this);
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

AxisMask ArchEthosU85::CanSubdivide(OpType opType, TransposeType transpose, ReverseType reverse)
{
    if ( (opType == OpType::FullyConnected || IsConvolution(opType) || IsElementwise(opType) || IsPooling(opType)) &&
         IsNone(transpose) && (reverse != ReverseType::H) )
    {
        return AxisMask::AxisY;
    }
    return AxisMask::None;
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
    return IsActivation(type) || type == OpType::Quantize || type == OpType::MemoryCopy || type == OpType::Transpose || type == OpType::Reverse;
}

bool ArchEthosU85::UseNullPool(OpType opType, int bits)
{
    return ((opType == OpType::MemoryCopy || opType == OpType::Rescale || opType == OpType::Transpose || opType == OpType::Reverse) && bits >= 32);
}

// Return true if kernel first mode has a better utilisation than depth first mode
static bool ChooseKernelFirst(const Shape &ifmShape, const Kernel *kernel, bool sparse)
{
    int k = kernel->ElementsWH();
    int s = sparse ? 10 : 5;
    int r = sparse ? 4 : 2;
    int k_rnd = (k / s) * s + std::max(k % s, r);
    double kernelFirstUtilisation = (ifmShape.Depth() / double(RoundAway(ifmShape.Depth(), 16)) * (k / double(k_rnd)));
    double depthFirstUtilisation = ifmShape.Depth() / double(RoundAway(ifmShape.Depth(), sparse ? 128 : 64));
    return kernelFirstUtilisation >= depthFirstUtilisation;
}

static Shape GetArchIFMBlockSize(const Shape &ofmBlock, const Kernel *kernel, const Shape &auBlock,
    const Shape &subkernelLimit, int upscale, int rounding, int ifmBlockDepth)
{
    Point2i dilatedSize = kernel->DilatedWH();

    // IFM block height
    int h = RequiredInputSize(ofmBlock.Height(), kernel->Stride().y, std::min(dilatedSize.y, subkernelLimit.Height()), upscale, rounding);
    h = RoundAway(h, auBlock.Height());

    // IFM block width
    int w = RequiredInputSize(ofmBlock.Width(), kernel->Stride().x, std::min(dilatedSize.x, subkernelLimit.Width()), upscale, rounding);
    w = RoundAway(w, auBlock.Width());

    return Shape(1, h, w, RoundAway(ifmBlockDepth ? ifmBlockDepth : ofmBlock.Depth(), auBlock.Depth()));
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
        return 0;
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
    unsigned reduceminmax = MaskForNpuOp(EthosU85NpuOp::ReduceMinMax);
    unsigned argmax = MaskForNpuOp(EthosU85NpuOp::ArgMax);
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
            conv | matmul | vectorprod | depthwise | pool | reducesum | elementwise | reduceminmax | argmax | resize,
            // 32 bit ifm
            reducesum | elementwise | reduceminmax | resize,
        };
        _uBlockToOpTable[b_1x1x16] = {
            depthwise | pool | elementwise | reduceminmax | argmax | resize,
            conv | vectorprod | elementwise | resize,  // convolution 1x1 kernel 16 bit ifm
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
            conv | matmul | vectorprod | depthwise | pool | reducesum | elementwise | reduceminmax | argmax | resize,
            reducesum | elementwise | reduceminmax | resize
        };
        _uBlockToOpTable[b_1x4x8] = {
            conv | matmul | vectorprod | reducesum | elementwise | resize,
            conv | matmul | vectorprod | depthwise | pool | reducesum | elementwise | reduceminmax | argmax | resize,
            reducesum | elementwise | reduceminmax | resize
        };
        _uBlockToOpTable[b_1x2x16] = {
            depthwise | pool | elementwise | reduceminmax | argmax | resize,
            conv | vectorprod | elementwise | resize,  // convolution 1x1 kernel 16 bit ifm
            elementwise | resize
        };
    }
    else if ( _macs == 512 )
    {
        unsigned b_2x2x16 = IndexForOfmUBlock(Shape(2, 2, 16));
        unsigned b_1x4x16 = IndexForOfmUBlock(Shape(1, 4, 16));
        _uBlockToOpTable[b_2x2x16] = {
            conv | depthwise | vectorprod | pool | reducesum | elementwise | reduceminmax | argmax | resize | matmul,
            conv | depthwise | vectorprod | pool | reducesum | elementwise | reduceminmax | argmax | resize | matmul,
            reducesum | elementwise | reduceminmax | resize,
        };
        _uBlockToOpTable[b_1x4x16] = {
            conv | depthwise | vectorprod | pool | reducesum | elementwise | reduceminmax | argmax | resize | matmul,
            conv | depthwise | vectorprod | pool | reducesum | elementwise | reduceminmax | argmax | resize | matmul,
            reducesum | elementwise | reduceminmax | resize
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
            conv | vectorprod | depthwise | pool | reducesum | elementwise | reduceminmax | argmax | resize,
            conv | vectorprod | depthwise | pool | reducesum | elementwise | reduceminmax | argmax | resize,
            reducesum | elementwise | reduceminmax | resize,
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
            conv | vectorprod | depthwise | pool | reducesum | elementwise | reduceminmax | argmax | resize,
            conv | vectorprod | depthwise | pool | reducesum | elementwise | reduceminmax | argmax | resize,
            reducesum | elementwise | reduceminmax | resize,
        };
    }
    // clang-format on
}

bool ArchEthosU85::IsUBlockValid(const OpType opType, int ifmBits, const Shape &ofmUBlock, bool hasIfm2, bool depthFirst1x1)
{
    EthosU85NpuOp npuOp = GetHWOp(opType);
    if ( npuOp == EthosU85NpuOp::None )
    {
        return false;
    }

    if ( UseNullPool(opType, ifmBits) )
    {
        // Implemented by none pooling op with IFM set to int8 (not used by the operation) and
        // input instead handled by ArchAccumulatorSource::Ifm2
        ifmBits = 8;
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

    // check special case 1x1 kernel convolution for 128 and 256
    if ( _macs == 128 && npuOp == EthosU85NpuOp::Convolution )
    {
        if ( ofmUBlock == Shape(1, 1, 16) && !depthFirst1x1 )
        {
            return false;
        }
    }
    else if ( _macs == 256 && npuOp == EthosU85NpuOp::Convolution )
    {
        if ( ofmUBlock == Shape(1, 2, 16) && !depthFirst1x1 )
        {
            return false;
        }
    }

    // one-hot encoded mask for NpuOp operations
    unsigned opmask = MaskForNpuOp(npuOp, hasIfm2);
    return bitsToOperations[bitIdx] & opmask;
}

Shape ArchEthosU85::FindUBlock(OpType opType, const ArchitectureConfigQuery &query, bool partKernel)
{
    const EthosU85NpuOp npuOp = GetHWOp(opType);
    const bool depthFirst1x1 = (query.kernel->Size().x == 1 && query.kernel->Size().y == 1) && !partKernel;
    assert(npuOp != EthosU85NpuOp::None);

    int bestWaste = std::numeric_limits<int>::max();
    Shape bestUblk;

    for ( int i = 0; i < _nOfmUBlocks; i++ )
    {
        const Shape &ublk = _ofmUBlocks[i];
        if ( !IsUBlockValid(opType, query.ifmBits, ublk, !!query.ifmShape[1], depthFirst1x1) )
        {
            continue;
        }

        // Minimum waste is better than aspect correct
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

static int GranularScale(int range, int granule, double ratio)
{
    assert(granule > 0);
    int granules = range / granule;
    granules = std::max(int(granules * ratio), 0);
    return granules * granule;
}

static int GranularTile(int range, int granule, int tile)
{
    assert(range >= 0 && granule > 0 && tile > 0);
    assert((tile % granule) == 0 && "tile must be multiple of granule");
    if ( range % tile == 0 ) return tile;
    int tiles = range / tile;
    return std::max(RoundAway(range / (tiles + 1), granule), granule);
}

// value    - how much we already have
// maxLimit - maximum amount that we want
// avail    - how much X we have available to reapportion
// required - lower limit on how much X we can give away
static int Reapportion(int value, int maxLimit, int &avail, int required)
{
    assert(required > 0 && value > 0);
    int excess = std::min(avail / required, maxLimit / value);
    if ( excess >= 2 )
    {
        avail /= excess;
        value *= excess;
    }
    return value;
}

static void FitAreaByAspect(double aspect, int &x, int &y, int fitInto, const Point2i &granule)
{
    assert(aspect);
    double w = std::sqrt(fitInto / aspect);
    double h = w * aspect;
    if ( h < 1.0 )
    {
        w *= h;
        h = 1;
    }
    else if ( w < 1.0 )
    {
        h *= w;
        w = 1;
    }
    x = RoundZero(std::max(int(w), granule.x), granule.x);
    y = RoundZero(std::max(int(h), granule.y), granule.y);
}

Shape ArchEthosU85::FindElementwiseConfig(const ArchitectureConfigQuery &query, const FindConfigCommon &common)
{
    LOG_TRACE2("Elementwise: OFM {}, ifm[0]={}, ifm[1]={}\n", query.ofmShape.ToString(), query.ifmShape[0].ToString(),
        query.ifmShape[1].ToString());
    LOG_INDENT(Logging::Out);
    assert(query.ifmBits >= 8);
    assert(common.granule.Depth() && common.ublock.Height());
    const Shape ofmShape = Shape::PadAxes(Shape::RoundAway(query.ofmShape, common.granule), 3, 1);
    Shape ofmBlockLimit = Shape::Min(ofmShape, common.ofmBlockMax);

    const bool isScalar = (query.ifmShape[0].Elements() == 1) || (query.ifmShape[1] && query.ifmShape[1].Elements() == 1);
    const int cbBricks = (_cbRamSizeBytes / CB_SLOTS) / (BRICK_ELEMENTS * (query.ifmBits / 8));
    const int obElements = (_obRamSizeBytes * 8) / query.ifmBits;
    // Width units are still 1-unit wide but represent a proportion of the buffer allocation
    const int obWidthUnits = obElements / common.granule.Depth();  // One row of depth granules
    const int cbWidthUnits = cbBricks / common.ublock.Height();    // One row of ublock-height pixels

    // Determine how to tile the block into the ofm shape. Constrain to the smaller of the ofm shape,
    // block limits and the output buffer size, on the given granularity.
    int hLimit = GranularTile(ofmShape.Height(), common.granule.Height(), ofmBlockLimit.Height());
    int wRequired = GranularTile(ofmShape.Width(), common.granule.Width(), std::min(obWidthUnits, ofmBlockLimit.Width()));
    int wLimit = obWidthUnits;
    int cLimit = Reapportion(common.granule.Depth(), ofmBlockLimit.Depth(), wLimit, std::max(wRequired, common.granule.Width()));

    // Binary elementwise, potentially broadcast
    if ( !isScalar && query.ifmShape[1] )
    {
        unsigned broadcastMask = query.ifmShape[0].LessMask(query.ofmShape);
        broadcastMask |= query.ifmShape[1].LessMask(query.ofmShape);

        // Broadcast in depth first
        if ( broadcastMask & 1 )
        {
            cLimit = GranularTile(ofmShape.Depth(), common.granule.Depth(), ofmBlockLimit.Depth());
            wLimit = cbWidthUnits;
            // This is only the depth axis
            if ( (broadcastMask & 1) == broadcastMask )
            {
                hLimit = Reapportion(common.ublock.Height(), ofmBlockLimit.Height(), wLimit,
                    std::max(wRequired, common.granule.Width()));
            }
        }
        // Broadcast in height
        else if ( broadcastMask & 4 )
        {
            wLimit = cbWidthUnits;
            cLimit = Reapportion(BRICK_ELEMENTS, ofmBlockLimit.Depth(), wLimit, std::max(wRequired, common.granule.Width()));
        }
        // Broadcast in width
        else if ( broadcastMask & 2 )
        {
            // No change
        }
    }

    Shape ofmBlock(1, hLimit, wLimit, cLimit);
    ofmBlock = Shape::RoundAway(ofmBlock, common.granule);
    LOG_TRACE2("Elementwise choice: ofmBlock = {}\n", ofmBlock.ToString());
    return ofmBlock;
}

static int BestTile(int range, int granule, int tile)
{
    assert(range >= 0 && granule > 0 && tile > 0);
    assert((tile % granule) == 0 && "tile must be multiple of granule");

    if ( range % tile == 0 ) return tile;
    int tiles = range / tile;
    int tile2 = std::max(RoundAway(range / (tiles + 1), granule), granule);
    int bestTile = tile;
    int bestRem = range % tile;
    int minTile = std::max(tile2 - (granule * 2), (tile2 + 1) / 2);
    int first = std::max(minTile, granule);
    for ( int t = tile; t >= first; t -= granule )
    {
        // Use absolute remainder, not % - we're interested
        // in comparing units of work, not relative waste.
        int xrem = range % t;
        if ( xrem == 0 ) xrem = t;
        if ( xrem > bestRem )
        {
            bestTile = t;
            bestRem = xrem;
        }
    }

    return bestTile;
}


Shape ArchEthosU85::AreaFit(const FindConfigCommon &common, const Shape &ofmShape, const Shape &ofmBlockLimit,
    const Shape &ifmShape, const Kernel *kernel)
{
    const int accElements = (_accRamSizeBytes * 8) / common.accBits;
    const int ibElements = (_ifmRamSizeBytes * 8) / common.ifmBits;
    const int ubAccElements = common.ublock.ElementsWH() * ACC_DEPTH_GRANULE;

    double aspect = double(ofmShape.Height()) / ofmShape.Width();
    bool prioritiseDepth = kernel->DilatedWH() == Point2i(1, 1);

    Point2i granule = common.granule.WH<int>();
    Shape fitShape;
    double bestMetric = std::numeric_limits<double>::max();
    int maxDepth = std::min(std::max(_macs, accElements / ubAccElements), ofmBlockLimit.Depth());
    double ofmArea = prioritiseDepth ? ofmShape.Width() * ofmShape.Depth() : ofmShape.Width() * ofmShape.Height();

    for ( int depth = ACC_DEPTH_GRANULE; (depth <= maxDepth); depth += ACC_DEPTH_GRANULE )
    {
        int width = 0, height = 0;
        int fitAcc = accElements;
        Shape ifmAllocUnit = CalcIfmAUSize(common.ifmBlockDepth ? common.ifmBlockDepth : depth, common.ifmBits, common.ublock);
        int ifmDepthGranule = ifmAllocUnit.Depth();
        int ifmVolume = Shape::RoundAway(ifmShape, ifmAllocUnit).Elements();
        bool fitted = false;
        int prevAccReq = -1;
        int retry = 25;
        while ( true )
        {
            FitAreaByAspect(aspect, width, height, fitAcc / depth, granule);
            width = std::min(width, ofmBlockLimit.Width());
            height = std::min(height, ofmBlockLimit.Height());

            // Regular width tiling is preferred, if possible
            int tmp = width * height;
            if ( width < ofmShape.Width() ) width = BestTile(ofmShape.Width(), common.granule.Width(), width);
            height = std::max(RoundZero(tmp / width, common.granule.Height()), common.granule.Height());

            // Accumulator Fit
            int accRequired = width * height * depth;
            double abRatio = double(accElements) / accRequired;

            // IFM Fit
            Shape ifmReq = GetArchIFMBlockSize(
                Shape(height, width, depth), kernel, ifmAllocUnit, _subkernelMax, 1, 0, common.ifmBlockDepth);
            int ibRequired = ifmReq.Elements();
            assert(ibRequired);
            ibRequired = std::min(ibRequired, ifmVolume);
            double ibRatio = double(ibElements) / ibRequired;

            if ( abRatio >= 1.0 && ibRatio >= 1.0 )
            {
                int ifmUsed = (depth % ifmDepthGranule) ? (depth % ifmDepthGranule) : ifmDepthGranule;
                double waste = 1.0 + (double(ifmDepthGranule - ifmUsed) / ifmDepthGranule) / 10;
                double fit = 1.0 + std::abs(1.0 - (double(ofmBlockLimit.Depth()) / depth)) / 10;
                int otherAxis = prioritiseDepth ? depth : height;  // Prioritise appropriate axis
                double coverage = 1.0 + std::abs(1.0 - (ofmArea / (width * otherAxis)));
                double metric = coverage * fit * waste;
                if ( (metric < bestMetric) )
                {
                    fitShape = Shape(height, width, depth);
                    bestMetric = metric;

                    // If it covers the entire OFM we can stop early
                    if ( depth >= ofmShape.Depth() && (width >= ofmShape.Width() && height >= ofmShape.Height()) )
                    {
                        fitShape = Shape::RoundAway(ofmShape, common.granule);
                        return fitShape;
                    }
                }
                fitted = true;
                break;
            }

            // If no subdivision progress was made for this depth
            // after a few iterations, force the scaling ratio to change
            if ( accRequired == prevAccReq )
            {
                if ( --retry <= 0 )
                {
                    ibRatio = 0.9;
                }
            }
            prevAccReq = accRequired;

            // Not met the IB requirement, fast reduce the
            // fitting limit.
            if ( (ibRatio < 1.0) && (abRatio > 1.0) )
            {
                fitAcc = std::min(fitAcc, accRequired);
            }

            // Didn't fit both ACC & IB, reduce the volume and retry
            double ratio = std::min(ibRatio * ibRatio, abRatio);
            int newAcc = GranularScale(fitAcc, ubAccElements, ratio);
            // Ratio didn't scale
            if ( newAcc == fitAcc )
            {
                newAcc = fitAcc - ubAccElements;
            }
            // No fit
            if ( newAcc <= ubAccElements )
            {
                break;
            }
            fitAcc = newAcc;
        }
        // Nothing fitted at this depth, increased depth won't improve
        if ( !fitted )
        {
            assert(!!fitShape && "No solution");
            break;
        }
    }
    return fitShape;
}


Shape ArchEthosU85::FindDepthwiseConfig(const ArchitectureConfigQuery &query, const FindConfigCommon &common, Shape &ifmBlock)
{
    LOG_TRACE2("Depthwise/Pooling: OFM {} (ublock={}  k={},{})\n", query.ofmShape.ToString(), common.ublock.ToString(),
        query.kernel->Size().x, query.kernel->Size().y);
    LOG_INDENT(Logging::Out);
    assert(common.accBits > 0 && query.ifmBits >= 8);
    assert(common.granule.Depth() && common.ublock.Height());
    const Shape ofmShape = Shape::PadAxes(Shape::RoundAway(query.ofmShape, common.granule), 3, 1);
    const Shape ofmBlockLimit = Shape::Min(ofmShape, common.ofmBlockMax);
    const int accElements = (_accRamSizeBytes * 8) / common.accBits;
    const int ibElements = (_ifmRamSizeBytes * 8) / query.ifmBits;
    const int ubAccElements = common.ublock.ElementsWH() * ACC_DEPTH_GRANULE;

    Shape fitShape = AreaFit(common, ofmShape, ofmBlockLimit, query.ifmShape[0], query.kernel);

    int depth = fitShape.Depth();
    int width = fitShape.Width();
    int height = fitShape.Height();

    Shape ifmAllocUnit;
    Shape ifmReq;
    unsigned forceReduce = 0;
    while ( true )
    {
        ifmAllocUnit = CalcIfmAUSize(common.ifmBlockDepth ? common.ifmBlockDepth : depth, query.ifmBits, common.ublock);
        int depthGranule = ifmAllocUnit.Depth();

        ifmReq = GetArchIFMBlockSize(
            Shape(height, width, depth), query.kernel, ifmAllocUnit, _subkernelMax, 1, 0, common.ifmBlockDepth);

        int ibRequired = ifmReq.Elements();
        assert(ibRequired);
        double ratio = double(ibElements) / ibRequired;

        // If more than one ofm block will be run, modify space to pre-buffer
        // another row of the IFM (where possible).
        if ( width < ofmShape.Width() || height < ofmShape.Height() || depth < ofmShape.Depth() )
        {
            Shape ifmRow = GetArchIFMBlockSize(Shape(common.granule.Height(), width, depth), query.kernel, ifmAllocUnit,
                _subkernelMax, 1, 0, common.ifmBlockDepth);
            // See if we can get read-buffering space for one extra row granule
            if ( (ibElements - ibRequired) < ifmRow.Elements() )
            {
                if ( (height != ofmShape.Height() || (forceReduce & 2)) && height > common.granule.Height() )
                {
                    height -= common.granule.Height();
                }
                else if ( (depth > width || (forceReduce & 1)) && depth > depthGranule )
                {
                    depth -= common.granule.Depth();
                }
                else if ( (width != ofmShape.Width() || (forceReduce & 4)) && width > common.granule.Width() )
                {
                    width -= common.granule.Width();
                }
                else if ( forceReduce != 7 )
                {
                    forceReduce = (forceReduce << 1) | 1;
                }
                else if ( depth > common.granule.Depth() )
                {
                    // Last resort, reducing depth
                    depth = std::max(depth / 2, common.granule.Depth());
                }
                else
                {
                    assert(false);
                    break;
                }
                continue;
            }
        }
        break;
    }

    Shape ofmBlock(height, width, depth);

    ifmBlock = ifmReq;

    assert(std::min(ifmBlock.Elements(), query.ifmShape[0].Elements()) <= ibElements);
    assert(ofmBlock.Elements() <= accElements);

    LOG_TRACE2("Depthwise choice: ofmBlock = {}\n", ofmBlock.ToString());
    return ofmBlock;
}

std::unique_ptr<ArchitectureOpConfig> ArchEthosU85::FindBlockConfig(OpType opType, const ArchitectureConfigQuery &query)
{
    LOG_TRACE2("FindBlockConfig: OPERATOR = {}\n", OpTypeToString(opType));
    LOG_INDENT(Logging::Out);

    constexpr int OFMSplitDepth = 16;  // Specific to this architecture
    assert(query.ifmBits > 0 && (query.ifmBits <= 32 || (query.ifmBits == 64 && opType == OpType::Rescale)));
    assert(query.ofmShape.Size() > 2 && "Insufficient dimensions to search for block config");
    assert(query.kernel != nullptr);

    EthosU85NpuOp npuOp = GetHWOp(opType);
    assert(npuOp != EthosU85NpuOp::None);
    if ( npuOp == EthosU85NpuOp::Dma ) return nullptr;  // DMA ops don't use block config

    // Elementwise larger-volume correction
    const Shape &ifmShape = (query.ifmShape[1].Elements() > query.ifmShape[0].Elements()) ? query.ifmShape[1] : query.ifmShape[0];

    // Operator typing help
    const bool isPooling = npuOp == EthosU85NpuOp::Pooling || npuOp == EthosU85NpuOp::ReduceMinMax || npuOp == EthosU85NpuOp::ArgMax;
    const bool isReduceSum = npuOp == EthosU85NpuOp::ReduceSum;
    const bool isDepthwise = npuOp == EthosU85NpuOp::Depthwise;
    const bool isElementwise = npuOp == EthosU85NpuOp::Elementwise;
    const bool isMatmul = (npuOp == EthosU85NpuOp::VectorProduct) && query.ifmShape[1];
    const bool isFullyConnected = (npuOp == EthosU85NpuOp::VectorProduct) && !isMatmul;

    // Accumulator settings
    EthosU85Accumulator accType = EthosU85Accumulator::Acc32;
    if ( (query.ifmBits == 16 && !isPooling && query.scaled) ||  // Normal 16-bit selection
         (query.ifmBits > 32) || (query.ofmBits > 32) )          // Special case for Rescale int48
    {
        accType = EthosU85Accumulator::Acc48;
    }

    const bool sparse = query.weightFormat & WeightFormat::Sparse2_4;
    const bool isPartKernel = (npuOp == EthosU85NpuOp::Convolution) && ChooseKernelFirst(ifmShape, query.kernel, sparse);

    const Shape ofmUBlock = FindUBlock(opType, query, isPartKernel);
    if ( !ofmUBlock )
    {
        // no valid ofm microblock found
        LOG_WARN("Could not find a valid OFM microblock for {} with {}-bit input.\n", OpTypeToString(opType), query.ifmBits);
        return nullptr;
    }

    // When using brick format and certain transposes, there are additional constraints to the block size, so we must
    // extend the search space to be able to find a valid block size.
    Shape ofmBlockGranule = ofmUBlock.WithDepth(ACC_DEPTH_GRANULE);
    if ( query.ofmFormat == TensorFormat::NHCWB16 )
    {
        if ( (query.transpose & TransposeType::MaskC) == TransposeType::W ) ofmBlockGranule[-2] = 16;
        if ( (query.transpose & TransposeType::MaskC) == TransposeType::H ) ofmBlockGranule[-3] = 16;
    }

    // Operator configuration to be returned
    auto config = std::make_unique<EthosU85OpConfig>();
    config->_ofmUBlock = ofmUBlock;
    config->_accumulatorType = accType;
    config->_accumulatorSource = query.accSource;
    config->_accumulatorOutputEnabled = query.accOutputEnabled;
    config->_ifmRamSizeBytes = _ifmRamSizeBytes;
    config->_traversal = EthosU85Traversal::DepthFirst;

    // Common search variables
    FindConfigCommon common;
    common.ofmBlockMax = _ofmBlockMax.Unpermute(uint32_t(query.transpose));
    common.ublock = ofmUBlock;
    common.granule = ofmBlockGranule;
    common.accBits = AccumulatorBits(accType);
    common.ifmBits = query.ifmBits;
    common.isPooling = isPooling;
    if ( query.reverse == ReverseType::C )
    {
        common.ofmBlockMax = common.ofmBlockMax.WithDepth(common.granule.Depth());
    }

    if ( isElementwise )
    {
        config->_ofmBlock = FindElementwiseConfig(query, common);
        config->_ifmBlock = config->_ofmBlock;
        assert(config->_ofmBlock.Width() % ofmBlockGranule[-2] == 0);
        return config;
    }

    if ( isDepthwise )
    {
        common.ifmBlockDepth = 0;
        config->_ofmBlock = FindDepthwiseConfig(query, common, config->_ifmBlock);
        assert(config->_ofmBlock.Width() % ofmBlockGranule[-2] == 0);
        config->_traversal = EthosU85Traversal::Depthwise;
        return config;
    }

    // Calculate fixed IFM block depth used for most non-depthwise/pooling operations
    int ifmBlockDepth = 64;
    if ( isPartKernel )
    {
        ifmBlockDepth = 16;
    }
    else if ( query.ifmBits == 32 || ((_macs == 128 || _macs == 256) && ofmUBlock.Depth() == 16 && !sparse) )
    {
        ifmBlockDepth = 32;
    }
    else if ( sparse && query.ifmBits == 8 )
    {
        assert(config->_traversal == EthosU85Traversal::DepthFirst);
        ifmBlockDepth = 128;
    }

    if ( (isPooling || isReduceSum) && (opType != OpType::MemoryCopy) )
    {
        common.ifmBlockDepth = isReduceSum ? ifmBlockDepth : 0;
        config->_ofmBlock = FindDepthwiseConfig(query, common, config->_ifmBlock);
        return config;
    }

    // Original Block Selection:
    const bool isConvolution = npuOp == EthosU85NpuOp::Convolution || npuOp == EthosU85NpuOp::Depthwise;
    const bool isResize = npuOp == EthosU85NpuOp::Resize;
    const bool isEqualDepthOp = isElementwise || isPooling || isDepthwise || isResize;

    EthosU85Traversal traversal = isDepthwise ? EthosU85Traversal::Depthwise : (isPartKernel ? EthosU85Traversal::PartKernel : EthosU85Traversal::DepthFirst);

    int accBits = AccumulatorBits(accType);
    int rounding;
    int upscale = UpscaleAndRounding(query.ifmResampling, rounding);
    int numBlocksInRam = 2;

    // Subkernel repeats of the IFM
    Point2i dilatedWH = query.kernel->DilatedWH();
    int ifmRepeats = DivRoundUp(dilatedWH.x, _subkernelMax.Width()) * DivRoundUp(dilatedWH.y, _subkernelMax.Height());

    // Weights fetch (for operators that have them)
    int weightFetchWH = isConvolution ? query.kernel->Size().AreaXY() : 0;

    int ofmUBlockDepth = ofmUBlock.Depth();

    Shape searchSpaceStep = Shape::Max(ofmUBlock, ofmBlockGranule);
    Shape ofmBlockMaxTp = _ofmBlockMax.Unpermute(uint32_t(query.transpose));
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
            Shape newAU = CalcIfmAUSize(depth, query.ifmBits, ofmUBlock);
            // clear wontFit if the IFM AU has changed
            if ( newAU != ifmAllocUnit )
            {
                wontFit.clear();
                ifmAllocUnit = newAU;
            }
        }

        for ( int height = searchSpaceStep.Height(); height <= searchSpaceEnd.Height(); height += searchSpaceStep.Height() )
        {
            for ( int width = searchSpaceStep.Width(); width <= searchSpaceEnd.Width(); width += searchSpaceStep.Width() )
            {
                // Avoid checking blocks that already didn't fit with a smaller depth.
                // i.e. if 8x4x16 didn't fit then 8x4x32 won't either.
                if ( wontFit.count(Point2i(height, width)) > 0 )
                {
                    continue;
                }

                // Calculate the IFM block dimensions required to feed this OFM block
                Shape ofmBlock = Shape(height, width, depth);

                Shape ifmBlock = GetArchIFMBlockSize(ofmBlock, query.kernel, ifmAllocUnit, _subkernelMax, upscale, rounding, ifmBlockDepth);

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
                    float relativeCost = 0.0;
                    if ( isFullyConnected )
                    {
                        relativeCost = 1.0f / (height * width);
                    }
                    else if ( isResize || isMatmul )
                    {
                        relativeCost = float(ofmElements) / (float(height) * width * depth);
                    }
                    else
                    {
                        relativeCost = (ifmFetch + weightFetch) / float(ofmElements);
                    }
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
                            if ( coverage <= bestCoverage && ((height <= 4 && width <= 4) || isMatmul) )
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
                            config->_ifmBlock = ifmBlock.WithDepth(ifmBlockDepth);
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
        if ( depth > searchSpaceEnd.Depth() && bestCost == std::numeric_limits<float>::infinity() && numBlocksInRam == 2 )
        {
            numBlocksInRam = 1;
            wontFit.clear();
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

Shape ArchEthosU85::CalcIfmAUSize(int ifmBlkDepth, int ifmBits, const Shape &ofmUBlk)
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
    unsigned blockIdx = IndexForOfmUBlock(ofmUBlk);
    assert(blockIdx < 3 && ifmu < 3);
    Shape &block = _uBlockToIfmAuTable[blockIdx][ifmu];
    return block.WithDepth(block.Depth() * 128 / ifmBits);
}

int ArchEthosU85::CalcResizeMaxOfmBlockWidth(int ifmBits, int scaleN, int scaleD)
{
    // Calculate the maximum OfmBlockWidth that still allows
    // the IFM block to fit in the chaining buffer
    assert(scaleN > 0);
    assert(scaleD > 0);
    const int cbBricks = (_cbRamSizeBytes / CB_SLOTS) / (BRICK_ELEMENTS * (ifmBits / 8));
    int maxOfmBlkW = int(std::ceil(((cbBricks - 2) * scaleN + 1) / double(scaleD)));
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
    int ifmAlignDepth = ifmAuDepth;
    int ifmBlockDepth = isEqualDepthOp ? ofmBlock.Depth() : std::min(ifmBlock.Depth(), ifmShape.Depth());
    ifmBlockDepth = RoundAway(ifmBlockDepth, ifmAlignDepth);
    int ifmBytes = ifmBlock.ElementsWH() * ifmBlockDepth * (ifmBits / 8) * numBlocksInRam;

    // Accumulator space
    int ofmBlockDepth = RoundAway(ofmBlock.Depth(), ACC_DEPTH_GRANULE);
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
        {OpType::DepthwiseConv2D, EthosU85NpuOp::Depthwise},
        {OpType::Conv2D, EthosU85NpuOp::Convolution},
        {OpType::ReduceSum, EthosU85NpuOp::ReduceSum},
        {OpType::FullyConnected, EthosU85NpuOp::VectorProduct},
        {OpType::MatMul, EthosU85NpuOp::VectorProduct},
        {OpType::MaxPool, EthosU85NpuOp::Pooling},
        {OpType::AvgPool, EthosU85NpuOp::Pooling},
        {OpType::QuantizedAvgPool, EthosU85NpuOp::Pooling},
        {OpType::QuantizedMaxPool, EthosU85NpuOp::Pooling},
        {OpType::ArgMax, EthosU85NpuOp::ArgMax},
        {OpType::ReduceMin, EthosU85NpuOp::ReduceMinMax},
        {OpType::ReduceMax, EthosU85NpuOp::ReduceMinMax},
        {OpType::ReduceAny, EthosU85NpuOp::ReduceMinMax},
        {OpType::ReduceAll, EthosU85NpuOp::ReduceMinMax},
        // TODO MLBEDSW-7986 add none pooling
        {OpType::Resize, EthosU85NpuOp::Resize},
        {OpType::Gather, EthosU85NpuOp::Dma},
        {OpType::Scatter, EthosU85NpuOp::Dma},
        {OpType::Tile, EthosU85NpuOp::Dma},
        {OpType::Rescale, EthosU85NpuOp::Pooling},
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

int EthosU85OpGroup::KeyToOpIndex(int key)
{
    if ( key > 0 )
    {
        key = -1;
    }

    else if ( key < 0 )
    {
        key = (-key) - 1;
    }

    if ( key >= _opsCount )
    {
        key = -1;
    }
    return key;
}

bool EthosU85OpGroup::CanStartChain(const ArchitectureOpGroupQuery &op)
{
    OpType opType = op.type;
    EthosU85NpuOp npuOp = ArchEthosU85::GetHWOp(opType);
    if ( npuOp == EthosU85NpuOp::None || npuOp == EthosU85NpuOp::Resize || npuOp == EthosU85NpuOp::Dma )
    {
        return false;
    }
    if ( npuOp == EthosU85NpuOp::Pooling && _arch->UseNullPool(opType, DataTypeSizeBits(op.ifm[0].type)) )
    {
        return false;
    }
    return true;
}

int EthosU85OpGroup::ExternalIfms(const ArchitectureOpGroupQuery &op)
{
    int externalInputs = op.inputs;
    for ( int i = 0; i < op.inputs; i++ )
    {
        if ( op.ifm[i].isConst && op.ifm[i].shape.Elements() == 1 &&
             _arch->SupportsScalar(op.type, op.ifm[i].type, MakeTensorUsage(TensorUsage::IFM, i)) )
        {
            externalInputs -= 1;
        }
    }
    return externalInputs;
}

bool EthosU85OpGroup::Fuse(const ArchitectureOpGroupQuery &op, const std::vector<int> &dependsOn)
{
    assert(_opsCount > 0);
    if ( _chainLength > 1 )
    {
        // TODO MLBEDSW-9142: support fusing on chained ops
        return false;
    }

    // activation fusing..
    if ( op.ifm[0].type == DataType::Int16 && (op.type == OpType::Sigmoid || op.type == OpType::Tanh) )
    {
        // Can not fuse int16 Sigmoid and Tanh LUT since they require special scaling done by AvgPoolNop
        return false;
    }

    if ( dependsOn.size() > 1 )
    {
        // Can only fuse with one op
        return false;
    }

    if ( op.inputs == 2 )
    {
        // can't fuse binary input
        return false;
    }

    int dep = KeyToOpIndex(dependsOn[0]);
    if ( dep < 0 )
    {
        // invalid key
        return false;
    }
    const EthosU85OpGroup::OpInfo &prevOp = _ops[dep];

    // Can't fuse activation with activation
    if ( IsActivation(op.type) && _hasFusedActivation )
    {
        return false;
    }

    // Can't fuse transpose/reverse with transpose/reverse
    if ( (op.type == OpType::Transpose || op.type == OpType::Reverse) && (_hasFusedTranspose || _hasFusedReverse) )
    {
        return false;
    }

    EthosU85Constraints *constraints = static_cast<EthosU85Constraints *>(_arch->_constraints.get());

    // Can't fuse a transpose or reverse type that's not supported by primaryOp in opgroup
    ArchOperatorQuery query;
    query.reverseMask = op.ofm.reverse;
    query.transposeMask = op.ofm.transpose;
    if ( !constraints->OperatorQuery(_ops[0].type, &query, nullptr).Any(QueryResult::Native) )
    {
        return false;
    }

    // Dependency without connection
    if ( prevOp.ofm.key != op.ifm[0].key )
    {
        return false;
    }

    // Can't fuse reshaped Tensors
    if ( prevOp.ofm.key == op.ifm[0].key && prevOp.ofm.shape != op.ifm[0].shape )
    {
        return false;
    }

    _hasFusedActivation = _hasFusedActivation || IsActivation(op.type);
    _hasFusedTranspose = _hasFusedTranspose || (op.type == OpType::Transpose && !IsNone(op.ofm.transpose));
    _hasFusedReverse = _hasFusedReverse || (op.type == OpType::Reverse && op.ofm.reverse != ReverseType::None);

    _fusedTensors.insert(prevOp.ofm.key);
    return true;
}

bool EthosU85OpGroup::Chain(const ArchitectureOpGroupQuery &op, const std::vector<int> &dependsOn, int externalInputs)
{
    // Op is considered for chaining
    EthosU85NpuOp npuOp = ArchEthosU85::GetHWOp(op.type);
    assert(_opsCount > 0);
    if ( _supportsChaining == false )
    {
        // primaryOp in opgroup doesnt support chaining
        return false;
    }
    if ( externalInputs == 0 )
    {
        // can only consider external (non-constant) inputs for chaining
        return false;
    }
    if ( _opsCount > _chainLength )
    {
        // TODO MLBEDSW-9142: support chaining on fused ops
        return false;
    }
    if ( npuOp != EthosU85NpuOp::Elementwise )
    {
        return false;
    }
    if ( _hasFusedTranspose || (op.type == OpType::Transpose && !IsNone(op.ofm.transpose)) )
    {
        return false;
    }
    if ( _hasFusedReverse || (op.type == OpType::Reverse && op.ofm.reverse != ReverseType::None) )
    {
        return false;
    }
    if ( (_chainLength + 1) > _maxChainLength )
    {
        return false;
    }
    if ( (_externalIfms + (externalInputs - int(dependsOn.size()))) > _maxExternalIfms )
    {
        return false;
    }
    for ( int key : dependsOn )
    {
        int dep = KeyToOpIndex(key);
        if ( dep < 0 )
        {
            return false;
        }
        const EthosU85OpGroup::OpInfo &prevOp = _ops[dep];

        if ( prevOp.ofm.shape != op.ofm.shape )
        {
            // cannot chain broadcasted ofm
            return false;
        }

        if ( prevOp.ofm.key == op.ifm[0].key )
        {
            _tensorCbMap[prevOp.ofm.key] = _chainIdx++;
            externalInputs--;
        }
        else if ( op.inputs == 2 && (prevOp.ofm.key == op.ifm[1].key) )
        {
            _tensorCbMap[prevOp.ofm.key] = _chainIdx++;
            externalInputs--;
        }
        else
        {
            // dependency without connection
            return false;
        }
    }
    _externalIfms += externalInputs;
    _chainLength += 1;
    return true;
}

int EthosU85OpGroup::Add(const ArchitectureOpGroupQuery &op, const std::vector<int> &dependsOn)
{
    int externalInputs = ExternalIfms(op);

    if ( !CanRunOnNPU(op) )
    {
        // Can only fuse NPU ops
        return 0;
    }

    if ( _opsCount == 0 )
    {
        _supportsChaining = CanStartChain(op);
        _externalIfms = externalInputs;
        _chainLength = 1;
        _hasFusedActivation = IsActivation(op.type);
        _hasFusedTranspose = (op.type == OpType::Transpose && !IsNone(op.ofm.transpose));
        _hasFusedReverse = (op.type == OpType::Reverse && op.ofm.reverse != ReverseType::None);
    }
    else if ( IsActivation(op.type) || op.type == OpType::Transpose || op.type == OpType::Reverse )
    {
        if ( Fuse(op, dependsOn) == false )
        {
            return 0;
        }
    }
    else
    {
        if ( Chain(op, dependsOn, externalInputs) == false )
        {
            return 0;
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

int EthosU85OpGroup::ChainingBuffer(UniqueId tensorUID)
{
    auto cb = _tensorCbMap.find(tensorUID);
    if ( cb != std::end(_tensorCbMap) )
    {
        return cb->second;
    }
    return -1;
}

bool EthosU85OpGroup::IsChained(UniqueId tensorUID)
{
    return ChainingBuffer(tensorUID) >= 0;
}

bool EthosU85OpGroup::IsFused(UniqueId tensorUID)
{
    return _fusedTensors.count(tensorUID) != 0;
}

bool EthosU85OpGroup::NeedsAllocation(UniqueId tensorUID)
{
    return !IsChained(tensorUID) && !IsFused(tensorUID);
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
            {DataType::Bool8, {DataType::Bool8, DataType::Int32, DataType::Int64}},
            {DataType::UInt8, {DataType::UInt8, DataType::Int32, DataType::Int64}},
            {DataType::Int8, {DataType::Int8, DataType::Int32, DataType::Int64}},
            {DataType::Int16, {DataType::Int16}},
        }},
    {EthosU85NpuOp::ReduceMinMax,
        {
            {DataType::Bool8, {DataType::Bool8, DataType::UInt8, DataType::Int8, DataType::Int16, DataType::Int32}},
            {DataType::UInt8, {DataType::Bool8, DataType::UInt8, DataType::Int8, DataType::Int16, DataType::Int32}},
            {DataType::Int8, {DataType::Bool8, DataType::UInt8, DataType::Int8, DataType::Int16, DataType::Int32}},
            {DataType::Int16, {DataType::Bool8, DataType::UInt8, DataType::Int8, DataType::Int16, DataType::Int32}},
            {DataType::Int32, {DataType::Bool8, DataType::UInt8, DataType::Int8, DataType::Int16, DataType::Int32}},
        }},
    {EthosU85NpuOp::ReduceSum,
        {
            {DataType::UInt8, {DataType::UInt8, DataType::Int8, DataType::Int16, DataType::Int32}},
            {DataType::Int8, {DataType::UInt8, DataType::Int8, DataType::Int16, DataType::Int32}},
            {DataType::Int16, {DataType::UInt8, DataType::Int8, DataType::Int16, DataType::Int32}},
            {DataType::Int32, {DataType::UInt8, DataType::Int8, DataType::Int16, DataType::Int32}},
        }},
    {EthosU85NpuOp::ArgMax,
        {
            {DataType::Bool8, {DataType::Int32, DataType::Int64}},
            {DataType::UInt8, {DataType::Int32, DataType::Int64}},
            {DataType::Int8, {DataType::Int32, DataType::Int64}},
            {DataType::Int16, {DataType::Int32, DataType::Int64}},
        }},
    {EthosU85NpuOp::Dma,
        {
            {DataType::Bool8, {DataType::Bool8}},
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

    if ( IsFloat(op.ifm[0].type | op.ifm[1].type | op.ofm.type) )
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

    if ( k->DepthMultiplier() > 1 )
    {
        return false;
    }

    switch ( npuOp )
    {
        case EthosU85NpuOp::Convolution:
        case EthosU85NpuOp::Depthwise:
        case EthosU85NpuOp::VectorProduct:
        case EthosU85NpuOp::Pooling:
        case EthosU85NpuOp::ReduceMinMax:
        case EthosU85NpuOp::ReduceSum:
        case EthosU85NpuOp::ArgMax:
        case EthosU85NpuOp::Elementwise:
        case EthosU85NpuOp::Resize:
        case EthosU85NpuOp::Dma:
            break;
        default:
            assert(false && "Unrecognized HWOp");
            return false;
    }

    // Validate that input/outputs shapes don't overflow
    if ( npuOp != EthosU85NpuOp::Dma )
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

    // Check allowed ifm/ofm data type mapping
    if ( npuOp != EthosU85NpuOp::Elementwise )
    {
        if ( op.type == OpType::LUT || op.type == OpType::MemoryCopy || op.type == OpType::Rescale || op.type == OpType::Tile )
        {  // TODO: LUT operations end up here due to UseAvgPoolNop although the rules are not the same as
           // for a Pooling operation, so skip checks for now.
            return true;
        }

        if ( op.type == OpType::Transpose || op.type == OpType::Reverse )
        {
            ArchOperatorQuery query;
            query.transposeMask = op.ofm.transpose;
            query.reverseMask = op.ofm.reverse;
            return _arch->_constraints->OperatorQuery(OpType::MemoryCopy, &query, nullptr).Any(QueryResult::Native);
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
        // TODO: Elementwise
    }

    return true;
}

}  // namespace regor
