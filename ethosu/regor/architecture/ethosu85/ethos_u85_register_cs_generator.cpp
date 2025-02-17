//
// SPDX-FileCopyrightText: Copyright 2023-2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
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

#include "ethos_u85_register_cs_generator.hpp"

#include "common/common.hpp"
#include "common/logging.hpp"

#include "architecture/ethos_u_scaling.hpp"
#include "common/data_type.hpp"
#include "compiler/high_level_command_stream.hpp"
#include "compiler/op_type.hpp"
#include "ethos_u85.hpp"
#define NPU_DISASSEMBLE
#define NPU_NAMESPACE ethosu85
#include "ethos_u85_interface.hpp"
#include "ethos_u85_scaling.hpp"

#include <deque>
#include <unordered_map>
#include <vector>

namespace regor
{

using namespace ethosu85;

constexpr int INVALID_CB = -1;

namespace
{
bool IsLUTType(OpType opType)
{
    return opType == OpType::LUT || opType == OpType::Sigmoid || opType == OpType::Tanh;
}

constexpr uint16_t OpCode(uint64_t cmd)
{
    return uint16_t(cmd & 0xFFFF);
}
}  // namespace

void EthosU85Emitter::ClearChainingRegisters()
{
    // The following commands need to be reset before and after any chained operation
    static const std::array<uint16_t, 27> resetCmds = {
        OpCode(isa::npu_set_ifm_precision_t()),
        OpCode(isa::npu_set_ifm_base0_t()),
        OpCode(isa::npu_set_ifm_base1_t()),
        OpCode(isa::npu_set_ifm_base2_t()),
        OpCode(isa::npu_set_ifm_base3_t()),
        OpCode(isa::npu_set_ifm_width0_m1_t()),
        OpCode(isa::npu_set_ifm_height0_m1_t()),
        OpCode(isa::npu_set_ifm_height1_m1_t()),
        OpCode(isa::npu_set_ifm_stride_x_t()),
        OpCode(isa::npu_set_ifm_stride_y_t()),
        OpCode(isa::npu_set_ifm_stride_c_t()),
        OpCode(isa::npu_set_ifm_region_t()),
        OpCode(isa::npu_set_ifm_zero_point_t()),
        OpCode(isa::npu_set_ifm2_precision_t()),
        OpCode(isa::npu_set_ifm2_base0_t()),
        OpCode(isa::npu_set_ifm2_base1_t()),
        OpCode(isa::npu_set_ifm2_base2_t()),
        OpCode(isa::npu_set_ifm2_base3_t()),
        OpCode(isa::npu_set_ifm2_width0_m1_t()),
        OpCode(isa::npu_set_ifm2_height0_m1_t()),
        OpCode(isa::npu_set_ifm2_height1_m1_t()),
        OpCode(isa::npu_set_ifm2_stride_x_t()),
        OpCode(isa::npu_set_ifm2_stride_y_t()),
        OpCode(isa::npu_set_ifm2_stride_c_t()),
        OpCode(isa::npu_set_ifm2_region_t()),
        OpCode(isa::npu_set_ifm2_zero_point_t()),
        OpCode(isa::npu_set_ofm_precision_t()),
    };
    for ( auto cmd : resetCmds )
    {
        _registers.erase(cmd);
    }
}

void EthosU85Emitter::Emit(uint32_t instr)
{
    uint16_t cmd = OpCode(instr);
    assert(IsCmd0(cmd));
    bool emit = IsOp(cmd) || SetRegister(cmd, instr);
    if ( emit )
    {
        _stream.push_back(instr);
    }
}

void EthosU85Emitter::Emit(uint64_t instr)
{
    uint16_t cmd = OpCode(instr);
    assert(IsCmd1(cmd));
    bool emit = IsOp(cmd) || SetRegister(cmd, instr);
    if ( emit )
    {
        _stream.push_back(uint32_t(instr));
        _stream.push_back(uint32_t(instr >> 32));
    }
}

void EthosU85Emitter::Clear()
{
    _stream.clear();
    _registers.clear();
}

bool EthosU85Emitter::SetRegister(uint16_t reg, uint64_t value)
{
    auto item = _registers.find(reg);
    bool isChanged = item == _registers.end() || item->second != value;
    if ( isChanged )
    {
        _registers[reg] = value;
    }
    return isChanged;
}

bool EthosU85Emitter::IsCmd0(uint16_t key)
{
    return (key >> 14) == uint16_t(cmd_ctrl::CMD0_CTRL);
}

bool EthosU85Emitter::IsCmd1(uint16_t key)
{
    return (key >> 14) == uint16_t(cmd_ctrl::CMD1_CTRL);
}

bool EthosU85Emitter::IsOp(uint16_t key)
{
    return IsCmd0(key) ? (key & (1 << 8)) == 0 : (key & (1 << 8)) != 0;
}


/// <summary>
/// Generates register command streams for Ethos U85.
/// </summary>
namespace
{
const std::unordered_map<OpType, elementwise_mode> kElementwiseMap = {
    {OpType::Add, elementwise_mode::ADD},
    {OpType::Sub, elementwise_mode::SUB},
    {OpType::Abs, elementwise_mode::ABS},
    {OpType::Mul, elementwise_mode::MUL},
    {OpType::Minimum, elementwise_mode::MIN},
    {OpType::Maximum, elementwise_mode::MAX},
    {OpType::LeakyRelu, elementwise_mode::LRELU},
    {OpType::CLZ, elementwise_mode::CLZ},
    {OpType::SHL, elementwise_mode::SHL},
    {OpType::SHR, elementwise_mode::LSR},
    {OpType::Div, elementwise_mode::DIV},
    {OpType::LogicalAnd, elementwise_mode::AND},
    {OpType::LogicalOr, elementwise_mode::OR},
    {OpType::LogicalXor, elementwise_mode::XOR},
    {OpType::LogicalNot, elementwise_mode::NOT},
    {OpType::And, elementwise_mode::AND},
    {OpType::Or, elementwise_mode::OR},
    {OpType::Xor, elementwise_mode::XOR},
    {OpType::Not, elementwise_mode::NOT},
    {OpType::Asr, elementwise_mode::SHR},
    {OpType::Equal, elementwise_mode::CMP_EQ},
    {OpType::Greater, elementwise_mode::CMP_GT},
    {OpType::GreaterEqual, elementwise_mode::CMP_GE},
    {OpType::NotEqual, elementwise_mode::CMP_NE},
    {OpType::AndNot, elementwise_mode::AND_NOT},
};

activation_type ToActivationType(DataType type)
{
    if ( IsSignedInteger(type) || IsBool(type) )
    {
        return activation_type::SIGNED;
    }
    else
    {
        assert(IsInteger(type));
        return activation_type::UNSIGNED;
    }
}

activation_format ToActivationFormat(TensorFormat format)
{
    if ( format == TensorFormat::NHCWB16 )
    {
        return activation_format::NHCWB16;
    }
    else
    {
        assert(format == TensorFormat::NHWC);
        return activation_format::NHWC;
    }
}

activation_precision ToActivationPrecision(DataType type)
{
    switch ( DataTypeSizeBits(type) )
    {
        case 8:
            return activation_precision::B8;
        case 16:
            return activation_precision::B16;
        case 32:
            return activation_precision::B32;
        case 48:
            [[fallthrough]];
        case 64:
            return activation_precision::B64;
        default:
            assert(false);
            return activation_precision::B64;
    }
}

activation_transpose ToActivationTranspose(TransposeType type)
{
    switch ( type )
    {
        case TransposeType::None:
            return activation_transpose::HWC;
        case TransposeType::NWHC:
            return activation_transpose::WHC;
        case TransposeType::NHCW:
            return activation_transpose::HCW;
        case TransposeType::NWCH:
            return activation_transpose::WCH;
        case TransposeType::NCHW:
            return activation_transpose::CHW;
        case TransposeType::NCWH:
            return activation_transpose::CWH;
        default:
            assert(false && "Unknown transpose mask");
            return activation_transpose::HWC;
    }
}

activation_reverse ToActivationReverse(ReverseType type)
{
    switch ( type )
    {
        case ReverseType::None:
            return activation_reverse::NONE;
        case ReverseType::H:
            return activation_reverse::H;
        case ReverseType::W:
            return activation_reverse::W;
        case ReverseType::C:
            return activation_reverse::C;
        default:
            assert(false && "Unknown reverse type");
            return activation_reverse::NONE;
    }
}

ifm_upscale_mode ToIfmUpscaleMode(ArchResampling resampling)
{
    if ( resampling == ArchResampling::Nearest )
    {
        return ifm_upscale_mode::NEAREST;
    }
    if ( resampling == ArchResampling::Zeros )
    {
        return ifm_upscale_mode::ZEROS;
    }
    return ifm_upscale_mode::NONE;
}

resize_mode ToResizeMode(ArchResizeMode mode)
{
    if ( mode == ArchResizeMode::Bilinear )
    {
        return resize_mode::BILINEAR;
    }
    if ( mode == ArchResizeMode::Nearest )
    {
        return resize_mode::NEAREST;
    }
    return resize_mode::REPLICATE;
}

round_mode_ofm GetOfmRoundingMode(const HLCOperation *op)
{
    switch ( op->ofm.rounding )
    {
        case HLCRoundMode::NATURAL:
            return round_mode_ofm::NATURAL;
        case HLCRoundMode::TRUNCATE:
            return round_mode_ofm::TRUNCATE_TO_ZERO;
        case HLCRoundMode::DBL:
            return round_mode_ofm::DOUBLE_SYMMETRIC;
        case HLCRoundMode::AUTO:
            return round_mode_ofm::DOUBLE_SYMMETRIC;
        case HLCRoundMode::TRUNCATE_TO_LOWER:
            return round_mode_ofm::TRUNCATE_TO_LOWER;
        case HLCRoundMode::DOUBLE_ASYMMETRIC:
            return round_mode_ofm::DOUBLE_ASYMMETRIC;
        case HLCRoundMode::SYMMETRIC:
            return round_mode_ofm::SYMMETRIC;
        default:
            return round_mode_ofm::DOUBLE_SYMMETRIC;
    }
}

round_mode_ifm GetIfmRoundingMode(const HLCOperation *op, int index)
{
    assert(index >= 0 && index < int(op->ifm.size()));
    switch ( op->ifm[index].rounding )
    {
        case HLCRoundMode::NATURAL:
            return round_mode_ifm::NATURAL;
        case HLCRoundMode::AUTO:
            [[fallthrough]];
        case HLCRoundMode::DBL:
            return round_mode_ifm::DOUBLE_SYMMETRIC;
        default:
            assert(false && "ifm with unsupported roundmode");
            return round_mode_ifm::DOUBLE_SYMMETRIC;
    }
}

pooling_mode GetPoolingMode(const HLCOperation *op)
{
    OpType opType = op->type;
    assert(IsPooling(opType));
    pooling_mode mode;
    if ( opType == OpType::AvgPool )
    {
        auto kernelSize = op->kernel.Size();
        // SUM when kernel size > 8x8
        mode = (kernelSize.x <= 8 && kernelSize.y <= 8) ? pooling_mode::AVERAGE : pooling_mode::SUM;
    }
    else if ( opType == OpType::MaxPool || opType == OpType::ReduceMax || opType == OpType::ReduceAll )
    {
        mode = pooling_mode::MAX;
    }
    else if ( opType == OpType::ArgMax )
    {
        auto axis = op->parameters.argmax.axis;
        assert(Flags<AxisMask>(AxisMask::AxisX, AxisMask::AxisY).All(axis) && "Argmax with unexpected axis");
        mode = (axis == AxisMask::AxisY) ? pooling_mode::ARGMAX_Y : pooling_mode::ARGMAX_X;
    }
    else if ( opType == OpType::ReduceMin || opType == OpType::ReduceAny )
    {
        mode = pooling_mode::MIN;
    }
    else
    {
        assert(opType == OpType::ReduceSum);
        mode = pooling_mode::REDUCE_SUM;
    }
    return mode;
}
}  // namespace

uint32_t EthosU85RCSGenerator::ConfigRegister(int macs, int cmdStreamVersion, int numAxiSram, int numAxiExt, int numWd, int product)
{
    return config_r{}
        .set_macs_per_cc(macs)
        .set_cmd_stream_version(cmdStreamVersion)
        .set_num_axi_sram(numAxiSram)
        .set_num_axi_ext(numAxiExt)
        .set_num_wd(numWd)
        .set_product(product);
}

uint32_t EthosU85RCSGenerator::IdRegister()
{
    return id_r{};
}

bool EthosU85RCSGenerator::IsSupportedElementwise(const OpType opType)
{
    return kElementwiseMap.count(opType) != 0;
}

EthosU85RCSGenerator::EthosU85RCSGenerator(ArchEthosU85 *arch) : _arch(arch)
{
}

void EthosU85RCSGenerator::Emit(uint32_t instr)
{
    _emit.Emit(instr);
}

void EthosU85RCSGenerator::Emit(uint64_t instr)
{
    _emit.Emit(instr);
}


int EthosU85RCSGenerator::GetDoubleBufferOffset(HLCWeights *weights, int rangeIndex)
{
    int doubleBufferOffset = 0;
    if ( weights->buffering == Buffering::Double )
    {
        assert(weights->subStreams > 0);
        int depthIndex = rangeIndex / weights->subStreams;
        if ( depthIndex % 2 == 1 )
        {
            doubleBufferOffset = weights->doubleBufferOffset;
        }
    }
    return doubleBufferOffset;
}


void EthosU85RCSGenerator::CheckAddressRange(ArchitectureMemory *memory, Address address, int size)
{
    assert(address >= 0);
    if ( address >= memory->SizeBytes() )
    {
        LOG_ERROR("Error: Address out of bounds, address {0}, memory '{1}' with size {2}\n", address, memory->Name(),
            memory->SizeBytes());
        // TODO: replace assert by error handling
        assert(false && "Address out of bounds");
    }
    assert(size >= 0);
    if ( address + size > memory->SizeBytes() )
    {
        LOG_ERROR("Error: Address offset out of bounds, address {0}, offset {1}, memory '{2}' with size {3}\n", address,
            size, memory->Name(), memory->SizeBytes());
        // TODO: replace assert by error handling
        assert(false && "address offset out of bounds");
    }
}

void EthosU85RCSGenerator::CheckAddresses(const HLCFeatureMap &fm)
{
    CheckAddressRange(fm.memArea.memory, fm.address, fm.AllocationSizeBytes());
    assert(fm.address % 16 == 0 || fm.format != TensorFormat::NHCWB16);
}

// Calculate the offset of the 4D slice at the position given by the leading dimensions
// of a possible 5D or 6D offset in the fm
static int Cube4DOffset(const HLCFeatureMap &fm, const Shape &strides)
{
    int offset = 0;
    int rank = fm.slice.offset.Size();
    if ( rank > 4 )
    {
        assert(fm.shape.Size() == fm.slice.offset.Size());
        int stride5D = strides.Batch() * fm.shape.Batch();

        offset += stride5D * fm.slice.offset[rank - 5];

        if ( rank > 5 )
        {
            int stride6D = stride5D * fm.shape[rank - 5];
            offset += stride6D * fm.slice.offset[rank - 6];
        }
    }
    return offset;
}

// Calculates the rolling buffer address of the given coordinate.
Address EthosU85RCSGenerator::AddressForCoordinate(const HLCFeatureMap &fm, const Shape &strides, const Shape &coord)
{
    Shape truncatedCoord = Shape(Shape::PadAxes(coord, 4, 0) % Shape::PadAxes(fm.shape, 4, 1), 4);
    int offset = Cube4DOffset(fm, strides);
    if ( fm.format == TensorFormat::NHWC )
    {
        offset += strides.Dot(truncatedCoord);
    }
    else if ( fm.format == TensorFormat::NHCWB16 )
    {
        constexpr int BRICK = 16;
        int elemSize = DataTypeSizeBits(fm.dataType) / 8;
        int strideX = BRICK * elemSize;
        offset +=
            truncatedCoord.Height() * strides.Height() + truncatedCoord.Width() * strideX +
            (truncatedCoord.Depth() / BRICK) * strides.Depth() + (truncatedCoord.Depth() % BRICK) * elemSize +
            truncatedCoord.Batch() * strides.Batch();
    }
    else
    {
        assert(false);
    }
    return fm.address + offset;
}

// Calculates tile sizes/addresses of a feature map
TileBox EthosU85RCSGenerator::GetTiles(const HLCFeatureMap &fm, const Shape &strides, const Box &area)
{
    int crossingY = RoundAway(area.Start().Height() + 1, fm.shape.Height());
    crossingY = std::min(crossingY, area.End().Height());
    int crossingX = RoundAway(area.Start().Width() + 1, fm.shape.Width());
    crossingX = std::min(crossingX, area.End().Width());
    TileBox tiles;
    auto height = crossingY - area.Start().Height();
    auto width = crossingX - area.Start().Width();
    tiles.height0 = (height + fm.stepXY.y - 1) / fm.stepXY.y;
    tiles.height1 = tiles.height0;
    tiles.width0 = (width + fm.stepXY.x - 1) / fm.stepXY.x;
    for ( int i = 0; i < 4; ++i )
    {
        tiles.address[i] = 0;
    }
    int fmSize = fm.AllocationSizeBytes();
    tiles.address[0] = AddressForCoordinate(fm, strides, area.Start());
    assert(fm.address <= tiles.address[0] && tiles.address[0] < fm.address + fmSize);
    if ( area.End().Width() > crossingX )
    {
        tiles.address[1] = AddressForCoordinate(fm, strides, area.Start().WithWidth(crossingX));
        assert(fm.address <= tiles.address[1] && tiles.address[1] < fm.address + fmSize);
        assert(false && "Striping in vertical direction is not supported");
    }
    if ( area.End().Height() > crossingY )
    {
        tiles.address[2] = AddressForCoordinate(fm, strides, area.Start().WithHeight(crossingY));
        assert(fm.address <= tiles.address[2] && tiles.address[2] < fm.address + fmSize);
    }
    if ( area.End().Width() > crossingX && area.End().Height() > crossingY )
    {
        tiles.address[3] = AddressForCoordinate(fm, strides, area.Start().WithWidth(crossingX).WithHeight(crossingY));
        assert(fm.address <= tiles.address[3] && tiles.address[3] < fm.address + fmSize);
    }
    if ( fm.format == TensorFormat::NHCWB16 )
    {
        for ( int i = 0; i < 4; ++i )
        {
            assert(tiles.address[i] % 16 == 0 && "NHCWB16 base address is not 16-byte aligned");
        }
    }
    return tiles;
}

MemoryAccess EthosU85RCSGenerator::ToMemoryAccess(const HLCFeatureMap &fm, const Box &area, AccessDirection direction)
{
    const auto &strides = fm.strides;
    Address start = AddressForCoordinate(fm, strides, area.Start());
    // Note: due to truncating of shape, AddressForCoordinate(fm, .., fm.shape) returns
    // fm.address; the - Shape(1, 1, 1) prevents this
    Address end = AddressForCoordinate(fm, strides, area.End() - Shape(1, 1, 1)) + DataTypeSizeBits(fm.dataType) / 8;
    if ( end < start )
    {
        // Area wraps around the end of the feature map
        start = fm.address;
        end = fm.address + fm.AllocationSizeBytes();
    }
    return MemoryAccess(direction, fm.memArea, start, end);
}

// Returns region number used in NPU_SET_..._REGION
uint32_t EthosU85RCSGenerator::ToRegion(const MemArea &memArea)
{
    auto region = BasePointerIndex::WeightTensor;
    if ( memArea == _arch->FeatureMapMemory() )
    {
        region = BasePointerIndex::ScratchTensor;
    }
    else if ( memArea == _arch->StagingMemory() )
    {
        region = BasePointerIndex::ScratchFastTensor;
    }
    else if ( memArea == _arch->LUTMemory() )
    {
        region = BasePointerIndex::Mem2Mem;
    }
    else
    {
        assert(memArea == _arch->ReadonlyMemory());
    }
    return uint32_t(region);
}

// Checks if the feature map is a scalar, and if so, returns the
// quantized value in scalarValue.
bool EthosU85RCSGenerator::IsScalar(const HLCFeatureMap &fm, int32_t &scalarValue)
{
    const auto &view = fm.bufferView;
    // A 1-sized feature map in constant memory is a scalar
    bool isScalar = fm.shape.Elements() == 1 && view.HasBuffer();
    if ( isScalar )
    {
        if ( fm.dataType == DataType::Int8 )
        {
            scalarValue = view.Values<int8_t>()[0];
        }
        else if ( fm.dataType == DataType::UInt8 )
        {
            scalarValue = view.Values<uint8_t>()[0];
        }
        else if ( fm.dataType == DataType::Int16 )
        {
            scalarValue = view.Values<int16_t>()[0];
        }
        else if ( fm.dataType == DataType::UInt16 )
        {
            scalarValue = view.Values<uint16_t>()[0];
        }
        else if ( fm.dataType == DataType::Int32 )
        {
            scalarValue = view.Values<int32_t>()[0];
        }
        else
        {  // Unsupported scalar value
            isScalar = false;
        }
    }
    return isScalar;
}


// Calculates waits for KERNEL_WAIT/DMA_WAIT, returns -1 if no wait is needed
// - opAccesses contains the memory accesses for the current operation
// - outstanding contains the memory accesses for ongoing "other" operations
//   (DMA operations if the current op is an NPU operation, NPU operations if the current op is a DMA operation)
// Note: NPU - NPU dependency is handled via blockdep
int EthosU85RCSGenerator::CalcCommandWaits(const MemoryAccesses &opAccesses, std::deque<MemoryAccesses> &outstanding)
{
    int waits = 0;
    for ( int index = int(outstanding.size()) - 1; index >= 0; ++waits, --index )
    {
        for ( const auto &access : opAccesses )
        {
            for ( const auto &outstandingAccess : outstanding[index] )
            {
                if ( access.Conflicts(outstandingAccess) )
                {
                    // Current op needs to wait, and after it has waited,
                    // outstanding[0..index] are not outstanding any longer
                    for ( int i = 0; i <= index; ++i )
                    {
                        outstanding.pop_front();
                    }
                    return waits;
                }
            }
        }
    }
    return -1;
}

// Returns LUT slot to be used for the given LUT operation.
// Sets alreadyInLutMem to true if the LUT is already in SHRAM.
int EthosU85RCSGenerator::AllocateLutSlot(
    std::vector<LutSlot> &lutSlots, const HLCOperation *op, int sizeInSlots, int timestamp, bool &alreadyInLutMem)
{
    alreadyInLutMem = false;
    int totalSlots = int(lutSlots.size());
    if ( sizeInSlots < 0 || sizeInSlots > totalSlots )
    {
        assert(false);
        return 0;
    }
    // Returns least recently used slot, unless the LUT is already in memory
    int allocatedSlot = 0;
    for ( int i = 0; i < totalSlots; i += sizeInSlots )
    {
        if ( lutSlots[i].hlcOp == op )
        {
            // LUT is already in SHRAM
            allocatedSlot = i;
            alreadyInLutMem = true;
            break;
        }
        if ( lutSlots[i].lastUsed < lutSlots[allocatedSlot].lastUsed )
        {
            allocatedSlot = i;
        }
    }
    for ( int j = allocatedSlot; j < allocatedSlot + sizeInSlots; ++j )
    {
        lutSlots[j].hlcOp = op;
        lutSlots[j].lastUsed = timestamp;
    }
    return allocatedSlot;
}

//----------------------------------------------------------------------
// Print
//----------------------------------------------------------------------

int EthosU85RCSGenerator::Disassemble(const uint32_t *in, std::string &op, std::vector<std::pair<std::string, std::string>> &fields)
{
    return isa::disassemble(in, op, fields);
}

//----------------------------------------------------------------------
// Scaling (OFM/IFM/IFM2_SCALE)
//----------------------------------------------------------------------

// Generates OFM_SCALE register for pooling operations
void EthosU85RCSGenerator::GenerateOFMScalingForPooling(HLCOperation *poolOp, bool useGlobalScale)
{
    QuantizedScale ofmScale(1, 0);
    pooling_mode mode = (poolOp->type == OpType::AvgPool && (poolOp->kernel.Size().x > 8 || poolOp->kernel.Size().y > 8)) ? pooling_mode::SUM : pooling_mode::NONE;

    if ( mode == pooling_mode::SUM && useGlobalScale )
    {
        uint32_t scale = 1;
        int shift = 0;
        QuantizePoolingScale(poolOp->kernel.ElementsWH(), ethosU85Scaling::GetScaleFactor(poolOp), 0, scale, shift, 31);
        ofmScale = QuantizedScale(int32_t(scale), shift);
    }
    else if ( poolOp->type == OpType::ArgMax && useGlobalScale )
    {
        // Argmax requires custom scaling to separate values and indices
        // We don't use RescalePooling as this is true regardless of QuantizationType
        const auto &unitQuant = Quantization::Unit();
        assert((poolOp->ofm.quantization.scales.empty() || poolOp->ofm.quantization.EqualScales(unitQuant)) && "Argmax without unit scale");
        ofmScale = QuantizedScale(1, 16);
    }
    else
    {
        bool isNoOp = _arch->UseAvgPoolNop(poolOp->type);
        ethosU85Scaling::RescalePooling(poolOp, isNoOp);
        if ( useGlobalScale && !poolOp->ofm.quantization.scales.empty() )
        {
            ofmScale = poolOp->ofm.quantization.scales[0];
            assert(unsigned(ofmScale.shift) < 64);
        }
    }
    Emit(isa::npu_set_ofm_scale_t(uint32_t(ofmScale.shift), 0, GetOfmRoundingMode(poolOp), ofmScale.scale));
}

// Generates OFM/IFM/IMF2_SCALE registers for elementwise operators.
void EthosU85RCSGenerator::GenerateScalingForElementwise(HLCOperation *op)
{
    auto opType = op->type;
    int ifmCnt = int(op->ifm.size());
    bool setIfmDoubleRound = op->ifm[0].quantization.type == QuantizationType::TFLITE;

    QuantizedScale input1Scale(QuantizedScale::Unit());
    QuantizedScale input2Scale(QuantizedScale::Unit());
    QuantizedScale outScale(QuantizedScale::Unit());
    ethosU85Scaling::RescaleElementwise(op);

    auto ifmRoundMode = GetIfmRoundingMode(op, 0);
    uint32_t ifmDoubleRound = 0;
    auto ofmRoundMode = GetOfmRoundingMode(op);
    uint32_t ofmDoubleRound = 0;

    if ( !op->ofm.quantization.scales.empty() ) outScale = op->ofm.quantization.scales[0];
    if ( !op->ifm[0].quantization.scales.empty() ) input1Scale = op->ifm[0].quantization.scales[0];
    if ( ifmCnt == 2 )
    {
        if ( !op->ifm[1].quantization.scales.empty() ) input2Scale = op->ifm[1].quantization.scales[0];
    }

    if ( opType == OpType::LeakyRelu )
    {
        // ifm1Scale is used for rescaling and ifm2Scale is used for alpha
        float alpha = op->parameters.leaky_relu.alpha;
        float ifm1Scale = float(input1Scale.Dequantize());
        input2Scale = QuantizedScale(alpha * ifm1Scale);
        ifmCnt = 2;
    }
    else if ( opType == OpType::Add || opType == OpType::Sub )
    {
        // Double round is used to compensate for the left shift that happens in AdvancedElementwiseAddSubScale
        if ( setIfmDoubleRound ) ifmDoubleRound = op->ifm[0].dataType == DataType::Int8 ? 20 : 15;
    }

    // Check that scaling is valid
    assert(!(opType == OpType::Div || opType == OpType::Mul) || (input1Scale == QuantizedScale::Unit()));
    assert(!(opType == OpType::Div || opType == OpType::Mul) || (input2Scale == QuantizedScale::Unit()));
    assert(!(opType == OpType::Div || opType == OpType::SHL || opType == OpType::SHR || opType == OpType::Asr) ||
           (outScale == QuantizedScale::Unit()));

    assert(unsigned(input1Scale.shift) < 64);
    Emit(isa::npu_set_ifm_scale_t(input1Scale.shift, ifmDoubleRound, ifmRoundMode, input1Scale.scale));
    if ( ifmCnt == 2 )
    {
        assert(unsigned(input2Scale.shift) < 64);
        // Use ifmRoundeMode since ifmCnt is forced to 2 for LeakyRelu
        auto ifm2RoundMode = opType == OpType::LeakyRelu ? ifmRoundMode : GetIfmRoundingMode(op, 1);
        Emit(isa::npu_set_ifm2_scale_t(input2Scale.shift, ifmDoubleRound, ifm2RoundMode, input2Scale.scale));
    }
    assert(unsigned(outScale.shift) < 64);
    Emit(isa::npu_set_ofm_scale_t(outScale.shift, ofmDoubleRound, ofmRoundMode, outScale.scale));
}

//----------------------------------------------------------------------
// BLOCKDEP calculation
//----------------------------------------------------------------------

static Shape CalcIFMJobShape(const Shape &ofmBlock, Kernel *kernel, int ifmBlockDepth)
{
    Point2i dilatedSize = kernel->DilatedWH();
    // TODO MLBEDSW-8498: Consider ifm_upscale_mode for job-shape calculations
    int h = RequiredInputSize(ofmBlock.Height(), kernel->Stride().y, dilatedSize.y, 1);
    int w = RequiredInputSize(ofmBlock.Width(), kernel->Stride().x, dilatedSize.x, 1);
    return Shape(1, h, w, ifmBlockDepth);
}

// Given the area and block size, adds the first/last jobs (depending on fromStart) to jobs.
// - area: total amount of work to perform
// - jobShape: size of each job
// - fromStart: if true, the first jobs are added, if false, the last jobs are added
//   (in that case, the very last job is added last)
void EthosU85RCSGenerator::GetJobs(const Box &area, const Shape &jobShape, int nrJobsToGet, bool fromStart, std::vector<Box> &jobs)
{
    Shape jobSplit = Shape::DivRoundUp(area.End() - area.Start(), jobShape);
    int z = jobSplit.Depth();
    int w = jobSplit.Width();
    int h = jobSplit.Height();
    int n = z * w * h;  // n = total number of jobs for the whole area
    const auto &start = area.Start().Extract(-3, -2, -1);
    const auto &end = area.End().Extract(-3, -2, -1);
    int firstJob = fromStart ? 0 : std::max(0, n - nrJobsToGet);
    int lastJob = fromStart ? std::min(n, nrJobsToGet) : n;
    for ( int i = firstJob; i < lastJob; ++i )
    {
        Shape from = Shape(start.Height() + (i / (z * w)) * jobShape.Height(),
            start.Width() + ((i / z) % w) * jobShape.Width(), start.Depth() + (i % z) * jobShape.Depth());

        jobs.emplace_back(from, Shape::Min(from + jobShape, end));
    }
}

// Calculates the value for the BLOCKDEP register
int EthosU85RCSGenerator::CalcBlockDep(HLCStripe *prevStripe, HLCStripe *stripe)
{
    if ( prevStripe == nullptr )
    {
        return 0;
    }
    const auto &op = stripe->operation;
    const auto &prevOp = prevStripe->operation;
    auto prevOfm = prevOp->ofm;

    if ( prevOp->subOps.size() )
    {
        prevOfm = prevOp->subOps.back().ofm;
    }
    // TODO MLBEDSW-9625: Compute block-dependency for transposed ofms
    if ( !IsNone(prevOfm.transpose) )
    {
        return 0;
    }
    // TODO MLBEDSW-9626: Compute block-dependency for reversed ofms
    if ( prevOfm.reverse != ReverseType::None )
    {
        return 0;
    }

    int ifmIndex = (op->ifm.size() > 1 && op->ifm[1].address == prevOfm.address && op->ifm[1].memArea == prevOfm.memArea) ? 1 : 0;
    const auto &ifm = op->ifm[ifmIndex];
    int maxJobs = _arch->MaxBlockdep();
    if ( ifm.address != prevOfm.address || ifm.memArea != prevOfm.memArea )
    {
        for ( const auto &fm : op->ifm )
        {
            if ( fm.memArea == prevOfm.memArea &&
                 Overlaps(fm.address, fm.address + fm.AllocationSizeBytes(), prevOfm.address, prevOfm.address + prevOfm.AllocationSizeBytes()) )
            {
                // Previous OFM overlaps with current IFM
                return 0;
            }
        }
        // Previous operation does not produce current operation's IFM
        return maxJobs;
    }
    if ( op->ifm.size() > 1 && ifm.AllocationSizeBytes() < op->ifm[1 - ifmIndex].AllocationSizeBytes() )
    {
        // Prev OFM produces IFM2 which is broadcasted (this should be rare)
        return 0;
    }
    if ( prevOfm.shape != ifm.shape )
    {
        // OFM has been reshaped; the job overlap calculations below do not work in this case
        return 0;
    }
    // Previous operation produces current operations IFM
    auto prevConfig = static_cast<EthosU85OpConfig *>(prevOp->config);
    if ( !prevConfig )
    {
        // Previous operation doesn't have a block config
        return 0;
    }
    Shape prevBlock = prevConfig->OfmBlock();
    auto config = static_cast<EthosU85OpConfig *>(op->config);
    if ( !config )
    {
        // Current operation doesn't have a block config
        return 0;
    }
    Shape currBlock = CalcIFMJobShape(config->OfmBlock(), &op->kernel, config->IfmBlock().Depth());
    // Get the last few jobs from the previous operation (each job produces a part of the current op's IFM)
    std::vector<Box> lastPrevJobs;
    GetJobs(prevStripe->ofmArea, prevBlock, maxJobs, false, lastPrevJobs);
    // Get the first few jobs from the current operation (each job consumes a part of the current op's IFM)
    std::vector<Box> firstCurrJobs;
    GetJobs(stripe->ifmAreas[ifmIndex], currBlock, maxJobs, true, firstCurrJobs);
    // Find the highest blockdep such that there is no overlap between
    // any job from the previous op with any job from the current op during blockdep jobs
    int sz = int(std::min(lastPrevJobs.size(), firstCurrJobs.size()));
    int prevLastIx = int(lastPrevJobs.size()) - 1;
    for ( int blockdep = 0; blockdep < sz; ++blockdep )
    {
        bool overlaps = false;
        for ( int i = 0; !overlaps && i <= blockdep; ++i )
        {
            for ( int j = blockdep - i; !overlaps && i + j <= blockdep; ++j )
            {
                if ( firstCurrJobs[i].Overlaps(lastPrevJobs[prevLastIx - j]) )
                {
                    overlaps = true;
                }
            }
        }
        if ( overlaps )
        {
            return blockdep;
        }
    }
    // No overlap found
    return sz;
}

//----------------------------------------------------------------------
// Register generation
//----------------------------------------------------------------------

void EthosU85RCSGenerator::GeneratePadding(const HLCPadding &padding)
{
    Emit(isa::npu_set_ifm_pad_top_t(padding.top));
    Emit(isa::npu_set_ifm_pad_left_t(padding.left));
    Emit(isa::npu_set_ifm_pad_bottom_t(padding.bottom));
    Emit(isa::npu_set_ifm_pad_right_t(padding.right));
}

// Generates ACTIVATION registers
void EthosU85RCSGenerator::GenerateActivation(const HLCStripe *stripe, MemoryAccesses &memoryAccesses)
{
    const HLCOperation *op = stripe->operation.get();
    OpType opType = OpType::None;
    const HLCParameters *parameters = nullptr;
    assert(stripe->opGroup != nullptr);
    EthosU85OpGroup *opGroup = static_cast<EthosU85OpGroup *>(stripe->opGroup);
    if ( IsActivation(op->type) )
    {
        // Non-fused activation
        opType = op->type;
        parameters = &op->parameters;
    }
    else
    {
        for ( auto &subOp : op->subOps )
        {
            if ( opGroup->IsFused(subOp.ifm[0].uid) && IsActivation(subOp.type) )
            {
                // Fused activation
                opType = subOp.type;
                parameters = &subOp.parameters;

                // We know there can be only one fused activation
                break;
            }
        }
    }

    auto &ofm = op->ofm;
    // Clamp quantMin/quantMax to valid range, but completely disable clipping if any are at the edge of the range
    int64_t quantizedMin = std::max(IntegerMin(ofm.dataType), IntegerMin(DataType::Int16));
    int64_t quantizedMax = std::min(IntegerMax(ofm.dataType), IntegerMax(DataType::UInt16));
    auto clipRange = DataTypeSizeBits(ofm.dataType) > 16 ? activation_clip_range::NONE : activation_clip_range::B16;
    if ( ofm.quantization.quantMin.size() )
    {
        if ( ofm.quantization.quantMin[0] > std::numeric_limits<int64_t>::min() )
            quantizedMin = std::max(quantizedMin, ofm.quantization.quantMin[0]);
        else clipRange = activation_clip_range::NONE;
    }
    if ( ofm.quantization.quantMax.size() )
    {
        if ( ofm.quantization.quantMax[0] < std::numeric_limits<int64_t>::max() )
            quantizedMax = std::min(quantizedMax, ofm.quantization.quantMax[0]);
        else clipRange = activation_clip_range::NONE;
    }

    auto act = activation_function::LUT_NONE;
    uint32_t tableIndex = 0;
    if ( IsLUTType(opType) )
    {
        auto &lutParams = parameters->lut;
        int lutSize = lutParams.sizeBytes;

        auto pos = _stripeToLutSlot.find(stripe);
        if ( pos != _stripeToLutSlot.end() )
        {
            tableIndex = pos->second;
        }
        else
        {
            assert(false && "Command uses lut, but no lut info found");
        }

        // tableIndex is based on 8 slots of size 256 and alignment is the same as the LUT size
        // Hardware expects 0-7 tables for 256 LUT
        //                  0-4 tables for 512 LUT
        //                  0-1 tables for  1k LUT
        //                    1 table  for  2k LUT
        // So for 512 and 1k the tableIndex is adjusted below
        switch ( ofm.dataType )
        {
            case DataType::Int8:
                assert(lutSize == 256);
                assert(lutParams.ifmType == DataType::Int8);
                act = activation_function::LUT_S8_S8;
                break;
            case DataType::UInt8:
                assert(lutSize == 256);
                assert(lutParams.ifmType == DataType::UInt8);
                act = activation_function::LUT_U8_U8;
                break;
            case DataType::Int16:
                if ( lutParams.ifmType == DataType::Int8 )
                {
                    assert(lutSize == 512 && tableIndex % 2 == 0);
                    act = activation_function::LUT_S8_S16;
                }
                else
                {
                    assert(lutSize == 2048 && tableIndex == 0);
                    assert(lutParams.ifmType == DataType::Int16);
                    if ( opType == OpType::LUT ) act = activation_function::LUT_S16_S16;
                    else if ( opType == OpType::Sigmoid ) act = activation_function::LUT_SIGMOID;
                    else act = activation_function::LUT_TANH;
                }
                break;
            case DataType::Int32:
                if ( lutParams.ifmType == DataType::Int8 )
                {
                    assert(lutSize == 1024 && tableIndex % 4 == 0);
                    act = activation_function::LUT_S8_S32;
                }
                else
                {
                    assert(lutSize == 2048 && tableIndex == 0);
                    assert(lutParams.ifmType == DataType::Int16);
                    act = activation_function::LUT_S16_S32;
                }
                break;
            default:
                assert(false && "Unsupported LUT table");
                break;
        }

        // Adjust table for 512 and 1k
        tableIndex = tableIndex / (lutSize / ArchEthosU85::LUT_SLOT_SIZE);

        Address lutStart = Address(tableIndex) * lutSize;
        memoryAccesses.emplace_back(AccessDirection::Read, _arch->LUTMemory(), lutStart, lutStart + lutSize);
    }
    assert(quantizedMin <= std::numeric_limits<uint16_t>::max());
    assert(quantizedMin >= std::numeric_limits<int16_t>::min());
    assert(quantizedMax <= std::numeric_limits<uint16_t>::max());
    assert(quantizedMax >= std::numeric_limits<int16_t>::min());
    Emit(isa::npu_set_activation_t(act, tableIndex, clipRange));
    Emit(isa::npu_set_activation_min_t(uint32_t(quantizedMin)));
    Emit(isa::npu_set_activation_max_t(uint32_t(quantizedMax)));
}

// Generates KERNEL related registers
void EthosU85RCSGenerator::GenerateKernel(const Kernel &kernel, bool partKernel)
{
    auto dilatedWH = kernel.DilatedWH();
    Emit(isa::npu_set_kernel_height_m1_t(dilatedWH.y - 1));
    Emit(isa::npu_set_kernel_width_m1_t(dilatedWH.x - 1));
    uint32_t stride_x_lsb = (kernel.Stride().x - 1) & 1;
    uint32_t stride_y_lsb = (kernel.Stride().y - 1) & 1;
    uint32_t stride_x_msb = ((kernel.Stride().x - 1) >> 1) & 1;
    uint32_t stride_y_msb = ((kernel.Stride().y - 1) >> 1) & 1;
    auto weightOrder = partKernel ? weight_order::PART_KERNEL_FIRST : weight_order::DEPTH_FIRST;
    kernel_dilation dilation_x = kernel_dilation(kernel.Dilation().x - 1);
    kernel_dilation dilation_y = kernel_dilation(kernel.Dilation().y - 1);
    kernel_decomposition decomposition = kernel_decomposition::D8X8;  //  Kernel decomposition
    Emit(isa::npu_set_kernel_stride_t(
        stride_x_lsb, stride_y_lsb, weightOrder, dilation_x, dilation_y, decomposition, stride_x_msb, stride_y_msb));
}


// Generates IFM_BROADCAST/IFM2_BROADCAST register for binary elementwise operations
static broadcast_mode CalculateBroadcast(const Shape &shape1, const Shape &shape2)
{
    uint8_t mode = uint8_t(broadcast_mode::NONE);
    if ( shape1.Height() < shape2.Height() && shape1.Height() == 1 )
    {
        // Broadcast in 'H' dimension
        mode |= uint8_t(broadcast_mode::H);
    }
    if ( shape1.Width() < shape2.Width() && shape1.Width() == 1 )
    {
        // Broadcast in 'W' dimension
        mode |= uint8_t(broadcast_mode::W);
    }
    if ( shape1.Depth() < shape2.Depth() && shape1.Depth() == 1 )
    {
        // Broadcast in 'C' dimension
        mode |= uint8_t(broadcast_mode::C);
    }
    return broadcast_mode(mode);
}

void EthosU85RCSGenerator::GenerateInputBroadcast(const Shape &ifmShape, const Shape &ifm2Shape, bool ifmIsScalar, bool ifm2IsScalar)
{
    assert(!(ifmIsScalar && ifm2IsScalar));
    broadcast_mode mode1 = ifmIsScalar ? broadcast_mode::SCALAR : broadcast_mode::NONE;
    broadcast_mode mode2 = ifm2IsScalar ? broadcast_mode::SCALAR : broadcast_mode::NONE;
    if ( ifmShape && ifm2Shape )
    {
        if ( !ifmIsScalar ) mode1 = CalculateBroadcast(ifmShape, ifm2Shape);
        if ( !ifm2IsScalar ) mode2 = CalculateBroadcast(ifm2Shape, ifmShape);
    }
    Emit(isa::npu_set_ifm_broadcast_t(mode1));
    Emit(isa::npu_set_ifm2_broadcast_t(mode2));
}

// Generates IFM_PRECISION register
void EthosU85RCSGenerator::GenerateIFMPrecision(const HLCFeatureMap &fm, bool chained, bool isScalar, DataType dataType)
{
    activation_type type = ToActivationType(dataType);
    activation_precision precision = ToActivationPrecision(dataType);
    activation_format format = ToActivationFormat(fm.format);
    activation_storage storage = activation_storage::TILE2X2;
    if ( chained )
    {
        storage = activation_storage::CHAINED;
    }
    else if ( isScalar )
    {
        storage = activation_storage::NONE;
    }
    Emit(isa::npu_set_ifm_precision_t(type, precision, format, storage));
}

// Generates IFM2_PRECISION register
void EthosU85RCSGenerator::GenerateIFM2Precision(const HLCFeatureMap &fm, bool chained, bool isScalar)
{
    activation_type type = ToActivationType(fm.dataType);
    activation_precision precision = ToActivationPrecision(fm.dataType);
    activation_format format = ToActivationFormat(fm.format);
    activation_storage storage = activation_storage::TILE2X2;
    if ( chained )
    {
        storage = activation_storage::CHAINED;
    }
    else if ( isScalar )
    {
        storage = activation_storage::NONE;
    }
    Emit(isa::npu_set_ifm2_precision_t(type, precision, format, storage));
}

// Generates OFM_PRECISION register
void EthosU85RCSGenerator::GenerateOFMPrecision(const HLCFeatureMap &fm, bool chained, bool useGlobalScale, bool enable_output)
{
    activation_type type = ToActivationType(fm.dataType);
    activation_precision precision = ToActivationPrecision(fm.dataType);
    activation_format format = ToActivationFormat(fm.format);
    auto scaleMode = useGlobalScale ? ofm_scale_mode::GLOBAL : ofm_scale_mode::PER_CHANNEL;
    activation_reverse reverse = ToActivationReverse(fm.reverse);
    activation_transpose transpose = ToActivationTranspose(fm.transpose);
    activation_storage storage = enable_output ? activation_storage::TILE2X2 : activation_storage::NONE;
    if ( chained )
    {
        storage = activation_storage::CHAINED;
        assert(reverse == activation_reverse::NONE && "Can't combine chaining and reverse");
        assert(transpose == activation_transpose::HWC && "Can't combine chaining and transpose");
    }
    if ( reverse != activation_reverse::NONE )
    {
        assert(transpose == activation_transpose::HWC && "Can't combine reverse and transpose");
        assert(storage != activation_storage::CHAINED && "Can't combine reverse and chaining");
    }
    if ( transpose != activation_transpose::HWC )
    {
        assert(reverse == activation_reverse::NONE && "Can't combine transpose and reverse");
        assert(storage != activation_storage::CHAINED && "Can't combine transpose and chaining");
    }
    Emit(isa::npu_set_ofm_precision_t(type, precision, format, scaleMode, reverse, transpose, storage));
}

// Generates common IFM registers
void EthosU85RCSGenerator::GenerateIFM(OpType opType, const HLCFeatureMap &fm, const Box &inputArea, bool isScalar,
    int32_t scalarValue, int chainBuffer, bool ifm2Chained)
{
    if ( isScalar )
    {
        Emit(isa::npu_set_op_scalar_t(uint32_t(scalarValue)));
    }
    else
    {
        if ( chainBuffer != INVALID_CB )
        {
            // chained ifm1
            Emit(isa::npu_set_ifm_region_t(chainBuffer));
        }
        else
        {
            CheckAddresses(fm);
            Emit(isa::npu_set_ifm_region_t(ToRegion(fm.memArea)));
            Shape strides = fm.strides;
            auto tiles = GetTiles(fm, strides, inputArea);
            auto boxSize = inputArea.SizeShape();
            // IFM_BASE registers
            Emit(isa::npu_set_ifm_base0_t(tiles.address[0]));
            Emit(isa::npu_set_ifm_base1_t(tiles.address[1]));
            Emit(isa::npu_set_ifm_base2_t(tiles.address[2]));
            Emit(isa::npu_set_ifm_base3_t(tiles.address[3]));
            // Tile related registers
            Emit(isa::npu_set_ifm_height0_m1_t(tiles.height0 - 1));
            Emit(isa::npu_set_ifm_height1_m1_t(tiles.height1 - 1));
            Emit(isa::npu_set_ifm_width0_m1_t(tiles.width0 - 1));
            if ( !ifm2Chained )
            {
                // set_ifm_depth is shared between ifm1/ifm2.
                // but should only be emitted if neither of the inputs are chained
                Emit(isa::npu_set_ifm_depth_m1_t(boxSize.Depth() - 1));
            }
            //  IFM_STRIDE registers
            Emit(isa::npu_set_ifm_stride_y_t(strides.Height() * fm.stepXY.y));
            Emit(isa::npu_set_ifm_stride_x_t(strides.Width() * fm.stepXY.x));
            Emit(isa::npu_set_ifm_stride_c_t(strides.Depth()));
        }
    }
    // IFM_ZERO_POINT register
    auto &quant = fm.quantization;
    uint32_t zp = quant.zeroPoints.empty() ? 0 : uint32_t(quant.zeroPoints[0]);

    // ifm zero-point is force-emitted if ofm is chained
    Emit(isa::npu_set_ifm_zero_point_t(zp));
}

// Generates common IFM2 registers
void EthosU85RCSGenerator::GenerateIFM2(
    OpType opType, const HLCFeatureMap &fm, const Box &inputArea, bool isScalar, int32_t scalarValue, int chainBuffer)
{
    if ( isScalar )
    {
        Emit(isa::npu_set_op_scalar_t(uint32_t(scalarValue)));
    }
    else
    {
        if ( chainBuffer != INVALID_CB )
        {
            // chained ifm2
            Emit(isa::npu_set_ifm2_region_t(chainBuffer));
        }
        else
        {
            CheckAddresses(fm);
            Emit(isa::npu_set_ifm2_region_t(ToRegion(fm.memArea)));
            Shape strides = fm.strides;
            auto tiles = GetTiles(fm, strides, inputArea);
            // IFM2_BASE registers
            Emit(isa::npu_set_ifm2_base0_t(tiles.address[0]));
            Emit(isa::npu_set_ifm2_base1_t(tiles.address[1]));
            Emit(isa::npu_set_ifm2_base2_t(tiles.address[2]));
            Emit(isa::npu_set_ifm2_base3_t(tiles.address[3]));
            // Tile related registers
            Emit(isa::npu_set_ifm2_height0_m1_t(tiles.height0 - 1));
            Emit(isa::npu_set_ifm2_height1_m1_t(tiles.height1 - 1));
            Emit(isa::npu_set_ifm2_width0_m1_t(tiles.width0 - 1));
            // IFM2_STRIDE registers
            Emit(isa::npu_set_ifm2_stride_y_t(strides.Height() * fm.stepXY.y));
            Emit(isa::npu_set_ifm2_stride_x_t(strides.Width() * fm.stepXY.x));
            Emit(isa::npu_set_ifm2_stride_c_t(strides.Depth()));
        }
    }
    // IFM2_ZERO_POINT register
    auto &quant = fm.quantization;
    uint32_t zp = quant.zeroPoints.empty() ? 0 : uint32_t(quant.zeroPoints[0]);

    // ifm zero-point is force-emitted if ofm is chained
    Emit(isa::npu_set_ifm2_zero_point_t(zp));
}

// Generates OFM registers
void EthosU85RCSGenerator::GenerateOFM(OpType opType, const HLCFeatureMap &fm, const Box &outputArea, int chainBuffer)
{
    auto boxSize = outputArea.SizeShape().Unpermute(uint32_t(fm.transpose));
    if ( chainBuffer != INVALID_CB )
    {
        Emit(isa::npu_set_ofm_region_t(chainBuffer));

        Emit(isa::npu_set_ofm_height_m1_t(DivRoundUp(boxSize.Height(), fm.stepXY.y) - 1));
        Emit(isa::npu_set_ofm_width_m1_t(DivRoundUp(boxSize.Width(), fm.stepXY.x) - 1));
        Emit(isa::npu_set_ofm_depth_m1_t(boxSize.Depth() - 1));
    }
    else
    {
        CheckAddresses(fm);
        Emit(isa::npu_set_ofm_region_t(ToRegion(fm.memArea)));
        Shape strides = fm.strides;
        auto tiles = GetTiles(fm, strides, outputArea);
        // OFM_BASE registers
        Emit(isa::npu_set_ofm_base0_t(tiles.address[0]));
        Emit(isa::npu_set_ofm_base1_t(tiles.address[1]));
        Emit(isa::npu_set_ofm_base2_t(tiles.address[2]));
        Emit(isa::npu_set_ofm_base3_t(tiles.address[3]));
        // OFM size (shape *before* transposition)
        Emit(isa::npu_set_ofm_height_m1_t(DivRoundUp(boxSize.Height(), fm.stepXY.y) - 1));
        Emit(isa::npu_set_ofm_width_m1_t(DivRoundUp(boxSize.Width(), fm.stepXY.x) - 1));
        Emit(isa::npu_set_ofm_depth_m1_t(boxSize.Depth() - 1));
        // Tile related registers (shape *after* transposition)
        Emit(isa::npu_set_ofm_height0_m1_t(tiles.height0 - 1));
        Emit(isa::npu_set_ofm_height1_m1_t(tiles.height1 - 1));
        Emit(isa::npu_set_ofm_width0_m1_t(tiles.width0 - 1));
        // OFM_STRIDE registers
        Emit(isa::npu_set_ofm_stride_y_t(strides.Height() * fm.stepXY.y));
        Emit(isa::npu_set_ofm_stride_x_t(strides.Width() * fm.stepXY.x));
        Emit(isa::npu_set_ofm_stride_c_t(strides.Depth()));
    }
    // OFM_ZERO_POINT register
    auto &quant = fm.quantization;
    uint32_t zp = quant.zeroPoints.empty() ? 0 : uint32_t(quant.zeroPoints[0]);
    Emit(isa::npu_set_ofm_zero_point_t(zp));
}

// Generates WEIGHT registers
void EthosU85RCSGenerator::GenerateWeights(const HLCStripe *stripe, MemoryAccesses &memoryAccesses)
{
    auto weights = stripe->operation->weights.get();
    if ( weights == nullptr )
    {
        return;
    }

    EthosU85OpConfig *config = static_cast<EthosU85OpConfig *>(stripe->operation->config);

    auto wgtFormat = (weights->format % WeightFormat::Fast) ? weight_format::FWD : weight_format::SWD;
    auto wgtSparsity = (weights->format % WeightFormat::Sparse2_4) ? weight_sparsity::SPARSE_2_4 : weight_sparsity::NONE;
    Emit(isa::npu_set_weight_format_t(wgtFormat, wgtSparsity));

    int depth = stripe->weightRangeDepth;
    Emit(isa::npu_set_weight_region_t(ToRegion(weights->memArea)));
    int offset = 0;
    for ( int i = 0; i < _arch->_cores; ++i )
    {
        Address address = 0;
        int length = 0;
        auto item = weights->encodedRanges.find(WeightKey(i, depth));
        if ( item != weights->encodedRanges.end() )
        {
            const auto &range = item->second;
            int doubleBufferOffset = GetDoubleBufferOffset(weights, range.index);
            address = weights->address + offset + range.weightOffset + doubleBufferOffset;
            length = RoundAway(range.weightBytes, 16);
            CheckAddressRange(weights->memArea.memory, address, length);
            memoryAccesses.emplace_back(AccessDirection::Read, weights->memArea, address, address + length);
            offset += RoundAway(range.TotalBytes(), 16);
        }

        switch ( i )
        {
            case 0:
                if ( length != 0 ) Emit(isa::npu_set_weight_base_t(address));
                Emit(isa::npu_set_weight_length_t(length));
                break;
            case 1:
                if ( length != 0 ) Emit(isa::npu_set_weight1_base_t(address));
                Emit(isa::npu_set_weight1_length_t(length));
                break;
            case 2:
                if ( length != 0 ) Emit(isa::npu_set_weight2_base_t(address));
                Emit(isa::npu_set_weight2_length_t(length));
                break;
            case 3:
                if ( length != 0 ) Emit(isa::npu_set_weight3_base_t(address));
                Emit(isa::npu_set_weight3_length_t(length));
                break;
            default:
                assert(false);
        }
    }
}

// Generates SCALE registers
void EthosU85RCSGenerator::GenerateScales(const HLCStripe *stripe, MemoryAccesses &memoryAccesses)
{
    auto scales = stripe->operation->scales.get();
    if ( scales == nullptr )
    {
        assert(!stripe->operation->weights);
        return;
    }
    int depth = stripe->weightRangeDepth;
    Emit(isa::npu_set_scale_region_t(ToRegion(scales->memArea)));
    auto item0 = scales->encodedRanges.find(WeightKey(0, depth));
    assert(item0 != scales->encodedRanges.end());
    auto &range0 = item0->second;
    int doubleBufferOffset = GetDoubleBufferOffset(scales, range0.index);
    Address address = scales->address + doubleBufferOffset;
    int length = RoundAway(range0.scaleBytes, 16);

    CheckAddressRange(scales->memArea.memory, address, length);
    Emit(isa::npu_set_scale_base_t(address));
    Emit(isa::npu_set_scale_length_t(length));
    memoryAccesses.emplace_back(AccessDirection::Read, scales->memArea, address, address + length);
}

// Generates OFM_BLK_HEIGHT/WIDTH/DEPTH registers
void EthosU85RCSGenerator::GenerateBlockConfig(const EthosU85OpConfig *config, const HLCFeatureMap &fm)
{
    Shape blk = config->OfmBlock();
    // Block constraints for transpose
    switch ( fm.transpose )
    {
        case TransposeType::NWHC:
            assert(blk.Width() <= _arch->_ofmBlockMax.Height() && "Illegal OFM block height");
            assert(blk.Height() <= _arch->_ofmBlockMax.Width() && "Illegal OFM block width");
            break;
        case TransposeType::NHCW:
            assert(blk.Depth() <= _arch->_ofmBlockMax.Width() && "Illegal OFM block width");
            assert(blk.Width() <= _arch->_ofmBlockMax.Depth() && "Illegal OFM block depth");
            // Width must be multiple of 16 if brick format
            assert((fm.format != TensorFormat::NHCWB16 || blk.Width() % 16 == 0) && "Illegal OFM block width for brick format");
            break;
        case TransposeType::NCWH:
            assert(blk.Depth() <= _arch->_ofmBlockMax.Height() && "Illegal OFM block height");
            assert(blk.Height() <= _arch->_ofmBlockMax.Depth() && "Illegal OFM block depth");
            // Height must be multiple of 16 if brick format
            assert((fm.format != TensorFormat::NHCWB16 || blk.Height() % 16 == 0) && "Illegal OFM block height for brick format");
            break;
        case TransposeType::NCHW:
            assert(blk.Depth() <= _arch->_ofmBlockMax.Height() && "Illegal OFM block height");
            assert(blk.Height() <= _arch->_ofmBlockMax.Width() && "Illegal OFM block width");
            assert(blk.Width() <= _arch->_ofmBlockMax.Depth() && "Illegal OFM block depth");
            // Width must be multiple of 16 if brick format
            assert((fm.format != TensorFormat::NHCWB16 || blk.Width() % 16 == 0) && "Illegal OFM block width for brick format");
            break;
        case TransposeType::NWCH:
            assert(blk.Width() <= _arch->_ofmBlockMax.Height() && "Illegal OFM block height");
            assert(blk.Depth() <= _arch->_ofmBlockMax.Width() && "Illegal OFM block width");
            assert(blk.Height() <= _arch->_ofmBlockMax.Depth() && "Illegal OFM block depth");
            // Height must be multiple of 16 if brick format
            assert((fm.format != TensorFormat::NHCWB16 || blk.Height() % 16 == 0) && "Illegal OFM block height for brick format");
            break;
        default:
            break;
    }
    // OFM block (shape *before* transposition)
    Emit(isa::npu_set_ofm_blk_height_m1_t(blk.Height() - 1));
    Emit(isa::npu_set_ofm_blk_width_m1_t(blk.Width() - 1));
    Emit(isa::npu_set_ofm_blk_depth_m1_t(blk.Depth() - 1));
}

// Generates ACC_FORMAT register
void EthosU85RCSGenerator::GenerateAccFormat(const HLCStripe *stripe)
{
    auto opType = stripe->operation->type;
    EthosU85OpConfig *config = static_cast<EthosU85OpConfig *>(stripe->operation->config);
    EthosU85Accumulator accType;
    ArchAccumulatorSource accSrc;

    if ( _arch->UseNullPool(opType, DataTypeSizeBits(stripe->operation->ifm[0].dataType)) )
    {
        accType = DataTypeSizeBits(stripe->operation->ifm[0].dataType) == 32 ? EthosU85Accumulator::Acc32 : EthosU85Accumulator::Acc48;
        accSrc = ArchAccumulatorSource::Ifm2;
    }
    else
    {
        accType = config->Acc();
        accSrc = config->AccSource();
        assert(
            accSrc != ArchAccumulatorSource::Ifm2 ||
            (stripe->operation->ifm[1].dataType == DataType::Int32 && accType == EthosU85Accumulator::Acc32) ||
            (stripe->operation->ifm[1].dataType == DataType::Int64 && accType == EthosU85Accumulator::Acc48));
    }

    acc_format format = accType == EthosU85Accumulator::Acc32 ? acc_format::I32 : acc_format::I48;

    auto w = config->OfmUBlock().Width();
    auto h = config->OfmUBlock().Height();
    microblock block = microblock::U1X1;

    switch ( unsigned(h) << 4 | w )
    {
        case 0x11:
            block = microblock::U1X1;
            break;
        case 0x12:
            block = microblock::U1X2;
            break;
        case 0x14:
            block = microblock::U1X4;
            break;
        case 0x22:
            block = microblock::U2X2;
            break;
        case 0x24:
            block = microblock::U2X4;
            break;
        case 0x44:
            block = microblock::U4X4;
            break;
        default:
            assert(false && "Invalid microblock");
    }
    acc_input input;
    switch ( accSrc )
    {
        case ArchAccumulatorSource::Acc:
            input = acc_input::KEEP;
            break;
        case ArchAccumulatorSource::Ifm2:
            input = acc_input::IFM2;
            break;
        case ArchAccumulatorSource::Reset:
        default:
            input = acc_input::RESET;
    }
    acc_output output = config->AccOutputEnabled() ? acc_output::ENABLE : acc_output::DISABLE;

    Emit(isa::npu_set_acc_format_t(format, input, output, block));
}

// Calculates and generates KERNEL_WAIT or DMA_WAIT register
void EthosU85RCSGenerator::GenerateWaits(bool isKernelWait, const MemoryAccesses &memoryAccesses, std::deque<MemoryAccesses> &outstandingAccesses)
{
    int waits = CalcCommandWaits(memoryAccesses, outstandingAccesses);
    if ( waits >= 0 )
    {
        if ( isKernelWait )
        {
            Emit(isa::npu_op_kernel_wait_t(waits));
        }
        else
        {
            Emit(isa::npu_op_dma_wait_t(waits));
        }
    }
}

void EthosU85RCSGenerator::UpdateMemoryAccesses(const MemoryAccesses &memoryAccesses, std::deque<MemoryAccesses> &accessesToUpdate, int maxWaits)
{
    accessesToUpdate.push_back(memoryAccesses);
    if ( int(accessesToUpdate.size()) > maxWaits )
    {
        accessesToUpdate.pop_front();
    }
}

// Inserts DMA commands for copying LUTs from constant memory
// to LUT memory
std::vector<std::unique_ptr<HighLevelCommand>>
EthosU85RCSGenerator::InsertLUTDMACommands(std::vector<std::unique_ptr<HighLevelCommand>> &cmds)
{
    std::vector<std::unique_ptr<HighLevelCommand>> result;
    int lutSlotSize = ArchEthosU85::LUT_SLOT_SIZE;
    int slots = int(_arch->_lutRam->SizeBytes() / lutSlotSize);
    std::vector<LutSlot> lutSlots(slots);
    int timestamp = 0;
    result.reserve(cmds.size());
    for ( auto &hlc : cmds )
    {
        ++timestamp;
        if ( hlc->IsStripe() )
        {
            auto stripe = static_cast<HLCStripe *>(hlc.get());
            auto op = stripe->operation;
            // TODO MLBEDSW-9142 LUT for chained subOps should be inserted before the primary Op
            const auto &subOps = stripe->operation->subOps;
            auto lutSubOp = std::find_if(
                subOps.begin(), subOps.end(), [](const auto &subOp) { return IsLUTType(subOp.type); });
            if ( IsLUTType(op->type) || (lutSubOp != subOps.end()) )
            {
                const auto &srcTens = IsLUTType(op->type) ? op->parameters.lut : lutSubOp->parameters.lut;
                assert(srcTens.sizeBytes % lutSlotSize == 0);
                bool alreadyInLutMem;
                int sizeInSlots = srcTens.sizeBytes / lutSlotSize;
                int slot = AllocateLutSlot(lutSlots, op.get(), sizeInSlots, timestamp, alreadyInLutMem);
                _stripeToLutSlot[stripe] = slot;

                if ( !alreadyInLutMem )
                {
                    auto dma = std::make_unique<HLCDMA>();
                    dma->srcMemArea = srcTens.memArea;
                    dma->srcAddress = srcTens.address;
                    dma->length = srcTens.sizeBytes;
                    dma->destMemArea = _arch->LUTMemory();
                    dma->destAddress = slot * lutSlotSize;
                    result.push_back(std::move(dma));
                }
            }
        }
        result.push_back(std::move(hlc));
    }
    return result;
}

// Converts TILE operations into 3D (or 2D) DMA operations
std::vector<std::unique_ptr<HighLevelCommand>>
EthosU85RCSGenerator::InsertTileDMACommands(std::vector<std::unique_ptr<HighLevelCommand>> &cmds)
{
    // reshape to 3D-tensor where the width-axis is being tiled
    static auto reshapeFunc = [](Shape &shape, int tiledAxis)
    {
        int height = 1;
        int channel = 1;
        // all axes before tiledAxis are reshaped to height
        for ( int i = 0; i < tiledAxis; i++ )
        {
            height *= shape[i];
        }
        // all axes after tiledAxis are reshaped to channel
        for ( int i = tiledAxis + 1; i < shape.Size(); i++ )
        {
            channel *= shape[i];
        }
        shape = {1, height, shape[tiledAxis], channel};
    };

    std::vector<std::unique_ptr<HighLevelCommand>> result;
    for ( auto &hlc : cmds )
    {
        if ( hlc->IsStripe() )
        {
            auto stripe = static_cast<HLCStripe *>(hlc.get());
            auto op = stripe->operation;
            if ( op->type == OpType::Tile )
            {
                // convert tile-operation to multiple DMA operations
                auto &ifm = op->ifm[0];
                auto &ofm = op->ofm;
                // max-height for 2D/3D DMA operations
                constexpr int maxHeight = (1 << 16) - 1;
                int elemSize = DataTypeSizeBits(ifm.dataType) / 8;
                assert(ifm.format == TensorFormat::NHWC);
                assert(ofm.format == TensorFormat::NHWC);
                const auto &tileParams = op->parameters.tile;
                reshapeFunc(ifm.shape, tileParams.axis);
                reshapeFunc(ofm.shape, tileParams.axis);
                auto srcStrides = Shape::GetStridesForShape(ifm.shape, {1, 1, 1, elemSize});
                auto dstStrides = Shape::GetStridesForShape(ofm.shape, {1, 1, 1, elemSize});
                int srcheightOffset = 0;
                int dstheightOffset = 0;
                int height = ifm.shape.Height();
                // Decompose height in slices if needed
                while ( height > 0 )
                {
                    int heightSlice = std::min(height, maxHeight);
                    for ( int i = 0; i < tileParams.multiplier; i++ )
                    {
                        // create 2D/3D DMA that copies ifm to ofm
                        int dstWidthOffset = i * ifm.shape.Width() * srcStrides.Width();
                        auto dma = std::make_unique<HLCDMA>();
                        dma->srcMemArea = ifm.memArea;
                        dma->srcAddress = ifm.address + srcheightOffset;
                        dma->srcStrides = srcStrides;
                        dma->length = ifm.shape.Depth() * elemSize;
                        dma->sizes = Shape(heightSlice, ifm.shape.Width());
                        dma->destMemArea = ofm.memArea;
                        dma->destAddress = ofm.address + dstheightOffset + dstWidthOffset;
                        dma->destStrides = dstStrides;
                        result.push_back(std::move(dma));
                    }
                    height -= heightSlice;
                    srcheightOffset += heightSlice * srcStrides.Height();
                    dstheightOffset += heightSlice * dstStrides.Height();
                }
                continue;
            }
        }
        result.push_back(std::move(hlc));
    }
    return result;
}

//----------------------------------------------------------------------
// Operations
//----------------------------------------------------------------------

// Generates NPU_OP_* command
void EthosU85RCSGenerator::GenerateOperationCode(const HLCOperation *op)
{
    auto opType = op->type;
    if ( opType == OpType::Resize )
    {
        resize_mode mode = ToResizeMode(op->parameters.resize.mode);
        Emit(isa::npu_op_resize_t(mode));
    }
    else if ( IsPooling(opType) )
    {
        Emit(isa::npu_op_pool_t(GetPoolingMode(op)));
    }
    else if ( IsDepthwise(opType) )
    {
        Emit(isa::npu_op_depthwise_t());
    }
    else if ( IsConvolution(opType) || IsVectorProduct(opType) )
    {
        // Dynamic weights when op->ifm.size() == 2 and acc source != ifm2, _weights_ifm2 parameter should be True
        auto accSource = static_cast<EthosU85OpConfig *>(op->config)->AccSource();
        Emit(isa::npu_op_conv_t(op->ifm.size() == 2 && accSource != ArchAccumulatorSource::Ifm2));
    }
    else if ( IsElementwise(opType) )
    {
        const auto &item = kElementwiseMap.find(opType);
        if ( item == kElementwiseMap.end() )
        {
            assert(false && "Unsupported elementwise operator");
        }
        else
        {
            Emit(isa::npu_op_elementwise_t(item->second));
        }
    }
    else if ( opType == OpType::Scatter || opType == OpType::Gather )
    {
        Emit(isa::npu_op_dma_start_t());
    }
    else if ( _arch->UseAvgPoolNop(opType) || opType == OpType::Rescale )
    {
        Emit(isa::npu_op_pool_t(_arch->UseNullPool(opType, DataTypeSizeBits(op->ifm[0].dataType)) ? pooling_mode::NONE : pooling_mode::SUM));
    }
    else
    {
        assert(false && "Unsupported operator");
    }
}

void EthosU85RCSGenerator::GenerateCommon(const HLCStripe *stripe, bool useGlobalScale, MemoryAccesses &memoryAccesses)
{
    auto op = stripe->operation.get();
    int32_t scalarValue = 0;
    bool isScalar = IsElementwise(op->type) && IsScalar(op->ifm[0], scalarValue);
    assert(stripe->opGroup != nullptr);
    EthosU85OpGroup *opGroup = static_cast<EthosU85OpGroup *>(stripe->opGroup);
    int ofmCb = opGroup->ChainingBuffer(op->ofm.uid);
    int ifmCb = opGroup->ChainingBuffer(op->ifm[0].uid);
    int ifm2Cb = -1;
    bool ofmChained = (ofmCb >= 0);
    bool ifmChained = (ifmCb >= 0);
    bool ifm2Chained = false;

    if ( op->ifm.size() == 2 )
    {
        ifm2Cb = opGroup->ChainingBuffer(op->ifm[1].uid);
        ifm2Chained = (ifm2Cb >= 0);
    }
    EthosU85OpConfig *config = static_cast<EthosU85OpConfig *>(stripe->operation->config);
    // Pool with input type >= 32 should use IFM int8 precision and none pooling op
    DataType
        dataType = _arch->UseNullPool(op->type, DataTypeSizeBits(op->ifm[0].dataType)) ? DataType::Int8 : op->ifm[0].dataType;
    GenerateIFMPrecision(op->ifm[0], ifmChained, isScalar, dataType);
    GenerateIFM(op->type, op->ifm[0], stripe->ifmAreas[0], isScalar, scalarValue, ifmCb, ifm2Chained);
    if ( !isScalar && !ifmChained )
    {
        memoryAccesses.push_back(ToMemoryAccess(op->ifm[0], stripe->ifmAreas[0], AccessDirection::Read));
    }
    ifm_upscale_mode upscaleMode = ToIfmUpscaleMode(op->ifm[0].resamplingMode);
    Emit(isa::npu_set_ifm_upscale_t(upscaleMode));
    if ( !IsElementwise(op->type) )
    {
        GeneratePadding(stripe->padding);
    }
    GenerateOFMPrecision(op->ofm, ofmChained, useGlobalScale, config->AccOutputEnabled());
    GenerateOFM(op->type, op->ofm, stripe->ofmArea, ofmCb);
    if ( !ofmChained )
    {
        memoryAccesses.push_back(ToMemoryAccess(op->ofm, stripe->ofmArea, AccessDirection::Write));
    }
    if ( !IsElementwise(op->type) && op->type != OpType::Resize )
    {
        GenerateKernel(op->kernel, config->Traversal() == EthosU85Traversal::PartKernel);
    }
    GenerateWeights(stripe, memoryAccesses);
    GenerateScales(stripe, memoryAccesses);
    GenerateActivation(stripe, memoryAccesses);
}

// Conv2D/Depthwise operations
void EthosU85RCSGenerator::GenerateConvolutionOp(const HLCStripe *stripe, MemoryAccesses &memoryAccesses)
{
    auto op = stripe->operation.get();
    EthosU85OpConfig *config = static_cast<EthosU85OpConfig *>(op->config);
    QuantizedScale ofmScale(1, 0);
    bool useGlobalScale = false;
    ethosU85Scaling::RescaleConvolution(op);

    if ( op->ifm.size() == 2 )
    {
        GenerateIFM2Precision(op->ifm[1], false, false);
        GenerateIFM2(op->type, op->ifm[1], stripe->ifmAreas[1], false, 0, -1);
        if ( config->AccSource() != ArchAccumulatorSource::Ifm2 )
        {
            // Dynamic weights
            assert(ToActivationPrecision(op->ifm[0].dataType) == ToActivationPrecision(op->ifm[1].dataType));
            useGlobalScale = true;
            Emit(isa::npu_set_weight_format_t(weight_format::SWD, weight_sparsity::NONE));  // Reset weight format
        }
    }

    if ( !op->ofm.quantization.scales.empty() )
    {
        ofmScale = op->ofm.quantization.scales[0];
        assert(unsigned(ofmScale.shift) < 64);
    }
    Emit(isa::npu_set_ofm_scale_t(ofmScale.shift, 0, GetOfmRoundingMode(op), ofmScale.scale));
    GenerateCommon(stripe, useGlobalScale, memoryAccesses);
}

// MaxPool/AvgPool or operations that are mapped to AvgPool
void EthosU85RCSGenerator::GeneratePoolingOp(HLCStripe *stripe, MemoryAccesses &memoryAccesses)
{
    auto opType = stripe->operation->type;
    auto op = stripe->operation.get();
    bool useGlobalScale = !op->scales;
    if ( _arch->UseNullPool(opType, DataTypeSizeBits(op->ifm[0].dataType)) )
    {
        GenerateIFM2Precision(op->ifm[0], false, false);
        GenerateIFM2(op->type, op->ifm[0], stripe->ifmAreas[0], false, 0, -1);
    }
    else if ( _arch->UseAvgPoolNop(op->type) )
    {
        assert(op->kernel.Size() == Point2i(1, 1));
        assert(op->kernel.Stride() == Point2i(1, 1));
        assert(op->kernel.Dilation() == Point2i(1, 1));
        assert(op->kernel.DepthMultiplier() == 1);
    }
    GenerateCommon(stripe, useGlobalScale, memoryAccesses);
    GenerateOFMScalingForPooling(op, useGlobalScale);
}

// Elementwise operations
void EthosU85RCSGenerator::GenerateElementwiseOp(HLCStripe *stripe, MemoryAccesses &memoryAccesses)
{
    auto op = stripe->operation.get();
    auto opType = op->type;
    constexpr bool useGlobalScale = true;
    if ( IsUnaryElementwise(opType) )
    {
        assert(op->ifm.size() == 1);
        GenerateScalingForElementwise(op);
        GenerateCommon(stripe, useGlobalScale, memoryAccesses);
        int32_t scalarValue = 0;
        bool ifmIsScalar = IsScalar(op->ifm[0], scalarValue);
        auto ifmShape = stripe->ifmAreas[0].SizeShape();
        GenerateInputBroadcast(ifmShape, Shape(), ifmIsScalar, false);
    }
    else
    {
        // Binary operation: generate IFM2 registers
        assert(op->ifm.size() == 2);
        assert(ToActivationPrecision(op->ifm[0].dataType) == ToActivationPrecision(op->ifm[1].dataType));
        assert(stripe->ifmAreas.size() == 2);
        assert(stripe->opGroup != nullptr);
        EthosU85OpGroup *opGroup = static_cast<EthosU85OpGroup *>(stripe->opGroup);
        int ifm2Cb = opGroup->ChainingBuffer(op->ifm[1].uid);
        bool ifm2Chained = (ifm2Cb >= 0);
        int32_t scalarValue = 0;
        auto ifmShape = stripe->ifmAreas[0].SizeShape();
        auto ifm2Shape = stripe->ifmAreas[1].SizeShape();
        GenerateScalingForElementwise(op);
        GenerateCommon(stripe, useGlobalScale, memoryAccesses);
        bool ifmIsScalar = IsScalar(op->ifm[0], scalarValue);
        bool ifm2IsScalar = IsScalar(op->ifm[1], scalarValue);
        GenerateIFM2Precision(op->ifm[1], ifm2Chained, ifm2IsScalar);
        GenerateIFM2(opType, op->ifm[1], stripe->ifmAreas[1], ifm2IsScalar, scalarValue, ifm2Cb);
        if ( !ifm2IsScalar && !ifm2Chained )
        {
            memoryAccesses.push_back(ToMemoryAccess(op->ifm[1], stripe->ifmAreas[1], AccessDirection::Read));
        }
        GenerateInputBroadcast(ifmShape, ifm2Shape, ifmIsScalar, ifm2IsScalar);
    }
}

// Resize operations
void EthosU85RCSGenerator::GenerateResizeOp(HLCStripe *stripe, MemoryAccesses &memoryAccesses)
{
    auto op = stripe->operation.get();
    auto opType = op->type;
    constexpr bool useGlobalScale = true;
    auto *config = static_cast<EthosU85OpConfig *>(op->config);
    Shape ofmBlock = config->_ofmBlock;

    auto ifmShape = stripe->ifmAreas[0].SizeShape();
    auto ofmShape = stripe->ofmArea.SizeShape();

    // operator-parameters
    const HLCParameters *params = &op->parameters;
    const auto &scale_w = params->resize.scaleX;
    const auto &scale_h = params->resize.scaleY;
    int offset_h = params->resize.offsetY;
    int offset_w = params->resize.offsetX;

    round_mode_ofm roundMode = GetOfmRoundingMode(op);

    // calculate ifm width read
    int ifmWidthRead = ((ofmShape.Width() - 1) * scale_w.d + offset_w) / scale_w.n + 2;

    // scaling is shift only and + 16
    QuantizedScale ofmScale = op->ofm.quantization.scales[0];
    int shift = 16 + ofmScale.shift;

    // X - width
    int one_step_int_w = scale_w.d / scale_w.n;
    int one_step_mod_w = scale_w.d % scale_w.n;
    int blk_step_int_w = ((ofmBlock.Width() - 1) * scale_w.d) / scale_w.n;
    int blk_step_mod_w = ((ofmBlock.Width() - 1) * scale_w.d) % scale_w.n;

    // Y - height
    int one_step_int_h = scale_h.d / scale_h.n;
    int one_step_mod_h = scale_h.d % scale_h.n;
    int blk_step_int_h = ((ofmBlock.Height() - 1) * scale_h.d) / scale_h.n;
    int blk_step_mod_h = ((ofmBlock.Height() - 1) * scale_h.d) % scale_h.n;

    // asserts
    assert(shift < (1 << 6));
    assert(ofmScale.scale == 1);
    assert(scale_w.n <= 2048);
    assert(scale_h.n <= 2048);
    assert(-scale_h.n <= offset_h);
    assert(offset_h < scale_h.n);
    assert(one_step_mod_h < scale_h.n);
    assert(-scale_w.n <= offset_w);
    assert(offset_w < scale_w.n);
    assert(one_step_mod_w < scale_w.n);
    assert(ToIfmUpscaleMode(op->ifm[0].resamplingMode) == ifm_upscale_mode::NONE);
    assert(ofmBlock.Height() == 1);
    assert(ToActivationTranspose(op->ofm.transpose) == activation_transpose::HWC);

    GenerateCommon(stripe, useGlobalScale, memoryAccesses);

    // Resize requires ifm2_zero_point 0
    Emit(isa::npu_set_ifm2_zero_point_t(0));
    Emit(isa::npu_set_ofm_scale_t(16 + ofmScale.shift, 0, roundMode, 1));

    // Resize specific registers
    Emit(isa::npu_set_resize_x_scale_n_m1_t(scale_w.n - 1));
    Emit(isa::npu_set_resize_y_scale_n_m1_t(scale_h.n - 1));
    Emit(isa::npu_set_resize_x_step_t(one_step_int_w, blk_step_int_w, one_step_mod_w, blk_step_mod_w));
    Emit(isa::npu_set_resize_y_step_t(one_step_int_h, blk_step_int_h, one_step_mod_h, blk_step_mod_h));
    Emit(isa::npu_set_resize_x_offset_t(offset_w));
    Emit(isa::npu_set_resize_y_offset_t(offset_h));
    Emit(isa::npu_set_kernel_height_m1_t(ifmShape.Height() - 1));
    Emit(isa::npu_set_kernel_width_m1_t(std::min(ifmShape.Width() - 1, ifmWidthRead - 1)));
}

bool EthosU85RCSGenerator::GenerateStripe(HLCStripe *stripe, MemoryAccesses &memoryAccesses)
{
    auto opType = stripe->operation->type;
    EthosU85NpuOp npuOp = ArchEthosU85::GetHWOp(opType);

    if ( npuOp == EthosU85NpuOp::Pooling || npuOp == EthosU85NpuOp::ReduceMinMax || npuOp == EthosU85NpuOp::ReduceSum || npuOp == EthosU85NpuOp::ArgMax )
    {
        GeneratePoolingOp(stripe, memoryAccesses);
    }
    else if ( npuOp == EthosU85NpuOp::Depthwise || npuOp == EthosU85NpuOp::Convolution || npuOp == EthosU85NpuOp::VectorProduct )
    {
        GenerateConvolutionOp(stripe, memoryAccesses);
    }
    else if ( npuOp == EthosU85NpuOp::Elementwise )
    {
        GenerateElementwiseOp(stripe, memoryAccesses);
    }
    else if ( npuOp == EthosU85NpuOp::Resize )
    {
        GenerateResizeOp(stripe, memoryAccesses);
    }
    else
    {
        LOG_ERROR("Register command stream generator: unsupported operator '{}'\n", OpTypeToString(opType));
        assert(false);
        return false;
    }
    EthosU85OpConfig *config = static_cast<EthosU85OpConfig *>(stripe->operation->config);
    GenerateBlockConfig(config, stripe->operation->ofm);
    GenerateAccFormat(stripe);
    return true;
}


std::shared_ptr<HLCStripe> EthosU85RCSGenerator::MakeStripeForSubOp(HLCStripe *stripe, HLCSubOperation &subOp)
{
    auto op = std::make_shared<HLCOperation>();
    op->type = subOp.type;
    op->ifm = subOp.ifm;
    op->ofm = subOp.ofm;
    op->_srcId = subOp._srcId;
    if ( IsLUTType(subOp.type) )
    {
        op->parameters.lut = subOp.parameters.lut;
    }
    else if ( subOp.type == OpType::LeakyRelu )
    {
        op->parameters.leaky_relu = subOp.parameters.leaky_relu;
    }
    op->config = stripe->operation->config;
    std::shared_ptr<HLCStripe> newStripe = std::make_shared<HLCStripe>(op);
    for ( int i = 0; i < int(subOp.ifm.size()); i++ )
    {
        // TODO MLBEDSW-9143 cascading + chaining requires striping area information for suboperations
        if ( subOp.ifm[i].slice.offset.IsValid() && subOp.ifm[i].slice.shape.IsValid() )
        {
            newStripe->ifmAreas.emplace_back(subOp.ifm[i].slice.offset, subOp.ifm[i].slice.shape);
        }
        else
        {
            newStripe->ifmAreas.emplace_back(subOp.ifm[i].shape);
        }
    }
    newStripe->ofmArea = stripe->ofmArea;
    newStripe->padding = stripe->padding;
    newStripe->opGroup = stripe->opGroup;
    return newStripe;
}

bool EthosU85RCSGenerator::GenerateOpGroup(HLCStripe *stripe, HLCStripe *prevOp, MemoryAccesses &memoryAccesses,
    std::deque<MemoryAccesses> &outstandingDmaAccesses, std::vector<std::pair<unsigned, std::string>> &debugInfo, CmdRanges *cmdRanges)
{
    assert(stripe->opGroup != nullptr);
    EthosU85OpGroup *opGroup = static_cast<EthosU85OpGroup *>(stripe->opGroup);
    const auto &subOps = stripe->operation->subOps;
    const bool isChained =
        subOps.end() !=
        std::find_if(subOps.begin(), subOps.end(), [](const auto &subOp) { return IsElementwise(subOp.type); });

    int blockdep = 0;
    if ( isChained )
    {
        _emit.ClearChainingRegisters();
        // TODO MLBEDSW-9162: calculate block-dependency for chained operations
    }
    else
    {
        blockdep = CalcBlockDep(prevOp, stripe);
    }

    // TODO MLBEDSW-9144 Compute MemoryAccesses for whole chain
    // and emit DMA waits for the whole chain before the first op.

    // Unroll Opgroup into stripes and generate commands for each subOp separately
    int idx = -1;
    std::shared_ptr<HLCStripe> subStripe = nullptr;
    while ( idx < int(subOps.size()) )
    {
        int emitStart = _emit.Position();
        if ( idx >= 0 && opGroup->IsChained(stripe->operation->ofm.uid) )
        {
            // chained sub operation
            HLCSubOperation subOp = subOps[idx];
            subStripe = MakeStripeForSubOp(stripe, subOp);
            stripe = subStripe.get();
        }
        // fuse next subOp if it's an activation, transpose or reverse
        if ( ((idx + 1) < int(subOps.size())) && opGroup->IsFused(subOps[idx + 1].ifm[0].uid) )
        {
            OpType type = subOps[idx + 1].type;
            assert(IsActivation(type) || type == OpType::Transpose || type == OpType::Reverse);
            if ( idx >= 0 )
            {
                HLCSubOperation activation = subOps[idx + 1];
                stripe->operation->subOps.push_back(std::move(activation));
            }
            idx++;
        }
        debugInfo.emplace_back(_emit.Position(), stripe->operation->ToString());
        if ( !GenerateStripe(stripe, memoryAccesses) )
        {
            return false;
        }

        Emit(isa::npu_set_blockdep_t(blockdep));

        GenerateWaits(false, memoryAccesses, outstandingDmaAccesses);
        GenerateOperationCode(stripe->operation.get());

        // Return command mapping information to the caller
        if ( cmdRanges )
        {
            cmdRanges->emplace_back(stripe->operation->_srcId, emitStart, _emit.Position());
        }

        if ( isChained )
        {
            // clear chain cache between every chained subOp
            _emit.ClearChainingRegisters();
        }
        idx++;
    }

    return true;
}

// Generates register commands for DMA operations
void EthosU85RCSGenerator::GenerateDMA(const HLCDMA *dma, MemoryAccesses &memoryAccesses)
{
    dma_region_mode srcRegionMode = dma->srcMemArea == _arch->LUTMemory() ? dma_region_mode::INTERNAL : dma_region_mode::EXTERNAL;
    dma_region_mode destRegionMode = dma->destMemArea == _arch->LUTMemory() ? dma_region_mode::INTERNAL : dma_region_mode::EXTERNAL;

    uint32_t size0 = dma->sizes.Size() > 0 ? dma->sizes[-1] : 1;
    uint32_t size1 = dma->sizes.Size() > 1 ? dma->sizes[-2] : 1;
    uint64_t srcStride0 = dma->srcStrides.Size() > 1 ? dma->srcStrides[-2] : 0;
    uint64_t srcStride1 = dma->srcStrides.Size() > 2 ? dma->srcStrides[-3] : 0;
    uint64_t destStride0 = dma->destStrides.Size() > 1 ? dma->destStrides[-2] : 0;
    uint64_t destStride1 = dma->destStrides.Size() > 2 ? dma->destStrides[-3] : 0;

    dma_stride_mode srcStrideMode;
    if ( size1 > 1 ) srcStrideMode = dma_stride_mode::D3;
    else if ( size0 > 1 ) srcStrideMode = dma_stride_mode::D2;
    else srcStrideMode = dma_stride_mode::D1;

    dma_idx_mode srcIndexMode = dma->srcIndexed ? dma_idx_mode::ENABLED : dma_idx_mode::DISABLED;
    dma_idx_mode destIndexMode = dma->destIndexed ? dma_idx_mode::ENABLED : dma_idx_mode::DISABLED;
    assert(!(srcIndexMode == dma_idx_mode::ENABLED && destIndexMode == dma_idx_mode::ENABLED));

    // Registers for 1D, 2D and 3D mode
    Emit(isa::npu_set_dma0_src_region_t(ToRegion(dma->srcMemArea), srcRegionMode, srcStrideMode, srcIndexMode));
    Emit(isa::npu_set_dma0_src_t(dma->srcAddress));
    Emit(isa::npu_set_dma0_dst_region_t(ToRegion(dma->destMemArea), destRegionMode, destIndexMode));
    Emit(isa::npu_set_dma0_dst_t(dma->destAddress));
    assert(dma->length > 0);
    Emit(isa::npu_set_dma0_len_t(dma->length));

    if ( srcStrideMode != dma_stride_mode::D1 )
    {
        // Registers for 2D and 3D mode
        assert(size0 > 0);
        Emit(isa::npu_set_dma0_size0_t(size0));
    }

    if ( srcStrideMode != dma_stride_mode::D1 || dma->srcIndexed )
    {
        // Registers for 2D and 3D mode, or src indexed operation
        Emit(isa::npu_set_dma0_src_stride0_t(srcStride0));
    }

    if ( srcStrideMode != dma_stride_mode::D1 || dma->destIndexed )
    {
        // Registers for 2D and 3D mode, or dest indexed operation
        Emit(isa::npu_set_dma0_dst_stride0_t(destStride0));
    }

    if ( srcStrideMode == dma_stride_mode::D3 )
    {
        // Registers for 3D mode
        assert(size1 > 0);
        Emit(isa::npu_set_dma0_size1_t(size1));
        Emit(isa::npu_set_dma0_src_stride1_t(srcStride1));
        Emit(isa::npu_set_dma0_dst_stride1_t(destStride1));
    }

    if ( dma->srcIndexed || dma->destIndexed )
    {
        // Registers for indexed operation
        Emit(isa::npu_set_dma0_idx_region_t(ToRegion(dma->idxMemArea)));
        assert(dma->idxMax >= 0);
        Emit(isa::npu_set_dma0_idx_max_t(dma->idxMax));
        Emit(isa::npu_set_dma0_idx_t(dma->idxAddress));
    }

    if ( srcStrideMode == dma_stride_mode::D3 && (dma->srcIndexed || dma->destIndexed) )
    {
        Emit(isa::npu_set_dma0_idx_skip1_t(dma->idxSkip1));
    }

    if ( srcStrideMode == dma_stride_mode::D1 )
    {
        // Address accesses for 1D mode
        CheckAddressRange(dma->srcMemArea.memory, dma->srcAddress, dma->length);
        CheckAddressRange(dma->destMemArea.memory, dma->destAddress, dma->length);
        memoryAccesses.emplace_back(AccessDirection::Read, dma->srcMemArea, dma->srcAddress, dma->srcAddress + dma->length);
        memoryAccesses.emplace_back(AccessDirection::Write, dma->destMemArea, dma->destAddress, dma->destAddress + dma->length);
    }
    else
    {
        // Address accesses for 2D and 3D mode
        CheckAddressRange(dma->srcMemArea.memory, dma->srcAddress, dma->srcStrides[0]);
        CheckAddressRange(dma->destMemArea.memory, dma->destAddress, dma->destStrides[0]);
        memoryAccesses.emplace_back(AccessDirection::Read, dma->srcMemArea, dma->srcAddress, dma->srcAddress + dma->srcStrides[0]);
        memoryAccesses.emplace_back(AccessDirection::Write, dma->destMemArea, dma->destAddress, dma->destAddress + dma->destStrides[0]);
    }

    if ( dma->srcIndexed || dma->destIndexed )
    {
        // Address accesses for indexed operation
        CheckAddressRange(dma->idxMemArea.memory, dma->idxAddress, size0 * size1);
        memoryAccesses.emplace_back(AccessDirection::Read, dma->idxMemArea, dma->idxAddress, dma->idxAddress + size0 * size1);
    }
}

std::vector<uint32_t> EthosU85RCSGenerator::GenerateCommandStream(
    std::vector<std::unique_ptr<HighLevelCommand>> &highLevelCommandStream, CmdRanges *cmdRanges, bool verbose)
{
    _emit.Clear();
    _stripeToLutSlot.clear();
    GenerateInitialRegisterSetup();
    auto cmds = InsertLUTDMACommands(highLevelCommandStream);
    cmds = InsertTileDMACommands(cmds);
    std::deque<MemoryAccesses> outstandingDmaAccesses;
    std::deque<MemoryAccesses> outstandingNpuAccesses;
    int maxOutstandingDMAOps = _arch->MaxOutstandingDMAOps();
    int maxOutstandingKernelOps = _arch->MaxOutstandingKernelOps();
    HLCStripe *prevOp = nullptr;
    std::vector<std::pair<unsigned, std::string>> debugInfo;

    for ( auto &hlc : cmds )
    {
        if ( hlc->IsStripe() )
        {
            MemoryAccesses memoryAccesses;
            auto stripe = static_cast<HLCStripe *>(hlc.get());
            if ( !GenerateOpGroup(stripe, prevOp, memoryAccesses, outstandingDmaAccesses, debugInfo, cmdRanges) )

            {
                return std::vector<uint32_t>();
            }

            prevOp = stripe;

            UpdateMemoryAccesses(memoryAccesses, outstandingNpuAccesses, maxOutstandingKernelOps);
        }
        else
        {
            MemoryAccesses dmaAccesses;
            auto dma = static_cast<HLCDMA *>(hlc.get());
            debugInfo.emplace_back(_emit.Position(), dma->ToString());
            GenerateDMA(dma, dmaAccesses);
            GenerateWaits(false, dmaAccesses, outstandingDmaAccesses);
            GenerateWaits(true, dmaAccesses, outstandingNpuAccesses);
            UpdateMemoryAccesses(dmaAccesses, outstandingDmaAccesses, maxOutstandingDMAOps);
            Emit(isa::npu_op_dma_start_t());
        }
    }
    Emit(isa::npu_op_stop_t(0xFFFF));
    if ( verbose )
    {
        PrintCommandStream(_emit.CommandStream(), debugInfo);
    }
    return _emit.CommandStream();
}

}  // namespace regor
