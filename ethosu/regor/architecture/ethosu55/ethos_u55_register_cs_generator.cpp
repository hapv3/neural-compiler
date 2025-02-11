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

#include "ethos_u55_register_cs_generator.hpp"

#include "common/common.hpp"
#include "common/logging.hpp"

#include "architecture/ethos_u_scaling.hpp"
#include "common/data_type.hpp"
#include "compiler/high_level_command_stream.hpp"
#include "compiler/op_type.hpp"
#include "ethos_u55.hpp"
#include "ethos_u55_scaling.hpp"
#define NPU_DISASSEMBLE
#define NPU_NAMESPACE ethosu55
// Note: Ethos-U55 and Ethos-U65 share interface definitions
#include "architecture/ethosu65/ethos_u65_interface.hpp"

#include <cstdint>
#include <deque>
#include <unordered_map>
#include <vector>

namespace regor
{
using namespace ethosu55;

void EthosU55Emitter::Emit(uint32_t instr)
{
    uint16_t cmd = instr & 0xFFFF;
    assert(IsCmd0(cmd));
    bool emit = IsOp(cmd) || SetRegister(cmd, instr);
    if ( emit )
    {
        _stream.push_back(instr);
    }
}

void EthosU55Emitter::Emit(uint64_t instr)
{
    uint16_t cmd = instr & 0xFFFF;
    assert(IsCmd1(cmd));
    bool emit = IsOp(cmd) || SetRegister(cmd, instr);
    if ( emit )
    {
        _stream.push_back(uint32_t(instr));
        _stream.push_back(uint32_t(instr >> 32));
    }
}

void EthosU55Emitter::Clear()
{
    _stream.clear();
    _registers.clear();
}


bool EthosU55Emitter::SetRegister(uint16_t reg, uint64_t value)
{
    auto item = _registers.find(reg);
    bool isChanged = item == _registers.end() || item->second != value;
    if ( isChanged )
    {
        _registers[reg] = value;
    }
    return isChanged;
}

bool EthosU55Emitter::IsCmd0(uint16_t key)
{
    return (key >> 14) == uint16_t(cmd_ctrl::CMD0_CTRL);
}

bool EthosU55Emitter::IsCmd1(uint16_t key)
{
    return (key >> 14) == uint16_t(cmd_ctrl::CMD1_CTRL);
}

bool EthosU55Emitter::IsOp(uint16_t key)
{
    return IsCmd0(key) ? (key & (1 << 8)) == 0 : (key & (1 << 8)) != 0;
}


/// <summary>
/// Generates register command streams for Ethos U55 and Ethos U65.
/// </summary>

namespace
{
const std::unordered_map<OpType, elementwise_mode> s_ElementwiseMap = {
    {OpType::Add, elementwise_mode::ADD},
    {OpType::Sub, elementwise_mode::SUB},
    {OpType::Abs, elementwise_mode::ABS},
    {OpType::Mul, elementwise_mode::MUL},
    {OpType::Minimum, elementwise_mode::MIN},
    {OpType::Maximum, elementwise_mode::MAX},
    {OpType::LeakyRelu, elementwise_mode::LRELU},
    {OpType::CLZ, elementwise_mode::CLZ},
    {OpType::SHL, elementwise_mode::SHL},
    {OpType::Asr, elementwise_mode::SHR},
};

activation_type ToActivationType(DataType type)
{
    if ( IsSignedInteger(type) )
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

ifm_scale_mode MapRcsIfmScaleModeToInterface(RCSIfmScaleMode rcsScaleMode)
{
    switch ( rcsScaleMode )
    {
        case RCSIfmScaleMode::OPA_OPB_16:
            return ifm_scale_mode::OPA_OPB_16;
        case RCSIfmScaleMode::OPA_32:
            return ifm_scale_mode::OPA_32;
        case RCSIfmScaleMode::OPB_32:
            return ifm_scale_mode::OPB_32;
        default:
            assert(0 && "Unexpected value, has the interface changed?");
            return ifm_scale_mode::OPA_32;
    }
}

round_mode MapHLCRoundModeToInterface(HLCRoundMode roundMode)
{
    switch ( roundMode )
    {
        case HLCRoundMode::NATURAL:
            return round_mode::NATURAL;
        case HLCRoundMode::TRUNCATE:
            return round_mode::TRUNCATE;
        case HLCRoundMode::AUTO:
            [[fallthrough]];
        case HLCRoundMode::DBL:
            return round_mode::DBL;
        default:
            assert(false && "usupported HLCRoundMode");
            return round_mode::DBL;
    }
}

}  // namespace

uint32_t EthosU55RCSGenerator::IdRegister()
{
    return id_r{};
}

bool EthosU55RCSGenerator::IsSupportedElementwise(const OpType opType)
{
    return s_ElementwiseMap.count(opType) != 0;
}

EthosU55RCSGenerator::EthosU55RCSGenerator(ArchEthosU55 *arch) : _arch(arch)
{
    int slots = (_arch->_shram.bankSizeBytes * _arch->_shram.lutBanks) / _arch->_shram.lutSlotSize;
    assert(slots);
    _lutSlots.resize(slots);
}


void EthosU55RCSGenerator::Emit(uint32_t instr)
{
    _emit.Emit(instr);
}

void EthosU55RCSGenerator::Emit(uint64_t instr)
{
    _emit.Emit(instr);
}

int EthosU55RCSGenerator::GetDoubleBufferOffset(HLCWeights *weights, int rangeIndex)
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

void EthosU55RCSGenerator::CheckAddressRange(ArchitectureMemory *memory, Address address, int size)
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

void EthosU55RCSGenerator::CheckAddresses(const HLCFeatureMap &fm)
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
Address EthosU55RCSGenerator::AddressForCoordinate(const HLCFeatureMap &fm, const Shape &strides, const Shape &coord)
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
TileBox EthosU55RCSGenerator::GetTiles(const HLCFeatureMap &fm, const Shape &strides, const Box &area)
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
    auto elementSize = DataTypeSizeBits(fm.dataType) / 8;
    auto depth = fm.shape.Depth();
    if ( fm.reverse == ReverseType::H )
    {
        assert(fm.format == TensorFormat::NHWC);
        tiles.address[0] += (tiles.height0 - 1) * fm.shape.Width() * depth * elementSize;
    }
    if ( fm.reverse == ReverseType::W )
    {
        assert(fm.format == TensorFormat::NHWC);
        tiles.address[0] += (tiles.width0 - 1) * depth * elementSize;
    }
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
        assert((tiles.address[0] | tiles.address[1] | tiles.address[2] | tiles.address[3]) % 16 == 0 && "NHCWB16 base address is not 16-byte aligned");
    }
    return tiles;
}

MemoryAccess EthosU55RCSGenerator::ToMemoryAccess(const HLCFeatureMap &fm, const Box &area, AccessDirection direction)
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
uint32_t EthosU55RCSGenerator::ToRegion(const MemArea &memArea)
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
bool EthosU55RCSGenerator::IsScalar(const HLCFeatureMap &fm, int32_t &scalarValue)
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
// Note: NPU - NPU dependency is handled via block dependency
int EthosU55RCSGenerator::CalcCommandWaits(const MemoryAccesses &opAccesses, std::deque<MemoryAccesses> &outstanding)
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
int EthosU55RCSGenerator::AllocateLutSlot(
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

int EthosU55RCSGenerator::Disassemble(const uint32_t *in, std::string &op, std::vector<std::pair<std::string, std::string>> &fields)
{
    return isa::disassemble(in, op, fields);
}

//----------------------------------------------------------------------
// Scaling (OFM/OPA/OPB_SCALE)
//----------------------------------------------------------------------

// Generates OFM_SCALE register for pooling operations
void EthosU55RCSGenerator::GenerateOFMScalingForPooling(HLCOperation *poolOp, bool useGlobalScale)
{
    QuantizedScale ofmScale(1, 0);
    bool isNoOp = _arch->UseAvgPoolNop(poolOp->type);
    ethosU55Scaling::RescalePooling(poolOp, isNoOp);

    if ( useGlobalScale && !poolOp->ofm.quantization.scales.empty() )
    {
        ofmScale = poolOp->ofm.quantization.scales[0];
        assert(unsigned(ofmScale.shift) < 64);
    }

    Emit(isa::npu_set_ofm_scale_t(uint32_t(ofmScale.shift), ofmScale.scale));
}

// Generates OFM/OPA/OPB_SCALE registers for elementwise operators.
// Returns the operator to scale
RCSIfmScaleMode EthosU55RCSGenerator::GenerateScalingForElementwise(HLCOperation *op, int ifm0Index)
{
    auto opToScale = RCSIfmScaleMode::OPA_OPB_16;
    auto opType = op->type;

    QuantizedScale ofmScale(1, 0);
    ethosU55Scaling::RescaleElementwise(op);
    int ifmCnt = int(op->ifm.size());
    bool allHaveScale =
        !op->ofm.quantization.scales.empty() && !op->ifm[0].quantization.scales.empty() && ifmCnt == 2 &&
        !op->ifm[1].quantization.scales.empty();

    if ( opType == OpType::Mul || opType == OpType::Abs )
    {
        if ( !op->ofm.quantization.scales.empty() )
        {
            ofmScale = op->ofm.quantization.scales[0];
        }
    }
    else if ( opType == OpType::LeakyRelu )
    {
        const HLCParameters *params = &op->parameters;
        double alpha = params->leaky_relu.alpha;
        ofmScale = QuantizedScale(alpha);
    }
    else if ( opType == OpType::Add || opType == OpType::Sub )
    {
        uint32_t opaScale = 1;
        uint32_t opbScale = 1;
        uint32_t opaShift = 0;
        if ( allHaveScale )
        {
            ofmScale = op->ofm.quantization.scales[0];
            QuantizedScale ifm1Scale = op->ifm[ifm0Index].quantization.scales[0];
            QuantizedScale ifm2Scale = op->ifm[1 - ifm0Index].quantization.scales[0];
            opaScale = ifm1Scale.scale;
            opaShift = ifm1Scale.shift;
            opbScale = ifm2Scale.scale;

            if ( ifm1Scale.scale == 0 || ifm2Scale.scale == 0 )
            {
                opbScale = 0;
                if ( ifm1Scale.scale == 0 )
                {
                    opToScale = RCSIfmScaleMode::OPB_32;
                    opaScale = ifm2Scale.scale;
                    opaShift = ifm2Scale.shift;
                }
                else
                {
                    opToScale = RCSIfmScaleMode::OPA_32;
                }
            }
            if ( ifm0Index == 1 )
            {
                // Reversed operands
                if ( opToScale == RCSIfmScaleMode::OPA_32 )
                {
                    opToScale = RCSIfmScaleMode::OPB_32;
                }
                else if ( opToScale == RCSIfmScaleMode::OPB_32 )
                {
                    opToScale = RCSIfmScaleMode::OPA_32;
                }
            }
        }
        assert(opaShift < 64);
        Emit(isa::npu_set_opa_scale_t(opaShift, opaScale));
        Emit(isa::npu_set_opb_scale_t(opbScale));
    }
    assert(unsigned(ofmScale.shift) < 64);
    Emit(isa::npu_set_ofm_scale_t(ofmScale.shift, ofmScale.scale));
    return opToScale;
}

//----------------------------------------------------------------------
// BLOCKDEP calculation
//----------------------------------------------------------------------

static Shape CalcIFMJobShape(const Shape &ofmBlock, Kernel *kernel, int ifmBlockDepth)
{
    // TODO MLBEDSW-8498: Consider ifm_upscale_mode for job-shape calculations
    Point2i dilatedSize = kernel->DilatedWH();
    int h = RequiredInputSize(ofmBlock.Height(), kernel->Stride().y, dilatedSize.y, 1);
    int w = RequiredInputSize(ofmBlock.Width(), kernel->Stride().x, dilatedSize.x, 1);
    return Shape(1, h, w, ifmBlockDepth);
}

// Given the area and block size, adds the first/last jobs (depending on fromStart) to jobs.
// - area: total amount of work to perform
// - jobShape: size of each job
// - fromStart: if true, the first jobs are added, if false, the last jobs are added
//   (in that case, the very last job is added last)
void EthosU55RCSGenerator::GetJobs(const Box &area, const Shape &jobShape, int nrJobsToGet, bool fromStart, std::vector<Box> &jobs)
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
int EthosU55RCSGenerator::CalcBlockDep(const HLCStripe *prevStripe, const HLCStripe *stripe)
{
    if ( prevStripe == nullptr )
    {
        return 0;
    }
    const auto &op = stripe->operation;
    const auto &prevOp = prevStripe->operation;
    const auto &prevOfm = prevOp->ofm;
    if ( _arch->_shram.reservedEndBanks == 0 )
    {
        // SHRAM has no reserved LUT banks
        if ( _stripeToLutSlot.count(prevStripe) && !_stripeToLutSlot.count(stripe) )
        {
            // Previous operation uses LUT, current does not
            return 0;  // Prevents corruption of the LUT
        }
    }

    int ifmIndex = (op->ifm.size() > 1 && op->ifm[1].address == prevOfm.address && op->ifm[1].memArea == prevOfm.memArea) ? 1 : 0;
    assert(size_t(ifmIndex) < op->ifm.size());
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
    auto prevConfig = static_cast<EthosU55OpConfig *>(prevOp->config);
    Shape prevBlock = prevConfig->OfmBlock();
    auto config = static_cast<EthosU55OpConfig *>(op->config);
    Shape currBlock = CalcIFMJobShape(config->OfmBlock(), &op->kernel, config->IfmBlock().Depth());
    // Get the last few jobs from the previous operation (each job produces a part of the current op's IFM)
    std::vector<Box> lastPrevJobs;
    GetJobs(prevStripe->ofmArea, prevBlock, maxJobs, false, lastPrevJobs);
    // Get the first few jobs from the current operation (each job consumes a part of the current op's IFM)
    std::vector<Box> firstCurrJobs;
    GetJobs(stripe->ifmAreas[ifmIndex], currBlock, maxJobs, true, firstCurrJobs);
    // Find the highest block dependency such that there is no overlap between
    // any job from the previous op with any job from the current op during block dependency jobs
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

void EthosU55RCSGenerator::GeneratePadding(const HLCPadding &padding)
{
    Emit(isa::npu_set_ifm_pad_top_t(padding.top));
    Emit(isa::npu_set_ifm_pad_left_t(padding.left));
    Emit(isa::npu_set_ifm_pad_bottom_t(padding.bottom));
    Emit(isa::npu_set_ifm_pad_right_t(padding.right));
}

// Generates ACTIVATION registers
void EthosU55RCSGenerator::GenerateActivation(const HLCStripe *stripe, MemoryAccesses &memoryAccesses)
{
    const HLCOperation *op = stripe->operation.get();
    assert(op->subOps.size() <= 1);
    OpType opType = OpType::None;
    const HLCParameters *parameters = nullptr;
    if ( IsActivation(op->type) )
    {
        // Non-fused activation
        opType = op->type;
        parameters = &op->parameters;
        assert(op->subOps.empty() || opType == op->subOps[0].type);
    }
    else if ( !op->subOps.empty() &&
              (stripe->opGroup && !static_cast<EthosU55OpGroup *>(stripe->opGroup)->NeedsAllocation(op->subOps[0].ifm[0].uid)) )
    {
        // Fused activation
        assert(IsActivation(op->subOps[0].type));
        opType = op->subOps[0].type;
        parameters = &op->subOps[0].parameters;
    }
    auto &ofm = op->ofm;
    int size = std::min(16, DataTypeSizeBits(ofm.dataType));
    assert(size > 0 && "Illegal data type");
    bool isSigned = bool(ofm.dataType & DataType::Signed);
    int64_t quantizedMin = isSigned ? -(1LL << (size - 1)) : 0;
    int64_t quantizedMax = isSigned ? (1LL << (size - 1)) - 1 : (1LL << size) - 1;

    auto act = activation_function::RELU;
    auto clipRange = activation_clip_range::OFM_PRECISION;
    if ( ofm.quantization.quantMin.size() )
    {
        quantizedMin = std::max(quantizedMin, ofm.quantization.quantMin[0]);
    }
    if ( ofm.quantization.quantMax.size() )
    {
        quantizedMax = std::min(quantizedMax, ofm.quantization.quantMax[0]);
    }

    if ( opType == OpType::Sigmoid )
    {
        act = activation_function::SIGMOID;
    }
    else if ( opType == OpType::Tanh )
    {
        act = activation_function::TANH;
    }
    else if ( opType == OpType::LUT )
    {
        auto &lutParams = parameters->lut;
        size = lutParams.sizeBytes;
        assert(size == 256 || size == 1024 || size == 2048);

        int tableIndex = 0;
        auto pos = _stripeToLutSlot.find(stripe);
        if ( pos != _stripeToLutSlot.end() )
        {
            tableIndex = pos->second;
        }
        else
        {
            assert(false && "Command uses lut, but no lut info found");
        }
        act = activation_function(int(activation_function::TABLE_0) + tableIndex);
        if ( ofm.dataType == DataType::Int32 )
        {
            // force INT8 range
            clipRange = activation_clip_range::FORCE_INT8;
            quantizedMin = std::max<int64_t>(quantizedMin, -128);
            quantizedMax = std::min<int64_t>(quantizedMax, 127);
        }
        auto &layout = static_cast<EthosU55OpConfig *>(op->config)->_layout;
        Address lutStart = Address(layout.lutStart) * _arch->_shram.bankSizeBytes + tableIndex * _arch->_shram.lutSlotSize;
        memoryAccesses.emplace_back(AccessDirection::Read, _arch->LUTMemory(), lutStart, lutStart + lutParams.sizeBytes);
    }
    assert(quantizedMin <= std::numeric_limits<uint16_t>::max());
    assert(quantizedMin >= std::numeric_limits<int16_t>::min());
    assert(quantizedMax <= std::numeric_limits<uint16_t>::max());
    assert(quantizedMax >= std::numeric_limits<int16_t>::min());
    Emit(isa::npu_set_activation_t(act, clipRange));
    Emit(isa::npu_set_activation_min_t(uint32_t(quantizedMin)));
    Emit(isa::npu_set_activation_max_t(uint32_t(quantizedMax)));
}

// Generates KERNEL related registers
void EthosU55RCSGenerator::GenerateKernel(const Kernel &kernel, bool partKernel)
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

// Generates IFM2_BROADCAST register for binary elementwise operations
void EthosU55RCSGenerator::GenerateIFM2Broadcast(const Shape &ifmShape, const Shape &ifm2Shape, bool reversedOperands, bool isScalar)
{
    auto broadcastH = broadcast_mode::DISABLE;
    auto broadcastW = broadcast_mode::DISABLE;
    auto broadcastC = broadcast_mode::DISABLE;
    auto order = reversedOperands ? ifm2_operand_order::ORDER_A : ifm2_operand_order::ORDER_B;
    auto isConstant = broadcast_mode::DISABLE;
    if ( isScalar )
    {
        isConstant = broadcast_mode::ENABLE;
    }
    else
    {
        if ( ifmShape.Height() != ifm2Shape.Height() )
        {
            // Broadcast in 'H' dimension
            broadcastH = broadcast_mode::ENABLE;
            assert(ifm2Shape.Height() == 1);
        }
        if ( ifmShape.Width() != ifm2Shape.Width() )
        {
            // Broadcast in 'W' dimension
            broadcastW = broadcast_mode::ENABLE;
            assert(ifm2Shape.Width() == 1);
        }
        if ( ifmShape.Depth() != ifm2Shape.Depth() )
        {
            // Broadcast in 'C' dimension
            broadcastC = broadcast_mode::ENABLE;
            assert(ifm2Shape.Depth() == 1);
        }
    }
    Emit(isa::npu_set_ifm2_broadcast_t(broadcastH, broadcastW, broadcastC, order, isConstant));
}

// Generates IFM_PRECISION register
void EthosU55RCSGenerator::GenerateIFMPrecision(const HLCFeatureMap &fm, RCSIfmScaleMode scaleMode, HLCRoundMode roundMode)
{
    activation_type type = ToActivationType(fm.dataType);
    activation_precision precision = ToActivationPrecision(fm.dataType);
    activation_format format = ToActivationFormat(fm.format);
    round_mode rounding = MapHLCRoundModeToInterface(roundMode);
    ifm_scale_mode interfaceScaleMode = MapRcsIfmScaleModeToInterface(scaleMode);
    Emit(isa::npu_set_ifm_precision_t(type, precision, format, interfaceScaleMode, rounding));
}

// Generates IFM2_PRECISION register
void EthosU55RCSGenerator::GenerateIFM2Precision(const HLCFeatureMap &fm)
{
    activation_type type = ToActivationType(fm.dataType);
    activation_precision precision = ToActivationPrecision(fm.dataType);
    activation_format format = ToActivationFormat(fm.format);
    Emit(isa::npu_set_ifm2_precision_t(type, precision, format));
}

// Generates OFM_PRECISION register
void EthosU55RCSGenerator::GenerateOFMPrecision(const HLCFeatureMap &fm, bool useGlobalScale)
{
    activation_type type = ToActivationType(fm.dataType);
    activation_precision precision = ToActivationPrecision(fm.dataType);
    activation_format format = ToActivationFormat(fm.format);
    round_mode roundMode = MapHLCRoundModeToInterface(fm.rounding);
    auto scaleMode = useGlobalScale ? ofm_scale_mode::GLOBAL : ofm_scale_mode::PER_CHANNEL;
    Emit(isa::npu_set_ofm_precision_t(type, precision, format, scaleMode, roundMode));
}

// Generates common IFM registers
void EthosU55RCSGenerator::GenerateIFM(OpType opType, const HLCFeatureMap &fm, const Box &inputArea)
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
    Emit(isa::npu_set_ifm_depth_m1_t(boxSize.Depth() - 1));
    // IFM_STRIDE registers
    Emit(isa::npu_set_ifm_stride_y_t(strides.Height() * fm.stepXY.y));
    Emit(isa::npu_set_ifm_stride_x_t(strides.Width() * fm.stepXY.x));
    Emit(isa::npu_set_ifm_stride_c_t(strides.Depth()));
    // IFM_ZERO_POINT register
    auto &quant = fm.quantization;
    uint32_t zp = quant.zeroPoints.empty() ? 0 : uint32_t(quant.zeroPoints[0]);
    Emit(isa::npu_set_ifm_zero_point_t(zp));
}

// Generates common IFM2 registers
void EthosU55RCSGenerator::GenerateIFM2(OpType opType, const HLCFeatureMap &fm, const Box &inputArea, bool isScalar, int32_t scalarValue)
{
    if ( isScalar )
    {
        Emit(isa::npu_set_ifm2_scalar_t(uint32_t(scalarValue)));
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
    // IFM2_ZERO_POINT register
    auto &quant = fm.quantization;
    uint32_t zp = quant.zeroPoints.empty() ? 0 : uint32_t(quant.zeroPoints[0]);
    Emit(isa::npu_set_ifm2_zero_point_t(zp));
}

// Generates OFM registers
void EthosU55RCSGenerator::GenerateOFM(OpType opType, const HLCFeatureMap &fm, const Box &outputArea)
{
    CheckAddresses(fm);
    Emit(isa::npu_set_ofm_region_t(ToRegion(fm.memArea)));
    Shape strides = fm.strides;
    auto tiles = GetTiles(fm, strides, outputArea);
    auto boxSize = outputArea.SizeShape();
    // OFM_BASE registers
    Emit(isa::npu_set_ofm_base0_t(tiles.address[0]));
    Emit(isa::npu_set_ofm_base1_t(tiles.address[1]));
    Emit(isa::npu_set_ofm_base2_t(tiles.address[2]));
    Emit(isa::npu_set_ofm_base3_t(tiles.address[3]));
    // OFM size
    unsigned heightM1 = DivRoundUp(boxSize.Height(), fm.stepXY.y) - 1;
    unsigned widthM1 = DivRoundUp(boxSize.Width(), fm.stepXY.x) - 1;
    assert(isa::npu_set_ofm_height_m1_t(heightM1).get_height_m1() == heightM1);
    assert(isa::npu_set_ofm_width_m1_t(widthM1).get_width_m1() == widthM1);
    Emit(isa::npu_set_ofm_height_m1_t(heightM1));
    Emit(isa::npu_set_ofm_width_m1_t(widthM1));
    // Tile related registers
    Emit(isa::npu_set_ofm_height0_m1_t(tiles.height0 - 1));
    Emit(isa::npu_set_ofm_height1_m1_t(tiles.height1 - 1));
    Emit(isa::npu_set_ofm_width0_m1_t(tiles.width0 - 1));
    unsigned depthM1 = boxSize.Depth() - 1;
    assert(isa::npu_set_ofm_depth_m1_t(depthM1).get_depth_m1() == depthM1);
    Emit(isa::npu_set_ofm_depth_m1_t(depthM1));
    // OFM_STRIDE registers
    // Make X/Y stride negative if the OFM should be reversed in that axis.
    if ( fm.reverse == ReverseType::H )
    {
        assert(fm.format == TensorFormat::NHWC);
        strides = strides.WithHeight(-strides.Height());
    }
    if ( fm.reverse == ReverseType::W )
    {
        assert(fm.format == TensorFormat::NHWC);
        strides = strides.WithWidth(-strides.Width());
    }
    Emit(isa::npu_set_ofm_stride_y_t(strides.Height() * fm.stepXY.y));
    Emit(isa::npu_set_ofm_stride_x_t(strides.Width() * fm.stepXY.x));
    Emit(isa::npu_set_ofm_stride_c_t(strides.Depth()));
    // OFM_ZERO_POINT register
    auto &quant = fm.quantization;
    uint32_t zp = quant.zeroPoints.empty() ? 0 : uint32_t(quant.zeroPoints[0]);
    Emit(isa::npu_set_ofm_zero_point_t(zp));
}

// Generates WEIGHT registers
void EthosU55RCSGenerator::GenerateWeights(const HLCStripe *stripe, MemoryAccesses &memoryAccesses)
{
    auto weights = stripe->operation->weights.get();
    if ( weights == nullptr )
    {
        return;
    }
    int depth = stripe->weightRangeDepth;
    Emit(isa::npu_set_weight_region_t(ToRegion(weights->memArea)));
    auto item0 = weights->encodedRanges.find(WeightKey(0, depth));
    assert(item0 != weights->encodedRanges.end());
    auto &range0 = item0->second;
    int doubleBufferOffset = GetDoubleBufferOffset(weights, range0.index);
    Address address = weights->address + range0.weightOffset + doubleBufferOffset;
    int length = RoundAway(range0.weightBytes, 16);
    CheckAddressRange(weights->memArea.memory, address, length);
    Emit(isa::npu_set_weight_base_t(address));
    Emit(isa::npu_set_weight_length_t(length));
    memoryAccesses.emplace_back(AccessDirection::Read, weights->memArea, address, address + length);
    auto item1 = weights->encodedRanges.find(WeightKey(1, depth));
    if ( item1 != weights->encodedRanges.end() )
    {
        auto &range1 = item1->second;
        Address address1 = weights->address + RoundAway(range0.TotalBytes(), 16) + range1.weightOffset + doubleBufferOffset;
        int length1 = RoundAway(range1.weightBytes, 16);
        CheckAddressRange(weights->memArea.memory, address1, length1);
        Emit(isa::npu_set_weight1_base_t(address1));
        Emit(isa::npu_set_weight1_length_t(length1));
        memoryAccesses.emplace_back(AccessDirection::Read, weights->memArea, address1, address1 + length1);
    }
    else if ( _arch->_cores > 1 )
    {
        Emit(isa::npu_set_weight1_length_t(0));
    }
}

// Generates SCALE registers
void EthosU55RCSGenerator::GenerateScales(const HLCStripe *stripe, MemoryAccesses &memoryAccesses)
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
    auto item1 = scales->encodedRanges.find(WeightKey(1, depth));
    if ( item1 != scales->encodedRanges.end() )
    {
        auto &range1 = item1->second;
        Address address1 = address + RoundAway(range0.TotalBytes(), 16);
        int length1 = RoundAway(range1.scaleBytes, 16);
        CheckAddressRange(scales->memArea.memory, address1, length1);
        Emit(isa::npu_set_scale1_base_t(address1));
        Emit(isa::npu_set_scale1_length_t(length1));
        memoryAccesses.emplace_back(AccessDirection::Read, scales->memArea, address1, address1 + length1);
    }
    else if ( _arch->_cores > 1 )
    {
        Emit(isa::npu_set_scale1_length_t(0));
    }
}

// Generates OFM_BLK_HEIGHT/WIDTH/DEPTH registers
void EthosU55RCSGenerator::GenerateBlockConfig(const EthosU55OpConfig *config)
{
    Emit(isa::npu_set_ofm_blk_height_m1_t(config->OfmBlock().Height() - 1));
    Emit(isa::npu_set_ofm_blk_width_m1_t(config->OfmBlock().Width() - 1));
    Emit(isa::npu_set_ofm_blk_depth_m1_t(config->OfmBlock().Depth() - 1));
}

// Generates IB_END/IB_START/AB_START/ACC_FORMAT registers
void EthosU55RCSGenerator::GenerateShramRegisters(const EthosU55OpConfig *config, bool hasIfm2)
{
    auto &layout = config->_layout;
    Emit(isa::npu_set_ifm_ib_end_t(layout.ibEnd));
    Emit(isa::npu_set_ab_start_t(layout.abStart));
    if ( hasIfm2 )
    {
        Emit(isa::npu_set_ifm2_ib_start_t(layout.ibStart2));
    }
    // ACC_FORMAT register
    auto accType = config->_accumulatorType;
    acc_format format;
    if ( accType == EthosU55SHRamElements::SHRAM_Acc16 )
    {
        format = acc_format::F16;
    }
    else if ( accType == EthosU55SHRamElements::SHRAM_Acc32 )
    {
        format = acc_format::I32;
    }
    else
    {
        assert(accType == EthosU55SHRamElements::SHRAM_Acc40);
        format = acc_format::I40;
    }
    Emit(isa::npu_set_acc_format_t(format));
}

// Calculates and generates KERNEL_WAIT or DMA_WAIT register
void EthosU55RCSGenerator::GenerateWaits(bool isKernelWait, const MemoryAccesses &memoryAccesses, std::deque<MemoryAccesses> &outstandingAccesses)
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

void EthosU55RCSGenerator::UpdateMemoryAccesses(const MemoryAccesses &memoryAccesses, std::deque<MemoryAccesses> &accessesToUpdate, int maxWaits)
{
    accessesToUpdate.push_back(memoryAccesses);
    if ( int(accessesToUpdate.size()) > maxWaits )
    {
        accessesToUpdate.pop_front();
    }
}

// Inserts DMA commands for copying LUTs from constant memory to LUT memory
void EthosU55RCSGenerator::InsertLUTDMACommand(
    int index, const HLCStripe *stripe, Temporaries &temps, std::vector<const HighLevelCommand *> &emitted)
{
    int lutSlotSize = _arch->_shram.lutSlotSize;
    auto op = stripe->operation;
    auto config = static_cast<EthosU55OpConfig *>(op->config);

    assert(op->type == OpType::LUT || (!op->subOps.empty() && op->subOps[0].type == OpType::LUT));

    const auto &srcTens = op->type == OpType::LUT ? op->parameters.lut : op->subOps[0].parameters.lut;
    assert(config->_layout.lutStart > 0);
    assert(srcTens.sizeBytes % lutSlotSize == 0);
    bool alreadyInLutMem;
    int sizeInSlots = srcTens.sizeBytes / lutSlotSize;
    int slot = AllocateLutSlot(_lutSlots, op.get(), sizeInSlots, index, alreadyInLutMem);
    _stripeToLutSlot[stripe] = slot;

    if ( !alreadyInLutMem )
    {
        auto dma = std::make_unique<HLCDMA>();
        dma->srcMemArea = srcTens.memArea;
        dma->srcAddress = srcTens.address;
        dma->length = srcTens.sizeBytes;
        dma->destMemArea = _arch->LUTMemory();
        dma->destAddress = _arch->_shram.bankSizeBytes * config->_layout.lutStart + slot * lutSlotSize;
        emitted.push_back(dma.get());
        temps.cmds.push_back(std::move(dma));
    }
}

// Inserts DMA commands to handle TILE operations
void EthosU55RCSGenerator::InsertTileDMACommand(const HLCStripe *stripe, Temporaries &temps, std::vector<const HighLevelCommand *> &emitted)
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

    auto op = stripe->operation;
    assert(op->type == OpType::Tile);

    auto &ifm = op->ifm[0];
    auto &ofm = op->ofm;

    assert(ifm.format == TensorFormat::NHWC);
    assert(ofm.format == TensorFormat::NHWC);

    const auto &tileParams = op->parameters.tile;

    reshapeFunc(ifm.shape, tileParams.axis);
    reshapeFunc(ofm.shape, tileParams.axis);

    int srcOffset = 0;
    int dstOffset = 0;
    int elemSize = DataTypeSizeBits(ifm.dataType) / 8;
    int rowBytes = ifm.shape[2] * ifm.shape[3] * elemSize;
    // each row in the IFM is copied separately
    // and duplicated based on the multiplier attribute.
    for ( int h = 0; h < ifm.shape.Height(); h++ )
    {
        for ( int i = 0; i < tileParams.multiplier; i++ )
        {
            auto dma = std::make_unique<HLCDMA>();
            dma->srcMemArea = ifm.memArea;
            dma->srcAddress = ifm.address + srcOffset;
            dma->length = rowBytes;
            dma->destMemArea = ofm.memArea;
            dma->destAddress = ofm.address + dstOffset;
            emitted.push_back(dma.get());
            temps.cmds.push_back(std::move(dma));
            dstOffset += rowBytes;
        }
        srcOffset += rowBytes;
    }
}

static inline int FirstSwapped(unsigned transpose, int &from)
{
    unsigned mask = unsigned(transpose) ^ unsigned(TransposeType::None);
    for ( int i = 0; i < 8; i++ )
    {
        if ( mask & 0xF )
        {
            from = (transpose >> (i * 4)) & 0xF;
            return i;
        }
        mask = mask >> 4;
    }
    from = 0;
    return -1;
}

void EthosU55RCSGenerator::InsertTransposeCommand(const HLCStripe *stripe, Temporaries &temps, std::vector<const HighLevelCommand *> &emitted)
{
    auto op = stripe->operation;
    auto &ifm = op->ifm[0];
    auto &ofm = op->ofm;

    assert(op->subOps.empty());
    assert(ifm.format == TensorFormat::NHWC);
    assert(ofm.format == TensorFormat::NHWC);
    assert(((ofm.transpose == TransposeType::NWHC) || !ifm.slice.shape || (ifm.shape == ifm.slice.shape)) && "Implementation cannot be sliced");
    ifm.shape = Shape::PadAxes(ifm.shape, 4, 1);
    assert((ifm.shape.AxisProduct(0, ifm.shape.Size() - 3) <= 1) && "Batch transposes unsupported");
    Shape outShape = ifm.shape.Permute(unsigned(ofm.transpose));

    // Which indexed axes have been swapped
    unsigned swapMask = unsigned(ofm.transpose) ^ unsigned(TransposeType::None);
    unsigned validMask = ofm.shape.ShapeMask();
    bool identity = (swapMask == 0) || (outShape.EqualMask(ofm.shape.WithOnes()) == validMask) || (ifm.shape.ElementsWH() == 1);
    if ( identity )
    {
        LOG_WARN("RCS: Emitting no-op transpose as a memory copy\n");
        auto dma = std::make_unique<HLCDMA>();
        dma->srcMemArea = ifm.memArea;
        dma->srcAddress = ifm.address;
        dma->length = DataTypeStorageSizeBytes(ifm.dataType, ifm.shape.Elements());
        dma->destMemArea = ofm.memArea;
        dma->destAddress = ofm.address;
        emitted.push_back(dma.get());
        temps.cmds.push_back(std::move(dma));
    }
    else
    {
        // Strided output on AveragePool can swap Height/Width over any channel depth by
        // adjusting the output strides to place the channel arrays in the required layout.
        //
        // IFM [h_pos, w_pos] = h_pos * ifm_stride_h + w_pos * ifm_stride_w
        // OFM [h_pos, w_pos] = h_pos * ofm_stride_w + w_pos * ofm_stride_h (stride has been swapped)
        //
        // Example:
        // Shape (2,3)            transposed to Shape (3,2)
        // |0|1|2| ifm_stride_w = 1             |0|3| ofm_stride_w = 1
        // |4|5|6| ifm_stride_h = 3             |1|4| ofm_stride_h = 2
        //                                      |2|5|
        //
        // This can be used to implement any 2-axis channel swap for 3 axes or fewer.
        //   NWHC - Transpose volume in a single pass.
        //   NHCW - Transpose 'height' number of CW1 slices.
        //   NCWH - Transpose 'width' number of HW1 slices (requires extra IFM striding).
        HLCFeatureMap inFM = ifm;
        HLCFeatureMap outFM = ofm;

        // Only two axis swaps can be achieved using AvgPool
        if ( NonZeroNybbles(swapMask) == 2 )
        {
            int elementSize = DataTypeSizeBits(ofm.dataType) / 8;
            // Activation element size must be supported or contiguous-channel bytes preserved
            // when transposed since Ethos-U55 ignores channel stride for NHWC tensors.
            assert(!((elementSize > 2) && (ofm.transpose == TransposeType::NCWH)));

            // Can only swap 2 axes at once using this method
            int from;
            int to = FirstSwapped(unsigned(ofm.transpose), from);
            from = ifm.shape.Size() - 1 - from;
            to = ifm.shape.Size() - 1 - to;

            // May be decomposing NWHC in depth
            Shape sliceShape = ifm.slice.shape ? ifm.slice.shape : ifm.shape;

            // Place the swappable axes in H/W (works in elements here)
            int depth = 1, slices = 1;
            int ifmStep = 0;
            int ofmStep = 0;

            // Not all elements participate in the transposed axes
            if ( (sliceShape[from] * sliceShape[to]) != sliceShape.Elements() )
            {
                if ( ofm.transpose == TransposeType::NWHC )
                {
                    depth = sliceShape.Depth();
                    slices = 1;
                    ifmStep = ofmStep = 0;
                    assert(from == 1 && to == 2);
                }
                else if ( ofm.transpose == TransposeType::NHCW )
                {
                    depth = 1;
                    slices = ifm.shape.Height();
                    ifmStep = ofmStep = ifm.shape.Depth() * ifm.shape.Width() * elementSize;
                    assert(from == 2 && to == 3);
                }
                else if ( ofm.transpose == TransposeType::NCWH )
                {
                    assert(elementSize <= 2);
                    depth = 1;
                    slices = ifm.shape.Width();
                    ifmStep = ifm.shape.Depth() * elementSize;
                    ofmStep = ifm.shape.Height() * elementSize;
                    assert(from == 1 && to == 3);
                }
                else
                {
                    assert(false && "Unsupported transpose");
                }
            }

            bool as16Bit = (outFM.dataType == DataType::Int16);
            // Recalculate destination as same as source but with output different strides
            outFM.shape = Shape(1, ifm.shape[from], ifm.shape[to], depth * (as16Bit ? 1 : elementSize));
            inFM.shape = outFM.shape;
            // Measure shapes in terms of bytes where necessary.
            if ( !as16Bit )
            {
                outFM.dataType = DataType::Int8;
                inFM.dataType = DataType::Int8;
            }
            // Input address (potential depth slices)
            if ( ifm.slice.offset )
            {
                inFM.address = AddressForCoordinate(ifm, ifm.strides, ifm.slice.offset);
                inFM.slice.offset = ifm.slice.offset.WithZeros();
            }
            // Output address (potential depth slices)
            if ( ofm.slice.offset )
            {
                outFM.address = AddressForCoordinate(ofm, ofm.strides, ofm.slice.offset);
                outFM.slice.offset = ofm.slice.offset.WithZeros();
            }

            // Special case for IFM with sparse strides
            if ( (slices > 1) && (ofm.transpose == TransposeType::NCWH) )
            {
                outFM.strides = Shape(1, elementSize, elementSize * ifm.shape.Width() * ifm.shape.Height(), elementSize);
                inFM.strides = Shape(1, elementSize * ifm.shape.Width() * ifm.shape.Depth(), elementSize, elementSize);
            }
            else if ( ofm.transpose == TransposeType::NWHC )
            {
                outFM.strides = Shape(1, elementSize * ofm.shape.Depth(), elementSize * ofm.shape.Depth() * outFM.shape.Height(), elementSize);
                inFM.strides = Shape(1, elementSize * ifm.shape.Depth() * inFM.shape.Width(), elementSize * ifm.shape.Depth(), elementSize);
            }
            else
            {
                outFM.strides = Shape(1, elementSize * depth, elementSize * depth * outFM.shape.Height(), elementSize);
                inFM.strides = Shape::GetStridesForShape(inFM.shape, (as16Bit ? elementSize : 1));
            }

            // Repeat the transpose at advancing offsets for each slice
            for ( int i = 0; i < slices; i++ )
            {
                // Create new stripe operations
                auto cmd = std::make_unique<HLCStripe>(*stripe);
                cmd->operation = std::make_shared<HLCOperation>();
                cmd->operation->kernel = Kernel::UnitKernel();
                cmd->operation->type = OpType::AvgPool;
                cmd->opGroup = nullptr;
                cmd->operation->ifm.push_back(inFM);
                cmd->operation->ofm = outFM;
                cmd->ofmArea = outFM.shape;
                cmd->ifmAreas[0] = inFM.shape;

                // Find a common block configuration
                if ( i == 0 )
                {
                    ArchitectureConfigQuery query{};
                    query.kernel = &cmd->operation->kernel;
                    query.ifmBits = DataTypeSizeBits(ifm.dataType);
                    query.ofmBits = DataTypeSizeBits(ofm.dataType);
                    query.ifmShape[0] = inFM.shape;
                    query.ofmShape = outFM.shape;
                    query.ofmFormat = TensorFormat::NHWC;
                    query.transpose = ofm.transpose;
                    temps.configs.push_back(_arch->FindBlockConfig(cmd->operation->type, query));
                }
                cmd->operation->config = temps.configs.back().get();
                // Add to CMD list
                emitted.push_back(cmd.get());
                temps.cmds.push_back(std::move(cmd));
                // Move to next slice
                inFM.address += ifmStep;
                outFM.address += ofmStep;
            }
        }
        else
        {
            assert(false && "3-axis swaps must be decomposed");
        }
    }
}


//----------------------------------------------------------------------
// Operations
//----------------------------------------------------------------------

// Generates NPU_OP_* command
void EthosU55RCSGenerator::GenerateOperationCode(OpType opType)
{
    if ( IsPooling(opType) )
    {
        pooling_mode mode;
        if ( opType == OpType::AvgPool || opType == OpType::ResizeBilinear )
        {
            mode = pooling_mode::AVERAGE;
        }
        else if ( opType == OpType::MaxPool )
        {
            mode = pooling_mode::MAX;
        }
        else
        {
            assert(opType == OpType::ReduceSum);
            mode = pooling_mode::REDUCE_SUM;
        }
        Emit(isa::npu_op_pool_t(mode));
    }
    else if ( IsDepthwise(opType) )
    {
        Emit(isa::npu_op_depthwise_t());
    }
    else if ( IsConvolution(opType) || IsVectorProduct(opType) )
    {
        Emit(isa::npu_op_conv_t());
    }
    else if ( IsElementwise(opType) )
    {
        const auto &item = s_ElementwiseMap.find(opType);
        if ( item == s_ElementwiseMap.end() )
        {
            assert(false && "Unsupported elementwise operator");
        }
        else
        {
            Emit(isa::npu_op_elementwise_t(item->second));
        }
    }
    else if ( _arch->UseAvgPoolNop(opType) || opType == OpType::Rescale )
    {
        // Implemented using AvgPool
        Emit(isa::npu_op_pool_t(pooling_mode::AVERAGE));
    }
    else
    {
        assert(false && "Unsupported operator");
    }
}

void EthosU55RCSGenerator::GenerateCommon(const HLCStripe *stripe, bool useGlobalScale, RCSIfmScaleMode opToScale,
    MemoryAccesses &memoryAccesses, int ifm0Index)
{
    auto op = stripe->operation.get();
    GenerateIFM(op->type, op->ifm[ifm0Index], stripe->ifmAreas[ifm0Index]);
    memoryAccesses.push_back(ToMemoryAccess(op->ifm[ifm0Index], stripe->ifmAreas[ifm0Index], AccessDirection::Read));

    // Select rounding based on RCSIfmScaleMode
    // rounding doesn't matter for RcsIfmScaleMode::OPA_OPB_16
    // as the scaling is not a fraction.
    HLCRoundMode rounding = op->ifm[0].rounding;
    if ( opToScale == RCSIfmScaleMode::OPB_32 )
    {
        assert(op->ifm.size() > 1);
        rounding = op->ifm[1].rounding;
    }
    GenerateIFMPrecision(op->ifm[ifm0Index], opToScale, rounding);
    ifm_upscale_mode upscaleMode = ToIfmUpscaleMode(op->ifm[0].resamplingMode);
    Emit(isa::npu_set_ifm_upscale_t(upscaleMode));
    if ( !IsElementwise(op->type) )
    {
        GeneratePadding(stripe->padding);
    }
    GenerateOFM(op->type, op->ofm, stripe->ofmArea);
    memoryAccesses.push_back(ToMemoryAccess(op->ofm, stripe->ofmArea, AccessDirection::Write));
    GenerateOFMPrecision(op->ofm, useGlobalScale);
    EthosU55OpConfig *config = static_cast<EthosU55OpConfig *>(stripe->operation->config);
    if ( !IsElementwise(op->type) )
    {
        GenerateKernel(op->kernel, config->Traversal() == EthosUTraversal::PartKernel);
    }
    GenerateWeights(stripe, memoryAccesses);
    GenerateScales(stripe, memoryAccesses);
    GenerateActivation(stripe, memoryAccesses);
    if ( _arch->_shram.reservedEndBanks == 0 )
    {
        // SHRAM has no reserved LUT banks; LUT is overwritten by accumulator buffer
        memoryAccesses.emplace_back(
            AccessDirection::Write, _arch->LUTMemory(), 0, _arch->_shram.bankSizeBytes * _arch->_shram.totalBanks);
    }
}

// Conv2D/Depthwise operations
void EthosU55RCSGenerator::GenerateConvolutionOp(const HLCStripe *stripe, MemoryAccesses &memoryAccesses)
{
    GenerateCommon(stripe, false, RCSIfmScaleMode::OPA_OPB_16, memoryAccesses);
}

// MaxPool/AvgPool/ResizeBilinear or operations that are mapped to AvgPool
void EthosU55RCSGenerator::GeneratePoolingOp(const HLCStripe *stripe, MemoryAccesses &memoryAccesses)
{
    auto op = stripe->operation.get();
    auto pad = stripe->padding;
    auto padSum = pad.top + pad.left + pad.bottom + pad.right;
    bool useGlobalScale = !op->scales;
    HLCStripe modifiedStripe(nullptr);

    if ( _arch->UseAvgPoolNop(op->type) )
    {
        assert(op->kernel.Size() == Point2i(1, 1));
        assert(op->kernel.Stride() == Point2i(1, 1));
        assert(op->kernel.Dilation() == Point2i(1, 1));
        assert(op->kernel.DepthMultiplier() == 1);
        assert(useGlobalScale);
        assert(op->ifm.size() > 0);
        // Op is being used as a 32-bit unscaled memory copy but
        // we do not support more than 16-bit activations so adjust
        // the tensor types and strides.
        if ( op->type == OpType::MemoryCopy && (op->ifm[0].dataType == op->ofm.dataType) && DataTypeSizeBits(op->ofm.dataType) == 32 )
        {
            assert(op->ifm[0].format == TensorFormat::NHWC);
            assert(op->ofm.format == TensorFormat::NHWC);
            modifiedStripe = *stripe;
            op->ifm[0].dataType = DataType::Int16;
            op->ifm[0].shape[-1] *= 2;
            op->ifm[0].strides[-1] /= 2;
            modifiedStripe.ifmAreas[0].Start() = modifiedStripe.ifmAreas[0].Start() * Shape(2);
            modifiedStripe.ifmAreas[0].End() = modifiedStripe.ifmAreas[0].End() * Shape(2);

            op->ofm.dataType = DataType::Int16;
            op->ofm.shape[-1] *= 2;
            op->ofm.strides[-1] /= 2;
            modifiedStripe.ofmArea.Start() = modifiedStripe.ofmArea.Start() * Shape(2);
            modifiedStripe.ofmArea.End() = modifiedStripe.ofmArea.End() * Shape(2);
            stripe = &modifiedStripe;
        }
    }
    GenerateCommon(stripe, useGlobalScale, RCSIfmScaleMode::OPA_OPB_16, memoryAccesses);
    GenerateOFMScalingForPooling(op, useGlobalScale);
}

// Elementwise operations
void EthosU55RCSGenerator::GenerateElementwiseOp(const HLCStripe *stripe, MemoryAccesses &memoryAccesses)
{
    auto op = stripe->operation.get();
    auto opType = op->type;
    bool useGlobalScale = opType == OpType::Add || opType == OpType::Sub || opType == OpType::Mul || opType == OpType::LeakyRelu || opType == OpType::Abs;
    if ( IsUnaryElementwise(opType) )
    {
        assert(op->ifm.size() == 1);
        auto opToScale = GenerateScalingForElementwise(op, 0);
        GenerateCommon(stripe, useGlobalScale, opToScale, memoryAccesses);
    }
    else
    {
        // Binary operation: generate IFM2 registers
        assert(op->ifm.size() == 2);
        assert(ToActivationPrecision(op->ifm[0].dataType) == ToActivationPrecision(op->ifm[1].dataType));
        assert(stripe->ifmAreas.size() == 2);
        int32_t scalarValue = 0;
        auto ifmShape = stripe->ifmAreas[0].SizeShape();
        auto ifm2Shape = stripe->ifmAreas[1].SizeShape();
        bool reversedOperands = IsScalar(op->ifm[0], scalarValue) || (ifmShape != ifm2Shape && ifmShape.IsSubShapeOf(ifm2Shape));
        int ifmIndex = 0;
        if ( reversedOperands )
        {
            // If reversed, the scalar/broadcasted feature map has to be the ifm2 tensor,
            // so switch ifm/ifm2
            ifmIndex = 1;
            std::swap(ifmShape, ifm2Shape);
        }
        auto opToScale = GenerateScalingForElementwise(op, ifmIndex);
        GenerateCommon(stripe, useGlobalScale, opToScale, memoryAccesses, ifmIndex);
        int ifm2Index = 1 - ifmIndex;
        assert(size_t(ifm2Index) < stripe->ifmAreas.size());
        const HLCFeatureMap &ifm2 = op->ifm.at(ifm2Index);
        bool isScalar = IsScalar(ifm2, scalarValue);
        GenerateIFM2(opType, ifm2, stripe->ifmAreas[ifm2Index], isScalar, scalarValue);
        if ( !isScalar )
        {
            memoryAccesses.push_back(ToMemoryAccess(ifm2, stripe->ifmAreas[ifm2Index], AccessDirection::Read));
        }
        GenerateIFM2Precision(ifm2);
        GenerateIFM2Broadcast(ifmShape, ifm2Shape, reversedOperands, isScalar);
    }
}

bool EthosU55RCSGenerator::GenerateStripe(const HLCStripe *stripe, const HLCStripe *prevStripe, AccessTracking &accesses)
{
    MemoryAccesses memoryAccesses;

    auto opType = stripe->operation->type;
    EthosU55NpuOp npuOp = ArchEthosU55::GetHWOp(opType);
    if ( npuOp == EthosU55NpuOp::Pooling || npuOp == EthosU55NpuOp::ReduceSum )
    {
        GeneratePoolingOp(stripe, memoryAccesses);
    }
    else if ( npuOp == EthosU55NpuOp::Depthwise || npuOp == EthosU55NpuOp::Convolution || npuOp == EthosU55NpuOp::VectorProduct )
    {
        GenerateConvolutionOp(stripe, memoryAccesses);
    }
    else if ( npuOp == EthosU55NpuOp::Elementwise )
    {
        GenerateElementwiseOp(stripe, memoryAccesses);
    }
    else
    {
        LOG_ERROR("Register command stream generator: unsupported operator '{}'\n", OpTypeToString(opType));
        assert(false);
        return false;
    }
    EthosU55OpConfig *config = static_cast<EthosU55OpConfig *>(stripe->operation->config);
    GenerateBlockConfig(config);
    GenerateShramRegisters(config, stripe->operation->ifm.size() >= 2);

    // BLOCKDEP register tracking
    int blockdep = CalcBlockDep(prevStripe, stripe);
    Emit(isa::npu_set_blockdep_t(blockdep));
    GenerateWaits(false, memoryAccesses, accesses.outstandingDmaAccesses);
    UpdateMemoryAccesses(memoryAccesses, accesses.outstandingNpuAccesses, accesses.maxOutstandingKernelOps);
    GenerateOperationCode(stripe->operation->type);
    return true;
}

// Generates register commands for DMA operations
void EthosU55RCSGenerator::GenerateDMA(const HLCDMA *dma, AccessTracking &accesses)
{
    MemoryAccesses memoryAccesses;

    auto srcRegionMode = dma_region_mode::EXTERNAL;
    auto destRegionMode = dma_region_mode::EXTERNAL;
    if ( dma->destMemArea == _arch->LUTMemory() )
    {
        destRegionMode = dma_region_mode::INTERNAL;
    }
    auto strideMode = dma_stride_mode::D1;
    CheckAddressRange(dma->srcMemArea.memory, dma->srcAddress, dma->length);
    CheckAddressRange(dma->destMemArea.memory, dma->destAddress, dma->length);
    Emit(isa::npu_set_dma0_src_region_t(ToRegion(dma->srcMemArea), srcRegionMode, strideMode));
    Emit(isa::npu_set_dma0_src_t(dma->srcAddress));
    Emit(isa::npu_set_dma0_dst_region_t(ToRegion(dma->destMemArea), destRegionMode, strideMode));
    Emit(isa::npu_set_dma0_dst_t(dma->destAddress));
    Emit(isa::npu_set_dma0_len_t(dma->length));

    // Track memory accesses
    memoryAccesses.emplace_back(AccessDirection::Read, dma->srcMemArea, dma->srcAddress, dma->srcAddress + dma->length);
    memoryAccesses.emplace_back(AccessDirection::Write, dma->destMemArea, dma->destAddress, dma->destAddress + dma->length);
    GenerateWaits(false, memoryAccesses, accesses.outstandingDmaAccesses);
    GenerateWaits(true, memoryAccesses, accesses.outstandingNpuAccesses);
    UpdateMemoryAccesses(memoryAccesses, accesses.outstandingDmaAccesses, accesses.maxOutstandingDMAOps);

    Emit(isa::npu_op_dma_start_t());
}

void EthosU55RCSGenerator::PrepareCommand(int index, HighLevelCommand *cmd, Temporaries &temps, std::vector<const HighLevelCommand *> &emitted)
{
    emitted.clear();

    if ( cmd->IsStripe() )
    {
        HLCStripe *stripe = static_cast<HLCStripe *>(cmd);
        auto op = stripe->operation;
        if ( op->type == OpType::Tile )
        {
            InsertTileDMACommand(stripe, temps, emitted);
            return;  // Return early to replace original op
        }
        else if ( op->type == OpType::LUT || (!op->subOps.empty() && op->subOps[0].type == OpType::LUT) )
        {
            InsertLUTDMACommand(index, stripe, temps, emitted);
        }
        else if ( op->type == OpType::Transpose )
        {
            InsertTransposeCommand(stripe, temps, emitted);
            return;
        }
        else if ( op->type == OpType::MatMul )
        {
            return;  // Delete until implemented
        }
        else if ( _arch->_shram.reservedEndBanks == 0 )
        {
            // LUT is overwritten by SHRAM accumulator buffers; clear slots
            for ( auto &slot : _lutSlots )
            {
                slot.hlcOp = nullptr;
                slot.lastUsed = 0;
            }
        }
    }

    // Emit original op
    emitted.push_back(cmd);
}


std::vector<uint32_t> EthosU55RCSGenerator::GenerateCommandStream(
    std::vector<std::unique_ptr<HighLevelCommand>> &highLevelCommandStream, CmdRanges *cmdRanges, bool verbose)
{
    _emit.Clear();
    _stripeToLutSlot.clear();
    // Clear lut slots at start of command stream generation
    for ( auto &slot : _lutSlots )
    {
        slot.hlcOp = nullptr;
        slot.lastUsed = 0;
    }

    GenerateInitialRegisterSetup();

    AccessTracking accesses;
    accesses.maxOutstandingDMAOps = _arch->MaxOutstandingDMAOps();
    accesses.maxOutstandingKernelOps = _arch->MaxOutstandingKernelOps();

    const HLCStripe *prevStripe = nullptr;
    std::vector<std::pair<unsigned, std::string>> debugInfo;

    Temporaries temporaries;
    std::vector<const HighLevelCommand *> emitted(4);

    int cmdIndex = 0;
    for ( const auto &cmd : highLevelCommandStream )
    {
        int emitStart = _emit.Position();

        PrepareCommand(cmdIndex, cmd.get(), temporaries, emitted);

        for ( auto hlc : emitted )
        {
            if ( hlc->IsStripe() )
            {
                auto stripe = static_cast<const HLCStripe *>(hlc);
                if ( verbose )
                {
                    debugInfo.emplace_back(_emit.Position(), stripe->operation->ToString());
                }
                if ( !GenerateStripe(stripe, prevStripe, accesses) )
                {
                    return std::vector<uint32_t>();
                }
                prevStripe = stripe;
            }
            else
            {
                auto dma = static_cast<const HLCDMA *>(hlc);
                if ( verbose )
                {
                    debugInfo.emplace_back(_emit.Position(), dma->ToString());
                }
                GenerateDMA(dma, accesses);
            }
        }

        // Return command mapping information to the caller
        if ( cmdRanges && cmd->IsStripe() )
        {
            cmdRanges->emplace_back(static_cast<HLCStripe *>(cmd.get())->operation->_srcId, emitStart, _emit.Position());
        }
        cmdIndex++;
    }
    Emit(isa::npu_op_stop_t(0xFFFF));
    if ( verbose )
    {
        PrintCommandStream(_emit.CommandStream(), debugInfo);
    }
    return _emit.CommandStream();
}

}  // namespace regor
