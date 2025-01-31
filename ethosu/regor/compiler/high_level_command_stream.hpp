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

#pragma once

#include "common/common.hpp"

#include "architecture/architecture.hpp"
#include "architecture/weight_encoder.hpp"
#include "common/box.hpp"
#include "common/data_type.hpp"
#include "common/reverse_type.hpp"
#include "common/shape.hpp"
#include "common/transpose_type.hpp"
#include "kernel.hpp"
#include "scheduler_operation.hpp"

#include <unordered_map>
#include <vector>

namespace regor
{

enum class HLCRoundMode : uint8_t
{
    DBL = 0,
    TRUNCATE = 1,
    NATURAL = 2,
    TRUNCATE_TO_LOWER = 3,
    DOUBLE_ASYMMETRIC = 4,
    SYMMETRIC = 5,
    AUTO = 0xff
};

struct HLCPadding
{
    int top = 0;
    int left = 0;
    int bottom = 0;
    int right = 0;

    std::string ToString() const
    {
        return fmt::format("[top:{},left:{},bottom:{},right:{}]", top, left, bottom, right);
    }
};

/// <summary>
/// IFM/OFM information needed to generate register commands
/// </summary>
struct HLCFeatureMap
{
    TensorUsage usage = TensorUsage::None;
    DataType dataType = DataType::None;
    TensorFormat format = TensorFormat::Unknown;
    TransposeType transpose = TransposeType::None;
    ReverseType reverse = ReverseType::None;
    TensorSlice slice;
    Shape shape;
    Shape strides;
    MemArea memArea;
    BufferView bufferView;
    Quantization quantization;
    Point2i stepXY = {1, 1};
    Address address = -1;
    UniqueId uid = ~0u;
    HLCRoundMode rounding = HLCRoundMode::AUTO;
    ArchResampling resamplingMode = ArchResampling::None;

    int AllocationSizeBytes() const { return TensorAllocationBytes(shape, format, dataType); }

    std::string ToString() const
    {
        return fmt::format("[{}], format: {}, {}:{}, address: {}", shape.ToString(), format, memArea.memory->Name(),
            memArea.usage.ToString(), address);
    }
};

/// <summary>
/// Information about encoded weights
/// </summary>
struct HLCWeights
{
    MemArea memArea;
    Buffering buffering;
    Flags<WeightFormat> format;
    Address address = -1;
    int doubleBufferOffset;  // When double buffering: offset to second buffer
    int subStreams = 1;
    std::unordered_map<int, WeightRange> encodedRanges;

    std::string ToString() const
    {
        return fmt::format("{} ranges, buffering: {}, {}:{}, address: {}, format: {}", encodedRanges.size(),
            int(buffering), memArea.memory->Name(), memArea.usage, address, format.ToString());
    }
};

// Parameters that apply only to particular sub operation
union HLCParameters
{
    // Alpha value for LeakyReLU
    struct
    {
        float alpha;
    } leaky_relu;

    // Location of the source LUT, to be DMAed to LUT memory
    struct LUT
    {
        MemArea memArea;
        Address address;
        int sizeBytes;
        DataType ifmType;
    } lut;

    struct
    {
        Fraction<int> scaleY;
        Fraction<int> scaleX;
        int offsetY;
        int offsetX;
        ArchResizeMode mode;
    } resize;

    struct
    {
        AxisMask axis;
    } argmax;

    struct
    {
        int axis;
        int multiplier;
    } tile;
};

/// <summary>
/// Sub operation
/// </summary>
struct HLCSubOperation
{
    OpType type = OpType::None;
    std::vector<HLCFeatureMap> ifm;
    HLCFeatureMap ofm;
    HLCParameters parameters = {};
    UniqueId _srcId = 0;
};

/// <summary>
/// Contains information needed to generate register commands for an NPU operation.
///
/// There is one HLCOperation for every SchedulerOperation. Each HLCOperation can be
/// associated with one (= non-cascaded) or more (= cascaded) HLCStripes
/// </summary>
struct HLCOperation : HLCSubOperation
{
    Kernel kernel;
    std::unique_ptr<HLCWeights> weights;
    std::unique_ptr<HLCWeights> scales;
    std::vector<HLCSubOperation> subOps;
    ArchitectureOpConfig *config = nullptr;

#ifndef NDEBUG
    std::string name;  // name of OFM
#endif

    std::string ToString() const
    {
        std::string k = kernel.Size().x == 0 ? "" : kernel.ToString();
#ifdef NDEBUG
        std::string name = "";
#endif
        std::string subOpStr = subOps.empty() ? " -" : "";
        for ( auto &subOp : subOps )
        {
            subOpStr += " " + OpTypeToString(subOp.type);
        }
        return fmt::format("{} {}, subOps:{}, {} {}", OpTypeToString(type), name, subOpStr, k, config ? config->ToString(false) : "");
    }
};

class HighLevelCommand
{
public:
    virtual ~HighLevelCommand() = default;
    virtual bool IsStripe() const { return false; }
    virtual std::string ToString() const = 0;
};

/// <summary>
/// High level command that performs part of or whole NPU operation,
/// depending on the box settings.
/// </summary>
class HLCStripe : public HighLevelCommand
{
public:
    std::shared_ptr<HLCOperation> operation;
    ArchitectureOpGroup *opGroup;
    std::vector<Box> ifmAreas;
    Box ofmArea;
    int weightRangeDepth = 0;  // Identifies depth slice
    HLCPadding padding;

public:
    HLCStripe(const std::shared_ptr<HLCOperation> &operation_) : operation(operation_), opGroup(nullptr) {}
    bool IsStripe() const override { return true; }

    std::string ToString() const override
    {
        std::string ofm = "";
#ifndef NDEBUG
        ofm = " -> " + operation->name;
#endif
        std::string extra = "";
        if ( ifmAreas.size() > 1 )
        {
            extra = fmt::format(", IFM2 {}", ifmAreas[1].ToString());
        }
        else if ( operation->weights != nullptr && operation->weights->buffering != Buffering::None )
        {
            extra = fmt::format(", Weight depth: {}", weightRangeDepth);
        }
        if ( padding.top != 0 || padding.bottom != 0 || padding.left != 0 || padding.right != 0 )
        {
            extra += fmt::format(", padding: {}", padding.ToString());
        }
        if ( ofmArea.SizeShape().Elements() != operation->ofm.shape.Elements() )
        {
            extra += (ofmArea.SizeShape().ElementsWH() == operation->ofm.shape.ElementsWH()) ? ", buffered" : ", cascaded";
        }
        return fmt::format("{}{} OFM area {}, IFM {}{}", OpTypeToString(operation->type), ofm, ofmArea.ToString(),
            ifmAreas[0].ToString(), extra);
    }
};

/// <summary>
/// High level command that performs a DMA operation.
/// </summary>
class HLCDMA : public HighLevelCommand
{
public:
    MemArea srcMemArea;
    Address srcAddress;
    Shape srcStrides;  // Only valid for Ethos U85
    bool srcIndexed;   // Only valid for Ethos U85
    MemArea destMemArea;
    Address destAddress;
    Shape destStrides;   // Only valid for Ethos U85
    bool destIndexed;    // Only valid for Ethos U85
    MemArea idxMemArea;  // Only valid for Ethos U85
    Address idxAddress;  // Only valid for Ethos U85
    int idxSkip1;        // Only valid for Ethos U85
    int idxMax;          // Only valid for Ethos U85
    int length;
    Shape sizes;  // Only valid for Ethos U85

    std::string ToString() const override
    {
        return fmt::format("DMA src: {}:{}, address: {}, dest: {}:{}, address: {}, sizes: ({}), length: {}",
            srcMemArea.memory->Name(), srcMemArea.usage, srcAddress, destMemArea.memory->Name(), destMemArea.usage,
            destAddress, sizes ? sizes.ToString() : "N/A", std::to_string(length));
    }
};

}  // namespace regor
