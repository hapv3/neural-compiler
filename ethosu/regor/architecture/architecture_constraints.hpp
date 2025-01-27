//
// SPDX-FileCopyrightText: Copyright 2024-2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
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

#include "architecture.hpp"
#include "common/data_type.hpp"
#include "common/reverse_type.hpp"
#include "common/scaling.hpp"
#include "common/shape.hpp"
#include "common/transpose_type.hpp"
#include "compiler/op_type.hpp"
#include "compiler/quantization.hpp"
#include "compiler/tensor_properties.hpp"

namespace regor
{

enum class TensorFormat : uint16_t;

/// <summary>
/// Simple Architecture feature map properties
/// </summary>
struct ArchFM
{
    Shape shape;
    DataType type = {};
    TensorFormat format = {};
};

/// <summary>
/// Information for querying support for Resize
/// </summary>
struct ResizeSupportQuery
{
    ArchResizeMode mode;
    GraphApi::FractionND scaleY;
    GraphApi::FractionND scaleX;
    int offsetY;
    int offsetX;
    Shape ifmShape;
};

const std::array<OpType, 10> elemWiseMainOps = {OpType::Minimum, OpType::Maximum, OpType::Add, OpType::Mul, OpType::Sub,
    OpType::Abs, OpType::Exp, OpType::LeakyRelu, OpType::Rsqrt, OpType::SquaredDifference};
/// <summary>
/// Information for querying whether an operation can be executed by the hardware
/// </summary>
struct ExecutionQuery
{
    OpType opType;
    OpType targetType;
    DataType ifmType;
    DataType ifm2Type;
    Shape ifmShape;
    Shape ifm2Shape;
    Shape ofmShape;
    DataType ofmType;
    TransposeType transposeType;
    ReverseType reverseTypeMask;
    ResizeSupportQuery resizeQuery;
    bool quantScalingInvalidOrUnequal = false;
};

struct ArchOperatorQuery
{
    ArchFM ifm[2];
    ArchFM ofm;
    ReverseType reverseMask = ReverseType::None;
    TransposeType transposeMask = TransposeType::None;
    struct
    {
        ResizeSupportQuery resize;
    } specific;
    ~ArchOperatorQuery(){};
};

enum class ArchRequirement
{
    None = 0,
    ScratchTensor = 1,
    OutputFormat = 2,
    OpSubstitution = 4,
};

struct ArchRequirements
{
    Flags<ArchRequirement> req;
    struct
    {
        Shape size;
        DataType type = DataType::None;
        TensorFormat format = TensorFormat::Unknown;
    } scratch;
    TensorFormat ofmFormat = TensorFormat::Unknown;
    OpType substitution = OpType::None;
};

enum class TransposeSupport
{
    None,
    NHWC = 1,
    NHCWB16 = 2,
    Any = NHWC | NHCWB16,
};

// Results for operator queries can return a combination of the
// following flags.
// Native - Operator supported natively in some or all cases (see other flags).
// Constrained - Not all operator cases have support (detailed queries may fail).
// HasRequirements - Cases are supported if architecture requirements are met.
enum class QueryResult
{
    None = 0,
    Unsupported = 1,
    Native = 2,
    Constrained = 4,
    HasRequirements = 8,
    NativeHasReq = Native | HasRequirements,
    NativeConstrained = Native | Constrained,
    NativeConstrainedHasReq = Native | Constrained | HasRequirements,
};

/// <summary>
/// Architecture capabilties query
/// </summary>
class IArchitectureConstraints
{
public:
    virtual ~IArchitectureConstraints() = default;

    virtual bool SupportsReverse(OpType opType, ReverseType reverseTypeMask) = 0;
    virtual bool SupportsFusedRescale(OpType opType, TensorUsage tensorUsage, DataType rescaleFromType,
        DataType rescaleToType, DataType opFromType, DataType opToType, const Quantization &quantization) = 0;
    virtual bool SupportsRescale(DataType fromType, DataType toType) = 0;
    virtual TransposeSupport SupportsTranspose(OpType opType, TransposeType transposeType) = 0;
    virtual bool SupportsAccumulatorSaveRestore() = 0;
    virtual bool SupportsLeakyRelu(bool quantized, DataType type) = 0;
    virtual bool SupportsNegativeStrides() = 0;
    virtual bool SupportsNot() = 0;
    virtual Flags<QueryResult> OperatorQuery(OpType opType, const ArchOperatorQuery *query, ArchRequirements *req = nullptr) = 0;

    bool CanExecute(const ExecutionQuery &query)
    {
        bool valid = true;
        if ( IsFloat(query.ifmType | query.ifm2Type | query.ofmType) )
        {
            return false;
        }

        switch ( query.opType )
        {
            case OpType::MatMul:
                valid = SupportsMatMul(query.opType);
                break;
            case OpType::ReverseV2:
                valid = SupportsReverse(query.targetType, query.reverseTypeMask);
                break;
            case OpType::Gather:
                valid = SupportsGather(query.opType);
                break;
            case OpType::Scatter:
                valid = SupportsScatter(query.opType);
                break;
            case OpType::ArgMax:
                valid = SupportsArgMax(query.opType);
                break;
            case OpType::Cast:
                valid = SupportsCast(query.opType, query.ifmType, query.ofmType);
                break;
            case OpType::Resize:
                valid = SupportsResize(query.resizeQuery);
                break;
            default:
                break;
        }
        if ( std::find(elemWiseMainOps.begin(), elemWiseMainOps.end(), query.opType) != elemWiseMainOps.end() )
        {
            valid = SupportsNonMatchingShapes(query.ifmShape, query.ifm2Shape, query.ofmShape);
        }
        return valid;
    }

protected:
    virtual bool SupportsMatMul(OpType opType) = 0;
    virtual bool SupportsGather(OpType opType) = 0;
    virtual bool SupportsScatter(OpType opType) = 0;
    virtual bool SupportsResize(const ResizeSupportQuery &query) = 0;
    virtual bool SupportsArgMax(OpType opType) = 0;
    virtual bool SupportsCast(OpType opType, DataType ifmType, DataType ofmType) = 0;
    virtual bool SupportsNonMatchingShapes(const Shape &ifmShape, const Shape &ifm2Shape, const Shape &ofmShape) = 0;
};

}  // namespace regor
