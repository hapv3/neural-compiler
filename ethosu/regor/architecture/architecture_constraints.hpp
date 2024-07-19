//
// SPDX-FileCopyrightText: Copyright 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
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


/// <summary>
/// Information for querying whether an operation can be executed by the hardware
/// </summary>
struct ExecutionQuery
{
    OpType opType;
    OpType targetType;
    DataType ifmType;
    Shape ifmShape;
    DataType ofmType;
    TransposeType transposeType;
    ReverseType reverseType;
    ResizeSupportQuery resizeQuery;
    bool quantScalingInvalidOrUnequal = false;
};

namespace Constraints
{

}  // namespace Constraints

/// <summary>
/// Architecture capabilties query
/// </summary>
class IArchitectureConstraints
{
public:
    virtual ~IArchitectureConstraints() = default;

    virtual bool SupportsTransposeHW(OpType opType, TransposeType transposeType) = 0;
    virtual bool SupportsReverse(OpType opType, ReverseType reverseType) = 0;
    virtual bool SupportsFusedRescale(OpType opType, TensorUsage tensorUsage, DataType fromType, DataType toType,
        const Quantization &quantization) = 0;

    bool CanExecute(const ExecutionQuery &query)
    {
        bool valid = true;
        switch ( query.opType )
        {
            case OpType::LeakyRelu:
                valid = SupportsLeakyRelu(query.quantScalingInvalidOrUnequal, query.ifmType);
                break;
            case OpType::MatMul:
                valid = SupportsMatMul(query.opType);
                break;
            case OpType::Transpose:
                valid = SupportsTranspose(query.targetType, query.transposeType, query.ifmShape);
                break;
            case OpType::ReverseV2:
                valid = SupportsReverse(query.targetType, query.reverseType);
                break;
            case OpType::Gather:
                valid = SupportsGather(query.opType);
                break;
            case OpType::Scatter:
                valid = SupportsScatter(query.opType);
                break;
            case OpType::Sigmoid:
                valid = SupportsSigmoidTanhLutInt16(query.opType);
                break;
            case OpType::ArgMax:
                valid = SupportsArgMax(query.opType);
                break;
            case OpType::ResizeNearestNeighbor:
                valid = SupportsResize(query.resizeQuery);
                break;
            case OpType::ResizeBilinear:
                valid = SupportsResize(query.resizeQuery);
                break;
            case OpType::Cast:
                valid = SupportsCast(query.opType, query.ifmType, query.ofmType);
                break;
            default:
                break;
        }
        return valid;
    }

protected:
    virtual bool SupportsLeakyRelu(bool quantized, DataType type) = 0;
    virtual bool SupportsMatMul(OpType opType) = 0;
    virtual bool SupportsGather(OpType opType) = 0;
    virtual bool SupportsScatter(OpType opType) = 0;
    virtual bool SupportsTranspose(OpType opType, TransposeType transposeType, Shape ifmShape) = 0;
    virtual bool SupportsSigmoidTanhLutInt16(OpType opType) = 0;
    virtual bool SupportsResize(const ResizeSupportQuery &query) = 0;
    virtual bool SupportsArgMax(OpType opType) = 0;
    virtual bool SupportsCast(OpType opType, DataType ifmType, DataType ofmType) = 0;
};

}  // namespace regor
