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
    const Quantization *quantization = nullptr;
};

struct ArchOperatorQuery
{
    ArchFM ifm[2];
    ArchFM ofm;
    ReverseType reverseMask = ReverseType::None;
    TransposeType transposeMask = TransposeType::None;
    WeightFormat weightFormat = WeightFormat::None;
    ArchAccumulatorSource accSrc = ArchAccumulatorSource::Reset;
    const Kernel *kernel = nullptr;
    int axis = 0;  // Uses negative notation: -1 = C, -2 = W, ...
    ~ArchOperatorQuery(){};
};

enum class ArchRequirement
{
    None = 0x00,
    Tensor = 0x01,          // Tensor requirement
    OpSubstitution = 0x02,  // Operator substitution
    Decompose = 0x04,       // Decompose
};

enum class ArchProperty
{
    None = 0,
    TensorAxis = 1 << 0,
    TensorDims = 1 << 1,
    KernelStride = 1 << 2,
    KernelDilation = 1 << 3,
    DepthMultiplier = 1 << 4,
    TransposeMask = 1 << 5,
    ReduceAxis = 1 << 6,
    Scaling = 1 << 7,
    NonConstantWeights = 1 << 8,
};

struct ArchTensorRequirement
{
    const ArchTensorRequirement *next = nullptr;
    TensorUsage usage = TensorUsage::None;
    TensorFormat format = TensorFormat::Unknown;
    DataType type = DataType::None;
    Shape shape;
};

struct ArchRequirements
{
    Flags<ArchRequirement> req;
    ArchTensorRequirement tensor;
    OpType substitution = OpType::None;
    Flags<ArchProperty> decomposeProps;
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
    virtual bool SupportsFusedRescale(OpType opType, TensorUsage tensorUsage, DataType rescaleFromType,
        DataType rescaleToType, DataType opFromType, DataType opToType, const Quantization &quantization) = 0;
    virtual bool SupportsAccumulatorSaveRestore() = 0;
    virtual bool SupportsNegativeStrides() = 0;
    virtual bool SupportsElementwiseLeakyRelu(bool quantized, DataType type) = 0;
    virtual bool SupportsRescale(DataType fromType, DataType toType) = 0;
    virtual Flags<QueryResult> OperatorQuery(OpType opType, const ArchOperatorQuery *query = nullptr, ArchRequirements *req = nullptr) = 0;
    virtual bool SupportedZeroPoint(int64_t zp, TensorUsage usage, DataType dType, OpType opType) = 0;
};

inline void Set(ArchTensorRequirement &req, TensorUsage usage, TensorFormat format)
{
    req.usage = usage;
    req.format = format;
    req.next = nullptr;
}

inline void Set(ArchTensorRequirement &req, TensorUsage usage, DataType type, TensorFormat format)
{
    Set(req, usage, format);
    req.type = type;
}

inline void Set(ArchTensorRequirement &req, TensorUsage usage, DataType type, TensorFormat format, const Shape &shape)
{
    Set(req, usage, type, format);
    req.shape = shape;
}

inline const ArchTensorRequirement *Get(const ArchTensorRequirement *req, TensorUsage usage)
{
    while ( req->usage != usage && req->next != nullptr )
    {
        req = req->next;
    }
    return req;
}

}  // namespace regor
