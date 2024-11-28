//
// SPDX-FileCopyrightText: Copyright 2022-2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
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

#include "graphapi_tosa_types.hpp"

#include <stddef.h>
#include <cstdint>

namespace GraphApi
{

static constexpr int MAX_TENSOR_DIMS = 6;  // Covers 3D padding

/// <summary>
/// Classification for how a Tensor is consumed by an operator
/// </summary>
enum class GraphTensorUsage : uint32_t
{
    None = 0,
    IFM = 0x01,
    OFM = 0x02,
    Weights = 0x03,
    Scales = 0x04,
    Params = 0x05,
    LUT = 0x06,
    State = 0x07,
    UserDefined = 0x1E,
    Last,
    TypeMask = 0x1F,
    IndexShift = 8,
    IndexMask = 0xFFFFF00,
    IFM0 = IFM,
    IFM1 = 0x0100 | IFM,
    IFM2 = 0x0200 | IFM,
    Params0 = Params,
    Params1 = 0x100 | Params,
    Params2 = 0x200 | Params,
};

constexpr inline GraphTensorUsage MakeTensorUsage(GraphTensorUsage type, int index)
{
    return GraphTensorUsage(uint32_t(type) | uint32_t(index << int(GraphTensorUsage::IndexShift)));
}

/// <summary>
/// Tensor axis ordering
/// </summary>
enum class AxisOrder : uint16_t
{
    Unknown = 0,
    OHWI = 1,
    IHWO = 2,
    OI = 3,
    HWCM = 4,
};

/// <summary>
/// Tensor data types
/// </summary>
enum class GraphDataType : uint16_t
{
    Bool8 = 1,
    Int4Packed8,
    Int8,
    Int16,
    Int32,
    Int48,
    Int64,
    UInt8,
    UInt16,
    UInt32,
    UInt48,
    UInt64,
    BFloat16,
    Float16,
    Float32,
};

enum class GraphTensorLayout : uint8_t
{
    Linear = 0,
    NHCWB16 = 1,
};

/// <summary>
/// Graph base axis shape
/// </summary>
struct GraphShape
{
    int32_t count;
    int32_t axisNHWC[MAX_TENSOR_DIMS];
};

/// <summary>
/// Graph buffer Handle
/// </summary>
struct GraphBuffer
{
};

/// <summary>
/// Graph tensor handle
/// </summary>
struct GraphTensor
{
    virtual ~GraphTensor() = default;
};

/// <summary>
/// Graph kernel definition
/// </summary>
struct GraphKernel
{
    int32_t sizeYXZ[3] = {0};
    int32_t strideYXZ[3] = {0};
    int32_t dilationYXZ[3] = {0};
    int32_t paddingTBLRNF[6] = {0};  // Top,Bottom,Left,Right,Near,Far
    double padValue = 0;
};

/// <summary>
/// Fraction of numerator and denominator
/// </summary>
struct FractionND
{
    int16_t n;
    int16_t d;
};

/// <summary>
/// 2 Dimensional X/Y point
/// </summary>
struct Point2
{
    int x, y;
};

#include "graphapi_attr.hpp"

/// <summary>
/// Graph operator
/// </summary>
struct GraphOperation
{
    virtual ~GraphOperation() = default;
};

/// <summary>
/// Buffer mapping strategy
/// </summary>
enum class BufferMapping
{
    Allocate = 0,
    Alias = 1,
};

// Freeform syntax versioning
static constexpr uint32_t VERSION_TOSA_0_60 = 0x00003C00;
static constexpr uint32_t VERSION_TOSA_0_80 = 0x00005000;
static constexpr uint32_t VERSION_TOSA_1_00 = 0x01000000;
static constexpr int32_t PROFILE_BASELINE = 0;
static constexpr int32_t PROFILE_MAIN = 1;

/// <summary>
/// Interface to a graph builder.
/// </summary>
struct IGraphBuilder
{
    virtual ~IGraphBuilder() = default;
    virtual bool RequireSyntaxVersion(uint32_t version, int32_t level) = 0;
    // Object factories
    virtual GraphOperation *CreateOp(tosa::Op opType, const GraphKernel *kernel) = 0;
    virtual GraphBuffer *CreateBuffer(size_t sizeBytes, BufferMapping mapping, const void *initialData) = 0;
    virtual GraphTensor *CreateTensor(const char *name, const GraphShape &shape, GraphTensorLayout layout,
        GraphDataType dataType, GraphBuffer *buffer = nullptr) = 0;
    // Set graph inputs/outputs
    virtual void AddInput(GraphTensor *graphTensor) = 0;
    virtual void AddOutput(GraphTensor *graphTensor) = 0;
    virtual void AddPersistent(GraphTensor *graphTensor) = 0;
    // Connect operator inputs/outputs
    virtual void AddInput(GraphOperation *graphOp, GraphTensorUsage usage, GraphTensor *graphTensor) = 0;
    virtual void AddOutput(GraphOperation *graphOp, GraphTensorUsage usage, GraphTensor *graphTensor) = 0;
    // Object attribute and properties
    virtual bool Set(GraphOperation *op, OpAttr attr, bool value) = 0;
    virtual bool Set(GraphOperation *op, OpAttr attr, int32_t value) = 0;
    virtual bool Set(GraphOperation *op, OpAttr attr, double value) = 0;
    virtual bool Set(GraphOperation *op, OpAttr attr, const GraphShape &value) = 0;
    virtual bool Set(GraphOperation *op, OpAttr attr, const FractionND &value) = 0;
    virtual bool Set(GraphOperation *op, OpAttr attr, const Point2 &value) = 0;
    virtual bool Set(GraphOperation *op, OpAttr attr, const char *value) = 0;
    virtual void SetZeroPoint(GraphOperation *op, GraphTensorUsage tensor, double zeroPoint) = 0;
    virtual void SetAxisOrder(GraphTensor *tensor, AxisOrder order) = 0;
    virtual void SetAxisStrides(GraphTensor *tensor, const GraphShape *axisStrides) = 0;
};

}  // namespace GraphApi
