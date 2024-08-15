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

#pragma once

#include "common/logging.hpp"

#include "architecture/architecture.hpp"
#include "common/scaling.hpp"
#include "graph.hpp"
#include "graph_optimiser.hpp"
#include "op_type.hpp"
#include "operation.hpp"
#include "softmax.hpp"
#include "tensor.hpp"

#include <fixedpoint/fixedpoint.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <memory>
#include <vector>

namespace regor
{
// Sigmoid clamp (-8, 8)
static double ClampSigmoid8(double value)
{
    return ClampSigmoid(value, 8.0);
};

/// <summary>
/// TFLite Graph optimiser
/// </summary>
class TFLiteGraphOptimiser : public GraphOptimiser
{
    using OpRewriteFunction = Operation *(TFLiteGraphOptimiser::*)(Graph *, Operation *);
    using TensorRewriteFunction = Tensor *(TFLiteGraphOptimiser::*)(Graph *, Tensor *);
    using GraphOptStepArray = std::vector<RewriteFunctions<TFLiteGraphOptimiser>>;

private:
    std::unique_ptr<Softmax> _softmax;

    // utility functions

    // Multiplies int with QuantizedScale with rounding.
    int MultiplyByQuantizedMultiplier(int x, QuantizedScale quantScale);

    Operation *MakeMulWithConstTensor(const std::string &name, const TensorConnection &ifmConn,
        const TensorConnection &ofmConn, const std::shared_ptr<Tensor> &constTens, const Quantization &quantization);

    // Helper function for converting operations
    Operation *MakeOperation(OpType opType, const TensorConnection *ifm0Conn, const TensorConnection *ifm1Conn, const TensorConnection *ofmConn);

    // Converts 16-bit Leaky ReLU
    Operation *ConvertLeakyRelu16bit(TensorConnection &ifmConn, TensorConnection &ofmConn, Operation *operation);

    // Get axis parameter for operator
    int GetAxis(const Operation *const operation);
    // Calculate the read shape and offset values for Slice.
    void SetSliceOffsetValues(Operation *const operation, Shape &readShape, Shape &readOffset);
    // Calculate the read shape and offset values for StridedSlice.
    void SetStridedSliceOffsetValues(Operation *const operation, const TensorConnection *const ifmConn, Shape &readShape, Shape &readOffset);
    // Creates MemoryCopy operation for the given ifm/ofm and write offset.
    std::shared_ptr<Operation> MakeMemoryCopyForConcat(
        const TensorConnection *const ofmConn, const TensorConnection *const ifmConn, const Shape &writeOffset);
    // Creates a MemoryCopy operation for the given ifm/ofm and readOffset.
    std::shared_ptr<Operation> MakeMemoryCopyForSplitOps(const TensorConnection *const ofmConn,
        const TensorConnection *const ifmConn, const Shape &readShape, const Shape &readOffset);

    // Creates the desired shape of either:
    // - Concat         (Input shape - supply IFM base shape)
    // - Split/SplitV   (Output shape - supply OFM base shape)
    //
    // returns the Desired shape.
    // Also calculates the axis4D, returned through supplied pointer.
    Shape MakeConcatSplitDesiredShape(int axis, const Shape &baseShape, int *const axis4D);

    // Creates the desired shape of either:
    // - pack   (Input shape - supply IFM base shape)
    // - unpack (Output shape - supply OFM base shape)
    //
    // returns the Desired shape.
    // Unpack keeps the unpacked dimension set to 1.
    // Also calculates the axis4D, returned through supplied pointer.
    Shape MakePackUnpackDesiredShape(int axis, const Shape &baseShape, int *const axis4D);

    // Creates the desired Output shape of StridedSlice.
    //
    // returns the Desired shape.
    Shape MakeStridedSliceDesiredShape(Operation *const operation, const Shape &baseShape);

    Operation *MakeDepthwiseMeanOp(const TensorConnection *ifmConn, const Shape &ifmShape4D, const Shape &readShape,
        const Shape &readOffset, const Shape &ofmShape4D, int w, int h, const std::string &name, std::shared_ptr<Tensor> &weightTensor,
        std::shared_ptr<Tensor> biasTensor, const Quantization &ifmQuant, const Quantization &weightQuant, const Quantization &ofmQuant);

    // Converts op to int8/uint8 LUT which is generated with the given function.
    Operation *ConvertToLUT8(Operation *op, std::function<double(double)> func, const std::string &name);

    // Converts op to int16 interpolating LUT which is generated with the given function.
    Operation *ConvertToInterpolatingLUT16(Operation *op, std::function<double(double)> func, const std::string &name);

    // Converts int16 Tanh/Sigmoid to LUT16
    Operation *ConvertTanhSigmoidToLUT16(Operation *const op);


    // Rewrite functions
    Operation *ConvertExpToLUT(Graph *const graph, Operation *const operation);
    Operation *RewritePack(Graph *const graph, Operation *const operation);
    Operation *RewriteSplit(Graph *const graph, Operation *const operation);
    Operation *RemoveReshape(Graph *const graph, Operation *const operation);
    Operation *ExtractTransposePermutation(Graph *const graph, Operation *const operation);
    Operation *RemoveTranspose(Graph *const graph, Operation *const operation);
    Operation *RemoveReverse(Graph *const graph, Operation *const operation);
    Operation *ConvertGather(Graph *const graph, Operation *const operation);
    Operation *ConvertScatter(Graph *const graph, Operation *const operation);
    Operation *ConvertResize(Graph *const graph, Operation *const operation);
    Operation *ConvertArgMax(Graph *const graph, Operation *const operation);

    // RewriteBatchMatMul must be called before rewrite of transpose
    Operation *CreateTransposeForMatMul(const std::shared_ptr<Tensor> &ifm, const Shape &ofmShape);
    Operation *RewriteBatchMatMul(Graph *const, Operation *const operation);
    Operation *RewriteSpaceToBatchConvBatchToSpace(Graph *const, Operation *const operation);
    Operation *FixupDilationGT2(Graph *const, Operation *const operation);
    Operation *FixupBias(Graph *const, Operation *const operation);

    // Rewrite FullyConnect with dynamic weights to MatMul
    Operation *RewriteFullyConnectDynamic(Graph *const, Operation *const operation);

    Operation *CreateCastToInt32(const TensorConnection *ifmConn);
    Operation *RewriteSquaredDifference(Graph *const, Operation *const operation);

    // Convert depthwise convolutions with a depth multiplier greater than 1 into a single Conv2D if:
    // - the input depth is 1; and
    // - the output depth equals the depth multiplier.
    Operation *RewriteDepthwise(Graph *const, Operation *const operation);

    // Check that no reshape like operations remain in graph.
    Operation *CheckReshapeOpsRemoved(Graph *const graph, Operation *const operation);

    Operation *ConvertSoftmaxOps(Graph *const graph, Operation *const operation);

    // Unroll convolution if stride is too great
    Operation *UnrollConv(Graph *const, Operation *const operation);

    Operation *ConvertMeanOps(Graph *const, Operation *const operation);

    // Converts int8/uint8 Sigmoid and Tanh to a LUT based solution
    Operation *ConvertTanhSigmoidToLUT(Graph *const, Operation *const operation);

    // Convert PReLU to (ReLU + Minimum + Mul + Add)
    Operation *ConvertPrelu(Graph *const graph, Operation *const operation);

    // Converts Leaky ReLU when needed (LUT based solution or mul + max).
    Operation *ConvertLeakyRelu(Graph *const graph, Operation *const operation);

    // Converts HardSwish to a LUT based solution.
    Operation *ConvertHardSwishToLUT(Graph *const graph, Operation *const operation);

    // Converts 8-bit LeakyRelu to a LUT based solution.
    Operation *Convert8bitLeakyReluToLUT(Graph *const graph, Operation *const operation, float alpha);

    // Converts RSqrt to a LUT based solution.
    Operation *ConvertRSqrtToLUT(Graph *const graph, Operation *const operation);

    // Based on explicit padding provided in a PAD operation, returns adjusted value for
    // padAfter that provides equivalent results when used with explicit padding
    int CalcPadAfter(int inputSize, int stride, int filterSize, int padBefore, int padAfter);

    // Tries to completely remove a PAD operator by using explicit padding.
    // E.g. a PAD operation that pads 1, followed by a CONV with VALID padding and kernel size 3
    // is rewritten such that the PAD is removed, and the CONV uses explicit padding.
    // Converts tens1 -> PAD -> tens2 -> CONV to tens1 -> CONV
    // This is the most efficient way to implement PAD, but cannot be done for all pad sizes.
    Operation *ReplacePadByExplicitPadding(Graph *const graph, Operation *const operation);

    void MakeMemoryCopyForPad(const char *name, const Operation *operation, TensorConnection *ofmConn,
        const Shape &shape, const Shape &offset);

    // Rewrites PAD operator to a MemoryCopy that copies the IFM to the OFM
    // + up to 4 MemoryCopy operators that fill the OFM with zeros at the borders.
    // This is done as fall-back for the PAD operators that remain after ReplacePadByExplicitPadding
    Operation *ConvertPad(Graph *const graph, Operation *const operation);

public:
    // The graph optimisation steps.
    // Order matters, array of rewrites processed in order.
    // clang-format off
    const GraphOptStepArray _graphOptimisationSteps =
    {{
        {
            {
#if LOG_TRACE1_ON
                &GraphOptimiser::VisitTensorLog
#endif
            },
            {
#if LOG_TRACE1_ON
                &GraphOptimiser::VisitOperatorLog,
#endif
            }
        },
        {
            {},
            {
                &TFLiteGraphOptimiser::RewritePack
            }
        },
        {
            {},
            {
                &TFLiteGraphOptimiser::RewriteSplit
            }
        },
        {
            {},
            {
                &TFLiteGraphOptimiser::RewriteBatchMatMul,
                &TFLiteGraphOptimiser::RewriteFullyConnectDynamic
            }
        },
        {
            {},
            {
                &TFLiteGraphOptimiser::RemoveReshape,
                &TFLiteGraphOptimiser::RemoveTranspose,
                &TFLiteGraphOptimiser::RemoveReverse,
            }
        },
        {
            {},
            {
                &TFLiteGraphOptimiser::RewriteSpaceToBatchConvBatchToSpace,
                &TFLiteGraphOptimiser::FixupDilationGT2,
                &TFLiteGraphOptimiser::FixupBias,
                &TFLiteGraphOptimiser::RewriteDepthwise,
                &TFLiteGraphOptimiser::ConvertArgMax,
                &TFLiteGraphOptimiser::ConvertExpToLUT,
                &TFLiteGraphOptimiser::ConvertTanhSigmoidToLUT,
                &TFLiteGraphOptimiser::ConvertSoftmaxOps,
                &TFLiteGraphOptimiser::ReplacePadByExplicitPadding,
                &TFLiteGraphOptimiser::ConvertMeanOps,
                &TFLiteGraphOptimiser::ConvertPrelu,
                &TFLiteGraphOptimiser::ConvertLeakyRelu,
                &TFLiteGraphOptimiser::ConvertHardSwishToLUT,
                &TFLiteGraphOptimiser::ConvertRSqrtToLUT,
                &TFLiteGraphOptimiser::ConvertGather,
                &TFLiteGraphOptimiser::RewriteSquaredDifference,
                &TFLiteGraphOptimiser::ConvertScatter,
                &TFLiteGraphOptimiser::ConvertResize,
                &TFLiteGraphOptimiser::UnrollConv,
            }
        },
        {
            {},
            {
                &TFLiteGraphOptimiser::ConvertPad,
            }
        },
        {
            {
#if LOG_TRACE1_ON
                &GraphOptimiser::VisitTensorLog
#endif
            },
            {
                &TFLiteGraphOptimiser::CheckReshapeOpsRemoved,
#if LOG_TRACE1_ON
                &GraphOptimiser::VisitOperatorLog,
#endif
                &GraphOptimiser::RecordOptimisation
            }
        }
    }};
    // clang-format on

    explicit TFLiteGraphOptimiser(IArchitectureConstraints *constraints, const GraphOptimiserOptions &options, OptimiserDatabase *db);

    const GraphOptStepArray &GraphOptimisationSteps() const { return _graphOptimisationSteps; }

    void OptimiseGraph(Graph *graph);
};

}  // namespace regor
