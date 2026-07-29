//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "architecture/architecture_constraints.hpp"
#include "architecture/neuralai/neural_ai_op_config.hpp"

#include <string>

namespace regor
{

class NeuralAIConstraints final : public IArchitectureConstraints
{
public:
    struct Classification
    {
        NeuralAIOpMode mode = NeuralAIOpMode::Unsupported;
        bool directNhwcInput = false;
        bool groupStationary = false;
        bool hasIcTail = false;
        bool hasOcTail = false;
        std::string diagnostic;

        explicit operator bool() const { return mode != NeuralAIOpMode::Unsupported; }
    };

    static Classification Classify(OpType opType, const Shape &ifmShape, const Shape &weightShape,
        const Shape &ofmShape, const Kernel *kernel);

    bool SupportsQuantization(OpType opType, const Quantization &ifmQuant, DataType ifmType,
        const Quantization &ifm2Quant, DataType ifm2Type, const Quantization &ofmQuant,
        DataType ofmType) override;
    bool SupportsQuantization(OpType opType, const Quantization &ifmQuant, DataType ifmType,
        const Quantization &ofmQuant, DataType ofmType) override;
    bool SupportsAccumulatorSaveRestore() override { return true; }
    bool SupportsNegativeStrides() override { return false; }
    bool SupportsElementwiseLeakyRelu(bool, DataType) override { return false; }
    bool SupportsRescale(DataType, DataType) override { return false; }
    bool SupportsDoubleBroadcast() override { return false; }
    bool SupportsFusedActivationClamping() const override { return true; }
    bool CanAliasReshape(const Shape &ifmShape, const Shape &ofmShape) const override
    {
        return ifmShape.Elements64() == ofmShape.Elements64() &&
               ifmShape.Depth() == ofmShape.Depth();
    }
    Flags<QueryResult> OperatorQuery(OpType opType, const ArchOperatorQuery *query = nullptr,
        ArchRequirements *req = nullptr) override;
    bool SupportedZeroPoint(int64_t zeroPoint, TensorUsage usage, DataType dataType, OpType opType) override;

private:
    static bool IsSupportedOp(OpType opType);
};

}  // namespace regor
