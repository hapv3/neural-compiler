//
// SPDX-FileCopyrightText: Copyright 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
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

#include "architecture/architecture.hpp"
#include "architecture/architecture_constraints.hpp"
#include "compiler/graph.hpp"
#include "compiler/operation.hpp"
#include "tflite_mapping.hpp"

namespace regor
{

class TfLiteSupportedOperators
{
    using OperatorCheck = bool (TfLiteSupportedOperators::*)(const Operation *);

protected:
    std::vector<OperatorCheck> _genericChecks;
    IArchitectureConstraints *_archConstraints;
    std::unordered_set<OpType> _supportedOpTypes;
    std::unordered_set<DataType> _supportedDataTypes;
    int64_t _maxWeightSum8Bit;
    int64_t _maxWeightSum16Bit;
    int64_t _maxBias;

public:
    TfLiteSupportedOperators(IArchitectureConstraints *constraints);
    virtual ~TfLiteSupportedOperators() = default;
    virtual bool Check(const Operation *) = 0;

protected:
    static void Failure(const Operation *op, const std::string &message = "", const std::string &constraint = "");

private:
    bool ConstraintOpType(const Operation *op);
    bool ConstraintTensDtypes(const Operation *op);
    bool ConstraintNumSplits(const Operation *op);
    bool ConstraintMustHaveIFM(const Operation *op);
    bool ConstraintMustHaveOFM(const Operation *op);
    bool ConstraintTensMustHaveShape(const Operation *op);
    bool ConstraintTensQuantized(const Operation *op);
    bool ConstraintFCWeightShape(const Operation *op);
    bool ConstraintPerAxisQuant(const Operation *op);
    bool ConstraintMatchingQuantization(const Operation *op);
    bool ConstraintZeroPoints(const Operation *op);
    bool ConstraintDepthMultiplier(const Operation *op);
    bool ConstraintWeightsPrecision(const Operation *op);
    bool ConstraintWeightSum(const Operation *op);
    bool ConstraintBias(const Operation *op);
    bool ConstraintAvgPool(const Operation *op);
    bool ConstraintMaxPool(const Operation *op);
    bool ConstraintTCStrides(const Operation *op);
    bool ConstraintTCShapes(const Operation *op);
    bool ConstraintRsqrt(const Operation *op);
    bool ConstraintConstParams(const Operation *op);
    bool ConstraintMean(const Operation *op);
    bool ConstraintSoftmax(const Operation *op);
    bool ConstraintPad(const Operation *op);
    bool ConstraintTransposeDims(const Operation *op);
    bool ConstraintStridedSlice(const Operation *op);
    bool ConstraintLog(const Operation *op);
    bool ConstraintLSTM(const Operation *op);
};

// Factory for supported-ops checkers
std::unique_ptr<TfLiteSupportedOperators> MakeSupportedOpsChecker(const std::string &target, IArchitectureConstraints *constraints);

}  // namespace regor
