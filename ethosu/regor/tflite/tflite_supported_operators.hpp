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

#include "common/ordered_map.hpp"
#include "compiler/graph.hpp"
#include "compiler/operation.hpp"
#include "tflite_mapping.hpp"

#include <functional>
#include <set>

namespace regor
{

class TfLiteSupportedOperators
{

public:
    virtual ~TfLiteSupportedOperators() = default;
    // Performs constraint-check for a given operation
    bool Check(const Operation *);
    // Maps opname to list of constraints
    ordered_map<OpType, std::vector<std::string>> Documentation();

protected:
    TfLiteSupportedOperators(int64_t maxWeightSum8Bit, int64_t maxWeightSum16Bit, int64_t maxBias,
        const std::set<DataType> &supportedDataTypes, const std::set<OpType> &supportedOpTypes);

    // protected subclass for constraintChecks
    // A ConstraintCheck object is represented by a check-function and a documentation string
    // The documentation string is emitted at failure, and can be used to generate documentation
    class ConstraintCheck
    {
    public:
        ConstraintCheck(std::function<bool(const Operation *)> checkFunc, const std::string &docString) :
                Check(std::move(checkFunc)), _documentation(docString){};
        ConstraintCheck(){};
        std::function<bool(const Operation *)> Check;

        std::string Documentation() { return _documentation; }

    private:
        std::string _documentation;
    };

    // Op-specific constraint map
    ordered_map<OpType, std::vector<ConstraintCheck *>> opConstraints;
    ConstraintCheck mustHaveIFM;
    ConstraintCheck mustHaveOFM;
    ConstraintCheck tensMustHaveShape;
    ConstraintCheck tensDimMustBeStatic;
    ConstraintCheck supportedDTypes;
    ConstraintCheck tensQuantized;
    ConstraintCheck quantizationScaleShiftPositive;
    ConstraintCheck fcWeightShape;
    ConstraintCheck perAxisQuant;
    ConstraintCheck matchingQuantization;
    ConstraintCheck weightsPrecision;
    ConstraintCheck weightSum;
    ConstraintCheck biasShape;
    ConstraintCheck biasConstant;
    ConstraintCheck biasPrecision;
    ConstraintCheck bias64BitRange;
    ConstraintCheck avgPoolPad;
    ConstraintCheck maxPool;
    ConstraintCheck transposeConvShape;
    ConstraintCheck rsqrtIFMPrecision;
    ConstraintCheck constParams;
    ConstraintCheck softmaxOverflow;
    ConstraintCheck padParams;
    ConstraintCheck transposeDims;
    ConstraintCheck logShapes;
    ConstraintCheck logPrecision;
    ConstraintCheck unitBatch;
    ConstraintCheck meanDepth;
    ConstraintCheck meanAxisSize;
    ConstraintCheck meanTotalElements;
    ConstraintCheck meanDataType;
    ConstraintCheck stridedSlice;
    ConstraintCheck lstmImplicitGateCalc;
    ConstraintCheck lstmPeephole;
    ConstraintCheck lstmProjection;
    ConstraintCheck lstmGateNorm;
    ConstraintCheck splitsMatchOutputs;
};

void Failure(const Operation *op, const std::string &extra = "");
// Factory for supported-ops checkers
std::unique_ptr<TfLiteSupportedOperators> MakeSupportedOpsChecker(const std::string &target);

}  // namespace regor
