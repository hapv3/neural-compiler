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

#include "architecture/architecture_constraints.hpp"
#include "compiler/operation.hpp"
#include "tflite_supported_operators.hpp"

namespace regor
{

class TfLiteSupportedOperatorsU85 : public TfLiteSupportedOperators
{
public:
    TfLiteSupportedOperatorsU85();

private:
    ConstraintCheck resizeACHPC;
    ConstraintCheck resizeScaleFactor;
    ConstraintCheck resizeBilinearPowerOfTwo;
    ConstraintCheck scatterShapeConst;
    ConstraintCheck scatterIndexConst;
    ConstraintCheck scatterIndexUnitChannel;
    ConstraintCheck scatterIndexUniqueElements;
    ConstraintCheck gatherAxis;
    ConstraintCheck ifmZeroPoints;
    ConstraintCheck ofmZeroPoints;
};
}  // namespace regor
