//
// SPDX-FileCopyrightText: Copyright 2021-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
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

#include "common/scaling.hpp"
#include "graph_optimiser.hpp"
#include "operation.hpp"
#include "operation_util.hpp"

#include <fixedpoint/fixedpoint.h>
#include <algorithm>
#include <vector>

namespace regor
{

/// <summary>
/// TFLite Graph optimiser Softmax rewriter
/// </summary>
class Softmax
{
private:
    OptimiserDatabase *_db = nullptr;
    IArchitectureConstraints *_constraints = nullptr;
    double _negExpRange;

public:
    Softmax(OptimiserDatabase *db, IArchitectureConstraints *constraints, double int16NegExpRange = 10.0);
    Operation *ConvertOp(Operation *const operation);

private:
    void RecordOptimisation(Operation *const operation, Operation *op);
    Operation *GetGraph8Bit(Operation *const operation, TensorConnection *ifmConn, TensorConnection *ofmConn);
    Operation *GetGraphInt16(Operation *const operation, TensorConnection *ifmConn, TensorConnection *ofmConn);
    std::vector<int32_t> GenerateExpTable(double beta, double inputScale);
    Operation *CreateTransposeMaxpool(Operation *const operation, TensorConnection *ifmConn, const Shape &ifmShape,
        const Shape &transposePerm, const Shape &transposeOFMShape, const Shape &maxPoolOFMStorageShape, const Quantization &noScaleQuant);
};

}  // namespace regor
