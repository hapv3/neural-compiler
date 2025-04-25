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

#include "graph_optimiser.hpp"
#include "operation.hpp"

namespace regor
{

/// <summary>
/// TFLite Graph optimiser LSTM rewriter
/// </summary>
class LSTM
{
private:
    Operation *_lstmOp = nullptr;
    OptimiserDatabase *_db = nullptr;
    Graph *_graph = nullptr;

    // Dimensions
    int _nFeature, _nTime, _nBatch;
    // Attributes
    int _cellClip;
    bool _isTimeMajor;
    // Input/Output
    TensorConnection *_ifmConn = nullptr;
    TensorConnection *_ofmConn = nullptr;

public:
    LSTM(Operation *operation, OptimiserDatabase *db, Graph *graph);
    Operation *ConvertOp();

private:
    void RecordOptimisation(Operation *op);
    TensorConnection *ExtractFeatureSlice(int time, int batch);
    TensorConnection *GetInitialState(TensorUsage stateUsage, int batch);
    void SetStateRead(Operation *op, int batch);
    void SetStateWrite(Operation *op, TensorUsage stateUsage, int batch);
    Operation *SetOutputWrite(TensorConnection *stateConn, int time, int batch);
    TensorConnection *CalculateGate(const std::string &name, TensorConnection *featureConn, TensorConnection *stateConn,
        TensorConnection *inputWeightConn, TensorConnection *inputBiasConn, TensorConnection *recurrentWeightConn,
        OpType activationType, int batch);
    TensorConnection *CalculateCellState(TensorConnection *cellStateConn, TensorConnection *inputGateConn,
        TensorConnection *forgetGateConn, TensorConnection *cellGateConn, int time, int batch);
    TensorConnection *CalculateOutputState(TensorConnection *outputGateConn, TensorConnection *cellStateConn, int time, int batch);
    std::pair<TensorConnection *, TensorConnection *> Step(TensorConnection *featureConn,
        TensorConnection *outputStateConn, TensorConnection *cellStateConn, int time, int batch);
};

}  // namespace regor
