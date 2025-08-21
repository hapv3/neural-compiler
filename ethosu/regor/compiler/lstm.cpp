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

#include "lstm.hpp"

#include "operation_util.hpp"
#include "quantization.hpp"

namespace regor
{

static constexpr double Q0_15_SCALE = 1.0 / (1 << 15);
static constexpr double Q3_12_SCALE = 1.0 / (1 << 12);

LSTM::LSTM(Operation *operation, OptimiserDatabase *db, Graph *graph) : _lstmOp(operation), _db(db), _graph(graph)
{
    assert(_lstmOp->Type() == OpType::UnidirectionalSequenceLstm);

    // Attributes
    assert(operation->HasAttribute<unidirectional_sequence_lstm_attr_t>());
    auto *attr = operation->Attribute<unidirectional_sequence_lstm_attr_t>();
    _isTimeMajor = attr->time_major;
    _cellClip = attr->cell_clip;

    // Input/Output
    _ifmConn = _lstmOp->Input(TensorUsage::IFM);
    _ofmConn = _lstmOp->Output(TensorUsage::OFM);

    // Input dimensions
    const auto &ifmShape = _ifmConn->shape;
    _nFeature = ifmShape[-1];
    _nTime = ifmShape[_isTimeMajor ? 0 : 1];
    _nBatch = ifmShape[_isTimeMajor ? 1 : 0];
}

void LSTM::RecordOptimisation(Operation *op)
{
    if ( _db )
    {
        _db->AddOptimised(*_lstmOp, op);
    }
}

Operation *LSTM::ConvertOp()
{
    Operation *returnOp = _lstmOp;
    int numBatches = _isTimeMajor ? 1 : _nBatch;
    for ( int batch = 0; batch < numBatches; batch++ )
    {
        TensorConnection *outputState = GetInitialState(TensorUsage::State, batch);
        TensorConnection *cellState = GetInitialState(MakeTensorUsage(TensorUsage::State, 1), batch);
        for ( int time = 0; time < _nTime; time++ )
        {
            TensorConnection *feature = ExtractFeatureSlice(time, batch);
            assert(feature);
            std::tie(outputState, cellState) = Step(feature, outputState, cellState, time, batch);
            returnOp = SetOutputWrite(outputState, time, batch);
        }
    }

    if ( returnOp != _lstmOp )
    {
        _lstmOp->Disconnect();
    }
    return returnOp;
}

// Extract time and batch slice of the input tensor.
TensorConnection *LSTM::ExtractFeatureSlice(int time, int batch)
{
    std::shared_ptr<Tensor> featureTensor = _ifmConn->tensor->Clone();
    featureTensor->SetName(fmt::format("{0}_feauture_b{1}.t{2}", featureTensor->Name(), batch, time));
    featureTensor->SetStorageShape({(_isTimeMajor ? _nBatch : 1), _nFeature});
    auto op = std::make_shared<Operation>(OpType::Slice);

    auto readShape = featureTensor->StorageShape();
    auto readOffset = _isTimeMajor ? Shape(time, 0, 0) : Shape(batch, time, 0);
    auto *attr = op->Attribute<slice_attr_t>();
    attr->size = readShape;
    attr->begin = readOffset;

    op->CopyInput(TensorUsage::IFM, *_ifmConn);
    op->ConnectOutput(TensorUsage::OFM, featureTensor).Set(_ifmConn->quantization);
    RecordOptimisation(op.get());
    return op->Output(TensorUsage::OFM);
}

// Get state tensor for provided state type and batch
TensorConnection *LSTM::GetInitialState(TensorUsage stateUsage, int batch)
{
    TensorConnection *stateConn = _lstmOp->Input(stateUsage);
    if ( _isTimeMajor )
    {
        // For time major, return the state tensor directly since all
        // batches are calculated in the same step.
        return stateConn;
    }
    else
    {
        // For batch major, return one batch slice of the state tensor.
        // The tensor has to be cloned in order to resolve graph dependencies correctly but
        // the clone will share the underlying buffer with the original state tensor which
        // ensure they are allocated to the same address.
        std::shared_ptr<Tensor> newStateTensor = stateConn->tensor->Clone();

        // Set read/write shape to be one batch and read/write offset to the current batch.
        TensorSlice slice({0, 0, batch, 0}, {1, 1, 1, stateConn->shape[-1]});
        const auto &stateQuant = stateConn->quantization;

        auto op = std::make_shared<Operation>(OpType::MemoryCopy);
        op->ConnectInput(TensorUsage::IFM, stateConn->tensor).Set(slice).Set(stateQuant);
        op->ConnectOutput(TensorUsage::OFM, newStateTensor).Set(slice).Set(stateQuant);

        // Mark the cloned tensor as persistent to require linear format and avoid fusing with
        // other tensors.
        _graph->AddPersistent(newStateTensor);
        RecordOptimisation(op.get());
        return op->Output(TensorUsage::OFM);
    }
}

// Setup the correct read shape and offset for reading from a state tensor.
void LSTM::SetStateRead(Operation *op, int batch)
{
    if ( !_isTimeMajor && _nBatch > 1 )
    {
        Shape cellStateShape = _lstmOp->Input(MakeTensorUsage(TensorUsage::State, 1))->shape;
        const Shape ifmShape = op->Input(TensorUsage::IFM)->shape;
        op->Input(TensorUsage::IFM)->Set(cellStateShape).Set({{0, 0, batch, 0}, {1, 1, 1, ifmShape[-1]}});
    }
}

// Write the state for the provided batch by pointing the operations ofm to the state tensor.
void LSTM::SetStateWrite(Operation *op, TensorUsage stateUsage, int batch)
{
    TensorConnection *stateConn = _lstmOp->Input(stateUsage);

    auto ofmConn = op->Output(TensorUsage::OFM);
    auto ofmShape = Shape::PadAxes(ofmConn->shape, 4, 1);

    std::shared_ptr<Tensor> newStateTensor = stateConn->tensor->Clone();
    op->ConnectOutput(TensorUsage::OFM, newStateTensor).Set(stateConn->shape).Set(stateConn->quantization);

    if ( !_isTimeMajor && _nBatch > 1 )
    {
        auto writeOffset = Shape(0, 0, batch, 0);
        ofmConn->Set({writeOffset, ofmShape});
    }

    // Mark the cloned tensor as persistent to require linear format and avoid fusing with
    // other tensors.
    _graph->AddPersistent(newStateTensor);
}

// Copy the output state to the time/batch slice of the final output.
Operation *LSTM::SetOutputWrite(TensorConnection *stateConn, int time, int batch)
{
    auto concatOp = std::make_shared<Operation>(OpType::MemoryCopy);

    auto concatIfmConn = &concatOp->ConnectInput(TensorUsage::IFM, stateConn->tensor).Set(stateConn->shape).Set(_ofmConn->quantization);

    if ( !_isTimeMajor && _nBatch > 1 )
    {
        Shape readOffset(0, 0, batch, 0);
        Shape readSize(1, 1, 1, stateConn->shape[-1]);
        concatIfmConn->Set({readOffset, readSize});
    }

    Shape writeOffset = _isTimeMajor ? Shape(0, time, 0, 0) : Shape(0, batch, time, 0);
    Shape writeShape = _isTimeMajor ? Shape(1, 1, stateConn->shape[-2], stateConn->shape[-1]) : Shape(1, 1, 1, stateConn->shape[-1]);
    concatOp->ConnectOutput(TensorUsage::OFM, _ofmConn->tensor)
        .Set(_ofmConn->shape)
        .Set(_ofmConn->quantization)
        .Set({writeOffset, writeShape})
        .Set(RoundMode::NATURAL);

    RecordOptimisation(concatOp.get());
    return concatOp.get();
}

// Generate a gate for the provided input and weights
// Activation( Add( FullyConnected(input feature), FullyConnected(output state) ) )
TensorConnection *LSTM::CalculateGate(const std::string &name, TensorConnection *featureConn, TensorConnection *stateConn,
    TensorConnection *inputWeightConn, TensorConnection *inputBiasConn, TensorConnection *recurrentWeightConn, OpType activationType, int batch)
{
    // Setup fullyconnected output quantization
    Quantization fcQuant;
    fcQuant.type = QuantizationType::TFLITE;
    fcQuant.scales = {Q3_12_SCALE};
    fcQuant.zeroPoints = {0};

    Operation *inputFC = CreateFullyConnected(fmt::format("{0}_feature_fc", name), featureConn->tensor,
        inputWeightConn->tensor, featureConn->quantization, inputWeightConn->quantization, fcQuant,
        featureConn->SliceShape(), DataType::Int16, inputBiasConn->tensor, inputBiasConn->quantization);
    TensorConnection *inputFCOfmConn = inputFC->Output(TensorUsage::OFM);

    Operation *recurrentFC = CreateFullyConnected(fmt::format("{0}_recurrent_fc", name), stateConn->tensor,
        recurrentWeightConn->tensor, stateConn->quantization, recurrentWeightConn->quantization, fcQuant,
        stateConn->SliceShape(), DataType::Int16);
    SetStateRead(recurrentFC, batch);
    TensorConnection *recurrentFCOfmConn = recurrentFC->Output(TensorUsage::OFM);

    Quantization addQuant;
    addQuant.type = QuantizationType::TFLITE;
    addQuant.scales = {1.0f};
    addQuant.zeroPoints = {0};
    Operation *add = CreateAdd(inputFCOfmConn->tensor, recurrentFCOfmConn->tensor, inputFCOfmConn->quantization,
        recurrentFCOfmConn->quantization, addQuant);

    // Create activation function
    Quantization activationQuant;
    activationQuant.type = QuantizationType::TFLITE;
    activationQuant.scales = {1.0f};
    activationQuant.zeroPoints = {0};

    auto activation = std::make_shared<Operation>(activationType);
    auto addOfmTensor = add->Output(TensorUsage::OFM)->tensor;

    activation->ConnectInput(TensorUsage::IFM, addOfmTensor).Set(addQuant);
    activation->ConnectOutput(TensorUsage::OFM, addOfmTensor->Clone()).Set(activationQuant);

    auto returnConn = activation->Output(TensorUsage::OFM);
    if ( activationType == OpType::Sigmoid )
    {
        // For Sigmoid we need to set the activation min/max values to match the possible range
        // in the reference. The values below are the quantized min/max values that the reference
        // can achive for the LUT based Sigmoid/Logistic. (The NPU does however have a larger range
        // due to intermediate higher precision.)
        auto clamp = std::make_shared<Operation>(OpType::Clamp);
        auto *attr = clamp->Attribute<clamp_attr_t>();
        attr->max = Quantize(32757.0f, activationQuant);
        attr->min = Quantize(11.0f, activationQuant);

        // Copying the input and output of the Add means the Clamp will also write to the cell state.
        clamp->CopyInput(TensorUsage::IFM, *returnConn);
        clamp->CopyOutput(TensorUsage::OFM, *returnConn);

        RecordOptimisation(clamp.get());
        returnConn = clamp->Output(TensorUsage::OFM);
    }

    RecordOptimisation(inputFC);
    RecordOptimisation(recurrentFC);
    RecordOptimisation(add);
    RecordOptimisation(activation.get());
    return returnConn;
}

// Calculate and update the cell state from the provided gates
// Clip( Add( Mul( cell state, forget gate ), Mul( cell gate, input gate ) ) )
TensorConnection *LSTM::CalculateCellState(TensorConnection *cellStateConn, TensorConnection *inputGateConn,
    TensorConnection *forgetGateConn, TensorConnection *cellGateConn, int time, int batch)
{
    const Quantization &cellStateQuant = cellStateConn->quantization;
    double cellStateScale = cellStateQuant.scales[0].Dequantize();
    // Calculate explicit scales based on the cell state quantization
    Quantization mulCFQuant;
    mulCFQuant.type = QuantizationType::TFLITE;
    mulCFQuant.scales = {ElementwiseMulScale(cellStateScale, Q0_15_SCALE, cellStateScale)};
    mulCFQuant.zeroPoints = {cellStateConn->quantization.zeroPoints[0]};
    // Create Mul(cell_state, forget_gate)
    Operation *mulCF = CreateMul(cellStateConn->tensor, forgetGateConn->tensor, mulCFQuant, mulCFQuant, mulCFQuant,
        DataType::None, &forgetGateConn->shape, &forgetGateConn->shape);
    SetStateRead(mulCF, batch);

    // Calculate explicit scales based on the cell state quantization
    Quantization mulCIQuant;
    mulCIQuant.type = QuantizationType::TFLITE;
    mulCIQuant.scales = {ElementwiseMulScale(Q0_15_SCALE, Q0_15_SCALE, cellStateScale)};
    mulCIQuant.zeroPoints = {cellStateConn->quantization.zeroPoints[0]};
    // Create Mul(cell_gate, input_gate)
    Operation *mulCI = CreateMul(cellGateConn->tensor, inputGateConn->tensor, mulCIQuant, mulCIQuant, mulCIQuant);

    // Create Add with cell state quantization
    Operation *add = CreateAdd(mulCF->Output(TensorUsage::OFM)->tensor, mulCI->Output(TensorUsage::OFM)->tensor,
        cellStateQuant, cellStateQuant, cellStateQuant);
    // Redirect the ofm of Add to cell state.
    SetStateWrite(add, MakeTensorUsage(TensorUsage::State, 1), batch);

    RecordOptimisation(mulCF);
    RecordOptimisation(mulCI);
    RecordOptimisation(add);

    TensorConnection *returnConn = add->Output(TensorUsage::OFM);
    if ( _cellClip != 0 )
    {
        // If the cell clip attribute is non-zero the output needs to be clamped.
        auto clamp = std::make_shared<Operation>(OpType::Clamp);
        auto *attr = clamp->Attribute<clamp_attr_t>();
        attr->max = Quantize(static_cast<float>(_cellClip), cellStateQuant);
        attr->min = Quantize(static_cast<float>(-_cellClip), cellStateQuant);

        // Copying the input and output of the Add means the Clamp will also write to the cell state.
        clamp->CopyInput(TensorUsage::IFM, *returnConn);
        clamp->CopyOutput(TensorUsage::OFM, *returnConn);

        RecordOptimisation(clamp.get());
        returnConn = clamp->Output(TensorUsage::OFM);
    }

    return returnConn;
}

// Calculate and update the output state from the provided gate output
// Mul( Tanh(cell state), output gate )
TensorConnection *LSTM::CalculateOutputState(TensorConnection *outputGateConn, TensorConnection *cellStateConn, int time, int batch)
{
    // Setup tanh quantization
    Quantization tanhQuant;
    tanhQuant.type = QuantizationType::TFLITE;
    tanhQuant.scales = {QuantizedScale(Q0_15_SCALE)};
    tanhQuant.zeroPoints = {0};

    // Create tanh(cell state)
    auto tanh = std::make_shared<Operation>(OpType::Tanh);
    tanh->ConnectInput(TensorUsage::IFM, cellStateConn->tensor).Set(cellStateConn->shape).Set(cellStateConn->quantization);

    // Tanh reads from the cell state. This may set an ifm slice which the ofm shape needs to honor.
    SetStateRead(tanh.get(), batch);
    auto tanhIfmConn = tanh->Input(TensorUsage::IFM);
    Shape tanhOfmShape = tanhIfmConn->SliceShape();
    // Create a new tensor for ofm instead of cloning, this ensures that the tanh output will not
    // overwrite the cell state.
    auto ofmName = fmt::format("{0}_tanh_b{1}.t{2}", cellStateConn->tensor->Name(), batch, time);
    auto tanhOfm = std::make_shared<Tensor>(ofmName, cellStateConn->tensor->Type(), tanhOfmShape);
    tanh->ConnectOutput(TensorUsage::OFM, tanhOfm).Set(tanhQuant);

    // Create Mul( Tanh, output gate )
    // Ofm quantization is based on the hidden scale.
    double hiddenScale = _lstmOp->Input(MakeTensorUsage(TensorUsage::Scratch, 4))->quantization.scales[0].Dequantize();
    auto mulQuant = _ofmConn->quantization;
    mulQuant.type = QuantizationType::TFLITE;
    mulQuant.scales = {ElementwiseMulScale(Q0_15_SCALE, Q0_15_SCALE, hiddenScale)};
    Operation *mul = CreateMul(tanhOfm, outputGateConn->tensor, tanhQuant, tanhQuant, mulQuant, _ifmConn->tensor->Type());

    // Save new output state
    SetStateWrite(mul, TensorUsage::State, batch);

    RecordOptimisation(tanh.get());
    RecordOptimisation(mul);
    return mul->Output(TensorUsage::OFM);
}


// Generate one step of the LSTM for the provided feature, batch and time
std::pair<TensorConnection *, TensorConnection *> LSTM::Step(TensorConnection *featureConn,
    TensorConnection *outputStateConn, TensorConnection *cellStateConn, int time, int batch)
{
    assert(outputStateConn && cellStateConn);
    auto suffix = fmt::format("b{0}.t{1}", batch, time);

    auto inputToInputWeightConn = _lstmOp->Input(MakeTensorUsage(TensorUsage::Weights, 0));
    auto recurrentToInputWeightConn = _lstmOp->Input(MakeTensorUsage(TensorUsage::Weights, 4));
    auto inputBiasConn = _lstmOp->Input(MakeTensorUsage(TensorUsage::Scales, 0));
    TensorConnection *inputGate = CalculateGate(fmt::format("input_gate_{0}", suffix), featureConn, outputStateConn,
        inputToInputWeightConn, inputBiasConn, recurrentToInputWeightConn, OpType::Sigmoid, batch);

    auto inputToForgetWeightConn = _lstmOp->Input(MakeTensorUsage(TensorUsage::Weights, 1));
    auto recurrentToForgetWeightConn = _lstmOp->Input(MakeTensorUsage(TensorUsage::Weights, 5));
    auto forgetBiasConn = _lstmOp->Input(MakeTensorUsage(TensorUsage::Scales, 1));
    TensorConnection *forgetGate = CalculateGate(fmt::format("forget_gate_{0}", suffix), featureConn, outputStateConn,
        inputToForgetWeightConn, forgetBiasConn, recurrentToForgetWeightConn, OpType::Sigmoid, batch);

    auto inputToCellWeightConn = _lstmOp->Input(MakeTensorUsage(TensorUsage::Weights, 2));
    auto recurrentToCellWeightConn = _lstmOp->Input(MakeTensorUsage(TensorUsage::Weights, 6));
    auto cellBiasConn = _lstmOp->Input(MakeTensorUsage(TensorUsage::Scales, 2));
    TensorConnection *cellGate = CalculateGate(fmt::format("cell_gate_{0}", suffix), featureConn, outputStateConn,
        inputToCellWeightConn, cellBiasConn, recurrentToCellWeightConn, OpType::Tanh, batch);

    // Calculate and update cell state
    cellStateConn = CalculateCellState(cellStateConn, inputGate, forgetGate, cellGate, time, batch);

    auto inputToOutputWeightConn = _lstmOp->Input(MakeTensorUsage(TensorUsage::Weights, 3));
    auto recurrentToOutputWeightConn = _lstmOp->Input(MakeTensorUsage(TensorUsage::Weights, 7));
    auto outputBiasConn = _lstmOp->Input(MakeTensorUsage(TensorUsage::Scales, 3));
    TensorConnection *outputGate = CalculateGate(fmt::format("output_gate_{0}", suffix), featureConn, outputStateConn,
        inputToOutputWeightConn, outputBiasConn, recurrentToOutputWeightConn, OpType::Sigmoid, batch);

    // Calculate and update ouput state
    assert(cellStateConn);
    outputStateConn = CalculateOutputState(outputGate, cellStateConn, time, batch);

    return {outputStateConn, cellStateConn};
}

}  // namespace regor
