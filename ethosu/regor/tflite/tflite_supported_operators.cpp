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

#include "tflite_supported_operators.hpp"

#include "common/common.hpp"
#include "common/logging.hpp"

#include "compiler/op_type.hpp"

namespace regor
{

bool TfLiteSupportedOperators::ConstraintOpType(const Operation *op)
{
    OpType opType = op->Type();
    if ( _supportedOpTypes.count(opType) == 0 )
    {
        Failure(op, "OpType is not supported", "");
        return false;
    }
    return true;
}

bool TfLiteSupportedOperators::ConstraintTensDtypes(const Operation *op)
{
    for ( const auto *list : {&op->Inputs(), &op->Outputs()} )
    {
        for ( const auto &item : list->pairs() )
        {
            auto usage = item.first;
            const auto &conn = item.second;
            auto type = conn.tensor->Type();
            if ( (IsIFM(usage) || IsOFM(usage)) && _supportedDataTypes.count(type) == 0 )
            {
                Failure(op, fmt::format("Operation has tensor with unsupported DataType {}", DataTypeToString(type)), "");
                return false;
            }
        }
    }
    return true;
}

bool TfLiteSupportedOperators::ConstraintNumSplits(const Operation *op)
{
    const char *constraint = "num_splits must match the number of outputs";
    const tflite::Operator *passthrough = static_cast<const tflite::Operator *>(op->Passthrough());
    OpType opType = op->Type();
    int numSplits = 0;
    if ( opType == OpType::Split )
    {
        assert(passthrough);
        const auto *opt = passthrough->builtin_options_as_SplitOptions();
        assert(opt);
        numSplits = opt->num_splits();
    }
    else if ( opType == OpType::SplitV )
    {
        assert(passthrough);
        const auto *opt = passthrough->builtin_options_as_SplitVOptions();
        assert(opt);
        numSplits = opt->num_splits();
    }
    else
    {
        return true;
    }
    int numOutputs = op->Outputs().size();
    if ( numSplits != numOutputs )
    {
        Failure(op, fmt::format("num_splits: {} does not match the number of outputs: {}", numSplits, numOutputs), constraint);
        return false;
    }
    return true;
}

bool TfLiteSupportedOperators::ConstraintMustHaveIFM(const Operation *op)
{
    const char *constraint = "Operations must have at least one IFM.";
    for ( const auto item : op->Inputs().pairs() )
    {
        auto usage = item.first;
        if ( IsIFM(usage) )
        {
            return true;
        }
    }
    Failure(op, "Operation without IFM", constraint);
    return false;
}

bool TfLiteSupportedOperators::ConstraintMustHaveOFM(const Operation *op)
{
    const char *constraint = "Operations must have at least one OFM.";
    for ( const auto item : op->Outputs().pairs() )
    {
        auto usage = item.first;
        if ( IsOFM(usage) )
        {
            return true;
        }
    }
    Failure(op, "Operation without OFM", constraint);
    return false;
}

bool TfLiteSupportedOperators::ConstraintTensMustHaveShape(const Operation *op)
{
    const char *constraint = "Tensors must have constant shape.";
    for ( const auto *list : {&op->Inputs(), &op->Outputs()} )
    {
        for ( const auto &item : list->pairs() )
        {
            auto usage = item.first;
            const auto &conn = item.second;
            if ( !conn.shape )
            {
                Failure(op, "Operation has shapeless tensor", constraint);
                return false;
            }
        }
    }
    return true;
}

bool TfLiteSupportedOperators::ConstraintTensQuantized(const Operation *op)
{
    const char *constraint = "Input(s), Output and Weight tensors must have quantization parameters";
    // Exceptions for this check
    switch ( op->Type() )
    {
        case OpType::ArgMax:
        case OpType::MirrorPad:
        case OpType::Quantize:
        case OpType::Shape:
        case OpType::Transpose:
        case OpType::GatherNd:
        case OpType::GatherV2:
            return true;
        default:
            break;
    }
    for ( const auto *list : {&op->Inputs(), &op->Outputs()} )
    {
        for ( const auto &item : list->pairs() )
        {
            auto usage = item.first;
            const auto &conn = item.second;
            if ( IsIFM(usage) || IsOFM(usage) || usage == TensorUsage::Weights )
            {
                const Quantization &quant = conn.quantization;
                if ( quant.scales.empty() || quant.zeroPoints.empty() )
                {
                    Failure(op, fmt::format("Operation has tensor {} with missing quantization parameters", conn.tensor->Name()), constraint);
                    return false;
                }
            }
        }
    }
    return true;
}

bool TfLiteSupportedOperators::ConstraintFCWeightShape(const Operation *op)
{
    const char *constraint = "FullyConnected weights must be on the form I,1,1,..,1,O";
    if ( op->Type() != OpType::FullyConnected )
    {
        return true;
    }
    auto weights = op->Input(TensorUsage::Weights);
    assert(weights);
    assert(weights->tensor);
    const auto &shape = weights->tensor->StorageShape();
    // Total elements must be equal to first-dim * last-dim
    if ( shape.Size() < 2 || (shape.Elements() != (shape[0] * shape[-1])) )
    {
        Failure(op, fmt::format("Unsupported weights shape: {}", shape.ToString()), constraint);
        return false;
    }
    return true;
}

void TfLiteSupportedOperators::Failure(const Operation *op, const std::string &message, const std::string &constraint)
{
    assert(op);
    auto ofmConn = op->Output(TensorUsage::OFM);
    const char *name = "N/A";
    if ( ofmConn && ofmConn->tensor )
    {
        name = ofmConn->tensor->Name().c_str();
    }
    auto tfLiteType = TfLiteMapping::OpTypeToBuiltinOperator(op->Type());
    assert(message.size() || constraint.size());
    LOG_WARN("\nWarning (supported operators) operator:{} ofm:{}\n", TfLiteMapping::BuiltinOperatorToString(tfLiteType), name);
    if ( message.size() )
    {
        LOG_WARN("Reason: {}\n", message);
    }
    if ( constraint.size() )
    {
        LOG_WARN("Constraint: {}\n", constraint);
    }
}

TfLiteSupportedOperators::TfLiteSupportedOperators(IArchitectureConstraints *constraints) :
        _archConstraints(constraints)
{
    _genericChecks = {
        &TfLiteSupportedOperators::ConstraintOpType,
        &TfLiteSupportedOperators::ConstraintTensDtypes,
        &TfLiteSupportedOperators::ConstraintNumSplits,
        &TfLiteSupportedOperators::ConstraintMustHaveIFM,
        &TfLiteSupportedOperators::ConstraintMustHaveOFM,
        &TfLiteSupportedOperators::ConstraintTensMustHaveShape,
        &TfLiteSupportedOperators::ConstraintFCWeightShape,
        &TfLiteSupportedOperators::ConstraintTensQuantized,
    };
}

namespace
{
void DisconnectActivation(std::shared_ptr<Operation> op)
{
    assert(TfLiteMapping::CanFuseActivationFunction(op.get()));
    // Op originally had a fused activation
    assert(op->Outputs().size() == 1);
    assert(op->OFM()->Readers().size() == 1);
    auto activation = op->OFM()->Readers().front();
    auto actOfm = activation->Output(TensorUsage::OFM);
    assert(actOfm);
    // bypass and disconnect the activation
    op->CopyOutput(TensorUsage::OFM, *actOfm);
    activation->SetPassthroughOp();
    activation->Disconnect();
}
}  // namespace

void TfLiteSupportedOperators::Process(Graph *graph)
{
    std::vector<std::shared_ptr<Operation>> operatorList;
    graph->GetAllOperations(operatorList);
    for ( auto &op : operatorList )
    {
        if ( op->Type() == OpType::Passthrough )
        {
            // Op is already passthrough
            // Only valid scenario is that op is a previously disconnected activation
            assert(op->Passthrough() == nullptr && "source-operation set to passthrough before supported-ops checks");
            assert(op->CountInputs(TensorUsage::IFM) == 0);
            assert(op->CountOutputs(TensorUsage::OFM) == 0);
            continue;
        }
        if ( !Check(op.get()) )
        {
            if ( TfLiteMapping::CanFuseActivationFunction(op.get()) )
            {
                // op originally had a fused activation
                // disconnect it from the graph as it will be handled by CPU
                DisconnectActivation(op);
            }
            else if ( op->IFM(0)->Writers().size() == 1 )
            {
                auto pred = op->IFM(0)->Writers().front();
                if ( TfLiteMapping::CanFuseActivationFunction(pred.get()) )
                {
                    // op is an activation function, disconnect op and set pred to passthrough
                    DisconnectActivation(pred);
                    pred->SetPassthroughOp();
                }
            }
            op->SetPassthroughOp();
        }
    }
}

}  // namespace regor
