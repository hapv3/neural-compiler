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
        case OpType::Select:
        case OpType::SelectV2:
        case OpType::ScatterNd:
        case OpType::Pad:
        case OpType::PadV2:
        case OpType::ReduceAll:
        case OpType::ReduceAny:
        case OpType::ExpandDims:
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

bool TfLiteSupportedOperators::ConstraintPerAxisQuant(const Operation *op)
{
    OpType opType = op->Type();
    if ( IsConvolution(opType) || opType == OpType::FullyConnected )
    {
        return true;
    }

    for ( const auto *list : {&op->Inputs(), &op->Outputs()} )
    {
        for ( const auto &[usage, conn] : list->pairs() )
        {
            if ( conn.quantization.scales.size() > 1 || conn.quantization.zeroPoints.size() > 1 )
            {
                Failure(op, "Operation does not support per-axis quantization", "");
                return false;
            }
        }
    }
    return true;
}

bool TfLiteSupportedOperators::ConstraintMatchingQuantization(const Operation *op)
{
    const char *constraint = "Both Input quantization parameters must match OFM quantization parameters";

    OpType opType = op->Type();

    if ( opType != OpType::Minimum && opType != OpType::Maximum )
    {
        return true;
    }

    const auto ofmConn = op->Output(TensorUsage::OFM);
    const auto ifmConn = op->Input(TensorUsage::IFM);
    const auto ifm2Conn = op->Input(TensorUsage::IFM1);
    assert(ofmConn);
    assert(ifmConn);
    assert(ifm2Conn);
    const auto &ofmQuant = ofmConn->quantization;
    const auto &ifmQuant = ifmConn->quantization;
    const auto &ifm2Quant = ifm2Conn->quantization;
    if ( ifmQuant != ofmQuant || ifm2Quant != ofmQuant )
    {
        Failure(op, "Operation has mismatching quantization parameters.", constraint);
        return false;
    }
    return true;
}

bool TfLiteSupportedOperators::ConstraintWeightsPrecision(const Operation *op)
{
    const char *constraint = "Weight tensors must be 8-bit precision";
    const auto wconn = op->Input(TensorUsage::Weights);
    if ( !wconn )
    {
        return true;
    }
    const auto type = wconn->tensor->Type();
    if ( DataTypeSizeBits(type) != 8 )
    {
        Failure(op, fmt::format("Weights tensor with precision: {}", DataTypeToString(type)), constraint);
        return false;
    }
    return true;
}

bool TfLiteSupportedOperators::ConstraintWeightSum(const Operation *op)
{
    std::string constraint = fmt::format(
        "The sum of absolute weights cannot exceed:\n"
        "\t{} for 8-bit IFM\n"
        "\t{} for 16-bit IFM",
        _maxWeightSum8Bit, _maxWeightSum16Bit);

    auto wConn = op->Input(TensorUsage::Weights);
    auto ifmConn = op->Input(TensorUsage::IFM);
    if ( !wConn || !ifmConn )
    {
        return true;
    }
    if ( !wConn->tensor->IsConstant() )
    {
        return true;
    }

    auto view = wConn->tensor->View();
    auto zeroPoints = wConn->quantization.zeroPoints;
    auto ifmType = ifmConn->tensor->Type();
    int ifmBits = DataTypeSizeBits(ifmType);
    int64_t maxWeightSum = ifmBits == 8 ? _maxWeightSum8Bit : _maxWeightSum16Bit;
    auto reader = view.Values<int>(wConn->tensor->Type());
    AxisOrder order = wConn->tensor->AxisOrder();
    Shape readShape = wConn->tensor->StorageShape();
    assert(readShape.Size() == 4);
    assert(order == AxisOrder::OHWI || order == AxisOrder::IHWO);

    int outChannels = readShape.Depth();
    int inChannels = readShape.Batch();
    if ( order == AxisOrder::OHWI )
    {
        std::swap(outChannels, inChannels);
    }
    // abort early if the readShape of the weights tensor guarantees no overflow.
    if ( (255 * readShape.Elements64() / outChannels) < maxWeightSum )
    {
        return true;
    }
    // Accumulate the weights in slices of output-channels
    // Fail if any slice overflows maxWeightSum
    for ( int out = 0; out < outChannels; out++ )
    {
        int64_t zeroPoint = 0;
        if ( !zeroPoints.empty() )
        {
            zeroPoint = zeroPoints.size() > 1 ? zeroPoints[out] : zeroPoints[0];
        }
        int64_t sum = 0;
        for ( int in = 0; in < inChannels; in++ )
        {
            for ( int h = 0; h < readShape.Height(); h++ )
            {
                for ( int w = 0; w < readShape.Width(); w++ )
                {
                    int64_t v;
                    if ( order == AxisOrder::OHWI )
                    {
                        v = reader[{out, h, w, in}];
                    }
                    else
                    {
                        v = reader[{in, h, w, out}];
                    }
                    sum += std::abs(v - zeroPoint);
                }
            }
        }
        if ( sum > maxWeightSum )
        {
            Failure(op, fmt::format("The absolute sum of weight-tensor elements: {} exceeds {}", sum, maxWeightSum), constraint);
            return false;
        }
    }
    return true;
}

bool TfLiteSupportedOperators::ConstraintBias(const Operation *op)
{
    auto bConn = op->Input(TensorUsage::Scales);
    if ( !bConn )
    {
        return true;
    }
    auto bShape = bConn->shape;
    if ( bShape.Elements() > bShape.Depth() )
    {
        Failure(op, fmt::format("Bias shape: {}", bShape.ToString()), "bias-values must be stored in channel axis");
        return false;
    }
    if ( !bConn->tensor->IsConstant() )
    {
        Failure(op, "Operation has non-constant bias tensor.", "The bias tensor must be constant");
        return false;
    }
    auto type = bConn->tensor->Type();
    if ( type != DataType::Int32 && type != DataType::Int64 )
    {
        Failure(op, fmt::format("Operation has bias with type:{}", DataTypeToString(type)), "The bias tensor precision must be Int32 or Int64");
        return false;
    }
    if ( type == DataType::Int64 )
    {
        // read bias values
        auto view = bConn->tensor->View();
        auto values = view.Values<int64_t>();
        for ( int64_t bias : values )
        {
            if ( bias > _maxBias )
            {
                std::string constraint = fmt::format("Int64 bias must be smaller than {}", _maxBias);
                Failure(op, fmt::format("Bias is out of range: {} > {}", bias, _maxBias), constraint);
                return false;
            }
        }
    }
    return true;
}

bool TfLiteSupportedOperators::ConstraintAvgPool(const Operation *op)
{
    OpType opType = op->Type();
    if ( opType != OpType::AvgPool )
    {
        return true;
    }
    auto kernel = op->Kernel();
    assert(kernel);
    auto [w, h] = kernel->Size();
    auto [sw, sh] = kernel->Stride();
    if ( kernel->Padding().IsZero() )
    {
        // VALID padding
        if ( h > 256 || h < 1 )
        {
            Failure(op, fmt::format("kernel height: {} out of range", h), "When padding=VALID, kernel-height must be in the range (1,256)");
            return false;
        }
        if ( h * w > 256 * 256 )
        {
            Failure(op, fmt::format("kernel product: {} out of range", h * w),
                "When padding=VALID, kernel product (H*W) must be in the range (1, 256*256)");
            return false;
        }
    }
    else
    {
        // SAME padding
        if ( w != sw && (w > 8 || w < 1) )
        {
            // kernel width out of range
            Failure(op, fmt::format("kernel width: {} out of range", w),
                "When padding=SAME, kernel width must be in the range (1,8) OR equal to the stride(width)");
            return false;
        }
        if ( h > 8 || h < 1 )
        {
            Failure(op, fmt::format("kernel height: {} out of range", h), "When padding=SAME, kernel height must be in the range (1,8)");
            return false;
        }
    }
    return true;
}

bool TfLiteSupportedOperators::ConstraintMaxPool(const Operation *op)
{
    OpType opType = op->Type();
    if ( opType != OpType::MaxPool )
    {
        return true;
    }
    auto kernel = op->Kernel();
    assert(kernel);
    auto [w, h] = kernel->Size();
    auto [sw, sh] = kernel->Stride();
    if ( h > 256 || h < 1 )
    {
        Failure(op, fmt::format("kernel height: {} out of range", h), "Kernel height must be in the range (1, 256)");
        return false;
    }
    if ( h * w > 256 * 256 )
    {
        Failure(op, fmt::format("kernel product: {} out of range", h * w), "Kernel product must be in the range (1, 256 * 256)");
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
    _maxWeightSum8Bit = 0;
    _maxWeightSum16Bit = 0;
    _maxBias = 0;
    _genericChecks = {
        &TfLiteSupportedOperators::ConstraintOpType,
        &TfLiteSupportedOperators::ConstraintTensDtypes,
        &TfLiteSupportedOperators::ConstraintNumSplits,
        &TfLiteSupportedOperators::ConstraintMustHaveIFM,
        &TfLiteSupportedOperators::ConstraintMustHaveOFM,
        &TfLiteSupportedOperators::ConstraintTensMustHaveShape,
        &TfLiteSupportedOperators::ConstraintFCWeightShape,
        &TfLiteSupportedOperators::ConstraintTensQuantized,
        &TfLiteSupportedOperators::ConstraintPerAxisQuant,
        &TfLiteSupportedOperators::ConstraintMatchingQuantization,
        &TfLiteSupportedOperators::ConstraintWeightsPrecision,
        &TfLiteSupportedOperators::ConstraintWeightSum,
        &TfLiteSupportedOperators::ConstraintBias,
        &TfLiteSupportedOperators::ConstraintAvgPool,
        &TfLiteSupportedOperators::ConstraintMaxPool,
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
