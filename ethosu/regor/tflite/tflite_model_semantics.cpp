//
// SPDX-FileCopyrightText: Copyright 2023-2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
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


#include "tflite_model_semantics.hpp"

#include "tflite_mapping.hpp"

#include <fmt/format.h>
#include <string>
#include <type_traits>
#include <unordered_set>
#include <utility>
#include <vector>

namespace tflite
{
using BufferOffsetRef = const flatbuffers::Vector<flatbuffers::Offset<Buffer>> &;

using OperatorFunction = void (*)(const Operator &op, const SubGraph &subgraph, const BuiltinOperator &builtinOperator, BufferOffsetRef buffers);
using OpCheckVec = std::vector<OperatorFunction>;

namespace
{
template<typename T>
T *CheckedPtr(T *p)
{
    if ( p == nullptr )
    {
        throw std::runtime_error(
            "Error: Null pointer exception encountered when reading TFLite file\n"
            "Failed to Parse TFLite file\n");
    }
    else
    {
        return p;
    }
}

template<typename T1, typename T2>
T1 CheckedAdd(T1 a, T2 b)
{
    if ( a > 0 && b > std::numeric_limits<T1>::max() - a )
    {
        throw std::runtime_error(
            "Integer overflow on addition\n"
            "Failed to parse TFLite file\n");
    }
    else if constexpr ( std::is_signed<T1>::value )
    {
        if ( a < 0 && b < std::numeric_limits<T1>::min() - a )
        {
            throw std::runtime_error(
                "Integer underflow on addition\n"
                "Failed to parse TFLite file\n");
        }
    }
    return T1(a + b);
}

template<typename T, typename U>
unsigned int BoundsCheckedIndex(T index, const U &vector)
{
    if ( (std::is_unsigned_v<T> || index >= 0) && static_cast<uint64_t>(index) < vector.size() ) return unsigned(index);
    throw std::runtime_error("Error: Out of bounds\n");
}

template<typename T, typename U>
unsigned int BoundsCheckedIndex(T index, const U &vector, const BuiltinOperator &builtinOperator)
{
    if ( (std::is_unsigned_v<T> || index >= 0) && static_cast<uint64_t>(index) < vector.size() ) return unsigned(index);
    std::string errorMessage = fmt::format(
        "Error: {0} Does not have valid TFLite Semantics.\n"
        " - Index out of bounds\n"
        "   Most likely missing inputs or output\n"
        "Failed to parse TFLite file\n",
        EnumNameBuiltinOperator(builtinOperator));

    throw std::runtime_error(errorMessage);
}

const Tensor *TensorFromUsage(regor::TensorUsage usage, const Operator &op, const BuiltinOperator &builtinOperator,
    const flatbuffers::Vector<flatbuffers::Offset<Tensor>> &tensors)
{
    auto opType = regor::TfLiteMapping::BuiltinOperatorToOpType(builtinOperator);
    if ( !regor::IsOFM(usage) )
    {
        const auto *inputs = CheckedPtr(op.inputs());
        if ( IsVariadic(opType) )
        {
            auto tensorIndex = BoundsCheckedIndex(regor::GetUsageIndex(usage), *inputs, builtinOperator);
            auto tensorsLookupIndex = BoundsCheckedIndex((*inputs)[tensorIndex], tensors, builtinOperator);
            return CheckedPtr(tensors[tensorsLookupIndex]);
        }

        auto mapping = regor::TfLiteMapping::InputTensorIndices(opType);
        auto at = std::find_if(mapping.begin(), mapping.end(),
            [usage](const std::pair<regor::OpType, regor::TensorUsage> &p) { return p.second == usage; });
        if ( at == mapping.end() )
        {
            return nullptr;
        }

        auto i = std::distance(mapping.begin(), at);
        auto tensorIndex = BoundsCheckedIndex(i, *inputs, builtinOperator);
        auto tensorsLookupIndex = BoundsCheckedIndex((*inputs)[tensorIndex], tensors, builtinOperator);
        return CheckedPtr(tensors[tensorsLookupIndex]);
    }
    else
    {
        const auto *outputs = CheckedPtr(op.outputs());
        auto tensorIndex = BoundsCheckedIndex(regor::GetUsageIndex(usage), *outputs, builtinOperator);
        auto tensorsLookupIndex = BoundsCheckedIndex((*outputs)[tensorIndex], tensors, builtinOperator);
        return CheckedPtr(tensors[tensorsLookupIndex]);
    }
}
}  // namespace

class InvalidTfLiteException
{
private:
    const std::string _constraint;
    const std::string _extra;
    const std::string _tensorName;
    const std::string _builtinOperatorName;

public:
    InvalidTfLiteException(const std::string &constraint, const std::string &extra, const Tensor &tensor) :
            _constraint(constraint), _extra(extra), _tensorName(CheckedPtr(tensor.name())->str())
    {
    }
    InvalidTfLiteException(const std::string &constraint, const std::string &extra, const Operator &op, const SubGraph &subgraph, const BuiltinOperator &builtinOperator) :
            _constraint(constraint), _extra(extra),
            _tensorName(
                CheckedPtr(TensorFromUsage(regor::TensorUsage::OFM, op, builtinOperator, *subgraph.tensors())->name())->str()),
            _builtinOperatorName(EnumNameBuiltinOperator(builtinOperator))
    {
    }

    const std::string &Constraint() const { return _constraint; }
    const std::string &Extra() const { return _extra; }
    const std::string &TensorName() const { return _tensorName; }
    const std::string &BuiltinOperatorName() const { return _builtinOperatorName; }
};

namespace
{

const std::array<BuiltinOperator, 1> convolutionOps = {BuiltinOperator::CONV_2D};

const std::array<BuiltinOperator, 2> poolingOps = {BuiltinOperator::AVERAGE_POOL_2D, BuiltinOperator::MAX_POOL_2D};

const std::array<BuiltinOperator, 1> concatOps = {BuiltinOperator::CONCATENATION};

const std::array<BuiltinOperator, 4> unaryElemWiseMainOps = {
    BuiltinOperator::ABS, BuiltinOperator::EXP, BuiltinOperator::LEAKY_RELU, BuiltinOperator::RSQRT};

const std::array<BuiltinOperator, 2> binaryElemWiseMinMaxOps = {BuiltinOperator::MINIMUM, BuiltinOperator::MAXIMUM};

const std::array<BuiltinOperator, 3> binaryElemWiseAddMulSub = {BuiltinOperator::ADD, BuiltinOperator::MUL, BuiltinOperator::SUB};

const std::array<BuiltinOperator, 10> elemWiseMainOps = {BuiltinOperator::MINIMUM, BuiltinOperator::MAXIMUM,
    BuiltinOperator::ADD, BuiltinOperator::MUL, BuiltinOperator::SUB, BuiltinOperator::ABS, BuiltinOperator::EXP,
    BuiltinOperator::LEAKY_RELU, BuiltinOperator::RSQRT, BuiltinOperator::SQUARED_DIFFERENCE};

const std::array<BuiltinOperator, 3> reshapeOps = {BuiltinOperator::RESHAPE, BuiltinOperator::SQUEEZE, BuiltinOperator::EXPAND_DIMS};


template<typename T>
T DataFromBuffer(BufferOffsetRef buffers, uint32_t bufferIndex, size_t index)
{
    bufferIndex = BoundsCheckedIndex(bufferIndex, buffers);
    auto buffer = CheckedPtr(buffers[bufferIndex]);
    const void *p = CheckedPtr(buffer->data())->data();
    if ( (uintptr_t(p) % alignof(T)) == 0 && index < CheckedPtr(buffer->data())->size() )
        return static_cast<const T *>(p)[index];
    throw std::runtime_error("Out of bounds\n");
}



bool IsInt(TensorType tensorType)
{
    return (tensorType == TensorType::INT4) || (tensorType == TensorType::INT8) || (tensorType == TensorType::INT16) ||
           (tensorType == TensorType::INT32) || (tensorType == TensorType::INT64);
}

bool IsUint(TensorType tensorType)
{
    return (tensorType == TensorType::UINT8) || (tensorType == TensorType::UINT16) ||
           (tensorType == TensorType::UINT32) || (tensorType == TensorType::UINT64);
}

Shape ShapeFromTens(const Tensor *tens)
{
    if ( tens->shape() )
    {
        size_t size = tens->shape()->size();
        if ( size > 0 ) return Shape(tens->shape()->data(), size);
    }
    return Shape();
}

void ConstraintEmptyConstTensors(const Model &m_model)
{
    std::unordered_set<int> writtenTensors;
    auto &buffers = *CheckedPtr(m_model.buffers());
    for ( const auto subgraph : *m_model.subgraphs() )
    {
        auto &tensors = *CheckedPtr(subgraph->tensors());
        if ( subgraph->inputs() && subgraph->operators() )
        {
            for ( auto input : *subgraph->inputs() )
            {
                writtenTensors.insert(input);
            }

            for ( auto op : *subgraph->operators() )
            {
                for ( auto output : *op->outputs() )
                {
                    writtenTensors.insert(output);
                }
            }
            for ( auto op : *subgraph->operators() )
            {
                for ( auto input : *op->inputs() )
                {
                    if ( input != -1 && !writtenTensors.count(input) )
                    {
                        auto tensor = tensors[BoundsCheckedIndex(input, tensors)];
                        auto buffer = buffers[BoundsCheckedIndex(tensor->buffer(), buffers)];
                        // Buffer 0 is a special buffer that is used for empty tensors
                        if ( (tensor->buffer() > 0 && (!buffer->data() || buffer->data()->size() == 0) && buffer->offset() <= 1) ||
                             (buffer->offset() > 1 && buffer->size() == 0) )
                        {
                            std::string constraint = "Constant tensors must not have empty buffers";
                            std::string extra = "Found Constant Tensor with empty buffer";
                            throw InvalidTfLiteException(constraint, extra, *tensor);
                        }
                    }
                }
            }
        }
    }
}

void ConstraintTensQuantScale(const Model &m_model)
{
    for ( const auto &subgraph : *m_model.subgraphs() )
    {
        for ( const auto &tensor : *subgraph->tensors() )
        {
            if ( tensor->quantization() && tensor->quantization()->scale() )
            {
                for ( float scale : *tensor->quantization()->scale() )
                {
                    if ( !std::isfinite(scale) )
                    {
                        std::string constraint = "Tensor Quantization scales must be finite";
                        std::string extra = fmt::format("Quantization scale={}", scale);
                        throw InvalidTfLiteException(constraint, extra, *tensor);
                    }
                }
            }
        }
    }
}

void ConstraintQuantScaleInf(const Operator &op, const SubGraph &subgraph, const BuiltinOperator &builtinOperator, BufferOffsetRef)
{
    auto *inputs = CheckedPtr(op.inputs());
    auto *outputs = CheckedPtr(op.outputs());
    if ( inputs->size() > 0 && outputs->size() > 0 )
    {
        auto ifm = TensorFromUsage(regor::TensorUsage::IFM, op, builtinOperator, *subgraph.tensors());
        auto ofm = TensorFromUsage(regor::TensorUsage::OFM, op, builtinOperator, *subgraph.tensors());
        if ( ifm && ifm->quantization() && ofm && ofm->quantization() )
        {
            auto ifmScales = ifm->quantization()->scale();
            auto ofmScales = ofm->quantization()->scale();
            if ( ifmScales && ifmScales->size() > 0 && ofmScales && ofmScales->size() > 0 )
            {
                float ifmScale = (*ifmScales)[0];
                float ofmScale = (*ofmScales)[0];
                if ( !std::isfinite(ifmScale / ofmScale) )
                {
                    std::string constraint = "Input and Output tensors must have quantization scales that fit within float32 precision";
                    std::string extra = fmt::format("(IFM Scale / OFM Scale)={}", ifmScale / ofmScale);
                    throw InvalidTfLiteException(constraint, extra, op, subgraph, builtinOperator);
                }
            }
        }
    }
}

void ConstraintConvGroupsIfmDepth(const Operator &op, const SubGraph &subgraph, const BuiltinOperator &builtinOperator, BufferOffsetRef)
{
    auto ifm = TensorFromUsage(regor::TensorUsage::IFM, op, builtinOperator, *subgraph.tensors());
    auto weights = TensorFromUsage(regor::TensorUsage::Weights, op, builtinOperator, *subgraph.tensors());
    auto ifmDepth = ShapeFromTens(ifm)[-1];
    auto kernelIc = ShapeFromTens(weights)[-1];
    if ( kernelIc == 0 || kernelIc < 0 || ifmDepth < 0 )
    {
        throw std::runtime_error("Error: Out of bounds\n");
    }
    if ( ifmDepth % kernelIc != 0 )
    {
        std::string constraint = "IFM depth must be a whole multiple of the filter kernel depth";
        std::string extra = fmt::format("IFM depth = {} and filter kernel depth = {}", ifmDepth, kernelIc);
        throw InvalidTfLiteException(constraint, extra, op, subgraph, builtinOperator);
    }
}

void ConstraintConvGroupsNumFilters(const Operator &op, const SubGraph &subgraph, const BuiltinOperator &builtinOperator, BufferOffsetRef)
{
    auto ifm = TensorFromUsage(regor::TensorUsage::IFM, op, builtinOperator, *subgraph.tensors());
    auto weights = TensorFromUsage(regor::TensorUsage::Weights, op, builtinOperator, *subgraph.tensors());
    auto ifmDepth = ShapeFromTens(ifm)[-1];
    auto kernelIc = ShapeFromTens(weights)[-1];
    auto kernelOc = ShapeFromTens(weights)[0];

    auto numConvGroups = ifmDepth / kernelIc;
    if ( numConvGroups == 0 || kernelOc < 0 || numConvGroups < 0 )
    {
        throw std::runtime_error("Error: Out of bounds\n");
    }
    if ( kernelOc % numConvGroups != 0 )
    {
        std::string constraint = "Number of filter kernels must be equally divisible by the number of convolution groups";
        std::string extra = fmt::format("Conv Groups = {}, filter kernels = {}", numConvGroups, kernelOc);
        throw InvalidTfLiteException(constraint, extra, op, subgraph, builtinOperator);
    }
}

void ConstraintDepthwiseConvOfmDepth(const Operator &op, const SubGraph &subgraph, const BuiltinOperator &builtinOperator, BufferOffsetRef)
{
    auto ifm = TensorFromUsage(regor::TensorUsage::IFM, op, builtinOperator, *subgraph.tensors());
    auto ofm = TensorFromUsage(regor::TensorUsage::OFM, op, builtinOperator, *subgraph.tensors());
    auto ifmDepth = ShapeFromTens(ifm)[-1];
    auto ofmDepth = ShapeFromTens(ofm)[-1];

    int depth_multiplier = CheckedPtr(op.builtin_options_as_DepthwiseConv2DOptions())->depth_multiplier();

    if ( ifmDepth < 0 || ofmDepth < 0 )
    {
        throw std::runtime_error("Error: Out of bounds\n");
    }
    if ( ifmDepth * depth_multiplier != ofmDepth )
    {
        std::string constraint = "OFM depth must be a equal to IFM depth times depth multiplier";
        std::string extra = fmt::format("OFM depth = {}, IFM depth = {} and depth multiplier = {}", ofmDepth, ifmDepth, depth_multiplier);
        throw InvalidTfLiteException(constraint, extra, op, subgraph, builtinOperator);
    }
}

void ConstraintMatchingInOutTypes(const Operator &op, const SubGraph &subgraph, const BuiltinOperator &builtinOperator, BufferOffsetRef)
{
    auto ifm = TensorFromUsage(regor::TensorUsage::IFM, op, builtinOperator, *subgraph.tensors());
    auto ofm = TensorFromUsage(regor::TensorUsage::OFM, op, builtinOperator, *subgraph.tensors());
    if ( ifm->type() != ofm->type() )
    {
        std::string constraint = "IFM and OFM types must match";
        std::string extra = fmt::format(
            "IFM type={} and OFM type={}", EnumNameTensorType(ifm->type()), EnumNameTensorType(ofm->type()));
        throw InvalidTfLiteException(constraint, extra, op, subgraph, builtinOperator);
    }
}

void ConstraintSoftmaxInOutTypes(const Operator &op, const SubGraph &subgraph, const BuiltinOperator &builtinOperator, BufferOffsetRef)
{
    auto ifm = TensorFromUsage(regor::TensorUsage::IFM, op, builtinOperator, *subgraph.tensors());
    auto ofm = TensorFromUsage(regor::TensorUsage::OFM, op, builtinOperator, *subgraph.tensors());
    if ( ifm->type() != ofm->type() && !(ifm->type() == TensorType::INT8 && ofm->type() == TensorType::INT16) )
    {
        std::string constraint = "IFM and OFM types must match or upcast";
        std::string extra = fmt::format(
            "IFM type={} and OFM type={}", EnumNameTensorType(ifm->type()), EnumNameTensorType(ofm->type()));
        throw InvalidTfLiteException(constraint, extra, op, subgraph, builtinOperator);
    }
}

void ConstraintBetaValueRange(const Operator &op, const SubGraph &subgraph, const BuiltinOperator &builtinOperator, BufferOffsetRef)
{
    auto beta = CheckedPtr(op.builtin_options_as_SoftmaxOptions())->beta();
    if ( beta < 0 )
    {
        std::string constraint = "Beta attr must to be positive";
        std::string extra = fmt::format("Attribute beta={}", beta);
        throw InvalidTfLiteException(constraint, extra, op, subgraph, builtinOperator);
    }
}

void ConstraintMatchingShapes(const Operator &op, const SubGraph &subgraph, const BuiltinOperator &builtinOperator, BufferOffsetRef)
{
    auto ifm = TensorFromUsage(regor::TensorUsage::IFM, op, builtinOperator, *subgraph.tensors());
    auto ofm = TensorFromUsage(regor::TensorUsage::OFM, op, builtinOperator, *subgraph.tensors());
    auto ifmShape = ShapeFromTens(ifm);
    auto ofmShape = ShapeFromTens(ofm);

    if ( ifmShape != ofmShape )
    {
        std::string constraint = "IFM and OFM shapes must match";
        std::string extra = fmt::format("IFM shape={} and OFM type={}", ofmShape.ToString(), ifmShape.ToString());
        throw InvalidTfLiteException(constraint, extra, op, subgraph, builtinOperator);
    }
}

void ConstraintSplitDim(const Operator &op, const SubGraph &subgraph, const BuiltinOperator &builtinOperator, BufferOffsetRef buffers)
{
    auto ifm = TensorFromUsage(regor::TensorUsage::IFM, op, builtinOperator, *subgraph.tensors());
    auto splitDimTensor = TensorFromUsage(regor::TensorUsage::Params, op, builtinOperator, *subgraph.tensors());

    int64_t splitDim = DataFromBuffer<int>(buffers, splitDimTensor->buffer(), 0);
    int64_t ifmRank = CheckedPtr(ifm->shape())->size();

    if ( splitDim <= -ifmRank || splitDim > ifmRank || ifmRank < 0 )
    {
        std::string constraint = "Split dim size must be in the interval [-rank(IFM),rank(IFM))";
        std::string extra = fmt::format("IFM rank={}, and Input split_dim={}", ifmRank, splitDim);
        throw InvalidTfLiteException(constraint, extra, op, subgraph, builtinOperator);
    }
}

void ConstraintSplitNumSplits(const Operator &op, const SubGraph &subgraph, const BuiltinOperator &builtinOperator, BufferOffsetRef buffers)
{
    auto ifm = TensorFromUsage(regor::TensorUsage::IFM, op, builtinOperator, *subgraph.tensors());
    auto ifmShape = ShapeFromTens(ifm);

    auto splitDimTensor = TensorFromUsage(regor::TensorUsage::Params, op, builtinOperator, *subgraph.tensors());
    auto splitDim = DataFromBuffer<int>(buffers, splitDimTensor->buffer(), 0);
    if ( splitDim < 0 )
    {
        splitDim = CheckedAdd(ifmShape.Size(), splitDim);
    }
    auto index = BoundsCheckedIndex(splitDim, *CheckedPtr(ifm->shape()));
    int splitDimSize = (*CheckedPtr(ifm->shape()))[index];

    int num_splits = CheckedPtr(op.builtin_options_as_SplitOptions())->num_splits();

    if ( splitDimSize == 0 || num_splits < 0 || splitDimSize < 0 )
    {
        throw std::runtime_error("Error: Out of bounds\n");
    }
    if ( num_splits != 0 && splitDimSize % num_splits != 0 )
    {
        std::string constraint = "Size of dim to split must be divisible by the number of splits";
        std::string extra = fmt::format("split_dim size={}, Attribute num_splits={}", splitDimSize, splitDim);
        throw InvalidTfLiteException(constraint, extra, op, subgraph, builtinOperator);
    }
}

void ConstraintSplitvInferred(const Operator &op, const SubGraph &subgraph, const BuiltinOperator &builtinOperator, BufferOffsetRef buffers)
{
    auto sizeSplits = TensorFromUsage(regor::TensorUsage::Params, op, builtinOperator, *subgraph.tensors());

    int sizeSplitsElems = (*CheckedPtr(sizeSplits->shape()))[0];
    int sizesToInfer = 0;
    bool valid = true;

    std::vector<int> vec;
    if ( sizeSplitsElems <= 0 )
    {
        throw std::runtime_error("Error: Out of Bounds\n");
    }
    vec.reserve(sizeSplitsElems);

    for ( unsigned int i = 0; static_cast<int>(i) < sizeSplitsElems; i++ )
    {
        if ( DataFromBuffer<int>(buffers, sizeSplits->buffer(), i) == -1 )
        {
            sizesToInfer++;
            vec.emplace_back(DataFromBuffer<int>(buffers, sizeSplits->buffer(), i));
            if ( sizesToInfer > 1 )
            {
                valid = false;
            }
        }
    }
    if ( !valid )
    {
        std::string constraint = "Only one size is allowed to be inferred";
        std::string extra = fmt::format("Input size_splits={}", Shape(vec.data(), sizeSplitsElems).ToString());
        throw InvalidTfLiteException(constraint, extra, op, subgraph, builtinOperator);
    }
}

void ConstraintAxisValid(const Operator &op, const SubGraph &subgraph, const BuiltinOperator &builtinOperator, BufferOffsetRef)
{
    auto ofm = TensorFromUsage(regor::TensorUsage::OFM, op, builtinOperator, *subgraph.tensors());
    int64_t ofmRank = CheckedPtr(ofm->shape())->size();

    int64_t axis = CheckedPtr(op.builtin_options_as_ConcatenationOptions())->axis();

    if ( axis <= -ofmRank || axis > ofmRank || ofmRank < 0 )
    {
        std::string constraint = "Axis must be in the interval [-rank(OFM),rank(OFM))";
        std::string extra = fmt::format("OFM rank={}, Attribute axis={}", ofmRank, axis);
        throw InvalidTfLiteException(constraint, extra, op, subgraph, builtinOperator);
    }
}

void ConstraintMatchingDimensionality(const Operator &op, const SubGraph &subgraph, const BuiltinOperator &builtinOperator, BufferOffsetRef)
{
    auto ofm = TensorFromUsage(regor::TensorUsage::OFM, op, builtinOperator, *subgraph.tensors());
    unsigned int ofmRank = CheckedPtr(ofm->shape())->size();

    for ( flatbuffers::uoffset_t i = 0; i < CheckedPtr(op.inputs())->size(); i++ )
    {
        auto ifm = TensorFromUsage(
            regor::MakeTensorUsage(regor::TensorUsage::IFM, int(i)), op, builtinOperator, *subgraph.tensors());
        unsigned int ifmRank = CheckedPtr(ifm->shape())->size();
        if ( ifmRank != ofmRank )
        {
            std::string constraint = "All Input ranks must match the OFM rank";
            std::string extra = fmt::format("Found rank mismatch: OFM rank={}, IFM rank={}", ofmRank, ifmRank);
            throw InvalidTfLiteException(constraint, extra, op, subgraph, builtinOperator);
        }
    }
}

void ConstraintValidDimensions(const Operator &op, const SubGraph &subgraph, const BuiltinOperator &builtinOperator, BufferOffsetRef)
{
    auto ofm = TensorFromUsage(regor::TensorUsage::OFM, op, builtinOperator, *subgraph.tensors());
    auto ofmShape = ShapeFromTens(ofm);

    int ofmRank = ofmShape.Size();
    int32_t axis = CheckedPtr(op.builtin_options_as_ConcatenationOptions())->axis();

    axis = CheckedAdd(axis, axis < 0 ? ofmRank : 0);

    int ifmAxisDimAcc = 0;
    for ( const auto input_index : *CheckedPtr(op.inputs()) )
    {
        auto index = BoundsCheckedIndex(input_index, *subgraph.tensors());
        auto input = (*subgraph.tensors())[index];
        auto inputShape = ShapeFromTens(input);
        for ( int dim = 0; dim < inputShape.Size(); dim++ )
        {
            if ( dim != axis )
            {
                if ( inputShape[dim] != ofmShape[dim] )
                {
                    std::string constraint = "All Input dimensions must match OFM dimension in all axes except the one defined by the axis attribute";
                    std::string extra = fmt::format(
                        "Found mismatch: IFM shape={}, OFM shape={}", inputShape.ToString(), ofmShape.ToString());
                    throw InvalidTfLiteException(constraint, extra, op, subgraph, builtinOperator);
                }
            }
            else
            {
                ifmAxisDimAcc = CheckedAdd(ifmAxisDimAcc, inputShape[axis]);
            }
        }
    }
    if ( ifmAxisDimAcc != ofmShape[axis] )
    {
        std::string constraint = "All Input dimensions must match OFM dimension in all axes except the one defined by the axis attribute";
        std::string extra = fmt::format("Found mismatch: sum of IFM dim size in concat axis={}, OFM concat axis dim size={}",
            ifmAxisDimAcc, ofmShape[axis]);
        throw InvalidTfLiteException(constraint, extra, op, subgraph, builtinOperator);
    }
}


void ConstraintStridedsliceInputCount(const Operator &op, const SubGraph &subgraph, const BuiltinOperator &builtinOperator, BufferOffsetRef)
{
    auto inputs = CheckedPtr(op.inputs());

    if ( inputs->size() != 4 )
    {
        std::string constraint = "Must have 4 input tensors";
        std::string extra = fmt::format("Number of inputs={}", inputs->size());
        throw InvalidTfLiteException(constraint, extra, op, subgraph, builtinOperator);
    }
}

void ConstraintPadInputCount(const Operator &op, const SubGraph &subgraph, const BuiltinOperator &builtinOperator, BufferOffsetRef)
{
    auto inputs = CheckedPtr(op.inputs());

    if ( inputs->size() != 2 )
    {
        std::string constraint = "Must have 2 input tensors";
        std::string extra = fmt::format("Number of inputs={}", inputs->size());
        throw InvalidTfLiteException(constraint, extra, op, subgraph, builtinOperator);
    }
}

void ConstraintPadOutputShape(const Operator &op, const SubGraph &subgraph, const BuiltinOperator &builtinOperator, BufferOffsetRef buffers)
{
    auto ifm = TensorFromUsage(regor::TensorUsage::IFM, op, builtinOperator, *subgraph.tensors());
    auto ofm = TensorFromUsage(regor::TensorUsage::OFM, op, builtinOperator, *subgraph.tensors());
    auto ifmShape = ShapeFromTens(ifm);
    auto ofmShape = ShapeFromTens(ofm);

    auto padTensor = TensorFromUsage(regor::TensorUsage::Params, op, builtinOperator, *subgraph.tensors());

    for ( unsigned int dim = 0; static_cast<int>(dim) < ifmShape.Size(); dim++ )
    {
        int paddedOutDim;
        if ( padTensor->type() == TensorType::INT32 )
        {
            paddedOutDim = CheckedAdd(CheckedAdd(ifmShape[dim], DataFromBuffer<int>(buffers, padTensor->buffer(), dim * 2)),
                DataFromBuffer<int>(buffers, padTensor->buffer(), dim * 2 + 1));
        }
        else
        {
            paddedOutDim = CheckedAdd(CheckedAdd(ifmShape[dim], DataFromBuffer<int64_t>(buffers, padTensor->buffer(), dim * 2)),
                DataFromBuffer<int64_t>(buffers, padTensor->buffer(), dim * 2 + 1));
        }

        if ( paddedOutDim != ofmShape[dim] )
        {
            std::string constraint = "Shape of OFM must equal the IFM shape plus padding";
            std::string extra = fmt::format("Found mismatch for dim={}, padded output size in dim={}, OFM size in dim={}",
                dim, paddedOutDim, ofmShape[dim]);
            throw InvalidTfLiteException(constraint, extra, op, subgraph, builtinOperator);
        }
    }
}

void ConstraintMatchingInputsTypes(const Operator &op, const SubGraph &subgraph, const BuiltinOperator &builtinOperator, BufferOffsetRef)
{
    auto ifm = TensorFromUsage(regor::TensorUsage::IFM, op, builtinOperator, *subgraph.tensors());
    auto ifm2 = TensorFromUsage(MakeTensorUsage(regor::TensorUsage::IFM, 1), op, builtinOperator, *subgraph.tensors());
    if ( ifm->type() != ifm2->type() )
    {
        std::string constraint = "Both Input data types must match";
        std::string extra = fmt::format(
            "Op has ifm_dtype={} and ifm2_dtype={}", EnumNameTensorType(ifm2->type()), EnumNameTensorType(ifm->type()));
        throw InvalidTfLiteException(constraint, extra, op, subgraph, builtinOperator);
    }
}

void ConstraintMatchingSigned(const Operator &op, const SubGraph &subgraph, const BuiltinOperator &builtinOperator, BufferOffsetRef)
{
    auto ifm = TensorFromUsage(regor::TensorUsage::IFM, op, builtinOperator, *subgraph.tensors());
    auto ofm = TensorFromUsage(regor::TensorUsage::OFM, op, builtinOperator, *subgraph.tensors());
    if ( IsInt(ifm->type()) && !IsInt(ofm->type()) )
    {
        std::string constraint = "For IFM that are signed, OFM must also be signed";
        std::string extra = fmt::format(
            "IFM type={} and OFM type={}", EnumNameTensorType(ifm->type()), EnumNameTensorType(ofm->type()));
        throw InvalidTfLiteException(constraint, extra, op, subgraph, builtinOperator);
    }
}

void ConstraintUnsignedValid(const Operator &op, const SubGraph &subgraph, const BuiltinOperator &builtinOperator, BufferOffsetRef)
{
    auto ifm = TensorFromUsage(regor::TensorUsage::IFM, op, builtinOperator, *subgraph.tensors());
    auto ofm = TensorFromUsage(regor::TensorUsage::OFM, op, builtinOperator, *subgraph.tensors());

    if ( IsUint(ifm->type()) && !(ifm->type() == ofm->type() || ofm->type() == TensorType::INT32) )
    {
        std::string constraint = "For IFM that are unsigned, OFM must either be the same type or int32";
        std::string extra = fmt::format(
            "IFM type={} and OFM type={}", EnumNameTensorType(ifm->type()), EnumNameTensorType(ofm->type()));
        throw InvalidTfLiteException(constraint, extra, op, subgraph, builtinOperator);
    }
}

void ConstraintInputSigned(const Operator &op, const SubGraph &subgraph, const BuiltinOperator &builtinOperator, BufferOffsetRef)
{
    auto ifm = TensorFromUsage(regor::TensorUsage::IFM, op, builtinOperator, *subgraph.tensors());

    if ( !(ifm->type() == TensorType::INT8 || ifm->type() == TensorType::INT16) )
    {
        std::string constraint = "IFM must be INT8 or INT16";
        std::string extra = fmt::format("IFM type={}", EnumNameTensorType(ifm->type()));
        throw InvalidTfLiteException(constraint, extra, op, subgraph, builtinOperator);
    }
}

void ConstraintInput8bit(const Operator &op, const SubGraph &subgraph, const BuiltinOperator &builtinOperator, BufferOffsetRef)
{
    auto ifm = TensorFromUsage(regor::TensorUsage::IFM, op, builtinOperator, *subgraph.tensors());

    if ( !(ifm->type() == TensorType::INT8 || ifm->type() == TensorType::UINT8) )
    {
        std::string constraint = "IFM has to be 8bit";
        std::string extra = fmt::format("IFM type={}, OFM type={}", EnumNameTensorType(ifm->type()));
        throw InvalidTfLiteException(constraint, extra, op, subgraph, builtinOperator);
    }
}

void ConstraintParams32bit(const Operator &op, const SubGraph &subgraph, const BuiltinOperator &builtinOperator, BufferOffsetRef)
{
    auto params = TensorFromUsage(regor::TensorUsage::Params, op, builtinOperator, *subgraph.tensors());

    if ( !(params->type() == TensorType::INT32 || params->type() == TensorType::UINT32) )
    {
        std::string constraint = "Params must be INT32 or UINT32";
        std::string extra = fmt::format("Params type={}", EnumNameTensorType(params->type()));
        throw InvalidTfLiteException(constraint, extra, op, subgraph, builtinOperator);
    }
}

void ConstraintArgmaxOutput(const Operator &op, const SubGraph &subgraph, const BuiltinOperator &builtinOperator, BufferOffsetRef)
{
    auto ofm = TensorFromUsage(regor::TensorUsage::OFM, op, builtinOperator, *subgraph.tensors());

    if ( !(ofm->type() == TensorType::INT32 || ofm->type() == TensorType::INT64) )
    {
        std::string constraint = "For IFM that are signed, OFM must also be signed";
        std::string extra = fmt::format("IFM type={}, OFM type={}", EnumNameTensorType(ofm->type()));
        throw InvalidTfLiteException(constraint, extra, op, subgraph, builtinOperator);
    }
}

void ConstraintGatherIndicesInput(const Operator &op, const SubGraph &subgraph, const BuiltinOperator &builtinOperator, BufferOffsetRef)
{
    auto ifm2 = TensorFromUsage(regor::TensorUsage::IFM1, op, builtinOperator, *subgraph.tensors());

    if ( !(ifm2->type() == TensorType::INT16 || ifm2->type() == TensorType::INT32 || ifm2->type() == TensorType::INT64) )
    {
        std::string constraint = "IFM2 must be INT16, INT32 or INT64";
        std::string extra = fmt::format("IFM2 type={}", EnumNameTensorType(ifm2->type()));
        throw InvalidTfLiteException(constraint, extra, op, subgraph, builtinOperator);
    }
}

void ConstraintTransposeParamsInput(const Operator &op, const SubGraph &subgraph, const BuiltinOperator &builtinOperator, BufferOffsetRef)
{
    auto ifm = TensorFromUsage(regor::TensorUsage::IFM, op, builtinOperator, *subgraph.tensors());
    auto params = TensorFromUsage(regor::TensorUsage::Params, op, builtinOperator, *subgraph.tensors());

    auto ifmShape = ShapeFromTens(ifm);
    auto paramsShape = ShapeFromTens(params);

    if ( ifmShape.Size() != paramsShape.Depth() )
    {
        std::string constraint = "Permutation vector size must match IFM rank";
        std::string extra = fmt::format("Params shape={}, IFM shape={}", paramsShape.ToString(), ifmShape.ToString());
        throw InvalidTfLiteException(constraint, extra, op, subgraph, builtinOperator);
    }
}

void ConstraintMatchingEitherShapes(const Operator &op, const SubGraph &subgraph, const BuiltinOperator &builtinOperator, BufferOffsetRef)
{
    auto ifm = TensorFromUsage(regor::TensorUsage::IFM, op, builtinOperator, *subgraph.tensors());
    auto ofm = TensorFromUsage(regor::TensorUsage::OFM, op, builtinOperator, *subgraph.tensors());
    auto ifmShape = ShapeFromTens(ifm);
    auto ofmShape = ShapeFromTens(ofm);
    bool valid = ifmShape == ofmShape;
    Shape ifm2Shape = Shape();
    auto ifm2 = TensorFromUsage(MakeTensorUsage(regor::TensorUsage::IFM, 1), op, builtinOperator, *subgraph.tensors());

    if ( ifm2 && !valid )
    {
        ifm2Shape = ShapeFromTens(ifm2);
        bool isBroadcastable = true;
        for ( auto i = 0; i < std::min(ifmShape.Size(), ifm2Shape.Size()); i++ )
        {
            isBroadcastable = isBroadcastable && (ifmShape[i] == ifm2Shape[i] || ifmShape[i] == 1 || ifm2Shape[i] == 1);
        }
        valid = ifm2Shape == ofmShape || (isBroadcastable && Shape::Max(ifmShape, ifm2Shape) == ofmShape);
    }
    if ( !valid )
    {
        std::string constraint = "At least one Input's shape must match the OFM's shape, or the union of the Input shapes must equal the OFM shape if Inputs are broadcast-able";
        std::string extra = fmt::format("IFM shape={}, IFM2 shape={} and OFM shape={}", ifmShape.ToString(),
            ifm2Shape.ToString(), ofmShape.ToString());
        throw InvalidTfLiteException(constraint, extra, op, subgraph, builtinOperator);
    }
}

void ConstraintMatchingInOutQuant(const Operator &op, const SubGraph &subgraph, const BuiltinOperator &builtinOperator, BufferOffsetRef)
{
    auto ifm = TensorFromUsage(regor::TensorUsage::IFM, op, builtinOperator, *subgraph.tensors());
    auto ofm = TensorFromUsage(regor::TensorUsage::OFM, op, builtinOperator, *subgraph.tensors());
    if ( ifm->quantization() && ifm->quantization()->scale() && ofm->quantization() && ofm->quantization()->scale() )
    {
        float ifmScale = (*ifm->quantization()->scale())[0];
        float ofmScale = (*ofm->quantization()->scale())[0];
        if ( ifmScale != ofmScale )
        {
            std::string constraint = "Input and output quantisation must match";
            std::string extra = fmt::format("IFM scale={}, OFM shape={}", ifmScale, ofmScale);
            throw InvalidTfLiteException(constraint, extra, op, subgraph, builtinOperator);
        }
    }
}

void ConstraintKeepDimIfmOfm(const Operator &op, const SubGraph &subgraph, const BuiltinOperator &builtinOperator, BufferOffsetRef)
{
    if ( CheckedPtr(op.builtin_options_as_FullyConnectedOptions())->keep_num_dims() )
    {
        auto ifm = TensorFromUsage(regor::TensorUsage::IFM, op, builtinOperator, *subgraph.tensors());
        auto ofm = TensorFromUsage(regor::TensorUsage::OFM, op, builtinOperator, *subgraph.tensors());
        auto ifmShape = ShapeFromTens(ifm);
        auto ofmShape = ShapeFromTens(ofm);
        if ( ifmShape.Size() != ofmShape.Size() )
        {
            std::string constraint = "IFM and OFM ranks must match";
            std::string extra = fmt::format("IFM rank={}, OFM rank={}", ifmShape.Size(), ofmShape.Size());
            throw InvalidTfLiteException(constraint, extra, op, subgraph, builtinOperator);
        }
    }
}

void ConstraintMatchingInOutElements(const Operator &op, const SubGraph &subgraph, const BuiltinOperator &builtinOperator, BufferOffsetRef)
{
    auto ifm = TensorFromUsage(regor::TensorUsage::IFM, op, builtinOperator, *subgraph.tensors());
    auto ofm = TensorFromUsage(regor::TensorUsage::OFM, op, builtinOperator, *subgraph.tensors());
    auto ifmShape = ShapeFromTens(ifm);
    auto ofmShape = ShapeFromTens(ofm);

    if ( ifmShape.Elements() != ofmShape.Elements() )
    {
        std::string constraint = "Input and output number of elements must match";
        std::string extra = fmt::format("IFM shape={}, OFM shape={}", ifmShape.ToString(), ofmShape.ToString());
        throw InvalidTfLiteException(constraint, extra, op, subgraph, builtinOperator);
    }
}

void ConstraintLstmInputRank(const Operator &op, const SubGraph &subgraph, const BuiltinOperator &builtinOperator, BufferOffsetRef)
{
    auto ifm = TensorFromUsage(regor::TensorUsage::IFM, op, builtinOperator, *subgraph.tensors());
    auto ofm = TensorFromUsage(regor::TensorUsage::OFM, op, builtinOperator, *subgraph.tensors());
    auto ifmShape = ShapeFromTens(ifm);
    auto ofmShape = ShapeFromTens(ofm);
    if ( !(ifmShape.Size() == 3 && ofmShape.Size() == 3) )
    {
        std::string constraint = "IFM and OFM must have 3D shape";
        std::string extra = fmt::format("IFM rank={}, OFM rank={}", ifmShape.Size(), ofmShape.Size());
        throw InvalidTfLiteException(constraint, extra, op, subgraph, builtinOperator);
    }
}

void ConstraintLstmInputs(const Operator &op, const SubGraph &subgraph, const BuiltinOperator &builtinOperator, BufferOffsetRef)
{
    auto input = CheckedPtr(op.inputs());
    if ( input->size() != 24 )
    {
        std::string constraint = "Must have 24 input const tensors";
        std::string extra = fmt::format("Number of inputs={}", input->size());
        throw InvalidTfLiteException(constraint, extra, op, subgraph, builtinOperator);
    }
}

void ConstraintLstmIntermediates(const Operator &op, const SubGraph &subgraph, const BuiltinOperator &builtinOperator, BufferOffsetRef)
{
    auto intermediates = CheckedPtr(op.intermediates());
    if ( intermediates->size() != 5 )
    {
        std::string constraint = "Must have 5 intermediate tensors";
        std::string extra = fmt::format("Number of intermediates={}", intermediates->size());
        throw InvalidTfLiteException(constraint, extra, op, subgraph, builtinOperator);
    }
}
// TODO: Implement ConstraintLstmVariables when adding LSTM support
void ConstraintLstmVariables(const Operator &, const SubGraph &, const BuiltinOperator &, BufferOffsetRef)
{
}

regor::ordered_map<BuiltinOperator, OpCheckVec> GetSpecificConstraints()
{

    regor::ordered_map<BuiltinOperator, OpCheckVec> specificOpConstraints(50);
    // Conv checks
    for ( auto opType : convolutionOps )
    {
        specificOpConstraints[opType].emplace_back(&ConstraintConvGroupsIfmDepth);
        specificOpConstraints[opType].emplace_back(&ConstraintConvGroupsNumFilters);
    }

    // Pooling checks
    for ( auto opType : poolingOps )
    {
        specificOpConstraints[opType].emplace_back(&ConstraintMatchingInOutTypes);
    }
    // Concat specific checks
    for ( auto opType : concatOps )
    {
        specificOpConstraints[opType].emplace_back(&ConstraintAxisValid);
        specificOpConstraints[opType].emplace_back(&ConstraintMatchingDimensionality);
        specificOpConstraints[opType].emplace_back(&ConstraintValidDimensions);
    }

    // ElementWiseMainOps checks
    for ( auto opType : elemWiseMainOps )
    {
        specificOpConstraints[opType].emplace_back(&ConstraintMatchingEitherShapes);
    }
    // UnaryElemWiseMainOps specific checks
    for ( auto opType : unaryElemWiseMainOps )
    {
        specificOpConstraints[opType].emplace_back(&ConstraintMatchingInOutTypes);
    }
    // BinaryElemWiseMinMaxOps specific checks
    for ( auto opType : binaryElemWiseMinMaxOps )
    {
        specificOpConstraints[opType].emplace_back(&ConstraintMatchingInOutTypes);
    }
    // BinaryElemWiseAddMulSub specific checks
    for ( auto opType : binaryElemWiseAddMulSub )
    {
        specificOpConstraints[opType].emplace_back(&ConstraintMatchingInputsTypes);
        specificOpConstraints[opType].emplace_back(&ConstraintMatchingSigned);
        specificOpConstraints[opType].emplace_back(&ConstraintUnsignedValid);
    }

    // OpsReshapingDimensions: Reshape, Squeeze, and ExpandDims
    for ( auto opType : reshapeOps )
    {
        specificOpConstraints[opType].emplace_back(&ConstraintMatchingInOutQuant);
        specificOpConstraints[opType].emplace_back(&ConstraintMatchingInOutElements);
    }

    // DepthwiseSpecificChecks
    specificOpConstraints[BuiltinOperator::DEPTHWISE_CONV_2D].emplace_back(&ConstraintDepthwiseConvOfmDepth);

    // SoftmaxSpecificChecks
    specificOpConstraints[BuiltinOperator::SOFTMAX].emplace_back(&ConstraintMatchingShapes);
    specificOpConstraints[BuiltinOperator::SOFTMAX].emplace_back(&ConstraintSoftmaxInOutTypes);
    specificOpConstraints[BuiltinOperator::SOFTMAX].emplace_back(&ConstraintBetaValueRange);

    // SplitSpecificChecks
    specificOpConstraints[BuiltinOperator::SPLIT].emplace_back(&ConstraintSplitDim);
    specificOpConstraints[BuiltinOperator::SPLIT].emplace_back(&ConstraintSplitNumSplits);


    // SplitVSpecificChecks
    specificOpConstraints[BuiltinOperator::SPLIT_V].emplace_back(&ConstraintSplitvInferred);

    // StridedSliceSpecificChecks
    specificOpConstraints[BuiltinOperator::STRIDED_SLICE].emplace_back(&ConstraintStridedsliceInputCount);

    // FullyConnectedSpecificChecks
    specificOpConstraints[BuiltinOperator::FULLY_CONNECTED].emplace_back(&ConstraintKeepDimIfmOfm);

    // PadSpecificChecks
    specificOpConstraints[BuiltinOperator::PAD].emplace_back(&ConstraintPadInputCount);
    specificOpConstraints[BuiltinOperator::PAD].emplace_back(&ConstraintPadOutputShape);

    // HardSwishSpecificChecks
    specificOpConstraints[BuiltinOperator::HARD_SWISH].emplace_back(&ConstraintInput8bit);
    specificOpConstraints[BuiltinOperator::HARD_SWISH].emplace_back(&ConstraintMatchingInOutTypes);

    // ArgMaxSpecificChecks
    specificOpConstraints[BuiltinOperator::ARGMAX].emplace_back(&ConstraintInput8bit);
    specificOpConstraints[BuiltinOperator::ARGMAX].emplace_back(&ConstraintArgmaxOutput);

    // UnidirectionalSequenceLstmSpecificChecks
    specificOpConstraints[BuiltinOperator::UNIDIRECTIONAL_SEQUENCE_LSTM].emplace_back(&ConstraintInputSigned);
    specificOpConstraints[BuiltinOperator::UNIDIRECTIONAL_SEQUENCE_LSTM].emplace_back(&ConstraintMatchingInOutTypes);
    specificOpConstraints[BuiltinOperator::UNIDIRECTIONAL_SEQUENCE_LSTM].emplace_back(&ConstraintLstmInputRank);
    specificOpConstraints[BuiltinOperator::UNIDIRECTIONAL_SEQUENCE_LSTM].emplace_back(&ConstraintLstmInputs);
    specificOpConstraints[BuiltinOperator::UNIDIRECTIONAL_SEQUENCE_LSTM].emplace_back(&ConstraintLstmIntermediates);
    specificOpConstraints[BuiltinOperator::UNIDIRECTIONAL_SEQUENCE_LSTM].emplace_back(&ConstraintLstmVariables);

    // GatherSpecificChecks
    specificOpConstraints[BuiltinOperator::GATHER].emplace_back(&ConstraintGatherIndicesInput);
    specificOpConstraints[BuiltinOperator::GATHER].emplace_back(&ConstraintMatchingInOutTypes);

    // TransposeSpecificChecks
    specificOpConstraints[BuiltinOperator::TRANSPOSE].emplace_back(&ConstraintParams32bit);
    specificOpConstraints[BuiltinOperator::TRANSPOSE].emplace_back(&ConstraintTransposeParamsInput);

    return specificOpConstraints;
}

OpCheckVec genericOpConstraints = {&ConstraintQuantScaleInf};

std::vector<std::function<void(const Model &model)>> modelConstraints{&ConstraintEmptyConstTensors, &ConstraintTensQuantScale};
}  // namespace

void TFLiteModelSemantics::Check()
{
    auto buffers = CheckedPtr(m_model->buffers());
    auto specificOpConstraints = GetSpecificConstraints();

    try
    {
        for ( const auto &constraintFunc : modelConstraints )
        {
            constraintFunc(*m_model);
        }
        for ( const auto &subgraph : *m_model->subgraphs() )
        {
            for ( const auto &op : *subgraph->operators() )
            {
                auto index = BoundsCheckedIndex(op->opcode_index(), *CheckedPtr(m_model->operator_codes()));
                auto opcode = (*CheckedPtr(m_model->operator_codes()))[index];
                auto builtinOperator =
                    opcode->builtin_code() == BuiltinOperator(0) ?
                        BuiltinOperator(opcode->deprecated_builtin_code()) :
                        opcode->builtin_code();

                for ( const auto &constraintFunc : genericOpConstraints )
                {
                    constraintFunc(*op, *subgraph, builtinOperator, *buffers);
                }

                for ( const auto &constraintFunc : specificOpConstraints[builtinOperator] )
                {
                    constraintFunc(*op, *subgraph, builtinOperator, *buffers);
                }
            }
        }
    }
    catch ( const InvalidTfLiteException &e )
    {
        std::string errorMessage = fmt::format(
            "Error: {0} '{1}' Does not have valid TFLite Semantics.\n"
            " - {2}\n"
            "   {3}\n"
            "Failed to parse TFLite file\n",
            e.BuiltinOperatorName(), e.TensorName(), e.Constraint(), e.Extra());

        throw std::runtime_error(errorMessage);
    }
}

}  // namespace tflite
