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

#include "common/common.hpp"
#include "common/logging.hpp"

#include "randomize.hpp"
#include "tflite/tflite_mapping.hpp"
#include "tflite/tflite_reader.hpp"
#include "tflite/tflite_schema_generated.hpp"
#include "tflite/tflite_supported_operators.hpp"
#include "tflite/tflite_writer.hpp"
#include "util.hpp"

#include <flatbuffers/minireflect.h>
#include <catch_all.hpp>
#include <iostream>
#include <limits>
#include <numeric>

using namespace regor;

namespace
{
// Generate random scalar integer in the range [1, 127]
template<typename T>
T GenerateRandomScalar()
{
    T scalar;
    randomize(scalar, T(1), T(127));
    return scalar;
}

// Generate random scalar boolean
template<>
bool GenerateRandomScalar()
{
    bool scalar;
    randomize(scalar, false, true);
    return scalar;
}
}  // namespace

// Generate a random enum from a TypeTable
template<typename E>
static E GenerateRandomEnum(const flatbuffers::TypeTable *typeTable)
{
    assert(typeTable->st == flatbuffers::SequenceType::ST_ENUM);
    assert(typeTable->num_elems > 0);

    return E(urandom_range(0, typeTable->num_elems - 1u));
}

// Generate a random string
static flatbuffers::Offset<> GenerateRandomString(flatbuffers::FlatBufferBuilder &fbb)
{
    return fbb.CreateString("string-" + std::to_string(urandom())).Union();
}

// Generate a random enum vector from a TypeTable
template<typename E>
static flatbuffers::Offset<>
GenerateRandomEnumVector(flatbuffers::FlatBufferBuilder &fbb, const size_t count, const flatbuffers::TypeTable *typeTable)
{
    assert(typeTable->st == flatbuffers::SequenceType::ST_ENUM);

    std::vector<E> v(count);
    for ( size_t i = 0; i < count; i++ )
    {
        v[i] = GenerateRandomEnum<E>(typeTable);
    }
    return fbb.CreateVector(v).Union();
}

// Generate a random vector from a TypeTable
template<typename E>
static flatbuffers::Offset<> GenerateRandomScalarVector(flatbuffers::FlatBufferBuilder &fbb, const size_t count)
{
    std::vector<E> v(count);
    for ( size_t i = 0; i < count; i++ )
    {
        v[i] = GenerateRandomScalar<E>();
    }
    return fbb.CreateVector(v).Union();
}

// Generate a random string vector
static flatbuffers::Offset<> GenerateRandomStringVector(flatbuffers::FlatBufferBuilder &fbb, const size_t count)
{
    std::vector<std::string> v(count);
    for ( size_t i = 0; i < count; i++ )
    {
        v[i] = "string-" + std::to_string(i);
    }
    return fbb.CreateVectorOfStrings(v).Union();
}

// Generate a randomized union member from a TypeTable
static flatbuffers::Offset<> GenerateRandomUnionMember(flatbuffers::FlatBufferBuilder &fbb, const flatbuffers::TypeTable *typeTable)
{
    // This function generates a flatbuffer Table by iterating over the type table two times. First time to generate
    // offsets to all non-scalar, repeating types and strings, then a second time to add all offsets and scalars. The
    // function is simplified in the sense that it can't generate all possible tables, but instead has enough
    // functionality to generate the tables that are used by TFLite's builtin_options (builtin_options2).

    // Can only generate tables
    assert(typeTable->st == flatbuffers::SequenceType::ST_TABLE);

    std::unordered_map<flatbuffers::voffset_t, flatbuffers::Offset<>> fieldToOffset;

    // Iterate over all types and generate offsets for non-scalar items
    for ( flatbuffers::voffset_t i = 0; i < typeTable->num_elems; i++ )
    {
        const auto name = typeTable->names[i];
        const auto type = flatbuffers::ElementaryType(typeTable->type_codes[i].base_type);
        const auto isRepeating = typeTable->type_codes[i].is_repeating != 0;
        const auto sequenceRef = typeTable->type_codes[i].sequence_ref;
        const auto field = flatbuffers::FieldIndexToOffset(i);

        // Can not generate unions
        assert(type != flatbuffers::ET_UTYPE);

        if ( isRepeating )
        {
            const size_t count = urandom_range(1, 5);

            if ( sequenceRef >= 0 )
            {
                LOG_TRACE1("Generating vector {} (vector of {})\n", name, flatbuffers::ElementaryTypeNames()[type]);

                // Can only generate ENUMs
                const auto *sequenceTypeTable = typeTable->type_refs[sequenceRef]();
                assert(sequenceTypeTable);
                assert(sequenceTypeTable->st == flatbuffers::SequenceType::ST_ENUM);

                if ( type == flatbuffers::ET_CHAR )
                    fieldToOffset[field] = GenerateRandomEnumVector<int8_t>(fbb, count, sequenceTypeTable);
                else if ( type == flatbuffers::ET_INT )
                    fieldToOffset[field] = GenerateRandomEnumVector<int32_t>(fbb, count, sequenceTypeTable);
                else if ( type == flatbuffers::ET_UINT )
                    fieldToOffset[field] = GenerateRandomEnumVector<uint32_t>(fbb, count, sequenceTypeTable);
                else
                    // Unsupported type (probably ET_SEQUENCE)
                    assert(false && "Unsupported elementary type");
            }
            else
            {
                LOG_TRACE1("Generating vector {} (vector of {})\n", name, flatbuffers::ElementaryTypeNames()[type]);

                // Vector of scalars/strings
                if ( type == flatbuffers::ET_BOOL ) fieldToOffset[field] = GenerateRandomScalarVector<bool>(fbb, count);
                else if ( type == flatbuffers::ET_CHAR )
                    fieldToOffset[field] = GenerateRandomScalarVector<int8_t>(fbb, count);
                else if ( type == flatbuffers::ET_UCHAR )
                    fieldToOffset[field] = GenerateRandomScalarVector<uint8_t>(fbb, count);
                else if ( type == flatbuffers::ET_SHORT )
                    fieldToOffset[field] = GenerateRandomScalarVector<int16_t>(fbb, count);
                else if ( type == flatbuffers::ET_USHORT )
                    fieldToOffset[field] = GenerateRandomScalarVector<uint16_t>(fbb, count);
                else if ( type == flatbuffers::ET_INT )
                    fieldToOffset[field] = GenerateRandomScalarVector<int32_t>(fbb, count);
                else if ( type == flatbuffers::ET_UINT )
                    fieldToOffset[field] = GenerateRandomScalarVector<uint32_t>(fbb, count);
                else if ( type == flatbuffers::ET_LONG )
                    fieldToOffset[field] = GenerateRandomScalarVector<int64_t>(fbb, count);
                else if ( type == flatbuffers::ET_ULONG )
                    fieldToOffset[field] = GenerateRandomScalarVector<uint64_t>(fbb, count);
                else if ( type == flatbuffers::ET_FLOAT )
                    fieldToOffset[field] = GenerateRandomScalarVector<float>(fbb, count);
                else if ( type == flatbuffers::ET_DOUBLE )
                    fieldToOffset[field] = GenerateRandomScalarVector<double>(fbb, count);
                else if ( type == flatbuffers::ET_STRING )
                    fieldToOffset[field] = GenerateRandomStringVector(fbb, count);
                else assert(false && "Unsupported elementary type");
            }
        }
        else
        {
            LOG_TRACE1("Generating string {} ({})\n", name, flatbuffers::ElementaryTypeNames()[type]);

            if ( type == flatbuffers::ET_STRING ) fieldToOffset[field] = GenerateRandomString(fbb);
        }
    }

    const auto tableOffset = fbb.StartTable();

    // Iterate over all types and add offsets and scalar types
    for ( flatbuffers::voffset_t i = 0; i < typeTable->num_elems; i++ )
    {
        const auto name = typeTable->names[i];
        const auto type = flatbuffers::ElementaryType(typeTable->type_codes[i].base_type);
        const auto isRepeating = typeTable->type_codes[i].is_repeating != 0;
        const auto sequenceRef = typeTable->type_codes[i].sequence_ref;
        const auto field = flatbuffers::FieldIndexToOffset(i);

        if ( fieldToOffset.count(field) == 0 )
        {
            // At this point it's too late for repeating types
            assert(!isRepeating);

            if ( sequenceRef >= 0 )
            {
                LOG_TRACE1("Generating and adding sequence {} ({})\n", name, flatbuffers::ElementaryTypeNames()[type]);

                const auto *sequenceTypeTable = typeTable->type_refs[sequenceRef]();
                assert(sequenceTypeTable);
                assert(sequenceTypeTable->st == flatbuffers::SequenceType::ST_ENUM);

                if ( type == flatbuffers::ET_CHAR )
                    fbb.AddElement(field, GenerateRandomEnum<uint8_t>(sequenceTypeTable));
                else if ( type == flatbuffers::ET_INT )
                    fbb.AddElement(field, GenerateRandomEnum<int32_t>(sequenceTypeTable));
                else if ( type == flatbuffers::ET_UINT )
                    fbb.AddElement(field, GenerateRandomEnum<uint32_t>(sequenceTypeTable));
                else
                    // Unsupported type (probably ET_SEQUENCE)
                    assert(false && "Unsupported elementary type");
            }
            else
            {
                LOG_TRACE1("Generating and adding scalar {} ({})\n", name, flatbuffers::ElementaryTypeNames()[type]);

                // Scalar
                if ( type == flatbuffers::ET_BOOL ) fbb.AddElement(field, GenerateRandomScalar<bool>());
                else if ( type == flatbuffers::ET_CHAR ) fbb.AddElement(field, GenerateRandomScalar<int8_t>());
                else if ( type == flatbuffers::ET_UCHAR ) fbb.AddElement(field, GenerateRandomScalar<uint8_t>());
                else if ( type == flatbuffers::ET_SHORT ) fbb.AddElement(field, GenerateRandomScalar<int16_t>());
                else if ( type == flatbuffers::ET_USHORT ) fbb.AddElement(field, GenerateRandomScalar<uint16_t>());
                else if ( type == flatbuffers::ET_INT ) fbb.AddElement(field, GenerateRandomScalar<int32_t>());
                else if ( type == flatbuffers::ET_UINT ) fbb.AddElement(field, GenerateRandomScalar<uint32_t>());
                else if ( type == flatbuffers::ET_LONG ) fbb.AddElement(field, GenerateRandomScalar<int64_t>());
                else if ( type == flatbuffers::ET_ULONG ) fbb.AddElement(field, GenerateRandomScalar<uint64_t>());
                else if ( type == flatbuffers::ET_FLOAT ) fbb.AddElement(field, GenerateRandomScalar<float>());
                else if ( type == flatbuffers::ET_DOUBLE ) fbb.AddElement(field, GenerateRandomScalar<double>());
                else assert(false && "Unsupported elementary type");
            }
        }
        else
        {
            LOG_TRACE1("Adding offset {} ({})\n", name, flatbuffers::ElementaryTypeNames()[type]);

            fbb.AddOffset(field, fieldToOffset[field]);
        }
    }

    return fbb.EndTable(tableOffset);
}

// Compare two lists of strings
static void CompareItems(const std::vector<std::string> &actual, const std::vector<std::string> &expected)
{
    static const std::array<std::string, 1> ignoredPrefixes = {
        "/model/description",
    };

    // Log all items
    for ( auto &item : actual )
        LOG_TRACE2("Actual: {}\n", item);
    for ( auto &item : expected )
        LOG_TRACE2("Expected: {}\n", item);

    // Check each item
    auto actualIter = actual.begin();
    auto expectedIter = expected.begin();
    while ( actualIter != actual.end() && expectedIter != expected.end() )
    {
        bool ignoreActual = false;
        bool ignoreExpected = false;
        for ( auto &prefix : ignoredPrefixes )
        {
            ignoreActual = ignoreActual || (actualIter->rfind(prefix, 0) != std::string::npos);
            ignoreExpected = ignoreExpected || (expectedIter->rfind(prefix, 0) != std::string::npos);
        }
        if ( ignoreActual )
        {
            LOG_TRACE1("Ignoring actual item: {}\n", *actualIter);
            actualIter++;
        }
        if ( ignoreExpected )
        {
            LOG_TRACE1("Ignoring expected item: {}\n", *expectedIter);
            expectedIter++;
        }
        if ( !ignoreActual && !ignoreExpected )
        {
            auto &a = *actualIter;
            auto &b = *expectedIter;
            REQUIRE(a == b);
            actualIter++;
            expectedIter++;
        }
    }

    // Make sure we have consumed all items
    const bool consumedAllActual = (actualIter == actual.end());
    const bool consumedAllExpected = (expectedIter == expected.end());

    // Log all items left
    for ( ; actualIter != actual.end(); actualIter++ )
        LOG_TRACE2("Unconsumed actual: {}\n", *actualIter);
    for ( ; expectedIter != expected.end(); expectedIter++ )
        LOG_TRACE2("Unconsumed expected: {}\n", *expectedIter);

    REQUIRE(consumedAllActual);
    REQUIRE(consumedAllExpected);
}

// Visitor that converts a flatbuffer to a vector of strings, one string per elementary item
class ToItemListVisitor : public flatbuffers::IterationVisitor
{
public:
    void StartSequence() { sequences.emplace_back(); }
    void EndSequence() { sequences.pop_back(); }
    void StartVector() { vectors.push_back(currentField); }
    void EndVector() { vectors.pop_back(); }

    void Field(size_t, size_t, flatbuffers::ElementaryType type, bool, const flatbuffers::TypeTable *, const char *name, const uint8_t *)
    {
        assert(name);

        currentField = name;
        currentFieldType = flatbuffers::ElementaryTypeNames()[type];
        sequences.back() = currentField;
    }

    void Element(size_t i, flatbuffers::ElementaryType type, const flatbuffers::TypeTable *, const uint8_t *)
    {
        currentFieldType = flatbuffers::ElementaryTypeNames()[type];
        sequences.back() = vectors.back() + "[" + std::to_string(i) + "]";
    }

    template<typename T>
    void HandleItem(T value, const char *name)
    {
        // Build the item path
        auto item = std::accumulate(sequences.begin(), sequences.end(), std::string(),
            [](const std::string &a, const std::string &b) { return a + "/" + b; });

        // Append item value (and type if available)
        if ( name ) item += fmt::format(" = {}", name);
        else item += fmt::format(" = {} ({})", value, currentFieldType);

        items.push_back(std::move(item));
    }

    void UType(uint8_t x, const char *name) { HandleItem(x, name); }
    void Bool(bool x) { HandleItem(x, nullptr); }
    void Char(int8_t x, const char *name) { HandleItem(x, name); }
    void UChar(uint8_t x, const char *name) { HandleItem(x, name); }
    void Short(int16_t x, const char *name) { HandleItem(x, name); }
    void UShort(uint16_t x, const char *name) { HandleItem(x, name); }
    void Int(int32_t x, const char *name) { HandleItem(x, name); }
    void UInt(uint32_t x, const char *name) { HandleItem(x, name); }
    void Long(int64_t x) { HandleItem(x, nullptr); }
    void ULong(uint64_t x) { HandleItem(x, nullptr); }
    void Float(float x) { HandleItem(x, nullptr); }
    void Double(double x) { HandleItem(x, nullptr); }
    void String(const flatbuffers::String *str) { HandleItem(str->string_view(), nullptr); }
    void Unknown(const uint8_t *) { HandleItem("?", "UNKNOWN"); }

    std::vector<std::string> items;

private:
    std::deque<std::string> sequences{"model"};
    std::deque<std::string> vectors;
    std::string currentField;
    std::string currentFieldType;
};

// Mark one op as passthrough and remove any associated activation function
static void MarkAsPassthrough(Operation *op)
{
    if ( TfLiteMapping::CanFuseActivationFunction(op) )
    {
        const auto ofm = op->OFM();
        assert(ofm);
        assert(ofm->IsSinglePath());
        const auto activation = ofm->Readers().front();
        const auto actOfmConn = activation->Output(TensorUsage::OFM);
        assert(actOfmConn);

        // Bypass and remove activation op
        op->CopyOutput(TensorUsage::OFM, *actOfmConn);
        activation->SetPassthroughOp();
        activation->Disconnect();
    }

    op->SetPassthroughOp();
}

TEST_CASE("passthrough")
{
    // This tests the passthrough functionality of Regor. Passthrough refers to outputting an unsupported operator
    // unchanged so that it can be executed by the TFLite or TFLite Micro framework on the CPU.
    //
    // This test ensures that functionality as follows:
    //
    // 1. Generate a TFLite network as a flatbuffer.
    // 2. Pass the flatbuffer to tflite_reader to obtain the GraphIR.
    // 3. Pass the GraphIR to tflite_writer to generate a flatbuffer again.
    // 4. Compare it with the flatbuffer from step 1. The contents should be identical.
    //
    // This is done for all operators in TFLite (together with their BuiltinOptions or BuiltinOptions2).

    DisableLogging disableLogging;

    // Use from_range to generate values from the array
    const tflite::BuiltinOperator op = GENERATE(
        from_range(std::begin(tflite::EnumValuesBuiltinOperator()), std::end(tflite::EnumValuesBuiltinOperator())));
    LOG_TRACE1("Testing operator {}\n", tflite::EnumNameBuiltinOperator(op));

    flatbuffers::FlatBufferBuilder fbb;

    // Per model
    std::vector<flatbuffers::Offset<tflite::OperatorCode>> operatorCodes;
    std::vector<flatbuffers::Offset<tflite::SubGraph>> subgraphs;
    std::vector<flatbuffers::Offset<tflite::Buffer>> buffers;

    // Per subgraph
    std::vector<flatbuffers::Offset<tflite::Operator>> operations;
    std::vector<flatbuffers::Offset<tflite::Tensor>> tensors;
    std::vector<flatbuffers::Offset<tflite::Metadata>> metadata;

    {
        // Generate 1 operator code
        const int8_t deprecatedBuiltinCode = std::min<int8_t>(int8_t(op), 127);
        const char *customCode = nullptr;
        const int32_t version = urandom_range(1, 5);
        const tflite::BuiltinOperator builtinCode = op;
        operatorCodes.push_back(tflite::CreateOperatorCodeDirect(fbb, deprecatedBuiltinCode, customCode, version, builtinCode));
    }

    // Generate empty first buffer
    buffers.push_back(tflite::CreateBufferDirect(fbb));  // Buffer 0

    {
        // Generate tensor
        const std::vector<int32_t> shape = {1, 9, 9, 3};
        const tflite::TensorType type = tflite::TensorType::INT8;
        const int bufferIndex = 0;
        const std::string name = "ifm0";

        // Create QuantizationParameters
        const std::vector<float> min = random_vector<float>(3, 0, 127);
        const std::vector<float> max = random_vector<float>(3, 128, 255);
        const std::vector<float> scale = {0.0042};
        const std::vector<int64_t> zeroPoint = {3, 7, 11};
        const tflite::QuantizationDetails detailsType = tflite::QuantizationDetails::CustomQuantization;
        const std::vector<uint8_t> custom = random_vector<uint8_t>(4, 0, 255);
        const auto details = tflite::CreateCustomQuantizationDirect(fbb, &custom).Union();
        const int32_t quantizedDimension = 3;
        const auto quantization = tflite::CreateQuantizationParametersDirect(
            fbb, &min, &max, &scale, &zeroPoint, detailsType, details, quantizedDimension);

        const bool isVariable = random_of(true, false);

        // Create SparsityParameters
        const std::vector<int32_t> traversalOrder = random_vector<int32_t>(4);
        const std::vector<int32_t> blockMap = random_vector<int32_t>(4);
        const tflite::DimensionType format = tflite::DimensionType::DENSE;
        const int32_t denseSize = urandom();
        const tflite::SparseIndexVector arraySegmentsType = tflite::SparseIndexVector::Uint16Vector;
        std::vector<uint16_t> arraySegmentsValues = random_vector<uint16_t>(4);
        const auto arraySegments = tflite::CreateUint16VectorDirect(fbb, &arraySegmentsValues).Union();
        const tflite::SparseIndexVector arrayIndicesType = tflite::SparseIndexVector::Uint16Vector;
        std::vector<uint16_t> arrayIndicesValues = random_vector<uint16_t>(4);
        const auto arrayIndices = tflite::CreateUint16VectorDirect(fbb, &arrayIndicesValues).Union();
        const std::vector<flatbuffers::Offset<tflite::DimensionMetadata>> dimMetadata = {
            tflite::CreateDimensionMetadata(fbb, format, denseSize, arraySegmentsType, arraySegments, arrayIndicesType, arrayIndices),
        };
        const auto sparsity = tflite::CreateSparsityParametersDirect(fbb, &traversalOrder, &blockMap, &dimMetadata);

        const std::vector<int32_t> shapeSignature = {1, 9, 9, 3};
        const bool hasRank = random_of(true, false);

        // Create VariantSubType
        const std::vector<int32_t> shape1 = random_vector<int32_t>(4, 1, 32);
        const tflite::TensorType type1 = random_of(tflite::TensorType::INT8, tflite::TensorType::INT16, tflite::TensorType::INT32);
        const bool hasRank1 = random_of(true, false);
        const std::vector<flatbuffers::Offset<tflite::VariantSubType>> variant_tensors = {
            tflite::CreateVariantSubTypeDirect(fbb, &shape1, type1, hasRank1),
        };

        tensors.push_back(tflite::CreateTensorDirect(fbb, &shape, type, bufferIndex, name.c_str(), quantization,
            isVariable, sparsity, &shapeSignature, hasRank, &variant_tensors));
    }

    for ( auto &index : {1, 2, 3} )
    {
        // Generate buffer with data
        std::vector<uint8_t> data = random_vector<uint8_t>(9, 0, 255);
        buffers.push_back(tflite::CreateBufferDirect(fbb, &data));

        // Generate simple constant tensor
        const std::vector<int32_t> shape = {1, 1, 3, 3};
        const tflite::TensorType type = tflite::TensorType::INT16;
        const int bufferIndex = buffers.size() - 1;
        const std::string name = "const-" + std::to_string(index);
        tensors.push_back(tflite::CreateTensorDirect(fbb, &shape, type, bufferIndex, name.c_str()));
    }

    {
        // Generate intermediate tensor (tensor index 4)
        // Intermediates cannot be constant and must use buffer 0
        const std::vector<int32_t> shape = {1, 11, 11, 3};
        const tflite::TensorType type = tflite::TensorType::INT16;
        const int bufferIndex = 0;
        const std::string name = "intermediate";
        tensors.push_back(tflite::CreateTensorDirect(fbb, &shape, type, bufferIndex, name.c_str()));
    }

    {
        // Generate simple output tensor (tensor index 5)
        const std::vector<int32_t> shape = {1, 11, 11, 3};
        const tflite::TensorType type = tflite::TensorType::FLOAT32;
        const int bufferIndex = 0;
        const std::string name = "ofm";
        tensors.push_back(tflite::CreateTensorDirect(fbb, &shape, type, bufferIndex, name.c_str()));
    }

    {
        std::vector<uint8_t> data2 = random_vector<uint8_t>(5, 0, 255);
        buffers.push_back(tflite::CreateBufferDirect(fbb, &data2));
        metadata.push_back(tflite::CreateMetadataDirect(fbb, "metadata1", uint32_t(buffers.size() - 1)));
    }

    {
        // Generate 1 operator
        const uint32_t opcodeIndex = 0;
        const std::vector<int32_t> inputs = {0, 1, 2, 3};
        const std::vector<int32_t> intermediates = {4};
        const std::vector<int32_t> outputs = {5};
        const std::vector<uint8_t> customOptions = random_vector<uint8_t>(5);
        const std::vector<uint8_t> mutatingVariableInputs = random_vector<uint8_t>(4);

        // Generate builtin_options or builtin_options2
        flatbuffers::Offset<> builtinOptions = 0;
        tflite::BuiltinOptions builtinOptionsType = TfLiteMapping::BuiltinOperatorToBuiltinOptions(op);
        flatbuffers::Offset<> builtinOptions2 = 0;
        tflite::BuiltinOptions2 builtinOptions2Type = TfLiteMapping::BuiltinOperatorToBuiltinOptions2(op);
        if ( builtinOptionsType != tflite::BuiltinOptions::NONE )
        {
            LOG_TRACE1("Generating union {}\n", tflite::EnumNameBuiltinOptions(builtinOptionsType));
            builtinOptions = GenerateRandomUnionMember(
                fbb, tflite::BuiltinOptionsTypeTable()->type_refs[size_t(builtinOptionsType) - 1]());
        }
        else if ( builtinOptions2Type != tflite::BuiltinOptions2::NONE )
        {
            LOG_TRACE1("Generating union {}\n", tflite::EnumNameBuiltinOptions2(builtinOptions2Type));
            builtinOptions2 = GenerateRandomUnionMember(
                fbb, tflite::BuiltinOptions2TypeTable()->type_refs[size_t(builtinOptions2Type) - 1]());
        }

        operations.push_back(tflite::CreateOperatorDirect(fbb, opcodeIndex, &inputs, &outputs, builtinOptionsType,
            builtinOptions, &customOptions, tflite::CustomOptionsFormat::FLEXBUFFERS, &mutatingVariableInputs,
            &intermediates, 0, 0, builtinOptions2Type, builtinOptions2));
    }

    {
        // Generate 1 subgraph
        const std::vector<int32_t> inputs = {0 /* ifm0 */};
        const std::vector<int32_t> outputs = {5 /* ofm */};
        const char *name = "subgraph1";
        subgraphs.push_back(tflite::CreateSubGraphDirect(fbb, &tensors, &inputs, &outputs, &operations, name));
    }

    // TODO: add metadata_buffer, metadata and signature_defs
    // Generate 1 model
    const auto model1 = tflite::CreateModelDirect(
        fbb, 3 /* Version */, &operatorCodes, &subgraphs, "description1", &buffers, nullptr, &metadata);

    // Create TFLite flatbuffer
    tflite::FinishModelBuffer(fbb, model1);
    const auto bufExpected = fbb.Release();
    LOG_TRACE1("Created network ({}, size {})\n", fmt::ptr(bufExpected.data()), bufExpected.size());

    // Read TFLite network as GraphIR
    std::vector<std::unique_ptr<Graph>> graphs1;
    TfLiteReader::LoadGraphs(bufExpected.data(), bufExpected.size(), graphs1, nullptr, true);
    LOG_TRACE1("Read network ({} subgraphs)\n", graphs1.size());

    // Check GraphIR
    REQUIRE(graphs1.size() == 1);
    REQUIRE(graphs1[0]->Inputs().size() == 1);
    REQUIRE(graphs1[0]->Outputs().size() == 1);
    std::vector<Operation *> ops1;
    graphs1[0]->GetAllOperations(ops1);
    REQUIRE((ops1.size() == 1 || ops1.size() == 2));
    MarkAsPassthrough(ops1[0]);
    ops1.clear();
    graphs1[0]->GetAllOperations(ops1);
    REQUIRE(ops1.size() == 1);
    REQUIRE(ops1[0]->Type() == OpType::Passthrough);

    // Write GraphIR as TFLite flatbuffer
    std::vector<std::unordered_map<const Tensor *, Address>> maps{{}};
    int64_t offset = 0;
    size_t size = 0;
    TfLiteWriter writer(1 << 31, true /* skip OfflineMemoryAllocation */);
    graphs1[0]->SetScheduledOrder(std::move(ops1));
    const auto bufActual = writer.Serialise(graphs1, maps, offset, size);
    LOG_TRACE1("Wrote network ({}, offset {}, size {})\n", fmt::ptr(bufActual.get()), offset, size);

    // Parse actual TFLite flatbuffer to a list of items
    flatbuffers::Verifier::Options options;
    flatbuffers::Verifier verifier1(bufActual.get() + offset, size, options);
    REQUIRE(tflite::VerifyModelBuffer(verifier1));
    auto modelActual = tflite::GetModel(bufActual.get() + offset);
    ToItemListVisitor toItemListVisitor1;
    IterateFlatBuffer(bufActual.get() + offset, modelActual->MiniReflectTypeTable(), &toItemListVisitor1);

    // Parse expected TFLite flatbuffer to a list of items
    flatbuffers::Verifier verifier2(bufExpected.data(), bufExpected.size(), options);
    REQUIRE(tflite::VerifyModelBuffer(verifier2));
    auto modelExpected = tflite::GetModel(bufExpected.data());
    ToItemListVisitor toItemListVisitor2;
    IterateFlatBuffer(bufExpected.data(), modelExpected->MiniReflectTypeTable(), &toItemListVisitor2);

    // Comapare actual and expected flatbuffer
    CompareItems(toItemListVisitor1.items, toItemListVisitor2.items);
}
