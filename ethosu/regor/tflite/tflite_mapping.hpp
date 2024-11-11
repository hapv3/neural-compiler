//
// SPDX-FileCopyrightText: Copyright 2021, 2023-2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
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

#include "common/data_type.hpp"
#include "compiler/op_type.hpp"
#include "compiler/operation.hpp"
#include "tflite_schema_generated.hpp"

#include <cassert>
#include <map>

namespace regor
{

class TfLiteMapping
{
public:
    TfLiteMapping(const TfLiteMapping &) = delete;  // Never constructed. Static members only.

    //
    // Conversions from TensorFlow Lite types to Regor types
    //
    static DataType TensorTypeToDataType(tflite::TensorType type) { return _tensorTypeToDataType.at(type); }
    static OpType ActivationFunctionToOpType(tflite::ActivationFunctionType type)
    {
        return _activationFunctionToOpType.at(type);
    }
    static OpType BuiltinOperatorToOpType(tflite::BuiltinOperator type)
    {
        auto it = _builtinOperatorToOpType.find(type);
        if ( it == _builtinOperatorToOpType.end() )
        {
            return OpType::None;
        }
        return it->second;
    }

    static std::string BuiltinOperatorToString(tflite::BuiltinOperator &type) { return EnumNameBuiltinOperator(type); }

    //
    // Conversions from Regor types to TensorFlow Lite types
    //
    static tflite::TensorType DataTypeToTensorType(DataType type)
    {
        if ( type == DataType::Int48 ) return tflite::TensorType::INT64;
        else if ( type == DataType::UInt48 ) return tflite::TensorType::UINT64;
        return _dataTypeToTensorType.at(type);
    }
    static tflite::ActivationFunctionType OpTypeToActivationFunction(OpType type)
    {
        auto it = _opTypeToActivationFunction.find(type);
        if ( it == _opTypeToActivationFunction.end() )
        {
            return tflite::ActivationFunctionType::NONE;
        }
        return it->second;
    }
    static tflite::BuiltinOperator OpTypeToBuiltinOperator(OpType type)
    {
        assert((type == OpType::CustomNpuOp || _opTypeToBuiltinOperator.count(type) == 1) && "Missing op type");

        return type == OpType::CustomNpuOp ? tflite::BuiltinOperator::CUSTOM : _opTypeToBuiltinOperator.at(type);
    }
    static tflite::BuiltinOptions OpTypeToBuiltinOptions(OpType type)
    {
        auto it1 = _opTypeToBuiltinOperator.find(type);
        if ( it1 == _opTypeToBuiltinOperator.end() )
        {
            return tflite::BuiltinOptions::NONE;
        }
        auto it2 = _builtinOperatorToBuiltinOptions.find(it1->second);
        if ( it2 == _builtinOperatorToBuiltinOptions.end() )
        {
            return tflite::BuiltinOptions::NONE;
        }
        return it2->second;
    }

    class InputTensorIndices;  // Usage: for (const auto& map_entry : InputTensorIndices(op_type)) {}

    // The order in which input tensors are referenced by a TensorFlow Lite operator depends on the operator type


    static bool CanFuseActivationFunction(const Operation *operation);

private:
    // Mappings from TensorFlow Lite types to Regor types are 1:1 (except for CUSTOM:[Custom|CustomNpuOp])
    // Mappings from Regor types to TensorFlow Lite types are 1:[0|1]
    // Regor types without a corresponding TensorFlow Lite type are not included in these maps
    //  (the Regor to TensorFlow Lite maps are created by inverting the TensorFlow Lite to Regor maps)

    static const std::map<tflite::TensorType, DataType> _tensorTypeToDataType;
    static const std::map<DataType, tflite::TensorType> _dataTypeToTensorType;

    static const std::map<tflite::ActivationFunctionType, OpType> _activationFunctionToOpType;
    static const std::map<OpType, tflite::ActivationFunctionType> _opTypeToActivationFunction;

    static const std::map<tflite::BuiltinOperator, OpType> _builtinOperatorToOpType;
    static const std::map<OpType, tflite::BuiltinOperator> _opTypeToBuiltinOperator;

    // Mapping from TensorFlow Lite operator type to TensorFlow Lite options type is N:1
    static const std::map<tflite::BuiltinOperator, tflite::BuiltinOptions> _builtinOperatorToBuiltinOptions;

    // The number of input tensors to a TensorFlow Lite operator depends on the operator type,
    // as does the order in which the different kinds of input tensor are listed.
    static const std::multimap<OpType, TensorUsage> _inputTensorIndices;

    // Assumes 1:1 mapping (else some entries will be lost)
    template<typename T1, typename T2>
    static std::map<T2, T1> InvertMap(std::map<T1, T2> map)
    {
        std::map<T2, T1> inverse_map;
        for ( const auto &pair : map )
        {
            inverse_map[pair.second] = pair.first;
        }
        return inverse_map;
    }
};


// Provides lookup into `_inputTensorIndices` as a range expression
class TfLiteMapping::InputTensorIndices
{
private:
    typedef std::multimap<OpType, TensorUsage>::const_iterator const_iterator;
    const_iterator _begin;
    const_iterator _end;

public:
    InputTensorIndices(OpType op_type)
    {
        auto pair = _inputTensorIndices.equal_range(op_type);
        _begin = pair.first;
        _end = pair.second;
    }
    const_iterator begin() const { return _begin; }
    const_iterator end() const { return _end; }
};

}  // namespace regor
