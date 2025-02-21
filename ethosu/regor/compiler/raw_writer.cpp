//
// SPDX-FileCopyrightText: Copyright 2024-2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
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

#include "raw_writer.hpp"

#include "common/logging.hpp"

#include <algorithm>
#include <memory>
#include <vector>

#include "include/regor.h"


namespace regor
{

static constexpr int WEIGHTS_REGION = 0;
static constexpr int SCRATCH_REGION = 1;
static constexpr int SCRATCH_FAST_REGION = 2;
static constexpr int INPUT_REGION = SCRATCH_REGION;
static constexpr int OUTPUT_REGION = SCRATCH_REGION;
static constexpr int VARIABLE_REGION = SCRATCH_REGION;

std::vector<std::pair<std::unique_ptr<const uint8_t[]>, size_t>> RawWriter::Serialise(
    const std::vector<std::unique_ptr<Graph>> &graphs, const std::vector<std::unordered_map<const Tensor *, Address>> &tensor_address_maps)
{
    if ( graphs.size() != 1 )
    {
        throw std::invalid_argument("RawWriter expects 1 graph");
    }
    const auto &graph = graphs[0];

    std::vector<Operation *> operations;
    graph->GetAllOperations(operations);
    if ( operations.size() != 1 )
    {
        throw std::invalid_argument("RawWriter expects graph with 1 operation");
    }

    if ( tensor_address_maps.size() != 1 )
    {
        throw std::invalid_argument("RawWriter expects 1 tensor address map");
    }
    const auto &tensor_address_map = tensor_address_maps[0];

    const Operation *customNpuOp = operations[0];
    if ( customNpuOp->Type() != OpType::CustomNpuOp )
    {
        throw std::invalid_argument("RawWriter expects graph with 1 CustomNpuOp");
    }

    // ethos_u_command_stream in TFLite format
    auto commandStreamTensorConnection = customNpuOp->Input(MakeTensorUsage(TensorUsage::Params, 0));
    auto commandStreamTensor = commandStreamTensorConnection->tensor.get();
    SerialiseCommandStreamTensor(commandStreamTensor);

    // read_only in TFLite format
    auto readOnlyTensorConnection = customNpuOp->Input(MakeTensorUsage(TensorUsage::Params, 1));
    auto readOnlyTensor = readOnlyTensorConnection->tensor.get();
    SerialiseReadOnlyTensor(readOnlyTensor);

    // scratch in TFLite format
    auto featureMapTensorConnection = customNpuOp->Input(MakeTensorUsage(TensorUsage::State, 0));
    auto featureMapTensor = featureMapTensorConnection->tensor.get();
    SerialiseScratchTensor(featureMapTensor, tensor_address_map.at(featureMapTensor));

    // scratch/scratch_fast TFLite format
    auto stagingTensorConnection = customNpuOp->Input(MakeTensorUsage(TensorUsage::State, 1));
    auto stagingTensor = stagingTensorConnection->tensor.get();
    if ( stagingTensor == featureMapTensor )
    {
        SerialiseScratchTensor(stagingTensor, tensor_address_map.at(stagingTensor));
    }
    else
    {
        SerialiseScratchFastTensor(stagingTensor, tensor_address_map.at(stagingTensor));
    }

    // Serialise input tensors
    for ( const auto &input : graph->Inputs() )
    {
        const Tensor *tensor = input.get();
        auto tensorUsage = customNpuOp->UsageOfTensor(tensor);
        if ( IsIFM(tensorUsage) && !tensor->IsConstant() )
        {
            SerialiseInputTensor(tensor, tensor_address_map.at(tensor));
        }
    }

    // Serialise output tensors
    for ( const auto &output : graph->Outputs() )
    {
        const Tensor *tensor = output.get();
        auto tensorUsage = customNpuOp->UsageOfTensor(tensor);
        if ( IsOFM(tensorUsage) )
        {
            SerialiseOutputTensor(tensor, tensor_address_map.at(tensor));
        }
    }

    // Serialise persistent tensors
    for ( const auto &persistent : graph->Persistent() )
    {
        const Tensor *tensor = persistent.get();
        auto tensorUsage = customNpuOp->UsageOfTensor(tensor);
        if ( (tensorUsage != TensorUsage::None) )
        {
            SerialiseVariableTensor(tensor, tensor_address_map.at(tensor));
        }
    }


    return std::move(_raw);
}

void RawWriter::SerialiseCommandStreamTensor(const Tensor *tensor)
{
    assert(tensor->IsConstant());

    // command_stream buffer and buffer size
    auto blob = tensor->View().Buffer()->Data<uint8_t>();
    auto blobSize = tensor->View().Buffer()->Size();

    const size_t serialisedTensorSize = sizeof(regor_raw_tensor_header_t) + blobSize;
    auto serialisedTensor = std::make_unique<uint8_t[]>(serialisedTensorSize);

    // Initialise header
    regor_raw_tensor_header_t header;
    header.type = regor_raw_tensor_header_t::RAW_TENSOR_TYPE_COMMAND_STREAM;
    header.tensor.command_stream.size = blobSize;

    // Copy header
    std::copy_n(reinterpret_cast<uint8_t *>(&header), sizeof(header), serialisedTensor.get());

    // Copy blob to right after header
    std::copy_n(blob, blobSize, serialisedTensor.get() + sizeof(regor_raw_tensor_header_t));

    _raw.emplace_back(std::move(serialisedTensor), serialisedTensorSize);
}

void RawWriter::SerialiseReadOnlyTensor(const Tensor *tensor)
{
    assert(tensor->View().HasBuffer());

    // read_only buffer and buffer size
    auto blob = tensor->View().Buffer()->Data<uint8_t>();
    auto blobSize = tensor->View().Buffer()->Size();

    const size_t serialisedTensorSize = sizeof(regor_raw_tensor_header_t) + blobSize;
    auto serialisedTensor = std::make_unique<uint8_t[]>(serialisedTensorSize);

    // Initialise read_only header
    regor_raw_tensor_header_t header;
    header.type = regor_raw_tensor_header_t::RAW_TENSOR_TYPE_READ_ONLY;
    header.tensor.read_only.region = WEIGHTS_REGION;
    header.tensor.read_only.size = blobSize;

    // Copy header
    std::copy_n(reinterpret_cast<uint8_t *>(&header), sizeof(header), serialisedTensor.get());

    // Copy blob to right after header
    std::copy_n(blob, blobSize, serialisedTensor.get() + sizeof(regor_raw_tensor_header_t));

    _raw.emplace_back(std::move(serialisedTensor), serialisedTensorSize);
}

void RawWriter::SerialiseScratchTensor(const Tensor *tensor, Address address)
{
    assert(!tensor->IsConstant());

    const size_t serialisedTensorSize = sizeof(regor_raw_tensor_header_t);
    auto serialisedTensor = std::make_unique<uint8_t[]>(serialisedTensorSize);

    // Initialise scratch header
    regor_raw_tensor_header_t header;
    header.type = regor_raw_tensor_header_t::RAW_TENSOR_TYPE_SCRATCH;
    header.tensor.scratch.region = SCRATCH_REGION;
    header.tensor.scratch.size = DataTypeStorageSizeBytes(tensor->Type(), tensor->StorageShape().Elements());
    header.tensor.scratch.address = address;

    // Copy header
    std::copy_n(reinterpret_cast<uint8_t *>(&header), sizeof(header), serialisedTensor.get());

    _raw.emplace_back(std::move(serialisedTensor), serialisedTensorSize);
}

void RawWriter::SerialiseScratchFastTensor(const Tensor *tensor, Address address)
{
    assert(!tensor->IsConstant());

    const size_t serialisedTensorSize = sizeof(regor_raw_tensor_header_t);
    auto serialisedTensor = std::make_unique<uint8_t[]>(serialisedTensorSize);

    // Initialise scratch_fast tensor header
    regor_raw_tensor_header_t header;
    header.type = regor_raw_tensor_header_t::RAW_TENSOR_TYPE_SCRATCH_FAST;
    header.tensor.scratch_fast.region = SCRATCH_FAST_REGION;
    header.tensor.scratch_fast.size = DataTypeStorageSizeBytes(tensor->Type(), tensor->StorageShape().Elements());
    header.tensor.scratch_fast.address = address;

    // Copy header
    std::copy_n(reinterpret_cast<uint8_t *>(&header), sizeof(header), serialisedTensor.get());

    _raw.emplace_back(std::move(serialisedTensor), serialisedTensorSize);
}

void RawWriter::SerialiseInputTensor(const Tensor *tensor, Address address)
{
    assert(!tensor->IsConstant());

    const size_t serialisedTensorSize = sizeof(regor_raw_tensor_header_t);
    auto serialisedTensor = std::make_unique<uint8_t[]>(serialisedTensorSize);

    // Initialise input tensor header
    regor_raw_tensor_header_t header;
    header.type = regor_raw_tensor_header_t::RAW_TENSOR_TYPE_INPUT;
    header.tensor.input.region = INPUT_REGION;
    header.tensor.input.element_size = DataTypeStorageSizeBytes(tensor->Type(), 1);
    auto shape = Shape::PadAxes(tensor->StorageShape(), 4, 1).ToList<uint32_t>();
    std::copy(shape.begin(), shape.end(), header.tensor.input.shape);
    header.tensor.input.size = DataTypeStorageSizeBytes(tensor->Type(), tensor->StorageShape().Elements());
    header.tensor.input.address = address;

    // Copy header
    std::copy_n(reinterpret_cast<uint8_t *>(&header), sizeof(header), serialisedTensor.get());

    _raw.emplace_back(std::move(serialisedTensor), serialisedTensorSize);
}

void RawWriter::SerialiseOutputTensor(const Tensor *tensor, Address address)
{
    assert(!tensor->IsConstant());

    const size_t serialisedTensorSize = sizeof(regor_raw_tensor_header_t);
    auto serialisedTensor = std::make_unique<uint8_t[]>(serialisedTensorSize);

    // Initialise output tensor header
    regor_raw_tensor_header_t header;
    header.type = regor_raw_tensor_header_t::RAW_TENSOR_TYPE_OUTPUT;
    header.tensor.output.region = OUTPUT_REGION;
    header.tensor.output.element_size = DataTypeStorageSizeBytes(tensor->Type(), 1);
    auto shape = Shape::PadAxes(tensor->StorageShape(), 4, 1).ToList<uint32_t>();
    std::copy(shape.begin(), shape.end(), header.tensor.output.shape);
    header.tensor.output.size = DataTypeStorageSizeBytes(tensor->Type(), tensor->StorageShape().Elements());
    header.tensor.output.address = address;

    // Copy header
    std::copy_n(reinterpret_cast<uint8_t *>(&header), sizeof(header), serialisedTensor.get());

    _raw.emplace_back(std::move(serialisedTensor), serialisedTensorSize);
}

void RawWriter::SerialiseVariableTensor(const Tensor *tensor, Address address)
{
    assert(!tensor->IsConstant());

    const size_t serialisedTensorSize = sizeof(regor_raw_tensor_header_t);
    auto serialisedTensor = std::make_unique<uint8_t[]>(serialisedTensorSize);

    // Initialise variable tensor header
    regor_raw_tensor_header_t header;
    header.type = regor_raw_tensor_header_t::RAW_TENSOR_TYPE_VARIABLE;
    header.tensor.variable.region = VARIABLE_REGION;
    header.tensor.variable.element_size = DataTypeStorageSizeBytes(tensor->Type(), 1);
    auto shape = Shape::PadAxes(tensor->StorageShape(), 4, 1).ToList<uint32_t>();
    std::copy(shape.begin(), shape.end(), header.tensor.variable.shape);
    header.tensor.variable.size = DataTypeStorageSizeBytes(tensor->Type(), tensor->StorageShape().Elements());
    header.tensor.variable.address = address;

    // Copy header
    std::copy_n(reinterpret_cast<uint8_t *>(&header), sizeof(header), serialisedTensor.get());

    _raw.emplace_back(std::move(serialisedTensor), serialisedTensorSize);
}

}  // namespace regor
