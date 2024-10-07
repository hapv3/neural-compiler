//
// SPDX-FileCopyrightText: Copyright 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
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

#include "quantization.hpp"
#include "tensor.hpp"

#include <vector>

namespace regor::GraphOptimisation
{

// Find specified tensor in Inputs() / Outputs() vectors.
// returns true if found in given vector.
bool IsTensorInVector(const std::vector<std::shared_ptr<Tensor>> &tensorVec, const Tensor *const tensorToFind);

// Insert a MemoryCopy operation after given ifm tensor. Returns a copy op shared_ptr.
// Will make a clone of ifm as ofm  and connects any other consumers of the ifm to it.
std::shared_ptr<Operation> InsertCopyOpAfterTensor(std::shared_ptr<Tensor> const &ifm, const Quantization &quantization);

// Connects output to operations in given list. Will not replace connection shape.
// Parameters:
// - producerList: List of producers.
// - tensorToReplace: if OFM on consumer match this tensor, replace it.
// - newTensor: The new output tensor to connect.
void ReplaceProducerOutput(std::vector<std::shared_ptr<Operation>> producerList, const Tensor *const tensorToReplace,
    std::shared_ptr<Tensor> newTensor);

// Connects input to operations in given list. Will not replace connection shape.
// Parameters:
// - exemptOperation: operation to exempt.
// - consumerList: List of consumers.
// - tensorToReplace: if IFM on consumer match this tensor, replace it.
// - newTensor: The new input tensor to connect.
void ReplaceConsumerInput(const Operation *const exemptOperation, std::vector<std::shared_ptr<Operation>> consumerList,
    const Tensor *const tensorToReplace, std::shared_ptr<Tensor> newTensor);

}  // namespace regor::GraphOptimisation
