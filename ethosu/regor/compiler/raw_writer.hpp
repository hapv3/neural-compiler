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

#include "architecture/architecture.hpp"
#include "compiler/graph.hpp"
#include "compiler/tensor.hpp"

#include <memory>
#include <unordered_map>
#include <vector>

namespace regor
{

class RawWriter
{
public:
    std::vector<std::pair<std::unique_ptr<const uint8_t[]>, size_t>> Serialise(const std::vector<std::unique_ptr<Graph>> &graphs,
        const std::vector<std::unordered_map<const Tensor *, Address>> &tensor_address_maps);

private:
    std::vector<std::pair<std::unique_ptr<const uint8_t[]>, size_t>> _raw;

    void SerialiseCommandStreamTensor(const Tensor *tensor);

    void SerialiseReadOnlyTensor(const Tensor *tensor);

    void SerialiseScratchTensor(const Tensor *tensor, Address address);

    void SerialiseScratchFastTensor(const Tensor *tensor, Address address);

    void SerialiseInputTensor(const Tensor *tensor, Address address);

    void SerialiseOutputTensor(const Tensor *tensor, Address address);

    void SerialiseVariableTensor(const Tensor *tensor, Address address);
};

}  // namespace regor
