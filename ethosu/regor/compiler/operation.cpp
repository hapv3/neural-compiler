//
// SPDX-FileCopyrightText: Copyright 2021-2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
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

#include "compiler/operation.hpp"

#include "common/ordered_map.hpp"
#include "common/scaling.hpp"
#include "graph_builder.hpp"
#include "kernel.hpp"
#include "op_type.hpp"
#include "tensor.hpp"

#include <memory>

namespace regor
{

Operation::Operation(OpType opType) : _type(opType)
{
    // Default 1x1 kernel for ops without a kernel
    _kernel = std::make_unique<class Kernel>(Point2i(1, 1), Point2i(1, 1), Point2i(1, 1));
}

Operation::Operation(const Operation &op) : Operation(op._type)
{
    _kernel = std::make_unique<class Kernel>(*op._kernel.get());
    _passthrough = op._passthrough;
}

TensorUsage Operation::UsageOfTensor(const Tensor *tensor) const
{
    for ( const auto &list : {_inputs.pairs(), _outputs.pairs()} )
    {
        for ( const auto &pair : list )
        {
            if ( tensor == pair.second.tensor.get() )
            {
                return pair.first;
            }
        }
    }
    return TensorUsage::None;
}

Tensor *Operation::IFM(int index) const
{
    auto conn = Input(MakeTensorUsage(TensorUsage::IFM, index));
    return conn ? conn->tensor.get() : nullptr;
}

Tensor *Operation::OFM() const
{
    return _outputs.at(TensorUsage::OFM).tensor.get();
}

void Operation::CopyInput(TensorUsage usage, const TensorConnection &tensorConnection)
{
    ConnectInput(usage, tensorConnection.tensor)
        .Set(tensorConnection.shape)
        .Set(tensorConnection.slice)
        .Set(tensorConnection.quantization);
}

TensorConnection &Operation::ConnectInput(TensorUsage usage, const std::shared_ptr<Tensor> &tensor)
{
    // Must create the new connection before destroying whatever it replaces,
    // because the existing connection (if present) might be the last remaining reference to this operation.
    tensor->AddReader(shared_from_this());

    if ( _inputs.contains(usage) && (_inputs[usage].tensor != tensor) )
    {
        _inputs[usage].tensor->RemoveReader(shared_from_this());
    }
    _inputs[usage].tensor = tensor;
    _inputs[usage].shape = tensor->StorageShape();
    return _inputs[usage];
}

void Operation::DisconnectInputInvalidatingInputs(TensorUsage usage)
{
    auto *inputConnection = Input(usage);
    if ( inputConnection )
    {
        if ( inputConnection->tensor ) inputConnection->tensor->RemoveReader(shared_from_this());
        _inputs.erase(usage);  // Invalidates all pointers to input connections.
    }
}

void Operation::CopyOutput(TensorUsage usage, const TensorConnection &tensorConnection)
{
    ConnectOutput(usage, tensorConnection.tensor)
        .Set(tensorConnection.shape)
        .Set(tensorConnection.slice)
        .Set(tensorConnection.quantization);
}

TensorConnection &Operation::ConnectOutput(TensorUsage usage, const std::shared_ptr<Tensor> &tensor)
{
    // Must create the new connection before destroying whatever it replaces,
    // because the existing connection (if present) might be the last remaining reference to this operation.
    tensor->AddWriter(shared_from_this());

    if ( _outputs.contains(usage) && (_outputs[usage].tensor != tensor) )
    {
        _outputs[usage].tensor->RemoveWriter(shared_from_this());
    }
    _outputs[usage].tensor = tensor;
    _outputs[usage].shape = tensor->StorageShape();

    return _outputs[usage];
}

void Operation::Disconnect()
{
    // This operation might be about to remove the last remaining references to itself,
    // so it must hold onto this one until it is finished disconnecting.
    auto self = shared_from_this();

    for ( auto &conn : _inputs )
    {
        conn.tensor->RemoveReader(self);
    }
    _inputs.clear();

    for ( auto &conn : _outputs )
    {
        conn.tensor->RemoveWriter(self);
    }
    _outputs.clear();
}

bool Operation::IsDisconnected() const
{
    return _inputs.empty() && _outputs.empty();
}

bool Operation::HasScaling() const
{
    bool scaled = true;
    for ( const auto &fm : _inputs.pairs() )
    {
        if ( fm.first == TensorUsage::IFM || fm.first == TensorUsage::IFM1 || fm.first == TensorUsage::OFM )
        {
            if ( fm.second.quantization.scales.empty() )
            {
                return false;
            }
        }
    }
    return scaled;
}

}  // namespace regor
