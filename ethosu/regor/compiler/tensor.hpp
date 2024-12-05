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

#pragma once

#include "common/common.hpp"

#include "common/buffer_view.hpp"
#include "common/data_type.hpp"
#include "common/shape.hpp"
#include "include/graphapi.hpp"
#include "tensor_properties.hpp"

#include <algorithm>
#include <string>
#include <vector>

namespace regor
{

class Operation;

/// <summary>
/// Graph tensor representation
/// </summary>
class Tensor : public GraphApi::GraphTensor, public std::enable_shared_from_this<Tensor>
{
private:
    std::string _name;
    DataType _type;
    UniqueId _uid;
    class Shape _storageShape;
    std::shared_ptr<class Buffer> _buffer;
    enum AxisOrder _axisOrder = AxisOrder::Unknown;
    const void *_passthrough = nullptr;  // Original flatbuffer description of this tensor (if it was loaded from one)

    std::vector<std::shared_ptr<Operation>> _readers;
    std::vector<std::shared_ptr<Operation>> _writers;

public:
    Tensor(const std::string &name, DataType type);
    Tensor(const std::string &name, DataType type, Shape shape);
    Tensor(const std::string &name, DataType type, Shape shape, const std::shared_ptr<Buffer> &buffer);

    const std::string &Name() const { return _name; }
    void SetName(const std::string &name) { _name = name; }
    DataType Type() const { return _type; }
    UniqueId Uid() const { return _uid; }

    const Shape &StorageShape() const { return _storageShape; }
    void SetStorageShape(const Shape &shape) { _storageShape = shape; }
    void SetBuffer(const std::shared_ptr<Buffer> &buffer) { _buffer = buffer; }

    BufferView View() const;
    bool IsConstant() const;
    void Reshape(const Shape &shape);
    void ChangeType(DataType newType);

    enum AxisOrder AxisOrder() const { return _axisOrder; }
    void SetAxisOrder(enum AxisOrder axisOrder) { _axisOrder = axisOrder; }

    const void *Passthrough() const { return _passthrough; }
    void SetPassthrough(const void *passthrough) { _passthrough = passthrough; }

    const std::vector<std::shared_ptr<Operation>> &Readers() const { return _readers; }
    const std::vector<std::shared_ptr<Operation>> &Writers() const { return _writers; }

    void AddReader(std::shared_ptr<Operation> reader);
    void AddWriter(std::shared_ptr<Operation> writer);
    void RemoveReader(std::shared_ptr<Operation> reader);
    void RemoveWriter(std::shared_ptr<Operation> writer);
    void RemoveReaders();
    void RemoveWriters();

    bool IsSinglePath() const { return _readers.size() == 1 && _writers.size() == 1; }

    std::unique_ptr<Tensor> Clone() const;
    std::string ToString() const;
};

}  // namespace regor
