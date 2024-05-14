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

#include "compiler/tensor.hpp"

#include "common/common.hpp"

#include "architecture/architecture.hpp"
#include "common/buffer_view.hpp"
#include "common/data_type.hpp"
#include "common/shape.hpp"

#include <algorithm>
#include <string>
#include <tuple>
#include <vector>

namespace regor
{

Tensor::Tensor(const std::string &name, DataType type) : _name(name), _type(type), _uid(GenerateUniqueId())
{
}

Tensor::Tensor(const std::string &name, DataType type, Shape shape) :
        _name(name), _type(type), _uid(GenerateUniqueId()), _storageShape(std::move(shape))
{
}

Tensor::Tensor(const std::string &name, DataType type, Shape shape, const std::shared_ptr<Buffer> &buffer) :
        _name(name), _type(type), _uid(GenerateUniqueId()), _storageShape(shape), _buffer(buffer)
{
    assert(DataTypeStorageSizeBytes(type, shape.Elements()) <= buffer->Size());
}

BufferView Tensor::View() const
{
    int elementBits = DataTypeSizeBits(_type) > 0 ? DataTypeSizeBits(_type) : 8;
    return BufferView(_buffer, 0, elementBits, _storageShape, Shape());
}

bool Tensor::IsConstant() const
{
    return _buffer && _buffer->Size();
}

void Tensor::Reshape(const Shape &shape)
{
    assert(shape.Elements() == StorageShape().Elements());
    SetStorageShape(shape);
}
void Tensor::ChangeType(DataType newType)
{
    assert(!IsConstant());
    _type = newType;
}

void Tensor::AddReader(std::shared_ptr<Operation> reader)
{
    if ( std::find(_readers.begin(), _readers.end(), reader) == _readers.end() )
    {
        _readers.push_back(reader);
    }
}
void Tensor::AddWriter(std::shared_ptr<Operation> writer)
{
    if ( std::find(_writers.begin(), _writers.end(), writer) == _writers.end() )
    {
        _writers.push_back(writer);
    }
}
void Tensor::RemoveReader(std::shared_ptr<Operation> reader)
{
    _readers.erase(std::remove(_readers.begin(), _readers.end(), reader), _readers.end());
}
void Tensor::RemoveWriter(std::shared_ptr<Operation> writer)
{
    _writers.erase(std::remove(_writers.begin(), _writers.end(), writer), _writers.end());
}
void Tensor::RemoveReaders()
{
    _readers.clear();
}
void Tensor::RemoveWriters()
{
    _writers.clear();
}

std::unique_ptr<Tensor> Tensor::Clone() const
{
    auto clone = std::make_unique<Tensor>(*this);
    clone->_uid = GenerateUniqueId();
    clone->RemoveReaders();
    clone->RemoveWriters();
    return clone;
}

std::string Tensor::ToString() const
{
    return fmt::format("<Tensor '{}': Shape = {}; Type = {};>", Name(), StorageShape().ToString(), DataTypeToString(Type()));
}

}  // namespace regor
