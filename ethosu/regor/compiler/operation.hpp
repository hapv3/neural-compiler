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

#include "attributes.hpp"
#include "common/ordered_map.hpp"
#include "common/reverse_type.hpp"
#include "common/scaling.hpp"
#include "common/transpose_type.hpp"
#include "include/graphapi.hpp"
#include "kernel.hpp"
#include "op_type.hpp"
#include "quantization.hpp"
#include "tensor.hpp"

#include <memory>

namespace regor
{

enum class RoundMode : uint8_t
{
    DBL = 0,
    TRUNCATE = 1,
    NATURAL = 2,
    TRUNCATE_TO_LOWER = 3,
    DOUBLE_ASYMMETRIC = 4,
    SYMMETRIC = 5,
    AUTO = 0xff
};


struct TensorSlice
{
    Shape offset;
    Shape shape;
};

struct TensorConnection
{
    std::shared_ptr<Tensor> tensor;
    Shape shape;
    // For operations accessing a slice of the tensor:
    // Writing: Concat and Pack
    // Reading: Split, SplitV, Unpack, Slice, and StridedSlice
    TensorSlice slice;
    Quantization quantization;
    TransposeType transpose = TransposeType::None;
    ReverseType reverse = ReverseType::None;

    TensorConnection &Set(const Shape &s)
    {
        shape = s;
        return *this;
    }
    TensorConnection &Set(const TensorSlice &s)
    {
        slice = s;
        return *this;
    }
    TensorConnection &Set(const Quantization &q)
    {
        quantization = q;
        return *this;
    }
    TensorConnection &Set(const TransposeType &t)
    {
        transpose = t;
        return *this;
    }
    TensorConnection &Set(const ReverseType &r)
    {
        reverse = r;
        return *this;
    }
};


/// <summary>
/// Graph Operation representation
/// </summary>
class Operation : public std::enable_shared_from_this<Operation>, public GraphApi::GraphOperation
{
private:
    ordered_map<TensorUsage, TensorConnection> _inputs;
    ordered_map<TensorUsage, TensorConnection> _outputs;
    OpType _type;
    std::unique_ptr<class Kernel> _kernel;
    DynamicRef _attr;
    RoundMode _rounding;
    const void *_passthrough = nullptr;  // Original flatbuffer description of this op (if it was loaded from one)

public:
    Operation(OpType opType);
    Operation(const Operation &op);
    OpType Type() const { return _type; }

    const ordered_map<TensorUsage, TensorConnection> &Outputs() const { return _outputs; }
    const ordered_map<TensorUsage, TensorConnection> &Inputs() const { return _inputs; }

    Tensor *IFM(int index) const;
    Tensor *OFM() const;

    TensorConnection *Input(TensorUsage usage) { return _inputs.try_ref(usage); }
    const TensorConnection *Input(TensorUsage usage) const { return _inputs.try_ref(usage); }
    TensorConnection *Output(TensorUsage usage) { return _outputs.try_ref(usage); }
    const TensorConnection *Output(TensorUsage usage) const { return _outputs.try_ref(usage); }

    TensorUsage UsageOfTensor(const Tensor *tensor) const;
    const class Kernel *Kernel() const { return _kernel.get(); }
    class Kernel *Kernel() { return _kernel.get(); }
    void SetKernel(std::unique_ptr<class Kernel> kernel) { _kernel = std::move(kernel); }

    RoundMode Rounding() const { return _rounding; }
    void SetRounding(RoundMode rounding) { _rounding = rounding; }

    const void *Passthrough() const { return _passthrough; }
    void SetPassthrough(const void *passthrough) { _passthrough = passthrough; }
    void SetPassthroughOp() { _type = OpType::Passthrough; }
    void CopyInput(TensorUsage usage, const TensorConnection &tensorConnection);
    TensorConnection &ConnectInput(TensorUsage usage, const std::shared_ptr<Tensor> &tensor);
    int CountInputs(TensorUsage usage) const { return CountUsage(_inputs, usage); }
    // Disconnecting input invalidates all pointers to inputs.
    void DisconnectInputInvalidatingInputs(TensorUsage usage);

    void CopyOutput(TensorUsage usage, const TensorConnection &tensorConnection);
    TensorConnection &ConnectOutput(TensorUsage usage, const std::shared_ptr<Tensor> &tensor);
    int CountOutputs(TensorUsage usage) const { return CountUsage(_outputs, usage); }

    void Disconnect();
    bool IsDisconnected() const;
    bool HasScaling() const;

    template<typename TYPE>
    TYPE *Attribute()
    {
        if ( !_attr )
        {
            _attr = CreateAttribute(TypeHash<TYPE>::HASH);
        }
        else if ( _attr.Info()->Hash() != TypeHash<TYPE>::HASH )
        {
            throw std::runtime_error("attribute already assigned for this operator");
        }

        return static_cast<TYPE *>(_attr.Instance());
    }

    template<typename TYPE>
    const TYPE *Attribute() const
    {
        if ( !_attr || (_attr.Info()->Hash() != TypeHash<TYPE>::HASH) )
        {
            throw std::runtime_error("requested attribute must be already assigned");
        }
        return static_cast<const TYPE *>(_attr.Instance());
    }

    DynamicRef *AttributeByKey(uint32_t hash)
    {
        if ( !_attr )
        {
            _attr = CreateAttribute(hash);
        }
        return &_attr;
    }

    const DynamicRef &AttributeRef() const { return _attr; }

private:
    int CountUsage(const ordered_map<TensorUsage, TensorConnection> &list, TensorUsage usage) const
    {
        int count = 0;
        for ( const auto &pair : list.pairs() )
        {
            if ( (pair.first & TensorUsage::TypeMask) == usage )
            {
                count++;
            }
        }
        return count;
    }
};

}  // namespace regor
