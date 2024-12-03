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

#pragma once

#include "common/common.hpp"

#include "common/data_type.hpp"
#include "common/dynamic_typing.hpp"
#include "common/numeric_util.hpp"
#include "common/shape.hpp"
#include "include/graphapi.hpp"

#include <forward_list>
#include <memory>
#include <string>

namespace tflite
{
enum class MirrorPadMode : int8_t;
}  // namespace tflite

namespace regor
{

#define GRAPHAPI_FUNCTION_DETAIL_ONLY
#include "include/graphapi_attr.hpp"

// MakeAttributeId places the type information in the lower 4-bits for the specific
// purpose of stripping it with a right shift. This allows us to compare fields
// of a different type ids.
#define ATTR_FIELD_ID(CLASS_, UNIQUE_) detail::MakeAttributeId(TypeHash<CLASS_>::HASH, 0, UNIQUE_)
#define ATTR_FIELD(MEMBER_, UNIQUE_) \
    { \
        offsetof(thisclass_t, MEMBER_), ATTR_FIELD_ID(thisclass_t, UNIQUE_) >> 4, \
            REGOR_FIELD_TYPE(decltype(reinterpret_cast<thisclass_t *>(0)->MEMBER_)) \
    } \
    ,

struct pooling_attr_t
{
    DataType accPrecision;
    BEGIN_FIELD_TABLE(pooling_attr_t)
        ATTR_FIELD(accPrecision, 0)
    END_FIELD_TABLE()
};

struct axis_attr_t
{
    int32_t axis;
    BEGIN_FIELD_TABLE(axis_attr_t)
        ATTR_FIELD(axis, 0)
    END_FIELD_TABLE()
};

struct reshape_attr_t
{
    Shape shape;
    BEGIN_FIELD_TABLE(reshape_attr_t)
        ATTR_FIELD(shape, 0)
    END_FIELD_TABLE()
};

struct slice_attr_t
{
    Shape begin;
    Shape size;
    BEGIN_FIELD_TABLE(slice_attr_t)
        ATTR_FIELD(begin, 0)
        ATTR_FIELD(size, 1)
    END_FIELD_TABLE()
};

struct resize_attr_t
{
    Fraction<int> scaleX;
    Fraction<int> scaleY;
    Point2i offset;
    Point2i border;
    tosa::ResizeMode mode;
    BEGIN_FIELD_TABLE(resize_attr_t)
        ATTR_FIELD(scaleX, 0)
        ATTR_FIELD(scaleY, 1)
        ATTR_FIELD(offset, 2)
        ATTR_FIELD(border, 3)
        ATTR_FIELD(mode, 4)
    END_FIELD_TABLE()
};

struct clamp_attr_t
{
    double min;
    double max;
    BEGIN_FIELD_TABLE(clamp_attr_t)
        ATTR_FIELD(min, 0)
        ATTR_FIELD(max, 1)
    END_FIELD_TABLE()
};

struct sign_attr_t
{
    bool input_unsigned;
    bool output_unsigned;
    BEGIN_FIELD_TABLE(sign_attr_t)
        ATTR_FIELD(input_unsigned, 0)
        ATTR_FIELD(output_unsigned, 1)
    END_FIELD_TABLE()
};

struct rescale_attr_t
{
    bool scale32;
    bool double_round;
    bool per_channel;
    BEGIN_FIELD_TABLE(rescale_attr_t)
        ATTR_FIELD(scale32, 0)
        ATTR_FIELD(double_round, 1)
        ATTR_FIELD(per_channel, 2)
    END_FIELD_TABLE()
};

struct mul_attr_t
{
    int32_t shift;
    BEGIN_FIELD_TABLE(mul_attr_t)
        ATTR_FIELD(shift, 0)
    END_FIELD_TABLE()
};

struct asr_attr_t
{
    bool round;
    BEGIN_FIELD_TABLE(asr_attr_t)
        ATTR_FIELD(round, 0)
    END_FIELD_TABLE()
};

struct cond_attr_t
{
    std::string then_branch;
    std::string else_branch;
    BEGIN_FIELD_TABLE(cond_attr_t)
        ATTR_FIELD(then_branch, 0)
        ATTR_FIELD(else_branch, 1)
    END_FIELD_TABLE()
};

struct while_attr_t
{
    std::string cond_branch;
    std::string body_branch;
    BEGIN_FIELD_TABLE(while_attr_t)
        ATTR_FIELD(cond_branch, 0)
        ATTR_FIELD(body_branch, 1)
    END_FIELD_TABLE()
};

struct transpose_conv2d_attr_t
{
    Shape outShape;
    Shape outPadTBLR;
    BEGIN_FIELD_TABLE(transpose_conv2d_attr_t)
        ATTR_FIELD(outShape, 0)
        ATTR_FIELD(outPadTBLR, 1)
    END_FIELD_TABLE()
};

struct transpose_attr_t
{
    Shape perm;
    BEGIN_FIELD_TABLE(transpose_attr_t)
        ATTR_FIELD(perm, 0)
    END_FIELD_TABLE()
};

struct fft_attr_t
{
    bool inverse;
    BEGIN_FIELD_TABLE(fft_attr_t)
        ATTR_FIELD(inverse, 0)
    END_FIELD_TABLE()
};

struct custom_attr_t
{
    std::string name;
    std::string domain;
    BEGIN_FIELD_TABLE(custom_attr_t)
        ATTR_FIELD(name, 0)
        ATTR_FIELD(domain, 1)
    END_FIELD_TABLE()
};

struct strided_slice_attr_t
{
    int begin_mask;
    int end_mask;
    int ellipsis_mask;
    int new_axis_mask;
    int shrink_axis_mask;
    BEGIN_FIELD_TABLE(strided_slice_attr_t)
        ATTR_FIELD(begin_mask, 0)
        ATTR_FIELD(end_mask, 1)
        ATTR_FIELD(ellipsis_mask, 2)
        ATTR_FIELD(new_axis_mask, 3)
        ATTR_FIELD(shrink_axis_mask, 4)
    END_FIELD_TABLE()
};

struct tflite_resize_t
{
    bool alignCorners;
    bool halfPixelCenters;
    BEGIN_FIELD_TABLE(tflite_resize_t)
        ATTR_FIELD(alignCorners, 0)
        ATTR_FIELD(halfPixelCenters, 1)
    END_FIELD_TABLE()
};

struct leaky_relu_attr_t
{
    float alpha;
    BEGIN_FIELD_TABLE(leaky_relu_attr_t)
        ATTR_FIELD(alpha, 0)
    END_FIELD_TABLE()
};

struct softmax_attr_t
{
    float beta;
    BEGIN_FIELD_TABLE(softmax_attr_t)
        ATTR_FIELD(beta, 0)
    END_FIELD_TABLE()
};

struct concat_attr_t
{
    int axis;
    BEGIN_FIELD_TABLE(concat_attr_t)
        ATTR_FIELD(axis, 0)
    END_FIELD_TABLE()
};

struct pack_unpack_attr_t
{
    int axis;
    BEGIN_FIELD_TABLE(pack_unpack_attr_t)
        ATTR_FIELD(axis, 0)
    END_FIELD_TABLE()
};

struct pad_attr_t
{
    double pad_const;
    BEGIN_FIELD_TABLE(pad_attr_t)
        ATTR_FIELD(pad_const, 0)
    END_FIELD_TABLE()
};

struct mirror_pad_mode_attr_t
{
    tflite::MirrorPadMode mode;
    BEGIN_FIELD_TABLE(mirror_pad_mode_attr_t)
        ATTR_FIELD(mode, 0)
    END_FIELD_TABLE()
};

#define REDUCED_HASH(hash) (hash & 0x000FFFFF)

DynamicRef CreateAttribute(uint32_t hash);

// Attribute container - maintains chains of
// dynamically allocated attribute objects.
struct Attributes
{
    mutable std::forward_list<DynamicRef> _chain;

    DynamicRef *Get(bool create, uint32_t hash) const
    {
        for ( auto pos = _chain.begin(); pos != _chain.end(); pos++ )
        {
            if ( *pos && (pos->Info()->Hash() == hash || REDUCED_HASH(pos->Info()->Hash()) == hash) ) return &(*pos);
        }
        if ( !create ) return nullptr;
        return &_chain.emplace_front(CreateAttribute(hash));
    }

    DynamicRef *Require(uint32_t hash) const
    {
        DynamicRef *ref = Get(false, hash);
        if ( !ref ) throw std::runtime_error("requested attribute must be already assigned");
        return ref;
    }

    Attributes &operator=(const Attributes &attr)
    {
        if ( &attr != this )
        {
            _chain = attr._chain;
        }
        return *this;
    }
};

// Mixin to make objects attributable
class Attributable
{
protected:
    Attributes _attr;

public:
    template<typename TYPE>
    TYPE *Attribute()
    {
        return static_cast<TYPE *>(_attr.Get(true, TypeHash<TYPE>::HASH)->Instance());
    }

    template<typename TYPE>
    const TYPE *Attribute() const
    {
        return static_cast<TYPE *>(_attr.Require(TypeHash<TYPE>::HASH)->Instance());
    }

    template<typename TYPE>
    bool HasAttribute() const
    {
        return _attr.Get(false, TypeHash<TYPE>::HASH) != nullptr;
    }

    DynamicRef *AttributeByKey(uint32_t hash) { return _attr.Get(true, hash); }

    const Attributes &AttributeRef() const { return _attr; }
};


}  // namespace regor
