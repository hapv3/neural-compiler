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

#include "attributes.hpp"

#include "common/logging.hpp"

namespace regor
{

#define CASE_MAKE_ATTR_INSTANCE(TYPE_) \
    case REDUCED_HASH(TypeHash<TYPE_>::HASH): \
        return DynamicRef(TypeInfoOf<TYPE_>::Get(true), TypeInfoOf<TYPE_>::SharedNew());

DynamicRef CreateAttribute(uint32_t reducedHash)
{
    reducedHash = REDUCED_HASH(reducedHash);
    switch ( reducedHash )
    {
        CASE_MAKE_ATTR_INSTANCE(asr_attr_t);
        CASE_MAKE_ATTR_INSTANCE(axis_attr_t);
        CASE_MAKE_ATTR_INSTANCE(reshape_attr_t);
        CASE_MAKE_ATTR_INSTANCE(clamp_attr_t);
        CASE_MAKE_ATTR_INSTANCE(concat_attr_t);
        CASE_MAKE_ATTR_INSTANCE(cond_attr_t);
        CASE_MAKE_ATTR_INSTANCE(custom_attr_t);
        CASE_MAKE_ATTR_INSTANCE(fft_attr_t);
        CASE_MAKE_ATTR_INSTANCE(leaky_relu_attr_t);
        CASE_MAKE_ATTR_INSTANCE(mul_attr_t);
        CASE_MAKE_ATTR_INSTANCE(pack_unpack_attr_t);
        CASE_MAKE_ATTR_INSTANCE(pad_attr_t);
        CASE_MAKE_ATTR_INSTANCE(pooling_attr_t);
        CASE_MAKE_ATTR_INSTANCE(rescale_attr_t);
        CASE_MAKE_ATTR_INSTANCE(sign_attr_t);
        CASE_MAKE_ATTR_INSTANCE(resize_attr_t);
        CASE_MAKE_ATTR_INSTANCE(slice_attr_t);
        CASE_MAKE_ATTR_INSTANCE(softmax_attr_t);
        CASE_MAKE_ATTR_INSTANCE(strided_slice_attr_t);
        CASE_MAKE_ATTR_INSTANCE(transpose_attr_t);
        CASE_MAKE_ATTR_INSTANCE(transpose_conv2d_attr_t);
        CASE_MAKE_ATTR_INSTANCE(while_attr_t);
        CASE_MAKE_ATTR_INSTANCE(mirror_pad_mode_attr_t);
        default:
            assert(false && "No attribute has this reduced hash");
            // Add a new XXX_attr_t struct to the header then
            // insert a new case entry in the statement above
            break;
    };
    return {};
}

}  // namespace regor
