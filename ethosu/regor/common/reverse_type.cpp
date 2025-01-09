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

#include "common/reverse_type.hpp"

#include "common/logging.hpp"

#include "common/bit_flags.hpp"

BEGIN_ENUM_TABLE(regor::ReverseType)
    ADD_ENUM_NAME(None)
    ADD_ENUM_NAME(C)
    ADD_ENUM_NAME(W)
    ADD_ENUM_NAME(H)
    ADD_ENUM_NAME(N)
    ADD_ENUM_NAME(B)
    ADD_ENUM_NAME(A)
    ADD_ENUM_NAME(Dynamic)
END_ENUM_TABLE()

namespace regor
{
// Calculate a bitmask of the reversed axes for the operation
Flags<ReverseType> ToReverseMask(const Shape &shape, int ofmRank)
{
    // Compose a bitmask with all the reverseTypes in shape
    Flags<ReverseType> mask = ReverseType::None;
    for ( int i = 0; i < shape.Size(); i++ )
    {
        // Convert axis to ReverseType
        // ReverseType = 0,1,2,4 represent C,W,H,N respectively
        // Axis = 0,1,2,3 represent the four outer axes of ofmShape
        // Compute a shift based on ofmShape and axis
        int axis = shape[i];
        assert(axis < ofmRank && "ToReverseMask axis out of bounds");
        assert(ofmRank < 32 && "ToReverseMask ofmRank out of bounds");
        unsigned shift = ofmRank - axis - 1;
        assert(shift < unsigned(ofmRank));
        mask |= ReverseType(1UL << shift);
    }
    return mask;
}

}  // namespace regor
